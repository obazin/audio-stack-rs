//! audio-stack-rs: a self-contained, host-agnostic audio backend.
//!
//! Decoding, DSP, output and transport for local files and web radio, plus
//! audio-file metadata parsing — behind one facade. There is no UI, no GUI
//! framework, and no database here: the host injects two traits ([`EventSink`]
//! for outbound transport events + visual frames, [`Store`] for loudness
//! persistence) and drives the [`AudioEngine`] with plain method calls.
//!
//! Threading, and why it is shaped this way:
//!
//! - The **engine thread** owns the `cpal::Stream`, the decoders and the
//!   queue. It is reached only by command over a `crossbeam-channel`, so the
//!   handle held by the host stays `Send + Sync`.
//! - The **cpal callback** is the only realtime context. It pops frames from a
//!   lock-free ring, runs the EQ and gain, and taps a mono copy for the
//!   analyser. It never allocates, locks, blocks — or logs.
//! - A small **owned tokio runtime** drives the detached network tasks
//!   (reconnect, now-playing pollers). The one async entry point,
//!   [`AudioEngine::play_stream`], is driven by the caller's runtime.

mod analyser;
#[cfg(feature = "analysis")]
mod analysis;
mod chain;
mod codecs;
#[cfg(feature = "convolution")]
mod convolution;
mod decode;
mod dsp;
mod engine;
mod events;
#[cfg(feature = "fir-eq")]
mod fireq;
mod loudness;
mod metadata;
mod nowplaying;
#[cfg(feature = "opus")]
mod opus;
mod output;
mod params;
#[cfg(feature = "pitch")]
mod pitch;
mod queue;
mod resample;
mod spectral;
mod stream;
#[cfg(feature = "stretch")]
mod stretch;

mod icy;

#[cfg(test)]
mod fixtures;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crossbeam_channel::Sender;

use engine::{Engine, EngineCommand};

// ── Facade surface (re-exports) ──────────────────────────────────────────────

pub use analyser::FRAME_BYTES;
pub use events::{EngineEvent, Mode};
pub use loudness::{db_to_linear, gain_db, parse_gain_db, parse_peak, Measured, Store};
pub use metadata::{
    audio_extension, read_cover, read_metadata, CoverArt, Metadata, AUDIO_EXTENSIONS,
    LOSSLESS_EXTENSIONS,
};
pub use nowplaying::Source;
pub use output::AudioDevice;
pub use params::{CENTER_FREQS, EQ_BAND_COUNT};
pub use queue::QueueEntry;

/// Where the engine sends what the host renders: transport [`EngineEvent`]s and
/// the raw visual frame (`FRAME_BYTES` bytes, ~60 Hz). The host implements this
/// and forwards each however it likes (an IPC channel, a callback, an `mpsc`).
///
/// Both methods are called from the engine thread and must not block it; a
/// dropped frame is invisible, a stalled send is audible.
pub trait EventSink: Send + Sync {
    /// Delivers one transport/state event.
    fn send_event(&self, event: EngineEvent);
    /// Delivers one visual frame (`FRAME_BYTES` bytes) at ~60 Hz.
    fn send_frame(&self, frame: &[u8]);
}

/// The handle to a running engine. Holds no audio state of its own —
/// everything lives on the engine thread, reached only by command.
pub struct AudioEngine {
    commands: Sender<EngineCommand>,
    params: Arc<params::Params>,
    /// Bumped whenever what is playing changes. A now-playing poller carries
    /// the value it started with and stops as soon as it no longer matches,
    /// which is how a poller for a station the listener has left dies.
    station_epoch: Arc<AtomicU64>,
    /// Cheap clone used to spawn the detached network tasks.
    handle: tokio::runtime::Handle,
    /// The owned runtime, taken and dropped without waiting on
    /// [`AudioEngine::shutdown`] / drop, so a long poll sleep cannot stall exit.
    runtime: Mutex<Option<tokio::runtime::Runtime>>,
    join: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl AudioEngine {
    // ── transport ────────────────────────────────────────────────────────────

    /// Replaces the queue and starts at `index`.
    pub fn load_queue(&self, entries: Vec<QueueEntry>, index: usize) {
        let _ = self.send(EngineCommand::LoadQueue { entries, index });
    }

    /// Connects to a station and hands the buffered reader to the engine.
    ///
    /// The one async entry point: it awaits the HTTP round trip and the initial
    /// prefetch so the engine thread never blocks on the network. It resolves
    /// once the station is buffered, so the caller can tell connecting from
    /// playing. `now_playing` names the station's metadata provider when it has
    /// one; the caller owns the station list, so it is the caller that knows.
    pub async fn play_stream(
        &self,
        station_id: String,
        url: String,
        now_playing: Option<Source>,
    ) -> Result<(), String> {
        // Claim the epoch *before* the connect. It retires the previous
        // station's poller, and it tags this request: the connect takes
        // seconds, and anything the listener starts in that window bumps the
        // epoch, so the engine can recognise this stream as abandoned and drop
        // it instead of tearing down whatever they chose to play instead.
        let epoch = self.next_station_epoch();
        let stream = stream::open(&url).await?;
        self.send(EngineCommand::PlayStream {
            epoch,
            station_id,
            url,
            stream: Box::new(stream),
            has_provider: now_playing.is_some(),
        })?;

        if let Some(source) = now_playing {
            nowplaying::spawn(
                &self.handle,
                self.commands.clone(),
                Arc::clone(&self.station_epoch),
                epoch,
                source,
            );
        }
        Ok(())
    }

    /// Resumes/starts playback of the current queue entry.
    pub fn play(&self) {
        let _ = self.send(EngineCommand::Play);
    }
    /// Pauses playback; the queue position is unchanged.
    pub fn pause(&self) {
        let _ = self.send(EngineCommand::Pause);
    }
    /// Toggles between [`play`](Self::play) and [`pause`](Self::pause).
    pub fn toggle(&self) {
        let _ = self.send(EngineCommand::Toggle);
    }
    /// Stops playback and releases the output device.
    pub fn stop(&self) {
        let _ = self.send(EngineCommand::Stop);
    }
    /// Advances to the next queue entry, honoring repeat/shuffle.
    pub fn next(&self) {
        let _ = self.send(EngineCommand::Next);
    }
    /// Returns to the previous queue entry, or restarts the current one if
    /// far enough into it.
    pub fn previous(&self) {
        let _ = self.send(EngineCommand::Previous);
    }
    /// Jumps directly to the queue entry at `index`.
    pub fn jump_to(&self, index: usize) {
        let _ = self.send(EngineCommand::JumpTo(index));
    }
    /// Seeks the current track to `position_secs`.
    pub fn seek(&self, position_secs: f64) {
        let _ = self.send(EngineCommand::Seek(position_secs));
    }
    /// Enables/disables shuffled queue order.
    pub fn set_shuffle(&self, enabled: bool) {
        let _ = self.send(EngineCommand::SetShuffle(enabled));
    }
    /// Enables/disables repeat of the queue.
    pub fn set_repeat(&self, enabled: bool) {
        let _ = self.send(EngineCommand::SetRepeat(enabled));
    }
    /// Enables/disables EBU R128 loudness normalization.
    pub fn set_normalize(&self, enabled: bool) {
        let _ = self.send(EngineCommand::SetNormalize(enabled));
    }
    /// Enables/disables gapless playback between queue entries.
    pub fn set_gapless(&self, enabled: bool) {
        let _ = self.send(EngineCommand::SetGapless(enabled));
    }
    /// Enables/disables crossfading between queue entries.
    pub fn set_crossfade(&self, enabled: bool) {
        let _ = self.send(EngineCommand::SetCrossfade(enabled));
    }
    /// Switches output to `device_id`, or the system default when `None`.
    pub fn set_device(&self, device_id: Option<String>) {
        let _ = self.send(EngineCommand::SetDevice(device_id));
    }

    /// Enables/disables time-stretch and sets the tempo ratio (1.0 = normal,
    /// 2.0 = double speed, clamped 0.25–2.0). Pitch is unaffected. A change
    /// is heard once the audio already buffered ahead has played — up to
    /// about half a second. Disabling ramps back to normal speed click-free;
    /// the effect leaves the signal path entirely at the next track change
    /// or seek. Echoed as [`EngineEvent::TimeStretch`].
    #[cfg(feature = "stretch")]
    pub fn set_time_stretch(&self, enabled: bool, ratio: f32) {
        let _ = self.send(EngineCommand::SetTimeStretch { enabled, ratio });
    }

    /// Enables/disables the linear-phase FIR EQ. It applies the same ten band
    /// gains as [`set_eq`](Self::set_eq) but with no inter-band phase
    /// distortion, taking over from the realtime biquad EQ while on. The cost
    /// is a constant ~43 ms latency (at 48 kHz): heard audio sits that far
    /// behind the reported position while enabled, and a change — enabling,
    /// disabling, or a slider move — is heard once the ~0.5 s already buffered
    /// has played. Echoed as [`EngineEvent::FirEq`] with the exact latency.
    #[cfg(feature = "fir-eq")]
    pub fn set_fir_eq(&self, enabled: bool) {
        let _ = self.send(EngineCommand::SetFirEq { enabled });
    }

    /// Enables/disables the convolution (impulse-response) effect and sets its
    /// IR and wet/dry mix (0.0 dry … 1.0 fully wet, equal-power). `ir_path` is
    /// any decodable audio file — a reverb, room/headphone-correction, or
    /// per-channel-filter IR; a mono IR applies to every channel, a stereo IR
    /// per channel. The file is decoded and resampled to the device rate on
    /// load; a load failure arrives as [`EngineEvent::Error`] and leaves the
    /// effect bypassed. Echoed as [`EngineEvent::Convolution`]. IRs are capped
    /// at ten seconds.
    #[cfg(feature = "convolution")]
    pub fn set_convolution(&self, enabled: bool, ir_path: Option<std::path::PathBuf>, mix: f32) {
        let _ = self.send(EngineCommand::SetConvolution {
            enabled,
            ir_path,
            mix,
        });
    }

    /// Enables/disables pitch-shift and sets the shift in **cents** (100 cents
    /// = one semitone, clamped to ±1200 = ±one octave). Duration is preserved,
    /// so the playhead is unaffected; disabling ramps back to normal pitch
    /// click-free and the effect leaves the signal path at the next track
    /// change or seek. A change is heard once the buffered audio ahead has
    /// played — up to about half a second. Echoed as [`EngineEvent::PitchShift`].
    #[cfg(feature = "pitch")]
    pub fn set_pitch_shift(&self, enabled: bool, cents: f32) {
        let _ = self.send(EngineCommand::SetPitchShift { enabled, cents });
    }

    /// Re-emit everything the host needs to render current state. Call after a
    /// fresh [`EventSink`] is attached so a reloaded UI catches up with audio
    /// that never stopped.
    pub fn describe(&self) {
        let _ = self.send(EngineCommand::Describe);
    }

    // ── realtime parameters (straight to the atomics) ─────────────────────────

    /// Master volume, `0.0..=1.0`. Audible on the next callback, no round trip
    /// through the engine thread.
    pub fn set_volume(&self, volume: f64) {
        self.params.set_volume(volume as f32);
    }

    /// The ten EQ band gains in dB. Audible on the next callback.
    pub fn set_eq(&self, gains: Vec<f64>) {
        let gains: Vec<f32> = gains.iter().map(|g| *g as f32).collect();
        self.params.set_eq_gains(&gains);
    }

    /// Frames the output callback has handed to the device — the playhead in
    /// device frames. Position normally arrives via [`EngineEvent::Position`];
    /// this is the raw counter, handy for telemetry and tests.
    pub fn frames_played(&self) -> u64 {
        self.params.frames_played()
    }

    // ── output devices ────────────────────────────────────────────────────────

    /// The output devices available, default first.
    pub fn devices(&self) -> Result<Vec<AudioDevice>, String> {
        output::list_devices()
    }

    // ── lifecycle ─────────────────────────────────────────────────────────────

    /// Stops the engine thread and waits for it, so the `cpal::Stream` is
    /// dropped while the process is still alive. Without this, some backends can
    /// call into freed memory during teardown.
    ///
    /// The wait is bounded: the engine thread can be parked inside a blocking
    /// radio read for up to the stream's read timeout, and this may run on a
    /// UI/event-loop thread during exit — an unbounded join there turns "quit
    /// while a station is stalled" into an app that never closes. Past the
    /// deadline the thread is abandoned; that re-opens the teardown hazard above
    /// for that one pathological case, which is the lesser evil. The network
    /// runtime is dropped without waiting, so a long poll sleep cannot stall
    /// exit either.
    pub fn shutdown(&self) {
        const SHUTDOWN_WAIT: std::time::Duration = std::time::Duration::from_secs(3);

        let _ = self.send(EngineCommand::Shutdown);
        if let Ok(mut guard) = self.join.lock() {
            if let Some(handle) = guard.take() {
                let deadline = std::time::Instant::now() + SHUTDOWN_WAIT;
                while !handle.is_finished() && std::time::Instant::now() < deadline {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                if handle.is_finished() {
                    let _ = handle.join();
                } else {
                    log::warn!("audio engine did not stop in time; exiting without it");
                }
            }
        }
        self.drop_runtime();
    }

    // ── internals ─────────────────────────────────────────────────────────────

    /// Claims the next epoch, invalidating any poller still running.
    fn next_station_epoch(&self) -> u64 {
        self.station_epoch.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Queues a command for the engine thread. Never blocks: the channel is
    /// unbounded.
    fn send(&self, command: EngineCommand) -> Result<(), String> {
        self.commands
            .send(command)
            .map_err(|_| "audio engine is not running".to_string())
    }

    /// Drops the network runtime immediately, without waiting for its tasks —
    /// a poller may be mid-sleep for minutes and we never want to block on that.
    fn drop_runtime(&self) {
        if let Ok(mut guard) = self.runtime.lock() {
            if let Some(runtime) = guard.take() {
                runtime.shutdown_background();
            }
        }
    }
}

impl Drop for AudioEngine {
    fn drop(&mut self) {
        // Safety net for the runtime if `shutdown` was never called.
        self.drop_runtime();
    }
}

/// Starts the engine thread. The output device is opened lazily, on first
/// play, so a machine with no sound card still constructs the engine.
///
/// `store` persists measured loudness; `sink` receives transport events and
/// visual frames; `device_id` selects an output by name (default when `None`).
pub fn init(
    store: Arc<dyn Store>,
    sink: Arc<dyn EventSink>,
    device_id: Option<String>,
) -> AudioEngine {
    let (tx, rx) = crossbeam_channel::unbounded();
    let params = Arc::new(params::Params::default());
    let station_epoch = Arc::new(AtomicU64::new(0));

    // A dedicated single-worker runtime for the detached network tasks. One
    // worker is plenty for a couple of HTTP pollers and the occasional
    // reconnect; the realtime audio path never touches it.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .thread_name("audio-stack-net")
        .enable_all()
        .build()
        .expect("build audio-stack network runtime");
    let handle = runtime.handle().clone();

    let engine = Engine::new(
        store,
        rx,
        tx.clone(),
        Arc::clone(&params),
        sink,
        Arc::clone(&station_epoch),
        handle.clone(),
        device_id,
    );
    let join = std::thread::Builder::new()
        .name("audio-stack-engine".into())
        .spawn(move || engine.run())
        .expect("spawn audio engine thread");

    AudioEngine {
        commands: tx,
        params,
        station_epoch,
        handle,
        runtime: Mutex::new(Some(runtime)),
        join: Mutex::new(Some(join)),
    }
}
