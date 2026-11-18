//! The engine thread: owns the output stream, the decoders and the queue.
//!
//! Everything that is `!Sync` or expensive to rebuild lives here, reached only
//! by [`EngineCommand`]. The thread loop is deliberately simple — drain
//! commands, top the ring up, emit whatever the UI is owed, sleep if there was
//! nothing to do — because the hard realtime work happens in the cpal callback
//! and the hard correctness work lives in [`super::queue`].

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, TryRecvError};

use super::analyser::{Analyser, FRAME_BYTES};
use super::chain::Chain;
use super::decode::Decoder;
use super::dsp::remap_channels;
use super::events::{EngineEvent, Mode, SeekDirection};
use super::icy;
use super::loudness::{self, Loudness, Store};
use super::nowplaying::Update;
use super::output::{self, Output};
use super::params::Params;
use super::queue::{Location, Queue, QueueEntry};
use super::resample::Resampler;
use super::source::{self, OpenTask, SourceResolver};
use super::stream::{NowPlaying, RadioStream};
use crate::EventSink;

/// Half a second of stereo audio at 48 kHz. Deep enough to ride out a disk
/// stall, shallow enough that a seek throws away little work.
const RING_SECONDS: f32 = 0.5;
/// Frames pulled from the decoder per pump. Small enough to stay responsive
/// to a seek, large enough that the loop is not all overhead.
const PUMP_FRAMES: usize = 2048;
/// Start opening the next track this long before the current one ends.
const PRELOAD_SECS: f64 = 5.0;
/// How long a crossfade overlaps two tracks. Kept under `PRELOAD_SECS` so
/// the incoming decoder is normally already open by the time this wants it.
const CROSSFADE_SECS: f64 = 4.0;
/// How many times to try getting a dropped station back before giving up and
/// telling the listener.
const MAX_RECONNECT_ATTEMPTS: u32 = 8;
/// Caps the backoff at 2^5 = 32 seconds.
const MAX_RECONNECT_BACKOFF_SHIFT: u32 = 5;
/// Cadence of `Position` events. The frontend interpolates between them.
const POSITION_INTERVAL: Duration = Duration::from_millis(100);

pub enum EngineCommand {
    LoadQueue {
        entries: Vec<QueueEntry>,
        index: usize,
    },
    SetSourceResolver(Arc<dyn SourceResolver>),
    RefreshCurrent {
        position_secs: f64,
        playing: bool,
    },
    RefreshTrack {
        track_id: i64,
        position_secs: f64,
        playing: bool,
    },
    /// Plays an already-connected station. The HTTP round trip happens in the
    /// command layer on Tauri's async runtime, so the engine thread only ever
    /// receives a reader that is ready to decode.
    PlayStream {
        /// The station epoch the command claimed *before* its connect. The
        /// connect can take seconds, and anything the listener starts in
        /// that window bumps the epoch — a stale value means this stream was
        /// abandoned and must be dropped, not played.
        epoch: u64,
        station_id: String,
        /// Kept so a dropped stream can be reopened without the frontend.
        url: String,
        stream: Box<RadioStream>,
        /// True when a now-playing poller was started for this station.
        has_provider: bool,
    },
    Play,
    Pause,
    Toggle,
    Stop,
    Next,
    Previous,
    JumpTo(usize),
    Seek(f64),
    SeekTrack {
        track_id: i64,
        position_secs: f64,
    },
    /// A seek relative to the playhead: `secs` (a magnitude) in `direction`,
    /// clamped to the track.
    SeekBy {
        secs: f64,
        direction: SeekDirection,
    },
    /// Loops `[start_secs, end_secs)`, jumping to the start now and back to
    /// it `repeats` more times on reaching the end (`None`: until cleared).
    SetLoop {
        start_secs: f64,
        end_secs: f64,
        repeats: Option<u32>,
    },
    ClearLoop,
    SetShuffle(bool),
    SetRepeat(bool),
    SetNormalize(bool),
    /// Whether a track boundary joins in-ring with no flush, or flushes to
    /// a deliberate gap. Ignored while crossfading — that always joins.
    SetGapless(bool),
    /// Whether a track boundary overlaps the next track in over
    /// `CROSSFADE_SECS`. Takes priority over `SetGapless` when both apply.
    SetCrossfade(bool),
    SetDevice(Option<String>),
    /// Enables/disables the time-stretch effect and sets its tempo ratio
    /// (1.0 = normal speed). Pitch is unaffected; the ratio is clamped to
    /// the range the effect supports.
    #[cfg(feature = "stretch")]
    SetTimeStretch {
        enabled: bool,
        ratio: f32,
    },
    /// Enables/disables the linear-phase FIR EQ. It applies the same ten band
    /// gains as the callback EQ but with linear phase and ~43 ms latency; while
    /// it is on the callback EQ runs flat so the two do not stack.
    #[cfg(feature = "fir-eq")]
    SetFirEq {
        enabled: bool,
    },
    /// Enables/disables the convolution effect and sets its impulse response
    /// and wet/dry mix. The IR file is decoded and resampled on load; a failure
    /// surfaces as `EngineEvent::Error` and leaves the effect bypassed.
    #[cfg(feature = "convolution")]
    SetConvolution {
        enabled: bool,
        ir_path: Option<std::path::PathBuf>,
        mix: f32,
    },
    /// Enables/disables pitch-shift and sets its cents (±1200 = ±one octave).
    /// Duration-preserving; disabling ramps to normal pitch.
    #[cfg(feature = "pitch")]
    SetPitchShift {
        enabled: bool,
        cents: f32,
    },
    /// A now-playing poller reporting what a station is playing. Carries the
    /// epoch it started under so a late answer about a station the listener
    /// has already left is discarded.
    StationMetadata {
        epoch: u64,
        update: Update,
    },
    /// A reconnect attempt succeeded. Carries the epoch it began under, so a
    /// station the listener has since left is discarded.
    Reconnected {
        epoch: u64,
        station_id: String,
        url: String,
        stream: Box<RadioStream>,
        has_provider: bool,
    },
    /// A reconnect attempt failed; try again if the station is still wanted.
    ReconnectFailed {
        epoch: u64,
    },
    /// Re-emit everything the UI needs to render current state. Sent on
    /// subscribe, so a reloaded webview catches up with audio that never
    /// stopped.
    Describe,
    Shutdown,
}

/// Where a track begins, measured in frames written to the ring. The UI is
/// told a track changed when this many frames have actually reached the
/// device, not when the decoder crossed the boundary — with gapless those are
/// up to half a second apart.
struct Boundary {
    frame: u64,
    index: usize,
    duration_secs: f64,
    /// Source seconds the track is at when this boundary reaches the device.
    /// 0 for the start of a track; the loop start for an A–B loop's jump
    /// back, whose audio resumes mid-track without a counter rebase.
    offset_secs: f64,
    /// The gain this track wants, in dB. Applied when the boundary reaches
    /// the device rather than when the decoder crossed it — at a gapless
    /// join the two are up to a ring-buffer apart, and applying it early
    /// would play the tail of the outgoing track at the next track's volume.
    gain_db: f64,
    /// Consumed once, when this discontinuity reaches the device.
    seeked: Option<(i64, f64)>,
}

/// An A–B loop on the current track. Detected at the decoder, not at the
/// device: the read that would cross `end_secs` is capped at it and the
/// decoder seeks back to `start_secs`, so the jump lands on the exact sample
/// and joins in-ring like a gapless track boundary — nothing is flushed.
struct LoopRegion {
    start_secs: f64,
    end_secs: f64,
    /// Jumps back still to make; `None` repeats until cleared.
    remaining: Option<u32>,
}

/// A crossfade in progress: the next track decoding and mixing into the
/// ring alongside the tail of the current one, so the join has no silence
/// and no click.
///
/// Deliberately carries no gain of its own — see [`Engine::push_pending`]:
/// the callback's existing gain (whatever the outgoing track set) keeps
/// applying to the whole mixed buffer for the crossfade's short span, and
/// the incoming track's own normalization gain takes over once it is fully
/// in, via the boundary [`Engine::complete_transition`] pushes. A track
/// normalized to the target loudness is never far from another, so this is
/// inaudible in practice and avoids a second gain stage in the mix.
struct Transition {
    decoder: Decoder,
    resampler: Option<Resampler>,
    /// The incoming track's own effect chain, mirroring the outgoing one's
    /// settings, adopted wholesale by `complete_transition`.
    chain: Chain,
    entry: QueueEntry,
    /// Where the queue cursor was before this transition advanced it —
    /// restored by `cancel_transition` if something supersedes the fade.
    outgoing_index: usize,
    duration_secs: f64,
    /// Device frames the crossfade should span.
    total_frames: u64,
    elapsed_frames: u64,
    decode_buf: Vec<f32>,
    mapped_buf: Vec<f32>,
    /// Resampled, not yet mixed into the ring.
    pending_in: Vec<f32>,
}

/// Where a seek of `secs` in `direction` from `position` lands, clamped to
/// `[0, duration]` (`duration` is only an upper bound when known, i.e. > 0).
/// `None` when there is nothing to do: a non-finite, zero or negative amount.
fn relative_seek_target(
    position: f64,
    secs: f64,
    direction: SeekDirection,
    duration: f64,
) -> Option<f64> {
    if !secs.is_finite() || secs <= 0.0 {
        return None;
    }
    let target = match direction {
        SeekDirection::Forward => position + secs,
        SeekDirection::Backward => position - secs,
    };
    let target = target.max(0.0);
    Some(if duration > 0.0 {
        target.min(duration)
    } else {
        target
    })
}

/// The cursor and source captured for an asynchronous open. Dropping the task
/// cancels work; checking the identity also protects against a moved cursor.
struct PendingCurrent {
    entry: QueueEntry,
    index: usize,
    task: OpenTask,
    position_secs: f64,
    attempts_left: usize,
    purpose: OpenPurpose,
    notify_seek: bool,
}

#[derive(Clone, Copy)]
enum OpenPurpose {
    Start,
    Advance,
    Refresh,
}

enum PreloadState {
    Opening(OpenTask),
    Failed,
}

pub struct Engine {
    loudness_store: Arc<dyn Store>,
    commands: Receiver<EngineCommand>,
    params: Arc<Params>,
    sink: Arc<dyn EventSink>,
    /// Spawns the detached reconnect task off the engine thread (which has no
    /// async context of its own).
    runtime: tokio::runtime::Handle,

    queue: Queue,
    resolver: Option<Arc<dyn SourceResolver>>,
    pending_current: Option<PendingCurrent>,
    mode: Mode,
    playing: bool,
    station_id: Option<String>,
    /// The playing station's url, so a dropped connection can be reopened.
    station_url: Option<String>,
    /// Consecutive failed reconnects, which sets how long to wait before the
    /// next one. Reset by any successful connection.
    reconnect_attempt: u32,
    /// Lets the engine hand a reopened stream back to itself.
    commands_tx: Sender<EngineCommand>,

    device_id: Option<String>,
    output: Option<Output>,
    producer: Option<rtrb::Producer<f32>>,
    analyser: Option<Analyser>,
    /// Set on Pause/Stop while a flush-driven fade is still in flight — the
    /// stream itself is only actually paused once the callback has serviced
    /// it. Pausing any earlier freezes the hardware output at whatever
    /// arbitrary sample the ring last held; the next `play()` then jumps
    /// straight from that frozen level into new audio with no fade between
    /// them, which is heard as a click at the moment playback resumes.
    pending_pause: bool,

    decoder: Option<Decoder>,
    /// Identity of the last installed local decoder, retained across a reopen.
    decoder_track_id: Option<i64>,
    /// Opened early so a gapless transition costs no probe or allocation.
    preloaded: Option<Decoder>,
    preloaded_entry: Option<QueueEntry>,
    /// Retained after failure so a pump never repeatedly resolves one source.
    preloading: Option<(QueueEntry, PreloadState)>,
    /// Written by the ICY reader as the station announces tracks.
    now_playing: Option<NowPlaying>,
    reported_title: Option<String>,
    /// Whether the normalization gain is applied at all. Consulted at the
    /// moment a gain is set, so the Settings toggle takes effect mid-track
    /// rather than at the next track.
    normalize: bool,
    /// The gain of the track currently being decoded, in dB.
    current_gain_db: f64,
    /// The A–B loop on the current track, if one is set.
    loop_region: Option<LoopRegion>,
    /// Loudness being measured for the playing track, when it has no gain of
    /// its own. Abandoned on a seek — a partial listen measures the wrong
    /// thing.
    measuring: Option<(i64, Loudness)>,
    /// Tempo/key being analysed for the playing local track, on the same
    /// lifecycle as `measuring` but for every local track (not only untagged
    /// ones). Abandoned on a seek — a partial listen is not a whole track.
    #[cfg(feature = "analysis")]
    analysing: Option<(i64, super::analysis::Analysis)>,
    /// Set when the station has a now-playing provider. The provider is then
    /// the only source: merging it with ICY, which often disagrees about
    /// timing, produces worse answers than either feed alone.
    has_provider: bool,
    station_epoch: Arc<AtomicU64>,
    resampler: Option<Resampler>,
    /// The effect chain between the resampler and the ring. Empty (and
    /// byte-transparent) until an effect is switched on.
    chain: Chain,

    /// Whether a track boundary joins in-ring (no flush) or flushes to a
    /// deliberate gap. Toggled by the Settings switch of the same name.
    gapless: bool,
    /// Whether a track boundary overlaps the next track in over
    /// `CROSSFADE_SECS` instead of joining or gapping. Takes priority over
    /// `gapless` when both could apply.
    crossfade: bool,
    /// A crossfade in progress, if any.
    transition: Option<Transition>,

    /// Total frames handed to the ring since the stream was built.
    frames_written: u64,
    boundaries: VecDeque<Boundary>,
    current_duration: f64,
    reported_index: Option<usize>,

    decode_buf: Vec<f32>,
    mapped_buf: Vec<f32>,
    /// Resampled samples that did not fit in the ring last turn.
    pending_out: Vec<f32>,

    last_position: Instant,
    last_frame_push: Instant,
}

impl Engine {
    // The engine is the hub every subsystem is injected into; a builder here
    // would be ceremony over a private constructor called from exactly one
    // place (`crate::init`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        loudness_store: Arc<dyn Store>,
        commands: Receiver<EngineCommand>,
        commands_tx: Sender<EngineCommand>,
        params: Arc<Params>,
        sink: Arc<dyn EventSink>,
        station_epoch: Arc<AtomicU64>,
        runtime: tokio::runtime::Handle,
        device_id: Option<String>,
    ) -> Self {
        Self {
            loudness_store,
            commands,
            params,
            sink,
            runtime,
            queue: Queue::new(),
            resolver: None,
            pending_current: None,
            mode: Mode::Idle,
            playing: false,
            station_id: None,
            station_url: None,
            reconnect_attempt: 0,
            commands_tx,
            device_id,
            output: None,
            producer: None,
            analyser: None,
            pending_pause: false,
            decoder: None,
            decoder_track_id: None,
            preloaded: None,
            preloaded_entry: None,
            preloading: None,
            now_playing: None,
            reported_title: None,
            normalize: true,
            current_gain_db: 0.0,
            loop_region: None,
            measuring: None,
            #[cfg(feature = "analysis")]
            analysing: None,
            has_provider: false,
            station_epoch,
            resampler: None,
            chain: Chain::new(),
            gapless: true,
            crossfade: true,
            transition: None,
            frames_written: 0,
            boundaries: VecDeque::new(),
            current_duration: 0.0,
            reported_index: None,
            decode_buf: Vec::new(),
            mapped_buf: Vec::new(),
            pending_out: Vec::new(),
            last_position: Instant::now(),
            last_frame_push: Instant::now(),
        }
    }

    pub fn run(mut self) {
        loop {
            let mut worked = false;
            loop {
                match self.commands.try_recv() {
                    Ok(EngineCommand::Shutdown) | Err(TryRecvError::Disconnected) => return,
                    Ok(command) => {
                        worked = true;
                        self.handle(command);
                    }
                    Err(TryRecvError::Empty) => break,
                }
            }

            self.service_pending_pause();
            worked |= self.poll_current();
            worked |= self.pump();
            self.emit_progress();
            self.push_visual_frame();

            if !worked {
                // Either idle or the ring is full; either way there is nothing
                // useful to do until the callback drains some of it.
                std::thread::sleep(Duration::from_millis(4));
            }
        }
    }

    // ── commands ────────────────────────────────────────────────────────

    fn handle(&mut self, command: EngineCommand) {
        match command {
            EngineCommand::SetSourceResolver(resolver) => {
                self.resolver = Some(resolver);
                self.cancel_transition();
                self.clear_preload();
                if let Some(pending) = self.pending_current.take() {
                    self.open_current(
                        pending.attempts_left,
                        pending.purpose,
                        pending.position_secs,
                        pending.notify_seek,
                    );
                }
            }
            EngineCommand::RefreshCurrent {
                position_secs,
                playing,
            } => {
                self.refresh_current(position_secs, playing);
            }
            EngineCommand::RefreshTrack {
                track_id,
                position_secs,
                playing,
            } => {
                if self.matches_track(track_id) {
                    self.refresh_current(position_secs, playing);
                }
            }
            EngineCommand::LoadQueue { entries, index } => {
                self.finish_measuring(false);
                self.pending_current = None;
                self.clear_preload();
                // The queue is being replaced wholesale, so a transition's
                // captured cursor position would point nowhere sensible —
                // drop it rather than try to restore it.
                self.transition = None;
                self.station_id = None;
                self.now_playing = None;
                self.reported_title = None;
                self.has_provider = false;
                self.station_epoch.fetch_add(1, Ordering::Relaxed);
                self.queue.load(entries, index);
                if self.queue.is_empty() {
                    self.stop();
                } else {
                    self.mode = Mode::Local;
                    self.start_current();
                }
            }
            EngineCommand::PlayStream {
                epoch,
                station_id,
                url,
                stream,
                has_provider,
            } => {
                // Anything the listener started while this station was
                // connecting — a local queue, another station, a stop —
                // bumped the epoch. A stale stream is one they abandoned;
                // playing it would tear down what they chose instead.
                if epoch != self.station_epoch.load(Ordering::Relaxed) {
                    return;
                }
                // A meter running for the local track that was playing must
                // be abandoned, exactly as LoadQueue does — otherwise it
                // would be fed the station and record the station's loudness
                // against that track's id.
                self.finish_measuring(false);
                // Crossfading is a local-queue concept; switching to radio
                // abandons whatever was fading in, same as LoadQueue.
                self.transition = None;
                self.pending_current = None;
                self.clear_preload();
                self.has_provider = has_provider;
                self.reconnect_attempt = 0;
                self.station_url = Some(url);
                self.start_stream(station_id, *stream);
            }
            EngineCommand::StationMetadata { epoch, update } => {
                // A poller for a station already left behind must not
                // overwrite what is playing now.
                if epoch == self.station_epoch.load(Ordering::Relaxed) {
                    self.emit(EngineEvent::StreamMetadata {
                        title: update.info.as_ref().map(|i| i.title.clone()),
                        artist: update.info.as_ref().and_then(|i| i.artist.clone()),
                        album: update.info.as_ref().and_then(|i| i.album.clone()),
                        cover: update.cover,
                    });
                }
            }
            EngineCommand::Play => self.set_playing(true),
            EngineCommand::Pause => self.set_playing(false),
            EngineCommand::Toggle => self.set_playing(!self.playing),
            EngineCommand::Stop => self.stop(),
            EngineCommand::Next => {
                // A manual skip supersedes any fade in progress — cancel it
                // (restoring the cursor) before advancing from there, so
                // "Next" during a crossfade means what it always means.
                self.cancel_transition();
                if self.mode == Mode::Local && self.queue.advance().is_some() {
                    self.start_current();
                }
            }
            EngineCommand::Previous => {
                self.cancel_transition();
                if self.mode == Mode::Local && self.queue.back().is_some() {
                    self.start_current();
                }
            }
            EngineCommand::JumpTo(index) => {
                self.cancel_transition();
                if self.mode == Mode::Local && self.queue.jump_to(index).is_some() {
                    self.start_current();
                }
            }
            EngineCommand::Seek(secs) => {
                self.seek(secs);
            }
            EngineCommand::SeekTrack {
                track_id,
                position_secs,
            } => {
                if self.matches_track(track_id) {
                    self.seek(position_secs);
                }
            }
            EngineCommand::SeekBy { secs, direction } => self.seek_by(secs, direction),
            EngineCommand::SetLoop {
                start_secs,
                end_secs,
                repeats,
            } => self.set_loop(start_secs, end_secs, repeats),
            EngineCommand::ClearLoop => self.clear_loop(),
            EngineCommand::SetShuffle(enabled) => {
                // A crossfade in progress keeps going: its incoming track is
                // already the cursor, and the order is rebuilt around it. Only
                // the preload, opened for the old successor, goes stale.
                self.clear_preload();
                self.queue.set_shuffle(enabled);
                self.refresh_preload();
                self.emit_state();
            }
            EngineCommand::SetRepeat(enabled) => {
                // A crossfade in progress keeps going: its incoming track is
                // already the cursor, and the order is rebuilt around it. Only
                // the preload, opened for the old successor, goes stale.
                self.clear_preload();
                self.queue.set_repeat(enabled);
                self.refresh_preload();
                self.emit_state();
            }
            EngineCommand::SetNormalize(enabled) => {
                self.normalize = enabled;
                // Re-apply what is playing so the switch is heard now rather
                // than at the next track.
                let gain = self.current_gain_db;
                self.apply_gain(gain);
            }
            EngineCommand::SetGapless(enabled) => {
                // Only ever consulted at the next `advance_or_stop`, so
                // nothing playing right now needs to change.
                self.gapless = enabled;
            }
            EngineCommand::SetCrossfade(enabled) => {
                self.crossfade = enabled;
                if !enabled {
                    // Predictable: turning it off stops a fade that is
                    // already under way, rather than letting it finish.
                    self.cancel_transition();
                }
            }
            #[cfg(feature = "stretch")]
            EngineCommand::SetTimeStretch { enabled, ratio } => {
                if !ratio.is_finite() {
                    return;
                }
                let (min, max) = super::stretch::TEMPO_RATIO_RANGE;
                let ratio = ratio.clamp(min, max);
                if let Err(message) = self.chain.set_time_stretch(enabled, ratio) {
                    self.emit(EngineEvent::Error { message });
                }
                // A crossfade's incoming track mirrors the setting, or it
                // would be adopted at the old tempo.
                if let Some(transition) = self.transition.as_mut() {
                    if let Err(message) = transition.chain.set_time_stretch(enabled, ratio) {
                        self.emit(EngineEvent::Error { message });
                    }
                }
                let (enabled, ratio) = self.chain.time_stretch();
                self.emit(EngineEvent::TimeStretch { enabled, ratio });
            }
            #[cfg(feature = "fir-eq")]
            EngineCommand::SetFirEq { enabled } => {
                // Flatten (or restore) the callback EQ first, so the two never
                // stack: the FIR effect then owns the band gains while on.
                self.params.set_fir_eq_active(enabled);
                if let Err(message) = self.chain.set_fir_eq(enabled, &self.params) {
                    self.emit(EngineEvent::Error { message });
                }
                // A crossfade's incoming track mirrors the setting, or it would
                // be adopted with the effect in the wrong state.
                if let Some(transition) = self.transition.as_mut() {
                    if let Err(message) = transition.chain.set_fir_eq(enabled, &self.params) {
                        self.emit(EngineEvent::Error { message });
                    }
                }
                let (enabled, latency_secs) = self.chain.fir_eq(self.device_rate());
                self.emit(EngineEvent::FirEq {
                    enabled,
                    latency_secs,
                });
            }
            #[cfg(feature = "convolution")]
            EngineCommand::SetConvolution {
                enabled,
                ir_path,
                mix,
            } => {
                if let Err(message) = self.chain.set_convolution(enabled, ir_path.clone(), mix) {
                    self.emit(EngineEvent::Error { message });
                }
                // A crossfade's incoming track mirrors the setting.
                if let Some(transition) = self.transition.as_mut() {
                    if let Err(message) =
                        transition
                            .chain
                            .set_convolution(enabled, ir_path.clone(), mix)
                    {
                        self.emit(EngineEvent::Error { message });
                    }
                }
                let (enabled, mix) = self.chain.convolution();
                self.emit(EngineEvent::Convolution { enabled, mix });
            }
            #[cfg(feature = "pitch")]
            EngineCommand::SetPitchShift { enabled, cents } => {
                if !cents.is_finite() {
                    return;
                }
                if let Err(message) = self.chain.set_pitch_shift(enabled, cents) {
                    self.emit(EngineEvent::Error { message });
                }
                if let Some(transition) = self.transition.as_mut() {
                    if let Err(message) = transition.chain.set_pitch_shift(enabled, cents) {
                        self.emit(EngineEvent::Error { message });
                    }
                }
                let (enabled, cents) = self.chain.pitch_shift();
                self.emit(EngineEvent::PitchShift { enabled, cents });
            }
            EngineCommand::SetDevice(id) => {
                self.device_id = id;
                // Rebuilding the stream is the only way to change device, and
                // it invalidates the ring, so re-seek to where we were.
                let position = self.position_secs();
                let was_playing = self.playing;
                self.output = None;
                self.producer = None;
                self.analyser = None;
                match self.ensure_output() {
                    Ok(()) => {
                        // The new device brings its own rate and channel
                        // count; the resampler must follow, or everything
                        // keeps being converted for the old device — wrong
                        // speed and pitch, and a scrambled interleave when
                        // the channel count changed.
                        if let Some(rate) = self.decoder.as_ref().map(|d| d.format().sample_rate) {
                            self.rebuild_resampler(rate);
                        }
                        // The chain's effects are (device rate, channels)-
                        // shaped too.
                        let (device_rate, device_channels) =
                            (self.device_rate(), self.device_channels() as usize);
                        if let Err(message) = self.chain.reconfigure(device_rate, device_channels) {
                            self.emit(EngineEvent::Error { message });
                        }
                        // Anything already resampled is shaped for the old
                        // device.
                        self.pending_out.clear();
                        if self.decoder.is_some() && self.mode == Mode::Local {
                            self.seek_internal(position, false);
                        }
                        // `output::build` returns a stopped stream and no
                        // other path starts it here (`set_playing` would
                        // early-return, `playing` never changed) — without
                        // this, playback freezes with the transport still
                        // showing playing.
                        if was_playing {
                            self.pending_pause = false;
                            if let Some(output) = self.output.as_ref() {
                                if let Err(message) = output.play() {
                                    self.emit(EngineEvent::Error { message });
                                }
                            }
                        }
                    }
                    Err(message) => {
                        // Surfaced like every other ensure_output failure —
                        // and the transport stops claiming playback that
                        // cannot be happening.
                        self.playing = false;
                        self.emit(EngineEvent::Error { message });
                        self.emit_state();
                    }
                }
            }
            EngineCommand::Reconnected {
                epoch,
                station_id,
                url,
                stream,
                has_provider,
            } => {
                if epoch == self.station_epoch.load(Ordering::Relaxed) {
                    self.has_provider = has_provider;
                    self.station_url = Some(url);
                    if self.start_stream(station_id, *stream) {
                        self.reconnect_attempt = 0;
                    } else {
                        // Connected, but the body does not decode — an HTML
                        // "stream offline" page served with a 200 is common.
                        // That is a failed attempt, not a recovery: keep the
                        // backoff running rather than parking for ever in
                        // playing-with-no-decoder with the counter reset.
                        self.reconnect();
                    }
                }
            }
            EngineCommand::ReconnectFailed { epoch } => {
                if epoch == self.station_epoch.load(Ordering::Relaxed) {
                    self.reconnect();
                }
            }
            EngineCommand::Describe => self.describe(),
            EngineCommand::Shutdown => {
                self.pending_current = None;
                self.clear_preload();
            }
        }
    }

    /// Reopens a station whose connection dropped, backing off between
    /// attempts.
    ///
    /// Stations drop routinely — a server restarts, a CDN node rotates — and
    /// falling silent is the wrong answer to something that fixes itself. The
    /// connect is async, so it happens off this thread and the reopened stream
    /// arrives back as an ordinary command.
    fn reconnect(&mut self) {
        let (Some(station_id), Some(url)) = (self.station_id.clone(), self.station_url.clone())
        else {
            self.stop();
            return;
        };

        if self.reconnect_attempt >= MAX_RECONNECT_ATTEMPTS {
            self.emit(EngineEvent::Error {
                message: format!("{} stopped responding", station_id),
            });
            self.stop();
            return;
        }

        // Doubling from a second, capped — long enough to outlast a restart,
        // short enough that a blip is barely noticed.
        let wait =
            Duration::from_secs(1u64 << self.reconnect_attempt.min(MAX_RECONNECT_BACKOFF_SHIFT));
        self.reconnect_attempt += 1;

        // The decoder is finished with; keep the output running so the ring
        // drains rather than cutting off mid-word.
        self.decoder = None;
        self.preloaded = None;

        let epoch = self.station_epoch.load(Ordering::Relaxed);
        let has_provider = self.has_provider;
        let commands = self.commands_tx.clone();

        self.runtime.spawn(async move {
            tokio::time::sleep(wait).await;
            match super::stream::open(&url).await {
                Ok(stream) => {
                    let _ = commands.send(EngineCommand::Reconnected {
                        epoch,
                        station_id,
                        url,
                        stream: Box::new(stream),
                        has_provider,
                    });
                }
                Err(message) => {
                    log::warn!("reconnect to {}: {}", station_id, message);
                    // Ask the engine to try again; it owns the attempt count.
                    let _ = commands.send(EngineCommand::ReconnectFailed { epoch });
                }
            }
        });
    }

    /// Begins measuring the playing track's loudness, when it is a local
    /// track that nothing has measured and no tag describes.
    ///
    /// Free: the engine is decoding these samples anyway. A track listened to
    /// end to end fills in its own gain for next time.
    fn start_measuring(&mut self, format: &super::decode::SourceFormat) {
        self.measuring = None;
        #[cfg(feature = "analysis")]
        {
            self.analysing = None;
        }
        if self.mode != Mode::Local {
            return;
        }
        let Some(entry) = self.queue.current() else {
            return;
        };
        let track_id = entry.track_id;
        // Tempo/key run for every local track, so this sits before the
        // loudness store's needs_measurement gate below.
        #[cfg(feature = "analysis")]
        if let Some(analysis) = super::analysis::Analysis::new(format.sample_rate, format.channels)
        {
            self.analysing = Some((track_id, analysis));
        }
        if !self.loudness_store.needs_measurement(track_id) {
            return;
        }
        if let Some(meter) = Loudness::new(format.sample_rate, format.channels) {
            self.measuring = Some((track_id, meter));
        }
    }

    /// Stores the measurement if the track was heard all the way through.
    ///
    /// Anything that skipped audio — a seek, a manual skip — abandons it
    /// instead: a partial listen measures the wrong thing, and a wrong gain
    /// is worse than none.
    fn finish_measuring(&mut self, complete: bool) {
        // Tempo/key: reported only for a full listen, then the state is dropped
        // either way (a partial listen is not a whole track).
        #[cfg(feature = "analysis")]
        if let Some((track_id, analysis)) = self.analysing.take() {
            if complete {
                let result = analysis.finish();
                self.emit(EngineEvent::TrackAnalysis {
                    track_id,
                    bpm: result.bpm,
                    bpm_confidence: result.bpm_confidence,
                    key: result.key,
                    key_confidence: result.key_confidence,
                });
            }
        }

        let Some((track_id, meter)) = self.measuring.take() else {
            return;
        };
        if !complete {
            return;
        }
        if let Some(measured) = meter.finish() {
            self.loudness_store.record(track_id, measured);
        }
    }

    /// Sets the mixer's gain for the track now audible, honouring the
    /// normalization switch. The clamp caps a boost at +12 dB so a broken tag
    /// cannot blow the output up.
    fn apply_gain(&mut self, gain_db: f64) {
        self.current_gain_db = gain_db;
        let linear = if self.normalize {
            loudness::db_to_linear(gain_db).clamp(0.0, 4.0)
        } else {
            1.0
        };
        self.params.set_track_gain(linear);
    }

    /// Replays current state onto a freshly attached subscriber.
    fn describe(&mut self) {
        if let Some(output) = self.output.as_ref() {
            self.sink.send_event(EngineEvent::Device {
                name: output.config.device_name.clone(),
                sample_rate: output.config.sample_rate,
                channels: output.config.channels,
            });
        }
        if let Some(format) = self.decoder.as_ref().map(|d| d.format().clone()) {
            self.emit(EngineEvent::Format {
                sample_rate: format.sample_rate,
                channels: format.channels,
                codec: format.codec,
            });
        }
        self.emit_loop();
        #[cfg(feature = "stretch")]
        {
            let (enabled, ratio) = self.chain.time_stretch();
            self.emit(EngineEvent::TimeStretch { enabled, ratio });
        }
        #[cfg(feature = "fir-eq")]
        {
            let (enabled, latency_secs) = self.chain.fir_eq(self.device_rate());
            self.emit(EngineEvent::FirEq {
                enabled,
                latency_secs,
            });
        }
        #[cfg(feature = "convolution")]
        {
            let (enabled, mix) = self.chain.convolution();
            self.emit(EngineEvent::Convolution { enabled, mix });
        }
        #[cfg(feature = "pitch")]
        {
            let (enabled, cents) = self.chain.pitch_shift();
            self.emit(EngineEvent::PitchShift { enabled, cents });
        }
        // Before `State`, whose index refers into this queue. The `State`
        // event alone cannot rebuild a reloaded webview's track list.
        if !self.queue.is_empty() {
            self.emit(EngineEvent::Queue {
                track_ids: self.queue.track_ids(),
            });
        }
        self.emit_state();
        self.emit_progress_now();
    }

    fn set_playing(&mut self, playing: bool) {
        if self.mode == Mode::Idle || self.playing == playing {
            return;
        }
        if playing
            && self.mode == Mode::Local
            && self.decoder.is_none()
            && self.pending_current.is_none()
        {
            // The queue played out earlier: the decoder is gone but the mode
            // was kept so the UI still shows the last track. Play means play
            // that track again — flipping the flag alone would report
            // `playing` over a pump with nothing to decode, a state with no
            // way out.
            self.start_current();
            return;
        }
        self.playing = playing;
        if playing {
            self.pending_pause = false;
            if let Some(output) = self.output.as_ref() {
                if let Err(message) = output.play() {
                    self.emit(EngineEvent::Error { message });
                }
            }
        } else {
            // See `pending_pause`: fade what is already in the ring instead
            // of freezing the stream on it outright.
            self.fade_then_pause();
        }
        self.emit_state();
    }

    fn stop(&mut self) {
        self.finish_measuring(false);
        self.pending_current = None;
        // Going idle; no cursor to preserve for a cancelled fade.
        self.transition = None;
        self.playing = false;
        self.mode = Mode::Idle;
        self.station_id = None;
        self.station_url = None;
        self.now_playing = None;
        self.reported_title = None;
        self.has_provider = false;
        self.station_epoch.fetch_add(1, Ordering::Relaxed);
        self.decoder = None;
        self.decoder_track_id = None;
        self.preloaded = None;
        self.preloaded_entry = None;
        self.preloading = None;
        self.pending_out.clear();
        self.chain.reset();
        self.boundaries.clear();
        self.clear_loop();
        self.fade_then_pause();
        self.emit_state();
    }

    /// Requests a flush — which fades whatever is already in the ring to
    /// silence rather than cutting it off, see [`super::params::Params`] —
    /// and defers the actual stream pause to [`Self::service_pending_pause`]
    /// once the callback has serviced it. See `pending_pause`.
    fn fade_then_pause(&mut self) {
        self.params.request_flush();
        self.pending_pause = true;
    }

    /// Finishes a pause once its fade has completed. Called once per turn of
    /// the run loop, since the callback (not this thread) is what clears the
    /// flush flag, typically within one device period.
    fn service_pending_pause(&mut self) {
        if self.pending_pause && !self.params.flush_pending() {
            self.pending_pause = false;
            if let Some(output) = self.output.as_ref() {
                let _ = output.pause();
            }
        }
    }

    /// Check before cancelling a crossfade: its cursor already points at the
    /// incoming entry, but a seek still acts on the outgoing decoder.
    fn matches_track(&self, track_id: i64) -> bool {
        if self.mode != Mode::Local {
            return false;
        }
        if self.transition.is_some() {
            return self.decoder_track_id == Some(track_id);
        }
        self.queue
            .current()
            .is_some_and(|entry| entry.track_id == track_id)
    }

    /// Returns whether the seek landed or a pending open accepted the target.
    fn seek(&mut self, secs: f64) -> bool {
        self.seek_internal(secs, true)
    }

    fn seek_internal(&mut self, secs: f64, notify: bool) -> bool {
        if self.mode != Mode::Local {
            return false;
        }
        // A seek acts on the track the listener is looking at — the
        // outgoing one, if a crossfade happens to be running — so any fade
        // in progress is cancelled first (restoring the cursor to it).
        self.cancel_transition();
        if let Some(pending) = self.pending_current.take() {
            if !secs.is_finite() || secs < 0.0 {
                self.pending_current = Some(pending);
                return false;
            }
            self.finish_measuring(false);
            self.open_current(pending.attempts_left, pending.purpose, secs, notify);
            self.emit_progress_now();
            return true;
        }
        let Some(decoder) = self.decoder.as_mut() else {
            return false;
        };
        // Whatever has been measured so far no longer describes a full
        // listen.
        self.measuring = None;
        #[cfg(feature = "analysis")]
        {
            self.analysing = None;
        }
        match decoder.seek(secs) {
            Ok(landed) => {
                // Drop everything already decoded past the old position, and
                // rebase the frame counter so the playhead jumps rather than
                // counting up from where it was.
                self.params.request_flush();
                self.pending_out.clear();
                self.chain.reset();
                self.boundaries.clear();
                let rate = self.device_rate().max(1) as f64;
                self.frames_written = (landed * rate) as u64;
                self.params.reset_frames(self.frames_written);
                // The counter was rebased to the landing point, so the
                // track's origin in that timeline is frame 0 — reusing
                // `frames_written` here would make `position_secs` read
                // `played - base = 0` for the rest of the track.
                self.boundaries.push_back(Boundary {
                    frame: 0,
                    index: self.queue.index(),
                    duration_secs: self.current_duration,
                    gain_db: self.current_gain_db,
                    offset_secs: 0.0,
                    seeked: None,
                });
                if notify {
                    self.emit_seeked(landed);
                }
                self.emit_progress_now();
                true
            }
            Err(message) => {
                self.emit(EngineEvent::Error { message });
                false
            }
        }
    }

    fn emit_seeked(&self, position_secs: f64) {
        if let Some(entry) = self.queue.current().filter(|_| self.mode == Mode::Local) {
            self.emit(EngineEvent::Seeked {
                track_id: entry.track_id,
                position_secs,
            });
        }
    }

    /// Seeks `secs` forward or backward from the playhead. The playhead is
    /// what the listener hears, not where the decoder has run ahead to, so
    /// "ten seconds back" lands ten seconds before the audio coming out of
    /// the speakers. During a crossfade `position_secs` still describes the
    /// outgoing track (no boundary is pushed until the fade completes),
    /// which is the track `seek` acts on once it cancels the fade — so the
    /// origin and the target agree.
    fn seek_by(&mut self, secs: f64, direction: SeekDirection) {
        if self.mode != Mode::Local || (self.decoder.is_none() && self.pending_current.is_none()) {
            return;
        }
        let Some(target) =
            relative_seek_target(self.position_secs(), secs, direction, self.current_duration)
        else {
            return;
        };
        self.seek(target);
    }

    // ── A–B loop ────────────────────────────────────────────────────────

    fn set_loop(&mut self, start_secs: f64, end_secs: f64, repeats: Option<u32>) {
        if self.mode != Mode::Local || self.decoder.is_none() {
            return;
        }
        let valid = start_secs.is_finite()
            && end_secs.is_finite()
            && start_secs >= 0.0
            && end_secs > start_secs;
        if !valid {
            self.emit(EngineEvent::Error {
                message: format!("loop: invalid region {start_secs}s to {end_secs}s"),
            });
            return;
        }
        self.loop_region = Some(LoopRegion {
            start_secs,
            end_secs,
            remaining: repeats,
        });
        // The first pass starts from the top, however far in the playhead
        // was. A start the track cannot reach leaves nothing to loop.
        if !self.seek(start_secs) {
            self.loop_region = None;
            return;
        }
        self.emit_loop();
    }

    fn clear_loop(&mut self) {
        if self.loop_region.take().is_some() {
            self.emit_loop();
        }
    }

    fn emit_loop(&mut self) {
        let (enabled, start_secs, end_secs, repeats_left) = match self.loop_region.as_ref() {
            Some(region) => (true, region.start_secs, region.end_secs, region.remaining),
            None => (false, 0.0, 0.0, None),
        };
        self.emit(EngineEvent::Loop {
            enabled,
            start_secs,
            end_secs,
            repeats_left,
        });
    }

    /// How many source frames the next decoder read may take before it
    /// crosses the loop end — `None` when no loop applies to where the
    /// decoder is. "Applies" means the decoder sits inside the region: a
    /// seek that lands outside it plays on past the end untouched, rather
    /// than snapping back the moment it is noticed.
    fn loop_read_frames(&self) -> Option<usize> {
        let region = self.loop_region.as_ref()?;
        let decoder = self.decoder.as_ref()?;
        let position = decoder.position_secs();
        let rate = decoder.format().sample_rate.max(1) as f64;
        // Half a frame of slack on both edges: the landing position is a
        // sum of packet time and sample offset, and an ulp short of the
        // start must still count as being at it.
        let half_frame = 0.5 / rate;
        if position + half_frame < region.start_secs || position + half_frame >= region.end_secs {
            return None;
        }
        // Rounded, not ceiled: a float hair over a whole frame must not read
        // one frame too many, and an end between two frames may land on
        // either.
        Some((((region.end_secs - position) * rate).round() as usize).max(1))
    }

    /// Whether a decoder that was inside the region has now reached its end
    /// — or the end of the track, for a region that runs past it.
    fn loop_end_reached(&self) -> bool {
        let (Some(region), Some(decoder)) = (self.loop_region.as_ref(), self.decoder.as_ref())
        else {
            return false;
        };
        let half_frame = 0.5 / decoder.format().sample_rate.max(1) as f64;
        decoder.is_exhausted() || decoder.position_secs() + half_frame >= region.end_secs
    }

    /// The jump back to the loop start, at the decoder. No flush: the audio
    /// already decoded up to the loop end is still to be heard, so the
    /// looped audio joins it in the ring exactly as a gapless track boundary
    /// does, and a boundary marks where the playhead reads `start_secs` again.
    fn loop_back(&mut self) {
        let Some(region) = self.loop_region.as_mut() else {
            return;
        };
        match region.remaining {
            Some(0) => {
                self.clear_loop();
                return;
            }
            Some(left) => region.remaining = Some(left - 1),
            None => {}
        }
        let start_secs = region.start_secs;
        // A repeated section is not a full listen.
        self.measuring = None;
        #[cfg(feature = "analysis")]
        {
            self.analysing = None;
        }
        let landed = match self.decoder.as_mut().map(|d| d.seek(start_secs)) {
            Some(Ok(landed)) => landed,
            Some(Err(message)) => {
                self.emit(EngineEvent::Error { message });
                self.clear_loop();
                return;
            }
            None => return,
        };
        // Same placement as the gapless join: past whatever is still waiting
        // to go into the ring, which all belongs to the pass just finished.
        let pending_frames = self.pending_out.len() as u64 / self.device_channels().max(1) as u64
            + self.chain.pending_output_frames();
        self.boundaries.push_back(Boundary {
            frame: self.frames_written + pending_frames,
            index: self.queue.index(),
            duration_secs: self.current_duration,
            gain_db: self.current_gain_db,
            offset_secs: landed,
            seeked: self.queue.current().map(|entry| (entry.track_id, landed)),
        });
        self.emit_loop();
    }

    // ── track lifecycle ─────────────────────────────────────────────────

    fn start_current(&mut self) {
        self.pending_current = None;
        self.clear_preload();
        self.pending_pause = false;
        self.finish_measuring(false);
        self.decoder = None;
        self.pending_out.clear();
        self.chain.reset();
        self.boundaries.clear();
        self.params.request_flush();
        self.playing = true;
        self.open_current(self.queue.len(), OpenPurpose::Start, 0.0, false);
    }

    /// Carries the one-pass failure bound across asynchronous opens as well.
    fn open_current(
        &mut self,
        mut attempts: usize,
        purpose: OpenPurpose,
        position: f64,
        mut notify_seek: bool,
    ) {
        loop {
            let Some(entry) = self.queue.current().cloned() else {
                self.stop();
                return;
            };
            if !matches!(purpose, OpenPurpose::Refresh) {
                self.current_duration = entry.duration_secs;
            }
            if matches!(purpose, OpenPurpose::Refresh)
                || position > 0.0
                || matches!(entry.location, Location::Deferred(_))
            {
                self.pending_current = Some(PendingCurrent {
                    task: source::open_in_background(
                        entry.location.clone(),
                        self.resolver.clone(),
                        self.runtime.clone(),
                        position,
                    ),
                    entry,
                    index: self.queue.index(),
                    position_secs: position,
                    attempts_left: attempts,
                    purpose,
                    notify_seek,
                });
                self.emit_state();
                return;
            }
            match self.open_entry(&entry) {
                Ok(decoder) => {
                    self.accept_current(decoder, entry, purpose, position, notify_seek);
                    return;
                }
                Err(message) => {
                    self.emit(EngineEvent::Error { message });
                    attempts = attempts.saturating_sub(1);
                    if attempts == 0 || self.queue.advance().is_none() {
                        self.stop();
                        return;
                    }
                    notify_seek = false;
                }
            }
        }
    }

    fn poll_current(&mut self) -> bool {
        let Some(pending) = self.pending_current.as_ref() else {
            return false;
        };
        if self.mode != Mode::Local
            || self.queue.index() != pending.index
            || !self.queue.current().is_some_and(|entry| {
                entry.track_id == pending.entry.track_id && entry.location == pending.entry.location
            })
        {
            self.pending_current = None;
            return true;
        }
        let result = match pending.task.try_recv() {
            Ok(result) => result,
            Err(std::sync::mpsc::TryRecvError::Empty) => return false,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                Err("source opener disconnected".to_owned())
            }
        };
        let pending = self.pending_current.take().expect("pending open");
        match result {
            Ok(opened) => self.accept_current(
                opened.decoder,
                pending.entry,
                pending.purpose,
                opened.position_secs,
                pending.notify_seek,
            ),
            Err(message) => {
                self.emit(EngineEvent::Error { message });
                if matches!(pending.purpose, OpenPurpose::Refresh) {
                    self.playing = false;
                    self.fade_then_pause();
                    self.emit_state();
                } else {
                    let attempts = pending.attempts_left.saturating_sub(1);
                    if attempts == 0 || self.queue.advance().is_none() {
                        self.stop();
                    } else {
                        self.open_current(attempts, pending.purpose, 0.0, false);
                    }
                }
            }
        }
        true
    }

    fn accept_current(
        &mut self,
        decoder: Decoder,
        entry: QueueEntry,
        purpose: OpenPurpose,
        position: f64,
        notify_seek: bool,
    ) {
        // Natural handovers already own output. Initial opens create it only
        // after a decoder is ready.
        if !matches!(purpose, OpenPurpose::Advance) {
            if let Err(message) = self.ensure_output() {
                self.emit(EngineEvent::Error { message });
                self.playing = false;
                self.emit_state();
                return;
            }
        }
        self.current_duration = decoder
            .format()
            .duration_secs
            .unwrap_or(entry.duration_secs);
        if matches!(purpose, OpenPurpose::Advance) && self.gapless && position == 0.0 {
            self.install_joined_decoder(decoder, entry.gain_db);
        } else {
            self.install_decoder_at(
                decoder,
                entry.gain_db,
                position,
                matches!(purpose, OpenPurpose::Refresh),
            );
        }
        if notify_seek {
            // An explicit seek supersedes any pending same-track restart.
            if let Some(boundary) = self.boundaries.back_mut() {
                boundary.seeked = None;
            }
            self.emit_seeked(position);
        }
        if matches!(purpose, OpenPurpose::Refresh) || position > 0.0 {
            self.finish_measuring(false);
        }
        if self.playing {
            self.begin_playback();
        } else {
            self.fade_then_pause();
            self.emit_state();
        }
    }

    fn refresh_current(&mut self, position: f64, playing: bool) {
        if self.mode != Mode::Local || !position.is_finite() || position < 0.0 {
            return;
        }
        self.cancel_transition();
        self.pending_current = None;
        self.clear_preload();
        self.finish_measuring(false);
        self.decoder = None;
        self.pending_out.clear();
        self.chain.reset();
        self.params.request_flush();
        self.frames_written = 0;
        self.params.reset_frames(0);
        self.boundaries.clear();
        self.boundaries.push_back(Boundary {
            frame: 0,
            index: self.queue.index(),
            duration_secs: self.current_duration,
            gain_db: self.current_gain_db,
            offset_secs: position,
            seeked: None,
        });
        self.playing = playing;
        if !playing {
            self.fade_then_pause();
        }
        self.open_current(1, OpenPurpose::Refresh, position, true);
        self.emit_progress_now();
    }

    /// Returns whether the station actually started playing, so a reconnect
    /// can tell a recovery from a stream that connected but does not decode.
    fn start_stream(&mut self, station_id: String, stream: RadioStream) -> bool {
        if let Err(message) = self.ensure_output() {
            self.emit(EngineEvent::Error { message });
            return false;
        }
        match Decoder::open(stream.source, stream.hint) {
            Ok(decoder) => {
                // The mode flips before `install_decoder`: `start_measuring`
                // guards on it, and with the old order the guard still read
                // `Local` — so a meter keyed to the previous local track was
                // built and fed the station's audio.
                self.mode = Mode::Radio;
                // A station has no length and no normalization gain.
                self.current_duration = 0.0;
                self.install_decoder(decoder, 0.0);
                self.station_id = Some(station_id);
                self.now_playing = Some(stream.now_playing);
                self.reported_title = None;
                self.begin_playback();
                true
            }
            Err(message) => {
                self.emit(EngineEvent::Error { message });
                false
            }
        }
    }

    /// Swaps in a freshly opened decoder and resets everything derived from
    /// the old one.
    fn install_decoder(&mut self, decoder: Decoder, gain_db: f64) {
        self.install_decoder_at(decoder, gain_db, 0.0, false);
    }

    fn install_decoder_at(
        &mut self,
        decoder: Decoder,
        gain_db: f64,
        position: f64,
        preserve_identity: bool,
    ) {
        let track_id = self
            .queue
            .current()
            .filter(|_| self.mode == Mode::Local)
            .map(|entry| entry.track_id);
        let restart = !preserve_identity && track_id.is_some() && self.decoder_track_id == track_id;
        self.decoder_track_id = track_id;
        // Defensive: every caller already clears this on its own path, but
        // a decoder swap and a crossfade cannot coexist.
        self.transition = None;
        self.clear_loop();
        let format = decoder.format().clone();
        self.start_measuring(&format);
        self.apply_gain(gain_db);
        self.rebuild_resampler(format.sample_rate);
        // The chain learns the device shape here (its effects are shaped by
        // it, not by the source) and starts the new track with fresh state.
        let (device_rate, device_channels) = (self.device_rate(), self.device_channels() as usize);
        if let Err(message) = self.chain.reconfigure(device_rate, device_channels) {
            self.emit(EngineEvent::Error { message });
        }
        self.decoder = Some(decoder);
        self.preloaded = None;
        self.preloaded_entry = None;
        self.preloading = None;

        self.params.request_flush();
        self.pending_out.clear();
        self.boundaries.clear();
        self.frames_written = 0;
        self.params.reset_frames(0);
        if !preserve_identity {
            self.reported_index = None;
        }
        self.boundaries.push_back(Boundary {
            frame: 0,
            index: self.queue.index(),
            duration_secs: self.current_duration,
            gain_db: self.current_gain_db,
            offset_secs: position,
            seeked: track_id.filter(|_| restart).map(|id| (id, position)),
        });

        self.emit(EngineEvent::Format {
            sample_rate: format.sample_rate,
            channels: format.channels,
            codec: format.codec,
        });
    }

    fn begin_playback(&mut self) {
        self.playing = true;
        // A fresh decoder means new, wanted audio is about to flow — any
        // pause `install_decoder`'s own flush is still fading out is moot.
        self.pending_pause = false;
        if let Some(output) = self.output.as_ref() {
            if let Err(message) = output.play() {
                self.emit(EngineEvent::Error { message });
            }
        }
        self.refresh_preload();
        self.emit_state();
    }

    /// Opens the next track early so a gapless join costs no probe.
    fn refresh_preload(&mut self) {
        if self.mode != Mode::Local || self.pending_current.is_some() {
            self.clear_preload();
            return;
        }
        let Some(next) = self.queue.peek_next().cloned() else {
            self.clear_preload();
            return;
        };
        if self
            .preloaded_entry
            .as_ref()
            .is_some_and(|entry| entry.track_id != next.track_id || entry.location != next.location)
            || self.preloading.as_ref().is_some_and(|(entry, _)| {
                entry.track_id != next.track_id || entry.location != next.location
            })
        {
            self.clear_preload();
        }
        let remaining =
            (self.current_duration - self.position_secs()) / self.chain.time_ratio().max(0.01);
        if self.preloaded.is_some() || remaining > PRELOAD_SECS {
            return;
        }
        match &next.location {
            Location::File(path) => match Decoder::open_file(path) {
                Ok(decoder) => {
                    self.preloaded = Some(decoder);
                    self.preloaded_entry = Some(next);
                }
                Err(message) => log::warn!("preload failed: {}", message),
            },
            Location::Remote(_) | Location::Deferred(_) => {
                if let Some((_, state)) = self.preloading.as_mut() {
                    let PreloadState::Opening(task) = state else {
                        return;
                    };
                    match task.try_recv() {
                        Ok(Ok(opened)) => {
                            self.preloaded = Some(opened.decoder);
                            self.preloaded_entry = Some(next);
                            self.preloading = None;
                        }
                        Ok(Err(message)) => {
                            log::warn!("preload failed: {}", message);
                            *state = PreloadState::Failed;
                        }
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                            log::warn!("preload failed: source opener disconnected");
                            *state = PreloadState::Failed;
                        }
                        Err(std::sync::mpsc::TryRecvError::Empty) => {}
                    }
                } else {
                    let task = source::open_in_background(
                        next.location.clone(),
                        self.resolver.clone(),
                        self.runtime.clone(),
                        0.0,
                    );
                    self.preloading = Some((next, PreloadState::Opening(task)));
                }
            }
        }
    }

    fn clear_preload(&mut self) {
        self.preloaded = None;
        self.preloaded_entry = None;
        self.preloading = None;
    }

    fn take_preloaded(&mut self, entry: &QueueEntry) -> Option<Decoder> {
        let matches = self.preloaded_entry.as_ref().is_some_and(|loaded| {
            loaded.track_id == entry.track_id && loaded.location == entry.location
        });
        self.preloaded_entry = None;
        let decoder = self.preloaded.take();
        if matches {
            decoder
        } else {
            None
        }
    }

    /// Opens a queued track where it lives. A remote one waits for the
    /// server, which is only done here where a decoder is needed now.
    fn open_entry(&self, entry: &QueueEntry) -> Result<Decoder, String> {
        source::open_blocking(&entry.location, self.resolver.clone(), &self.runtime)
    }

    // ── the pump ────────────────────────────────────────────────────────

    /// Moves audio from the decoder into the ring. Returns whether it made
    /// progress, so the caller knows whether to sleep.
    fn pump(&mut self) -> bool {
        if !self.playing || self.decoder.is_none() {
            return false;
        }
        // The flush handshake: while a flush is pending the callback is about
        // to drain the ring, so anything written now would be swept away with
        // the stale audio (and desync `frames_written` from `frames_played`
        // for the rest of the track). Hold off until the callback lowers the
        // flag; it does so within one device period.
        if self.params.flush_pending() {
            return false;
        }
        let channels = self.device_channels() as usize;
        let Some(producer) = self.producer.as_ref() else {
            return false;
        };
        if producer.slots() < PUMP_FRAMES * channels {
            return false;
        }

        let source_channels = self
            .decoder
            .as_ref()
            .map(|d| d.format().channels as usize)
            .unwrap_or(channels)
            .max(1);

        // An effect can produce more frames than it was fed (time-stretch
        // slowing down), so ring room alone no longer bounds `pending_out`.
        // Hold the decoder until the backlog clears rather than piling up
        // latency; without effects the backlog never reaches this.
        let backlog = self.pending_out.len() >= PUMP_FRAMES * channels * 2;

        // An A–B loop caps the read at the loop end, so the jump back below
        // lands on the exact sample rather than up to a pump buffer late.
        let loop_frames = self.loop_read_frames();
        let frames = PUMP_FRAMES.min(loop_frames.unwrap_or(PUMP_FRAMES));
        self.decode_buf.resize(frames * source_channels, 0.0);
        let read = if backlog {
            0
        } else {
            match self.decoder.as_mut().unwrap().read(&mut self.decode_buf) {
                Ok(read) => read,
                Err(message) => {
                    self.emit(EngineEvent::Error { message });
                    self.advance_or_stop();
                    return true;
                }
            }
        };

        if read > 0 {
            // Measured at the source's own rate and channel count: the answer
            // should describe the file, not whatever device it happens to be
            // playing through.
            if let Some((_, meter)) = self.measuring.as_mut() {
                meter.feed(&self.decode_buf[..read]);
            }
            #[cfg(feature = "analysis")]
            if let Some((_, analysis)) = self.analysing.as_mut() {
                analysis.feed(&self.decode_buf[..read]);
            }

            self.mapped_buf.clear();
            remap_channels(
                &self.decode_buf[..read],
                source_channels,
                channels,
                &mut self.mapped_buf,
            );

            // `pending_out` is appended to, never cleared here: upsampling
            // produces more frames than were read, so the ring may not have
            // room for all of them this turn. The remainder waits rather than
            // being dropped, which would be an audible gap.
            let result = self.chain.process(
                self.resampler.as_mut(),
                &self.mapped_buf,
                &mut self.pending_out,
                self.frames_written,
            );
            if let Err(message) = result {
                self.emit(EngineEvent::Error { message });
            }
        }

        if loop_frames.is_some() && self.loop_end_reached() {
            self.loop_back();
        }

        let pushed = self.push_pending(channels);

        // An exhausted decoder can leave a tail inside the chain's effects,
        // invisible to the emptiness check below. Flush it — except into a
        // gapless join, where the chain survives the handover and its tail
        // flows seamlessly into the next track instead (mirroring the kept
        // resampler in `retune_resampler_for_join`).
        if self.decoder.as_ref().is_some_and(|d| d.is_exhausted()) {
            let gapless_join = self.gapless
                && self.mode == Mode::Local
                && self.transition.is_none()
                && self.queue.peek_next().is_some();
            if !gapless_join {
                if let Err(message) = self.chain.drain(&mut self.pending_out) {
                    self.emit(EngineEvent::Error { message });
                }
            }
        }

        // Only move on once the tail of this track has actually been handed
        // over — otherwise the last fraction of a second is lost, which is
        // exactly the join gapless is supposed to make seamless.
        let exhausted =
            self.decoder.as_ref().is_some_and(|d| d.is_exhausted()) && self.pending_out.is_empty();
        if exhausted && self.transition.is_some() {
            self.complete_transition();
        } else if exhausted {
            self.advance_or_stop();
        } else if self.transition.is_none() && !self.maybe_start_transition() {
            self.refresh_preload();
        }

        read > 0 || pushed
    }

    /// Moves as much of `pending_out` into the ring as fits, keeping the
    /// rest. During a crossfade, mixes it against the incoming track's own
    /// resampled buffer first — see [`Transition`].
    ///
    /// Always writes a whole number of frames: the ring is a flat stream of
    /// interleaved samples, so committing a partial frame would shift every
    /// later frame's channel alignment and swap the stereo image.
    fn push_pending(&mut self, channels: usize) -> bool {
        if channels == 0 {
            return false;
        }
        if self.transition.is_some() {
            self.fill_transition(channels);
        }
        if self.pending_out.is_empty() {
            return false;
        }
        let Some(producer) = self.producer.as_mut() else {
            return false;
        };

        let mut room = (producer.slots().min(self.pending_out.len()) / channels) * channels;
        if let Some(transition) = self.transition.as_ref() {
            // Never write more than the incoming track has ready — a
            // partial mix would need frames that do not exist yet.
            room = room.min((transition.pending_in.len() / channels) * channels);
        }
        if room == 0 {
            return false;
        }
        let Ok(chunk) = producer.write_chunk_uninit(room) else {
            return false;
        };

        if let Some(transition) = self.transition.as_mut() {
            let total = transition.total_frames.max(1) as f32;
            let start_frame = transition.elapsed_frames;
            let outgoing = &self.pending_out[..room];
            let incoming = &transition.pending_in[..room];
            chunk.fill_from_iter((0..room).map(|i| {
                let t = ((start_frame + (i / channels) as u64) as f32 / total).min(1.0);
                let (fade_out, fade_in) = equal_power(t);
                outgoing[i] * fade_out + incoming[i] * fade_in
            }));
            transition.pending_in.drain(..room);
            transition.elapsed_frames += (room / channels) as u64;
        } else {
            chunk.fill_from_iter(self.pending_out[..room].iter().copied());
        }

        self.pending_out.drain(..room);
        self.frames_written += (room / channels) as u64;
        true
    }

    /// Tops up the incoming track's resampled buffer, mirroring the decode
    /// → remap → resample steps `pump` runs for the outgoing decoder.
    fn fill_transition(&mut self, channels: usize) {
        let mut error = None;
        if let Some(transition) = self.transition.as_mut() {
            if transition.pending_in.len() / channels.max(1) < PUMP_FRAMES {
                let source_channels = (transition.decoder.format().channels as usize).max(1);
                transition
                    .decode_buf
                    .resize(PUMP_FRAMES * source_channels, 0.0);
                match transition.decoder.read(&mut transition.decode_buf) {
                    Ok(read) if read > 0 => {
                        transition.mapped_buf.clear();
                        remap_channels(
                            &transition.decode_buf[..read],
                            source_channels,
                            channels,
                            &mut transition.mapped_buf,
                        );
                        // Ring position 0 is a placeholder: a mirror chain
                        // records no timeline markers until it is adopted.
                        let result = transition.chain.process(
                            transition.resampler.as_mut(),
                            &transition.mapped_buf,
                            &mut transition.pending_in,
                            0,
                        );
                        if let Err(message) = result {
                            error = Some(message);
                        }
                    }
                    Ok(_) => {}
                    Err(message) => error = Some(message),
                }
            }
        }
        if let Some(message) = error {
            self.emit(EngineEvent::Error { message });
        }
    }

    /// Starts crossfading the next track in, once the current one is close
    /// enough to its end. Consumes the preloaded decoder when one is ready,
    /// so this costs no probe on the common path.
    fn maybe_start_transition(&mut self) -> bool {
        // A loop keeps the track from ending, so a fade into the next one
        // would pull the listener out of it.
        if !self.crossfade
            || self.transition.is_some()
            || self.mode != Mode::Local
            || self.loop_region.is_some()
        {
            return false;
        }
        let Some(next) = self.queue.peek_next().cloned() else {
            return false;
        };
        // `remaining` is source time; the fade spans wall-clock time, and a
        // time-changing effect makes them differ by its ratio.
        let remaining =
            (self.current_duration - self.position_secs()) / self.chain.time_ratio().max(0.01);
        if remaining > CROSSFADE_SECS {
            return false;
        }

        self.refresh_preload();
        let decoder = match self.take_preloaded(&next) {
            Some(decoder) => decoder,
            None if matches!(next.location, Location::Remote(_) | Location::Deferred(_)) => {
                // Never stall the ring on resolution, HTTP, or probing. If
                // the worker misses this window the natural join will wait.
                return false;
            }
            None => match self.open_entry(&next) {
                Ok(decoder) => decoder,
                Err(message) => {
                    log::warn!("crossfade preload failed: {}", message);
                    return false;
                }
            },
        };
        self.preloading = None;

        let format = decoder.format().clone();
        let device_rate = self.device_rate();
        let channels = self.device_channels() as usize;
        let resampler = match open_resampler(format.sample_rate, device_rate, channels) {
            Ok(resampler) => resampler,
            Err(message) => {
                self.emit(EngineEvent::Error { message });
                return false;
            }
        };

        // Spans whatever is actually left of the outgoing track, capped at
        // the target duration — a track shorter than the crossfade window
        // still gets a full, click-free fade rather than an abrupt cut.
        let total_frames =
            (remaining.clamp(0.05, CROSSFADE_SECS) * device_rate.max(1) as f64) as u64;

        let outgoing_index = self.queue.index();
        self.queue.advance();

        self.transition = Some(Transition {
            duration_secs: format.duration_secs.unwrap_or(next.duration_secs),
            decoder,
            resampler,
            chain: self.chain.spawn_mirror(),
            entry: next,
            outgoing_index,
            total_frames: total_frames.max(1),
            elapsed_frames: 0,
            decode_buf: Vec::new(),
            mapped_buf: Vec::new(),
            pending_in: Vec::new(),
        });
        true
    }

    /// Finishes an in-progress crossfade: the incoming decoder takes over
    /// as the decoder of record, indistinguishable from an ordinary track
    /// that just started — decoding, resampling and (from the boundary
    /// below) gain all follow the normal single-track path from here.
    fn complete_transition(&mut self) {
        let Some(transition) = self.transition.take() else {
            return;
        };
        // The outgoing track played in full — a crossfade overlaps the next
        // track in, it never cuts the outgoing one short.
        self.finish_measuring(true);

        self.decoder = Some(transition.decoder);
        self.resampler = transition.resampler;
        self.chain = transition.chain;
        self.clear_loop();
        // The adopted mirror starts its timeline fresh from the ring
        // position — the same origin approximation the boundary below makes.
        self.chain.adopt_timeline();
        self.pending_out = transition.pending_in;
        self.current_duration = transition.duration_secs;
        let restart = self.decoder_track_id == Some(transition.entry.track_id);
        self.decoder_track_id = Some(transition.entry.track_id);

        // See `Transition`'s doc comment: the mix carried no gain of its
        // own, so this boundary is what hands normalization back to the
        // callback for the incoming track, exactly as a gapless join does.
        self.boundaries.push_back(Boundary {
            frame: self.frames_written,
            index: self.queue.index(),
            duration_secs: self.current_duration,
            gain_db: transition.entry.gain_db,
            offset_secs: 0.0,
            seeked: restart.then_some((transition.entry.track_id, 0.0)),
        });

        if let Some(format) = self.decoder.as_ref().map(|d| d.format().clone()) {
            self.start_measuring(&format);
            self.emit(EngineEvent::Format {
                sample_rate: format.sample_rate,
                channels: format.channels,
                codec: format.codec,
            });
        }
        self.refresh_preload();
    }

    /// Drops an in-progress crossfade and restores the queue cursor, as if
    /// it had never started. Called by any command that supersedes a
    /// transition — a seek, a manual skip, a device change — so those act
    /// on the track the listener actually asked for rather than the one
    /// quietly fading in.
    fn cancel_transition(&mut self) {
        if let Some(transition) = self.transition.take() {
            self.queue.jump_to(transition.outgoing_index);
            self.emit_state();
        }
    }

    /// End of a track: hand over to the preloaded decoder without flushing —
    /// that continuity is what makes the join gapless.
    fn advance_or_stop(&mut self) {
        // Reaching here from an exhausted decoder means the whole track was
        // decoded, which is the only case worth recording.
        let complete = self.decoder.as_ref().is_some_and(|d| d.is_exhausted());
        self.finish_measuring(complete);

        if self.mode == Mode::Radio {
            // A station does not end; the connection dropped. Get it back
            // rather than falling silent.
            self.reconnect();
            return;
        }
        if self.mode != Mode::Local {
            self.stop();
            return;
        }

        let Some(next) = self.queue.peek_next().cloned() else {
            self.decoder = None;
            self.clear_preload();
            self.playing = false;
            self.emit_state();
            return;
        };
        let preloaded = self.take_preloaded(&next);
        let preloading = self
            .preloading
            .take()
            .and_then(|(entry, state)| match state {
                PreloadState::Opening(task)
                    if entry.track_id == next.track_id && entry.location == next.location =>
                {
                    Some((entry, task))
                }
                _ => None,
            });
        self.queue.advance();
        self.current_duration = next.duration_secs;
        self.decoder = None;
        if let Some(decoder) = preloaded {
            self.current_duration = decoder.format().duration_secs.unwrap_or(next.duration_secs);
            if self.gapless {
                self.install_joined_decoder(decoder, next.gain_db);
            } else {
                self.install_decoder(decoder, next.gain_db);
            }
            self.refresh_preload();
        } else if let Some((entry, task)) = preloading {
            self.pending_current = Some(PendingCurrent {
                entry,
                index: self.queue.index(),
                task,
                position_secs: 0.0,
                attempts_left: self.queue.len(),
                purpose: OpenPurpose::Advance,
                notify_seek: false,
            });
        } else {
            self.open_current(self.queue.len(), OpenPurpose::Advance, 0.0, false);
        }
    }

    /// Keeps the ring and its timeline intact at a natural queue boundary.
    fn install_joined_decoder(&mut self, decoder: Decoder, gain_db: f64) {
        let track_id = self.queue.current().map(|entry| entry.track_id);
        let restart = track_id.is_some() && self.decoder_track_id == track_id;
        self.decoder_track_id = track_id;
        let format = decoder.format().clone();
        self.retune_resampler_for_join(format.sample_rate);
        self.decoder = Some(decoder);
        self.clear_preload();
        self.start_measuring(&format);
        self.clear_loop();
        let pending_frames = self.pending_out.len() as u64 / self.device_channels().max(1) as u64
            + self.chain.pending_output_frames();
        self.boundaries.push_back(Boundary {
            frame: self.frames_written + pending_frames,
            index: self.queue.index(),
            duration_secs: self.current_duration,
            gain_db,
            offset_secs: 0.0,
            seeked: track_id.filter(|_| restart).map(|id| (id, 0.0)),
        });
        self.emit(EngineEvent::Format {
            sample_rate: format.sample_rate,
            channels: format.channels,
            codec: format.codec,
        });
    }

    // ── output plumbing ─────────────────────────────────────────────────

    fn ensure_output(&mut self) -> Result<(), String> {
        if self.output.is_some() {
            return Ok(());
        }
        let rate_hint = 48_000;
        let ring = (rate_hint as f32 * RING_SECONDS) as usize * 2;
        let (producer, consumer) = rtrb::RingBuffer::<f32>::new(ring.max(PUMP_FRAMES * 4));
        // Roughly 170 ms of mono at 48 kHz: the FFT window plus headroom.
        let (tap_producer, tap_consumer) = rtrb::RingBuffer::<f32>::new(8192);

        let output = output::open(
            self.device_id.as_deref(),
            Arc::clone(&self.params),
            consumer,
            tap_producer,
        )?;

        self.emit(EngineEvent::Device {
            name: output.config.device_name.clone(),
            sample_rate: output.config.sample_rate,
            channels: output.config.channels,
        });

        self.producer = Some(producer);
        self.analyser = Some(Analyser::new(tap_consumer));
        self.output = Some(output);
        Ok(())
    }

    /// The resampler policy at a gapless join. A matching resampler is kept —
    /// its staged frames are the tail of the outgoing track, and keeping them
    /// is what makes a same-rate album join truly seamless. On a real change
    /// the old instance is drained into `pending_out` first, so those tail
    /// frames are heard instead of vanishing with it.
    fn retune_resampler_for_join(&mut self, source_rate: u32) {
        let device_rate = self.device_rate();
        let channels = self.device_channels() as usize;
        match self.resampler.as_ref() {
            Some(existing) if existing.matches(source_rate, device_rate, channels) => return,
            None if source_rate == device_rate && device_rate != 0 => return,
            _ => {}
        }
        if let Some(resampler) = self.resampler.as_mut() {
            if let Err(message) = resampler.drain(&mut self.pending_out) {
                self.emit(EngineEvent::Error { message });
            }
        }
        self.rebuild_resampler(source_rate);
    }

    fn rebuild_resampler(&mut self, source_rate: u32) {
        let device_rate = self.device_rate();
        let channels = self.device_channels() as usize;
        match open_resampler(source_rate, device_rate, channels) {
            Ok(resampler) => self.resampler = resampler,
            Err(message) => {
                self.emit(EngineEvent::Error { message });
                self.resampler = None;
            }
        }
    }

    fn device_rate(&self) -> u32 {
        self.output
            .as_ref()
            .map(|o| o.config.sample_rate)
            .unwrap_or(0)
    }

    fn device_channels(&self) -> u16 {
        self.output.as_ref().map(|o| o.config.channels).unwrap_or(2)
    }

    // ── reporting ───────────────────────────────────────────────────────

    /// Seconds into the current track, derived from what the callback has
    /// actually played rather than from what the decoder has run ahead to.
    ///
    /// At 1× this is a plain division: one device frame played is one source
    /// frame of musical time. A time-changing effect breaks that, so when the
    /// chain has recorded timeline markers the position is the difference in
    /// mapped source seconds instead; frames no marker covers were 1:1 by
    /// construction and keep the division.
    fn position_secs(&self) -> f64 {
        if let Some(pending) = self.pending_current.as_ref() {
            return pending.position_secs;
        }
        let rate = self.device_rate();
        if rate == 0 {
            return self.boundaries.back().map(|b| b.offset_secs).unwrap_or(0.0);
        }
        let played = self.params.frames_played();
        let origin = self.boundaries.iter().rev().find(|b| b.frame <= played);
        let base = origin.map(|b| b.frame).unwrap_or(0);
        // Where in the track the origin sits: 0 for a track start, the loop
        // start for a loop's jump back.
        let offset = origin.map(|b| b.offset_secs).unwrap_or(0.0);
        let since_origin = match (self.chain.stream_secs(played), self.chain.stream_secs(base)) {
            (None, None) => (played.saturating_sub(base)) as f64 / rate as f64,
            (played_secs, base_secs) => {
                let played_secs = played_secs.unwrap_or(played as f64 / rate as f64);
                let base_secs = base_secs.unwrap_or(base as f64 / rate as f64);
                (played_secs - base_secs).max(0.0)
            }
        };
        offset + since_origin
    }

    fn emit_progress(&mut self) {
        if self.last_position.elapsed() < POSITION_INTERVAL {
            return;
        }
        self.emit_progress_now();
    }

    fn emit_progress_now(&mut self) {
        self.last_position = Instant::now();

        // Drop every boundary the device has already passed except the most
        // recent, which stays as the origin for `position_secs`. Announcing
        // the crossing here rather than when the decoder reached it is what
        // makes the UI flip at the moment the listener hears the new track.
        let played = self.params.frames_played();
        while self.boundaries.len() > 1 && self.boundaries[1].frame <= played {
            if let Some((track_id, position_secs)) = self
                .boundaries
                .front_mut()
                .and_then(|boundary| boundary.seeked.take())
            {
                self.emit(EngineEvent::Seeked {
                    track_id,
                    position_secs,
                });
            }
            self.boundaries.pop_front();
        }
        // The timeline only ever gets looked up at `played` and at the
        // current boundary origin; markers behind both are done with.
        let keep_from = self
            .boundaries
            .front()
            .map(|b| b.frame.min(played))
            .unwrap_or(played);
        self.chain.prune_timeline(keep_from);

        if let Some(boundary) = self.boundaries.front().filter(|b| b.frame <= played) {
            let (index, duration, gain_db) =
                (boundary.index, boundary.duration_secs, boundary.gain_db);
            self.current_duration = duration;
            if self.reported_index != Some(index) {
                self.reported_index = Some(index);
                // The new track is only now reaching the speakers, so this is
                // the moment its gain becomes the right one. `Gain` ramps
                // across the buffer, so the change is inaudible.
                self.apply_gain(gain_db);
                self.emit(EngineEvent::TrackChanged { index });
                // Decode can already be several short tracks ahead. Keep the
                // announced identity at this audible boundary, but capabilities
                // still describe what a command can execute at the queue cursor.
                self.emit_state_at(index);
            }
        }
        if let Some((track_id, position_secs)) = self
            .boundaries
            .front_mut()
            .filter(|boundary| boundary.frame <= played)
            .and_then(|boundary| boundary.seeked.take())
        {
            self.emit(EngineEvent::Seeked {
                track_id,
                position_secs,
            });
        }

        self.emit_stream_metadata();

        if self.mode != Mode::Idle {
            self.emit(EngineEvent::Position {
                position_secs: self.position_secs(),
                duration_secs: if self.mode == Mode::Radio {
                    0.0
                } else {
                    self.current_duration
                },
            });
        }
    }

    /// Announces a station's current track when it changes.
    ///
    /// Skipped entirely when a now-playing provider is polling: it reports the
    /// same track with more detail, and two sources racing would flicker.
    fn emit_stream_metadata(&mut self) {
        if self.has_provider {
            return;
        }
        let Some(source) = self.now_playing.as_ref() else {
            return;
        };
        // Clone out and release the lock before emitting — the ICY reader
        // takes this same lock from inside a decode read.
        let title = match source.lock() {
            Ok(guard) => guard.clone(),
            Err(_) => return,
        };
        if title == self.reported_title {
            return;
        }
        self.reported_title = title.clone();

        let parsed = title.as_deref().and_then(icy::parse);
        self.emit(EngineEvent::StreamMetadata {
            title: parsed.as_ref().map(|p| p.title.clone()),
            artist: parsed.as_ref().and_then(|p| p.artist.clone()),
            album: parsed.as_ref().and_then(|p| p.album.clone()),
            cover: None,
        });
    }

    /// Builds and sends one visualiser frame, at roughly 60 Hz.
    fn push_visual_frame(&mut self) {
        let Some(analyser) = self.analyser.as_mut() else {
            return;
        };
        analyser.drain();
        if self.last_frame_push.elapsed() < Duration::from_millis(16) {
            return;
        }
        self.last_frame_push = Instant::now();
        let frame = analyser.frame();
        debug_assert_eq!(frame.len(), FRAME_BYTES);
        self.sink.send_frame(&frame);
    }

    fn emit_state(&self) {
        self.emit_state_at(self.queue.index());
    }

    fn emit_state_at(&self, index: usize) {
        self.emit(EngineEvent::State {
            playing: self.playing,
            mode: self.mode,
            index,
            queue_len: self.queue.len(),
            shuffle: self.queue.shuffle(),
            repeat: self.queue.repeat(),
            has_next: self.mode == Mode::Local && self.queue.peek_next().is_some(),
            station_id: self.station_id.clone(),
        });
    }

    fn emit(&self, event: EngineEvent) {
        self.sink.send_event(event);
    }
}

/// Builds a resampler for one source rate against the device, or `None`
/// when the rates already match. Shared by the main decode path and a
/// crossfade transition, which each need their own independent instance.
fn open_resampler(
    source_rate: u32,
    device_rate: u32,
    channels: usize,
) -> Result<Option<Resampler>, String> {
    if source_rate == 0 || device_rate == 0 {
        return Ok(None);
    }
    let resampler = Resampler::new(source_rate, device_rate, channels)?;
    Ok(if resampler.is_passthrough() {
        None
    } else {
        Some(resampler)
    })
}

/// Equal-power crossfade curve: `fade_out` and `fade_in` at `t` (0..=1) sum
/// their squares to a constant, so the mix does not dip in perceived
/// loudness at the midpoint the way a straight linear fade would.
fn equal_power(t: f32) -> (f32, f32) {
    let angle = t.clamp(0.0, 1.0) * std::f32::consts::FRAC_PI_2;
    (angle.cos(), angle.sin())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;
    use std::future::Future;
    use std::io::Write;

    /// These tests are about audio reaching the device, not about remembering
    /// how loud it was.
    fn no_store() -> Arc<dyn Store> {
        Arc::new(super::super::loudness::NoStore)
    }

    /// A no-op sink: the transport tests don't inspect emitted events.
    struct NoSink;
    impl crate::EventSink for NoSink {
        fn send_event(&self, _event: EngineEvent) {}
        fn send_frame(&self, _frame: &[u8]) {}
    }

    fn no_sink() -> Arc<dyn crate::EventSink> {
        Arc::new(NoSink)
    }

    /// A shared runtime handle for the few tests that spawn network work. The
    /// pure transport tests never touch it.
    fn test_runtime() -> tokio::runtime::Handle {
        use std::sync::OnceLock;
        static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
        RT.get_or_init(|| tokio::runtime::Runtime::new().expect("test runtime"))
            .handle()
            .clone()
    }

    /// An engine with no output device, no real sink and no thread — the
    /// transport logic (queue advance, seek bookkeeping, boundary crossing)
    /// needs none of them, so it can be exercised directly.
    fn bare_engine() -> Engine {
        bare_engine_with(no_store())
    }

    fn bare_engine_with(store: Arc<dyn Store>) -> Engine {
        let (tx, rx) = crossbeam_channel::unbounded();
        Engine::new(
            store,
            rx,
            tx,
            Arc::new(Params::default()),
            no_sink(),
            Arc::new(AtomicU64::new(0)),
            test_runtime(),
            None,
        )
    }

    /// Collects emitted events, for the tests that assert on the echo.
    struct CaptureSink(std::sync::Mutex<Vec<EngineEvent>>);

    impl crate::EventSink for CaptureSink {
        fn send_event(&self, event: EngineEvent) {
            self.0.lock().unwrap().push(event);
        }
        fn send_frame(&self, _frame: &[u8]) {}
    }

    type ResolveReply = tokio::sync::oneshot::Sender<Result<crate::RemoteFile, String>>;

    #[derive(Debug)]
    struct ResolveCall {
        id: String,
        reply: ResolveReply,
        thread: std::thread::ThreadId,
    }

    struct ControlledResolver(std::sync::mpsc::Sender<ResolveCall>);

    impl SourceResolver for ControlledResolver {
        fn resolve<'a>(
            &'a self,
            id: &'a str,
        ) -> std::pin::Pin<Box<dyn Future<Output = Result<crate::RemoteFile, String>> + Send + 'a>>
        {
            Box::pin(async move {
                let (reply, receiver) = tokio::sync::oneshot::channel();
                self.0
                    .send(ResolveCall {
                        id: id.to_owned(),
                        reply,
                        thread: std::thread::current().id(),
                    })
                    .expect("test receives calls");
                receiver
                    .await
                    .unwrap_or_else(|_| Err("cancelled by test".to_owned()))
            })
        }
    }

    fn controlled_resolver() -> (
        Arc<dyn SourceResolver>,
        std::sync::mpsc::Receiver<ResolveCall>,
    ) {
        let (sender, receiver) = std::sync::mpsc::channel();
        (Arc::new(ControlledResolver(sender)), receiver)
    }

    fn deferred_entry(track_id: i64, id: &str) -> QueueEntry {
        QueueEntry {
            track_id,
            location: Location::Deferred(crate::DeferredSource { id: id.to_owned() }),
            duration_secs: 1.0,
            gain_db: 0.0,
        }
    }

    fn next_resolve(calls: &std::sync::mpsc::Receiver<ResolveCall>) -> ResolveCall {
        calls
            .recv_timeout(Duration::from_secs(5))
            .expect("resolver is polled")
    }

    fn assert_cancelled(mut reply: ResolveReply) {
        test_runtime().block_on(async {
            tokio::time::timeout(Duration::from_secs(5), reply.closed())
                .await
                .expect("superseded resolution is dropped");
        });
    }

    fn until(mut done: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !done() {
            assert!(Instant::now() < deadline, "worker did not complete");
            std::thread::yield_now();
        }
    }

    #[test]
    fn deferred_current_is_cancellable_and_never_resolves_on_the_engine_thread() {
        let mut engine = bare_engine();
        let (resolver, calls) = controlled_resolver();
        engine.handle(EngineCommand::SetSourceResolver(resolver));
        engine.handle(EngineCommand::LoadQueue {
            entries: vec![deferred_entry(1, "first"), deferred_entry(2, "second")],
            index: 0,
        });
        let first = next_resolve(&calls);
        assert_eq!(first.id, "first");
        assert_ne!(first.thread, std::thread::current().id());
        assert!(engine.output.is_none(), "resolution precedes output setup");
        engine.handle(EngineCommand::Next);
        let second = next_resolve(&calls);
        assert_eq!(second.id, "second");
        assert_cancelled(first.reply);
        engine.handle(EngineCommand::Stop);
        assert_cancelled(second.reply);
        assert!(!engine.poll_current());
        assert!(engine.decoder.is_none());
        assert_eq!(engine.mode, Mode::Idle);
    }

    #[test]
    fn replacing_the_resolver_restarts_only_the_wanted_open() {
        let mut engine = bare_engine();
        let (resolver, old_calls) = controlled_resolver();
        engine.handle(EngineCommand::SetSourceResolver(resolver));
        engine.handle(EngineCommand::LoadQueue {
            entries: vec![deferred_entry(1, "stable")],
            index: 0,
        });
        let old = next_resolve(&old_calls);
        let (replacement, new_calls) = controlled_resolver();
        engine.handle(EngineCommand::SetSourceResolver(replacement));
        let new = next_resolve(&new_calls);
        assert_cancelled(old.reply);
        assert_eq!(new.id, "stable");
        engine.stop();
        assert_cancelled(new.reply);
    }

    #[test]
    fn repeating_deferred_failures_try_one_queue_pass() {
        let (mut engine, sink) = captured_engine();
        let (resolver, calls) = controlled_resolver();
        engine.handle(EngineCommand::SetSourceResolver(resolver));
        engine.queue.set_repeat(true);
        engine.handle(EngineCommand::LoadQueue {
            entries: vec![deferred_entry(1, "first"), deferred_entry(2, "second")],
            index: 0,
        });
        for (id, track_id) in [("first", 1), ("second", 2)] {
            let call = next_resolve(&calls);
            assert_eq!(call.id, id);
            call.reply.send(Err("unavailable".to_owned())).unwrap();
            until(|| {
                engine.poll_current();
                engine.mode == Mode::Idle
                    || engine
                        .pending_current
                        .as_ref()
                        .is_some_and(|p| p.entry.track_id != track_id)
            });
        }
        assert_eq!(engine.mode, Mode::Idle);
        assert!(!engine.playing);
        assert!(calls.try_recv().is_err());
        assert_eq!(
            sink.0
                .lock()
                .unwrap()
                .iter()
                .filter(|event| { matches!(event, EngineEvent::Error { .. }) })
                .count(),
            2
        );
    }

    #[test]
    fn a_natural_transition_adopts_the_pending_preload_without_resolving_again() {
        let mut engine = bare_engine();
        let (resolver, calls) = controlled_resolver();
        engine.resolver = Some(resolver);
        engine.queue.load(
            vec![deferred_entry(1, "outgoing"), deferred_entry(2, "incoming")],
            0,
        );
        engine.mode = Mode::Local;
        engine.playing = true;
        engine.current_duration = 1.0;
        engine.refresh_preload();
        let call = next_resolve(&calls);
        assert_eq!(call.id, "incoming");
        assert!(
            !engine.maybe_start_transition(),
            "a blocked resolver cannot stall the ring"
        );
        engine.advance_or_stop();
        assert!(engine.pending_current.is_some());
        for _ in 0..10 {
            assert!(!engine.poll_current());
            assert!(!engine.pump());
        }
        assert!(
            calls.try_recv().is_err(),
            "one actual open, not one per pump"
        );
        engine.stop();
        assert_cancelled(call.reply);
    }

    #[test]
    fn deferred_preload_decodes_real_audio_and_keeps_the_queue_identity() {
        let mut engine = bare_engine();
        let (resolver, calls) = controlled_resolver();
        engine.resolver = Some(resolver);
        let incoming = deferred_entry(2, "incoming");
        engine
            .queue
            .load(vec![deferred_entry(1, "outgoing"), incoming.clone()], 0);
        engine.mode = Mode::Local;
        engine.current_duration = 1.0;
        engine.refresh_preload();
        let call = next_resolve(&calls);
        assert_eq!(call.id, "incoming");
        let address = crate::remote::test_server::serve(fixtures::wav_bytes(
            8_000,
            1,
            &fixtures::silence(1, 8_000),
        ));
        call.reply
            .send(Ok(crate::RemoteFile {
                url: format!("{address}/track.wav"),
                extension: Some("wav".to_owned()),
                bounded_range: None,
            }))
            .unwrap();
        until(|| {
            engine.refresh_preload();
            engine.preloaded.is_some()
        });
        assert_eq!(engine.preloaded_entry.as_ref(), Some(&incoming));
        assert_eq!(engine.queue.peek_next(), Some(&incoming));
        engine.advance_or_stop();
        assert_eq!(engine.queue.current(), Some(&incoming));
        assert_eq!(engine.decoder.as_ref().unwrap().format().sample_rate, 8_000);
        assert!(calls.try_recv().is_err());
    }

    #[test]
    fn a_same_track_id_with_a_new_location_discards_the_old_preload() {
        let mut engine = bare_engine();
        let (resolver, calls) = controlled_resolver();
        engine.resolver = Some(resolver);
        engine.mode = Mode::Local;
        engine.current_duration = 1.0;
        engine.queue.load(
            vec![deferred_entry(1, "current"), deferred_entry(2, "old")],
            0,
        );
        engine.refresh_preload();
        let old = next_resolve(&calls);
        engine.queue.load(
            vec![deferred_entry(1, "current"), deferred_entry(2, "new")],
            0,
        );
        engine.refresh_preload();
        let new = next_resolve(&calls);
        assert_eq!(new.id, "new");
        assert_cancelled(old.reply);
        engine.stop();
        assert_cancelled(new.reply);
    }

    #[test]
    fn refresh_preserves_pause_position_queue_and_reported_identity_while_pending() {
        let (mut engine, sink) = captured_engine();
        let (resolver, calls) = controlled_resolver();
        engine.resolver = Some(resolver);
        engine
            .queue
            .load(vec![deferred_entry(1, "one"), deferred_entry(2, "two")], 0);
        engine.queue.set_shuffle(true);
        engine.queue.set_repeat(true);
        engine.mode = Mode::Local;
        engine.reported_index = Some(engine.queue.index());
        let index = engine.queue.index();
        let next = engine.queue.peek_next().cloned();
        engine.handle(EngineCommand::RefreshCurrent {
            position_secs: 0.5,
            playing: false,
        });
        let first = next_resolve(&calls);
        assert!(!engine.playing);
        assert_eq!(engine.position_secs(), 0.5);
        engine.set_playing(true);
        engine.set_playing(false);
        assert!(engine.seek(0.75));
        let second = next_resolve(&calls);
        assert_cancelled(first.reply);
        assert!(!engine.playing);
        assert_eq!(engine.position_secs(), 0.75);
        assert_eq!(engine.reported_index, Some(index));
        assert_eq!(engine.queue.index(), index);
        assert_eq!(engine.queue.peek_next(), next.as_ref());
        assert!(engine.queue.shuffle() && engine.queue.repeat());
        assert!(!sink
            .0
            .lock()
            .unwrap()
            .iter()
            .any(|event| matches!(event, EngineEvent::TrackChanged { .. })));
        engine.stop();
        assert_cancelled(second.reply);
    }

    #[test]
    #[ignore = "requires an audio output device"]
    fn refresh_completion_keeps_pause_position_identity_and_discards_partial_metering() {
        let (path, entry) = wav_track(1, "audio-stack-refresh-worker");
        let (mut engine, sink) = captured_engine();
        engine.loudness_store = Arc::new(EagerStore);
        engine.params.set_volume(0.0);
        engine
            .ensure_output()
            .expect("an output device is available");
        engine.queue.load(vec![entry.clone()], 0);
        engine.mode = Mode::Local;
        let decoder = Decoder::open_file(&path).expect("initial native decoder");
        engine.current_duration = decoder.format().duration_secs.unwrap();
        engine.install_decoder(decoder, entry.gain_db);
        engine.reported_index = Some(0);
        assert!(
            engine.measuring.is_some(),
            "a full initial listen is being measured"
        );

        sink.0.lock().unwrap().clear();
        engine.handle(EngineCommand::RefreshTrack {
            track_id: entry.track_id,
            position_secs: 0.5,
            playing: false,
        });
        assert!(
            seeked_events(&sink).is_empty(),
            "opening is not a confirmation"
        );
        until(|| {
            engine.poll_current();
            engine.pending_current.is_none()
        });
        let decoder = engine
            .decoder
            .as_ref()
            .expect("refresh installed its decoder");
        assert!((decoder.position_secs() - 0.5).abs() < 0.01);
        let landed = decoder.position_secs();
        assert_eq!(seeked_events(&sink), vec![(entry.track_id, landed)]);
        engine.emit_progress_now();
        engine.emit_progress_now();
        assert_eq!(seeked_events(&sink), vec![(entry.track_id, landed)]);
        {
            let events = sink.0.lock().unwrap();
            let confirmation = events
                .iter()
                .position(|event| matches!(event, EngineEvent::Seeked { .. }))
                .unwrap();
            assert!(events[confirmation + 1..]
                .iter()
                .any(|event| { matches!(event, EngineEvent::Position { .. }) }));
        }
        assert!(!engine.playing);
        assert!((engine.position_secs() - 0.5).abs() < 0.01);
        assert_eq!(engine.reported_index, Some(0));
        assert!(engine.measuring.is_none());
        assert!(!sink
            .0
            .lock()
            .unwrap()
            .iter()
            .any(|event| matches!(event, EngineEvent::TrackChanged { .. })));
        std::fs::remove_file(path).unwrap();
    }

    fn captured_engine() -> (Engine, Arc<CaptureSink>) {
        let sink = Arc::new(CaptureSink(std::sync::Mutex::new(Vec::new())));
        let (tx, rx) = crossbeam_channel::unbounded();
        let engine = Engine::new(
            no_store(),
            rx,
            tx,
            Arc::new(Params::default()),
            sink.clone(),
            Arc::new(AtomicU64::new(0)),
            test_runtime(),
            None,
        );
        (engine, sink)
    }

    #[cfg(feature = "stretch")]
    #[test]
    fn set_time_stretch_clamps_the_ratio_and_echoes_the_setting() {
        let (mut engine, sink) = captured_engine();
        engine.handle(EngineCommand::SetTimeStretch {
            enabled: true,
            ratio: 9.0,
        });

        let events = sink.0.lock().unwrap();
        let echoed = events
            .iter()
            .find_map(|e| match e {
                EngineEvent::TimeStretch { enabled, ratio } => Some((*enabled, *ratio)),
                _ => None,
            })
            .expect("the setting must be echoed");
        assert_eq!(echoed, (true, 2.0), "9.0 clamps to the supported maximum");
        assert_eq!(engine.chain.time_stretch(), (true, 2.0));
    }

    #[cfg(feature = "stretch")]
    #[test]
    fn a_non_finite_ratio_is_dropped_not_applied() {
        let (mut engine, sink) = captured_engine();
        engine.handle(EngineCommand::SetTimeStretch {
            enabled: true,
            ratio: f32::NAN,
        });
        assert!(sink.0.lock().unwrap().is_empty(), "no echo, no effect");
        assert_eq!(engine.chain.time_stretch(), (false, 1.0));
    }

    #[cfg(feature = "stretch")]
    #[test]
    fn describe_recovers_the_time_stretch_setting() {
        let (mut engine, sink) = captured_engine();
        engine.handle(EngineCommand::SetTimeStretch {
            enabled: true,
            ratio: 1.25,
        });
        sink.0.lock().unwrap().clear();

        engine.describe();
        let events = sink.0.lock().unwrap();
        assert!(
            events.iter().any(|e| matches!(
                e,
                EngineEvent::TimeStretch {
                    enabled: true,
                    ratio,
                } if (*ratio - 1.25).abs() < 1e-6
            )),
            "a reloaded UI must recover the setting from describe()"
        );
    }

    #[cfg(feature = "fir-eq")]
    #[test]
    fn set_fir_eq_echoes_and_flattens_the_callback_eq() {
        let (mut engine, sink) = captured_engine();
        engine.handle(EngineCommand::SetFirEq { enabled: true });

        let events = sink.0.lock().unwrap();
        let enabled = events
            .iter()
            .find_map(|e| match e {
                EngineEvent::FirEq { enabled, .. } => Some(*enabled),
                _ => None,
            })
            .expect("the setting must be echoed");
        assert!(enabled, "enabling must be echoed as enabled");
        assert!(
            engine.params.fir_eq_active(),
            "the callback EQ must be flattened while the FIR EQ owns the gains"
        );
    }

    #[cfg(feature = "fir-eq")]
    #[test]
    fn disabling_fir_eq_restores_the_callback_eq() {
        let (mut engine, _sink) = captured_engine();
        engine.handle(EngineCommand::SetFirEq { enabled: true });
        assert!(engine.params.fir_eq_active());
        engine.handle(EngineCommand::SetFirEq { enabled: false });
        assert!(
            !engine.params.fir_eq_active(),
            "disabling must hand the gains back to the callback EQ"
        );
    }

    #[cfg(feature = "fir-eq")]
    #[test]
    fn describe_recovers_the_fir_eq_setting() {
        let (mut engine, sink) = captured_engine();
        engine.handle(EngineCommand::SetFirEq { enabled: true });
        sink.0.lock().unwrap().clear();

        engine.describe();
        let events = sink.0.lock().unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EngineEvent::FirEq { enabled: true, .. })),
            "a reloaded UI must recover the setting from describe()"
        );
    }

    /// Writes a one-sample unit-impulse IR, enough to build a backend from.
    /// `name` must be unique per test: tests run in parallel threads, and a
    /// shared path would race a truncating write against a read.
    #[cfg(feature = "convolution")]
    fn write_delta_ir(name: &str) -> std::path::PathBuf {
        let bytes = fixtures::wav_bytes(48_000, 1, &[i16::MAX]);
        let path = std::env::temp_dir().join(format!("audio-stack-rs-engine-conv-{name}.wav"));
        std::fs::write(&path, bytes).expect("write ir");
        path
    }

    #[cfg(feature = "convolution")]
    #[test]
    fn set_convolution_echoes_the_setting() {
        let (mut engine, sink) = captured_engine();
        engine.handle(EngineCommand::SetConvolution {
            enabled: true,
            ir_path: Some(write_delta_ir("echo-setting")),
            mix: 0.4,
        });

        let events = sink.0.lock().unwrap();
        let echoed = events
            .iter()
            .find_map(|e| match e {
                EngineEvent::Convolution { enabled, mix } => Some((*enabled, *mix)),
                _ => None,
            })
            .expect("the setting must be echoed");
        assert!(echoed.0, "enabling must be echoed as enabled");
        assert!(
            (echoed.1 - 0.4).abs() < 1e-6,
            "and carry the mix: {}",
            echoed.1
        );
    }

    #[cfg(feature = "convolution")]
    #[test]
    fn a_bad_ir_surfaces_an_error_and_leaves_convolution_off() {
        let (mut engine, sink) = captured_engine();
        engine.handle(EngineCommand::SetConvolution {
            enabled: true,
            ir_path: Some(std::path::PathBuf::from("/no/such/impulse.wav")),
            mix: 0.5,
        });

        let events = sink.0.lock().unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EngineEvent::Error { .. })),
            "a missing IR must surface an error"
        );
    }

    #[cfg(feature = "convolution")]
    #[test]
    fn a_failed_ir_swap_echoes_disabled_not_enabled() {
        // Replacing a working IR with a bad file surfaces the error, and the
        // echo that follows must say disabled — not claim an IR that never
        // loaded.
        let (mut engine, sink) = captured_engine();
        engine.handle(EngineCommand::SetConvolution {
            enabled: true,
            ir_path: Some(write_delta_ir("failed-swap")),
            mix: 0.5,
        });
        sink.0.lock().unwrap().clear();

        engine.handle(EngineCommand::SetConvolution {
            enabled: true,
            ir_path: Some(std::path::PathBuf::from("/no/such/replacement.wav")),
            mix: 0.5,
        });
        let events = sink.0.lock().unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EngineEvent::Error { .. })),
            "the failed load must surface an error"
        );
        let echoed = events
            .iter()
            .find_map(|e| match e {
                EngineEvent::Convolution { enabled, .. } => Some(*enabled),
                _ => None,
            })
            .expect("the setting must still be echoed");
        assert!(!echoed, "the echo must report disabled after a failed swap");
    }

    #[cfg(feature = "convolution")]
    #[test]
    fn describe_recovers_the_convolution_setting() {
        let (mut engine, sink) = captured_engine();
        engine.handle(EngineCommand::SetConvolution {
            enabled: true,
            ir_path: Some(write_delta_ir("describe")),
            mix: 0.6,
        });
        sink.0.lock().unwrap().clear();

        engine.describe();
        let events = sink.0.lock().unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EngineEvent::Convolution { enabled: true, .. })),
            "a reloaded UI must recover the setting from describe()"
        );
    }

    #[cfg(feature = "analysis")]
    #[test]
    fn a_full_listen_emits_a_track_analysis() {
        let (mut engine, sink) = captured_engine();
        // Arm analysis by hand for track 7 and feed it a click track, the way
        // the pump would while it played.
        let mut analysis = super::super::analysis::Analysis::new(44_100, 2).expect("analysis");
        let click: Vec<f32> = (0..44_100 * 8)
            .flat_map(|n| {
                let period = 44_100 / 2; // 120 BPM
                let s = if n % period < 200 { 0.8 } else { 0.0 };
                [s, s]
            })
            .collect();
        analysis.feed(&click);
        engine.analysing = Some((7, analysis));

        engine.finish_measuring(true);
        let events = sink.0.lock().unwrap();
        let reported = events
            .iter()
            .find_map(|e| match e {
                EngineEvent::TrackAnalysis { track_id, bpm, .. } => Some((*track_id, *bpm)),
                _ => None,
            })
            .expect("a full listen must report analysis");
        assert_eq!(reported.0, 7, "the event carries the track id");
        assert!(reported.1.is_some(), "a click track has a tempo");
        assert!(engine.analysing.is_none(), "the analysis state is consumed");
    }

    #[cfg(feature = "analysis")]
    #[test]
    fn a_partial_listen_reports_no_analysis() {
        let (mut engine, sink) = captured_engine();
        engine.analysing = Some((
            3,
            super::super::analysis::Analysis::new(44_100, 2).expect("analysis"),
        ));

        engine.finish_measuring(false); // a seek or skip, not a full listen
        assert!(
            !sink
                .0
                .lock()
                .unwrap()
                .iter()
                .any(|e| matches!(e, EngineEvent::TrackAnalysis { .. })),
            "a partial listen must not report analysis"
        );
        assert!(engine.analysing.is_none(), "but the state is still dropped");
    }

    #[cfg(feature = "pitch")]
    #[test]
    fn set_pitch_shift_clamps_the_cents_and_echoes_the_setting() {
        let (mut engine, sink) = captured_engine();
        engine.handle(EngineCommand::SetPitchShift {
            enabled: true,
            cents: 5_000.0,
        });

        let echoed = sink
            .0
            .lock()
            .unwrap()
            .iter()
            .find_map(|e| match e {
                EngineEvent::PitchShift { enabled, cents } => Some((*enabled, *cents)),
                _ => None,
            })
            .expect("the setting must be echoed");
        assert_eq!(echoed, (true, 1_200.0), "5000 cents clamps to one octave");
        assert_eq!(engine.chain.pitch_shift(), (true, 1_200.0));
    }

    #[cfg(feature = "pitch")]
    #[test]
    fn a_non_finite_cents_is_dropped() {
        let (mut engine, sink) = captured_engine();
        engine.handle(EngineCommand::SetPitchShift {
            enabled: true,
            cents: f32::NAN,
        });
        assert!(sink.0.lock().unwrap().is_empty(), "no echo, no effect");
        assert_eq!(engine.chain.pitch_shift(), (false, 0.0));
    }

    #[cfg(feature = "pitch")]
    #[test]
    fn describe_recovers_the_pitch_shift_setting() {
        let (mut engine, sink) = captured_engine();
        engine.handle(EngineCommand::SetPitchShift {
            enabled: true,
            cents: 300.0,
        });
        sink.0.lock().unwrap().clear();

        engine.describe();
        assert!(
            sink.0.lock().unwrap().iter().any(|e| matches!(
                e,
                EngineEvent::PitchShift { enabled: true, cents } if (*cents - 300.0).abs() < 1e-3
            )),
            "a reloaded UI must recover the setting from describe()"
        );
    }

    /// A store that wants everything measured — what arms the measuring path
    /// the way an untagged, never-played WAV does in production.
    struct EagerStore;

    impl Store for EagerStore {
        fn needs_measurement(&self, _track_id: i64) -> bool {
            true
        }
        fn record(&self, _track_id: i64, _measured: super::super::loudness::Measured) {}
    }

    #[test]
    fn seek_rebases_the_playhead_and_keeps_the_boundary_at_the_origin() {
        let mut engine = bare_engine();
        engine.mode = Mode::Local;
        let wav = fixtures::wav_bytes(44_100, 2, &fixtures::tone(44_100, 2, 3 * 44_100, 440.0));
        let mut hint = symphonia::core::formats::probe::Hint::new();
        hint.with_extension("wav");
        engine.decoder =
            Some(Decoder::open(Box::new(std::io::Cursor::new(wav)), hint).expect("wav decodes"));

        engine.seek(2.0);

        // With no device open the rate falls back to 1 frame per second, so
        // the rebased counter lands on the seek target (give or take the
        // packet boundary the accurate seek snapped to).
        assert!(engine.frames_written >= 1, "the counter was rebased");
        assert_eq!(
            engine.params.frames_played(),
            engine.frames_written,
            "the playhead counter is rebased together with frames_written"
        );
        assert_eq!(
            engine.boundaries.front().map(|b| b.frame),
            Some(0),
            "the track's origin in the rebased timeline is frame 0 — anything \
             else makes position_secs read `played - base = 0` for the rest \
             of the track"
        );
    }

    fn engine_with_tone(secs: usize) -> Engine {
        let mut engine = bare_engine();
        engine.mode = Mode::Local;
        engine.current_duration = secs as f64;
        let wav = fixtures::wav_bytes(44_100, 2, &fixtures::tone(44_100, 2, secs * 44_100, 440.0));
        let mut hint = symphonia::core::formats::probe::Hint::new();
        hint.with_extension("wav");
        engine.decoder =
            Some(Decoder::open(Box::new(std::io::Cursor::new(wav)), hint).expect("wav decodes"));
        engine
    }

    #[test]
    fn relative_seek_target_moves_from_the_playhead_and_clamps_to_the_track() {
        use SeekDirection::{Backward, Forward};
        assert_eq!(relative_seek_target(10.0, 5.0, Forward, 60.0), Some(15.0));
        assert_eq!(relative_seek_target(10.0, 5.0, Backward, 60.0), Some(5.0));
        assert_eq!(
            relative_seek_target(2.0, 5.0, Backward, 60.0),
            Some(0.0),
            "a backward seek never goes before the start"
        );
        assert_eq!(
            relative_seek_target(58.0, 5.0, Forward, 60.0),
            Some(60.0),
            "a forward seek never goes past the end"
        );
        assert_eq!(
            relative_seek_target(58.0, 5.0, Forward, 0.0),
            Some(63.0),
            "an unknown duration (0) is no upper bound"
        );
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                relative_seek_target(10.0, bad, Forward, 60.0),
                None,
                "{bad} seconds is not a seek"
            );
        }
    }

    #[test]
    fn seek_by_rebases_the_playhead_like_an_absolute_seek() {
        let mut engine = engine_with_tone(3);

        engine.seek_by(2.0, SeekDirection::Forward);

        // No device, so the rate falls back to 1 frame per second and the
        // counter lands on the target (give or take the packet the accurate
        // seek snapped to).
        assert!(engine.frames_written >= 1, "the counter was rebased");
        assert_eq!(engine.params.frames_played(), engine.frames_written);
        assert_eq!(engine.boundaries.front().map(|b| b.frame), Some(0));
    }

    #[test]
    fn seek_by_past_the_end_clamps_to_the_track_without_an_error() {
        let (mut engine, sink) = captured_engine();
        let tone = engine_with_tone(3);
        engine.mode = Mode::Local;
        engine.current_duration = tone.current_duration;
        engine.decoder = tone.decoder;

        engine.seek_by(30.0, SeekDirection::Forward);

        let events = sink.0.lock().unwrap();
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, EngineEvent::Error { .. })),
            "clamping to the end must not surface a seek error: {events:?}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EngineEvent::Position { .. })),
            "the seek reports the new position"
        );
    }

    #[test]
    fn seek_by_before_the_start_lands_at_zero() {
        let mut engine = engine_with_tone(3);
        engine.seek(2.0);
        assert!(engine.frames_written >= 1, "precondition: not at the start");

        engine.seek_by(30.0, SeekDirection::Backward);

        assert_eq!(
            engine.frames_written, 0,
            "clamped to the start of the track"
        );
        assert_eq!(engine.params.frames_played(), 0);
    }

    #[test]
    fn seek_by_ignores_a_station_and_a_bad_amount() {
        let mut engine = engine_with_tone(3);
        engine.mode = Mode::Radio;
        engine.seek_by(2.0, SeekDirection::Forward);
        assert_eq!(engine.frames_written, 0, "a station cannot be seeked");

        engine.mode = Mode::Local;
        engine.seek_by(f64::NAN, SeekDirection::Forward);
        engine.seek_by(-2.0, SeekDirection::Forward);
        assert_eq!(
            engine.frames_written, 0,
            "a bad amount is dropped, not applied"
        );
        assert!(
            engine.boundaries.is_empty(),
            "a dropped seek must not touch the timeline"
        );
    }

    // ── A–B loop ────────────────────────────────────────────────────────

    /// A ramp short enough that neighbouring frames decode to distinct
    /// values, so a test can tell exactly which source frame it is hearing.
    const RAMP_FRAMES: usize = 4000;

    fn ramp_value(frame: usize) -> f32 {
        ((frame as f32 / RAMP_FRAMES as f32) * 30_000.0) as i16 as f32 / 32_768.0
    }

    fn ramp_secs(frame: usize) -> f64 {
        frame as f64 / 44_100.0
    }

    /// A playing engine over the ramp, with a ring to pump into and a sink
    /// capturing the echoes. No device, so the ring is drained by hand.
    fn looping_engine() -> (Engine, Arc<CaptureSink>, rtrb::Consumer<f32>) {
        let (mut engine, sink) = captured_engine();
        engine.mode = Mode::Local;
        engine.playing = true;
        engine.current_duration = ramp_secs(RAMP_FRAMES);
        let wav = fixtures::wav_bytes(44_100, 2, &fixtures::ramp(2, RAMP_FRAMES));
        let mut hint = symphonia::core::formats::probe::Hint::new();
        hint.with_extension("wav");
        engine.decoder =
            Some(Decoder::open(Box::new(std::io::Cursor::new(wav)), hint).expect("wav decodes"));
        let (producer, consumer) = rtrb::RingBuffer::<f32>::new(48_000 * 4);
        engine.producer = Some(producer);
        (engine, sink, consumer)
    }

    /// Pumps until the track ends or `max_frames` have reached the ring,
    /// playing the callback's part (draining the ring, finishing flushes).
    /// Returns the left channel of what was heard.
    fn heard(
        engine: &mut Engine,
        consumer: &mut rtrb::Consumer<f32>,
        max_frames: usize,
    ) -> Vec<f32> {
        let mut out = Vec::new();
        for _ in 0..100_000 {
            engine.params.finish_flush();
            engine.pump();
            while let Ok(sample) = consumer.pop() {
                out.push(sample);
            }
            if engine.decoder.is_none() || out.len() / 2 >= max_frames {
                break;
            }
        }
        out.iter().step_by(2).copied().collect()
    }

    fn assert_heard_frames(heard: &[f32], expected: impl Iterator<Item = usize>) {
        let expected: Vec<usize> = expected.collect();
        assert_eq!(heard.len(), expected.len(), "frames heard");
        for (i, (sample, frame)) in heard.iter().zip(&expected).enumerate() {
            assert!(
                (sample - ramp_value(*frame)).abs() < 1e-4,
                "output frame {i} is {sample}, not source frame {frame} ({})",
                ramp_value(*frame)
            );
        }
    }

    fn loop_events(sink: &CaptureSink) -> Vec<(bool, f64, f64, Option<u32>)> {
        sink.0
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                EngineEvent::Loop {
                    enabled,
                    start_secs,
                    end_secs,
                    repeats_left,
                } => Some((*enabled, *start_secs, *end_secs, *repeats_left)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn set_loop_jumps_to_the_start_and_echoes_the_region() {
        let (mut engine, sink) = captured_engine();
        let tone = engine_with_tone(3);
        engine.mode = Mode::Local;
        engine.current_duration = tone.current_duration;
        engine.decoder = tone.decoder;

        engine.set_loop(2.0, 3.0, Some(2));

        assert_eq!(loop_events(&sink), vec![(true, 2.0, 3.0, Some(2))]);
        assert!(
            engine.params.frames_played() >= 1,
            "the playhead was rebased to the loop start (1 frame/s with no device)"
        );

        engine.describe();
        assert_eq!(
            loop_events(&sink).len(),
            2,
            "describe re-emits the loop so a reloaded UI recovers it"
        );
    }

    #[test]
    fn an_invalid_loop_region_is_rejected_with_an_error() {
        let (mut engine, sink, _consumer) = looping_engine();
        for (start, end) in [
            (2.0, 1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
            (f64::NAN, 1.0),
            (0.0, f64::INFINITY),
        ] {
            engine.set_loop(start, end, None);
            assert!(
                engine.loop_region.is_none(),
                "{start}..{end} is not a region"
            );
        }
        let errors = sink
            .0
            .lock()
            .unwrap()
            .iter()
            .filter(|e| matches!(e, EngineEvent::Error { .. }))
            .count();
        assert_eq!(errors, 5, "each rejection is reported");
        assert!(loop_events(&sink).is_empty(), "nothing to echo");
    }

    #[test]
    fn a_loop_repeats_the_section_sample_exactly_then_plays_on() {
        let (mut engine, sink, mut consumer) = looping_engine();
        engine.set_loop(ramp_secs(1000), ramp_secs(2000), Some(1));

        let out = heard(&mut engine, &mut consumer, usize::MAX);

        // 1000..2000 twice — the jump lands on frame 1000 exactly, with no
        // frame lost or doubled at the seam — then the rest of the track.
        assert_heard_frames(
            &out,
            (1000..2000).chain(1000..2000).chain(2000..RAMP_FRAMES),
        );
        assert!(engine.decoder.is_none(), "the track ran to its end");
        assert!(
            engine.loop_region.is_none(),
            "a finished loop clears itself"
        );
        assert_eq!(
            loop_events(&sink),
            vec![
                (true, ramp_secs(1000), ramp_secs(2000), Some(1)),
                (true, ramp_secs(1000), ramp_secs(2000), Some(0)),
                (false, 0.0, 0.0, None),
            ],
            "set, one jump back, then cleared on reaching the end again"
        );
        assert!(
            engine
                .boundaries
                .iter()
                .any(|b| (b.offset_secs - ramp_secs(1000)).abs() < 1e-9),
            "the jump back leaves a boundary carrying the loop start as its offset"
        );
    }

    #[test]
    fn a_zero_repeat_loop_plays_the_section_once_from_its_start() {
        let (mut engine, _sink, mut consumer) = looping_engine();
        engine.set_loop(ramp_secs(1000), ramp_secs(2000), Some(0));

        let out = heard(&mut engine, &mut consumer, usize::MAX);

        assert_heard_frames(&out, 1000..RAMP_FRAMES);
        assert!(engine.loop_region.is_none());
    }

    #[test]
    fn a_loop_past_the_end_of_the_track_loops_at_the_end() {
        let (mut engine, _sink, mut consumer) = looping_engine();
        engine.set_loop(ramp_secs(3000), 60.0, Some(1));

        let out = heard(&mut engine, &mut consumer, usize::MAX);

        assert_heard_frames(&out, (3000..RAMP_FRAMES).chain(3000..RAMP_FRAMES));
        assert!(engine.decoder.is_none(), "and then the track ends normally");
    }

    #[test]
    fn a_loop_without_a_count_repeats_until_cleared() {
        let (mut engine, _sink, mut consumer) = looping_engine();
        engine.set_loop(ramp_secs(1000), ramp_secs(2000), None);

        // Three times the whole file's length, all of it inside the region.
        let out = heard(&mut engine, &mut consumer, RAMP_FRAMES * 3);
        assert!(out.len() >= RAMP_FRAMES * 3);
        assert!(engine.decoder.is_some(), "still playing");
        assert!(engine.loop_region.is_some(), "still looping");
        let expected = (0..out.len()).map(|i| 1000 + i % 1000);
        assert_heard_frames(&out, expected);

        engine.clear_loop();
        let rest = heard(&mut engine, &mut consumer, usize::MAX);
        assert!(
            engine.decoder.is_none(),
            "cleared, the track runs to its end"
        );
        assert!(
            (rest.last().copied().unwrap() - ramp_value(RAMP_FRAMES - 1)).abs() < 1e-4,
            "ending on the last frame of the file"
        );
    }

    #[test]
    fn a_seek_outside_the_region_plays_on_past_the_end() {
        let (mut engine, _sink, mut consumer) = looping_engine();
        engine.set_loop(ramp_secs(1000), ramp_secs(2000), None);
        engine.seek(ramp_secs(2500));

        let out = heard(&mut engine, &mut consumer, usize::MAX);

        assert_heard_frames(&out, 2500..RAMP_FRAMES);
        assert!(
            engine.decoder.is_none(),
            "no snap back from outside the region"
        );
    }

    #[test]
    fn stopping_clears_the_loop() {
        let (mut engine, sink, _consumer) = looping_engine();
        engine.set_loop(ramp_secs(1000), ramp_secs(2000), None);

        engine.stop();

        assert!(engine.loop_region.is_none());
        assert_eq!(
            loop_events(&sink).last().copied(),
            Some((false, 0.0, 0.0, None))
        );
    }

    #[test]
    fn no_crossfade_starts_while_a_loop_is_active() {
        let (mut engine, _consumer, path0, path1) = crossfading_engine("janis-crossfade-loop-test");
        engine.loop_region = Some(LoopRegion {
            start_secs: 0.0,
            end_secs: 0.5,
            remaining: None,
        });

        engine.pump();

        let _ = std::fs::remove_file(&path0);
        let _ = std::fs::remove_file(&path1);
        assert!(
            engine.transition.is_none(),
            "a loop keeps the track from ending, so nothing fades in over it"
        );
        assert_eq!(engine.queue.index(), 0);
    }

    fn missing_entries(count: usize) -> Vec<QueueEntry> {
        (0..count)
            .map(|i| QueueEntry {
                track_id: i as i64,
                location: std::path::PathBuf::from(format!("/nonexistent/janis-test-{i}.flac"))
                    .into(),
                duration_secs: 1.0,
                gain_db: 0.0,
            })
            .collect()
    }

    #[test]
    fn a_repeating_queue_of_unopenable_files_stops_instead_of_recursing() {
        let mut engine = bare_engine();
        engine.queue.load(missing_entries(30), 0);
        engine.queue.set_repeat(true);
        engine.mode = Mode::Local;
        engine.playing = true;

        // Repeat wraps the cursor forever, so before the retry bound this
        // recursed per failed entry until the engine thread blew its stack
        // and aborted the whole process.
        engine.advance_or_stop();

        assert!(engine.decoder.is_none());
        assert!(!engine.playing, "nothing playable means not playing");
        assert_eq!(engine.mode, Mode::Idle, "a full failed pass stops playback");
    }

    #[test]
    fn play_after_the_queue_ends_never_reports_ghost_playback() {
        let mut engine = bare_engine();
        engine.queue.load(missing_entries(1), 0);
        engine.mode = Mode::Local;

        // The last track just finished: the decoder is gone but the mode
        // stays Local so the UI keeps showing it.
        engine.advance_or_stop();
        assert!(!engine.playing);

        // Play must either start a track or leave the transport stopped —
        // reporting `playing` with nothing to decode is a state with no way
        // out (pump returns immediately forever).
        engine.set_playing(true);
        assert!(
            !engine.playing || engine.decoder.is_some(),
            "playing with no decoder must be unreachable"
        );
    }

    #[test]
    fn clicking_a_station_never_measures_it_against_the_previous_local_track() {
        // The corruption this guards against: a local WAV is playing and
        // being measured; the user clicks a station without stopping; the
        // meter keyed to the WAV's id would then be fed the station and its
        // loudness written permanently into the WAV's row.
        let mut engine = bare_engine_with(Arc::new(EagerStore));
        engine.queue.load(missing_entries(1), 0);
        engine.mode = Mode::Local;
        engine.measuring = Some((0, Loudness::new(44_100, 2).expect("meter")));

        let wav = fixtures::wav_bytes(44_100, 2, &fixtures::silence(2, 4_410));
        let mut hint = symphonia::core::formats::probe::Hint::new();
        hint.with_extension("wav");
        let stream = RadioStream {
            source: Box::new(std::io::Cursor::new(wav)),
            hint,
            now_playing: Arc::new(std::sync::Mutex::new(None)),
        };
        engine.handle(EngineCommand::PlayStream {
            epoch: engine.station_epoch.load(Ordering::Relaxed),
            station_id: "test-station".into(),
            url: "http://localhost/never-used".into(),
            stream: Box::new(stream),
            has_provider: false,
        });

        assert!(
            engine.measuring.is_none(),
            "a station has no track row to record against — the local \
             track's meter must be abandoned, not fed the stream"
        );
    }

    #[test]
    fn a_stale_play_stream_is_dropped_instead_of_overriding_the_listener() {
        let mut engine = bare_engine();
        // The command claimed epoch 0 before its connect; while it was
        // connecting the listener started something else, which bumped it.
        let claimed = engine.station_epoch.load(Ordering::Relaxed);
        engine.station_epoch.fetch_add(1, Ordering::Relaxed);

        let wav = fixtures::wav_bytes(44_100, 2, &fixtures::silence(2, 4_410));
        let mut hint = symphonia::core::formats::probe::Hint::new();
        hint.with_extension("wav");
        let stream = RadioStream {
            source: Box::new(std::io::Cursor::new(wav)),
            hint,
            now_playing: Arc::new(std::sync::Mutex::new(None)),
        };
        engine.handle(EngineCommand::PlayStream {
            epoch: claimed,
            station_id: "abandoned".into(),
            url: "http://localhost/never-used".into(),
            stream: Box::new(stream),
            has_provider: false,
        });

        assert_eq!(
            engine.mode,
            Mode::Idle,
            "a station the listener abandoned while it connected must not start"
        );
        assert!(engine.station_id.is_none());
    }

    #[test]
    fn a_reconnect_that_fails_to_probe_gives_up_cleanly_instead_of_parking() {
        let mut engine = bare_engine();
        engine.mode = Mode::Radio;
        engine.playing = true;
        engine.station_id = Some("s".into());
        engine.station_url = Some("http://localhost/x".into());
        engine.reconnect_attempt = MAX_RECONNECT_ATTEMPTS;

        // Connected, but the body is an HTML error page, not audio.
        let stream = RadioStream {
            source: Box::new(std::io::Cursor::new(
                b"<html>stream offline</html>".to_vec(),
            )),
            hint: symphonia::core::formats::probe::Hint::new(),
            now_playing: Arc::new(std::sync::Mutex::new(None)),
        };
        engine.handle(EngineCommand::Reconnected {
            epoch: engine.station_epoch.load(Ordering::Relaxed),
            station_id: "s".into(),
            url: "http://localhost/x".into(),
            stream: Box::new(stream),
            has_provider: false,
        });

        // The old code reset the attempt counter before probing and absorbed
        // the failure, leaving mode Radio / playing / no decoder — a state
        // nothing could ever leave. A failed probe now keeps the backoff
        // running, and an exhausted backoff stops playback honestly.
        assert_eq!(engine.mode, Mode::Idle);
        assert!(!engine.playing);
    }

    #[test]
    fn a_remote_next_track_is_preloaded_off_the_engine_thread() {
        let address = crate::remote::test_server::serve(fixtures::wav_bytes(
            8_000,
            1,
            &fixtures::silence(1, 8_000),
        ));
        let mut engine = bare_engine();
        engine.queue.load(
            vec![
                QueueEntry {
                    track_id: 1,
                    location: std::path::PathBuf::from("/nonexistent/playing.wav").into(),
                    duration_secs: 1.0,
                    gain_db: 0.0,
                },
                QueueEntry {
                    track_id: 2,
                    location: Location::Remote(crate::RemoteFile {
                        url: format!("{address}/track.wav?api_key=secret"),
                        extension: Some("wav".to_owned()),
                        bounded_range: None,
                    }),
                    duration_secs: 1.0,
                    gain_db: 0.0,
                },
            ],
            0,
        );
        engine.mode = Mode::Local;
        // The playing track is within the preload window.
        engine.current_duration = 1.0;

        engine.refresh_preload();
        assert!(
            engine.preloaded.is_none() && engine.preloading.is_some(),
            "the first call starts the open and returns at once"
        );
        let deadline = Instant::now() + Duration::from_secs(10);
        while engine.preloaded.is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
            engine.refresh_preload();
        }
        assert!(engine.preloaded.is_some(), "the remote decoder arrives");
        assert!(engine.preloading.is_none());
    }

    #[test]
    fn a_gapless_advance_starts_measuring_the_incoming_track() {
        let dir = std::env::temp_dir().join("janis-gapless-measure-test");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("next.wav");
        let wav = fixtures::wav_bytes(44_100, 2, &fixtures::silence(2, 4_410));
        std::fs::File::create(&path)
            .and_then(|mut f| f.write_all(&wav))
            .expect("write test wav");

        let mut engine = bare_engine_with(Arc::new(EagerStore));
        engine.queue.load(
            vec![
                QueueEntry {
                    track_id: 1,
                    location: std::path::PathBuf::from("/nonexistent/finished.wav").into(),
                    duration_secs: 1.0,
                    gain_db: 0.0,
                },
                QueueEntry {
                    track_id: 2,
                    location: path.clone().into(),
                    duration_secs: 0.1,
                    gain_db: 0.0,
                },
            ],
            0,
        );
        engine.mode = Mode::Local;

        // Track 1 just ran out; the join installs track 2 by hand, bypassing
        // install_decoder to keep the ring intact — it must still arm a meter,
        // or only the first track of any queue is ever measured.
        engine.advance_or_stop();

        let _ = std::fs::remove_file(&path);
        assert_eq!(
            engine.measuring.as_ref().map(|(id, _)| *id),
            Some(2),
            "the incoming track gets its own meter at a gapless join"
        );
    }

    #[test]
    fn equal_power_curve_stays_at_constant_power_through_the_fade() {
        let (out0, in0) = equal_power(0.0);
        assert!((out0 - 1.0).abs() < 1e-6 && in0.abs() < 1e-6);

        let (out1, in1) = equal_power(1.0);
        assert!(out1.abs() < 1e-6 && (in1 - 1.0).abs() < 1e-6);

        let (outm, inm) = equal_power(0.5);
        assert!(
            (outm * outm + inm * inm - 1.0).abs() < 1e-5,
            "the squares must sum to one everywhere, or the mix gets \
             louder or quieter mid-fade: got {outm} and {inm}"
        );

        assert!(equal_power(-1.0).0 <= 1.0, "out-of-range input must clamp");
        assert!(equal_power(2.0).1 <= 1.0, "out-of-range input must clamp");
    }

    /// Writes a decodable one-second WAV to a fresh temp path, returning it
    /// alongside a `QueueEntry` for `path`.
    fn wav_track(track_id: i64, dir_name: &str) -> (std::path::PathBuf, QueueEntry) {
        let dir = std::env::temp_dir().join(dir_name);
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join(format!("{track_id}.wav"));
        let wav = fixtures::wav_bytes(44_100, 2, &fixtures::tone(44_100, 2, 44_100, 440.0));
        std::fs::File::create(&path)
            .and_then(|mut f| f.write_all(&wav))
            .expect("write test wav");
        (
            path.clone(),
            QueueEntry {
                track_id,
                location: path.into(),
                duration_secs: 1.0,
                gain_db: 0.0,
            },
        )
    }

    /// A two-track engine with crossfade on, already playing track 0 with
    /// `current_duration` short enough that the very next `pump()` call
    /// falls inside the crossfade window (position stays at 0 with no real
    /// device advancing `frames_played`, so `remaining == current_duration`
    /// for as long as the test runs). Returns the ring's consumer so a test
    /// can read back what was mixed into it.
    fn crossfading_engine(
        dir_name: &str,
    ) -> (
        Engine,
        rtrb::Consumer<f32>,
        std::path::PathBuf,
        std::path::PathBuf,
    ) {
        let (path0, entry0) = wav_track(0, dir_name);
        let (path1, entry1) = wav_track(1, dir_name);

        let mut engine = bare_engine();
        engine.mode = Mode::Local;
        engine.playing = true;
        engine.crossfade = true;
        engine.current_duration = 1.0;
        engine.queue.load(vec![entry0.clone(), entry1], 0);
        engine.decoder = Some(engine.open_entry(&entry0).expect("wav decodes"));

        let (producer, consumer) = rtrb::RingBuffer::<f32>::new(48_000 * 4);
        engine.producer = Some(producer);

        (engine, consumer, path0, path1)
    }

    #[test]
    fn a_crossfade_starts_immediately_when_already_within_its_window() {
        let (mut engine, _consumer, path0, path1) =
            crossfading_engine("janis-crossfade-start-test");

        engine.pump();

        let _ = std::fs::remove_file(&path0);
        let _ = std::fs::remove_file(&path1);
        assert!(
            engine.transition.is_some(),
            "current_duration (1.0s) is already inside CROSSFADE_SECS, so \
             the very first pump should start fading the next track in"
        );
        assert_eq!(
            engine.queue.index(),
            1,
            "the cursor moves to the incoming track immediately, like a \
             gapless join — only the UI-visible state is deferred"
        );
    }

    #[test]
    fn a_crossfade_mixes_and_completes_into_the_incoming_track() {
        let (mut engine, mut consumer, path0, path1) =
            crossfading_engine("janis-crossfade-complete-test");

        // Generous bound: a one-second tone at 44.1 kHz decodes in a
        // handful of PUMP_FRAMES-sized reads.
        for _ in 0..500 {
            engine.pump();
            if engine.transition.is_none() && engine.frames_written > 0 {
                break;
            }
        }

        let _ = std::fs::remove_file(&path0);
        let _ = std::fs::remove_file(&path1);

        assert!(
            engine.transition.is_none(),
            "the crossfade must complete once the outgoing track exhausts"
        );
        assert_eq!(engine.queue.index(), 1, "the incoming track is now current");
        assert!(
            engine.boundaries.iter().any(|b| b.index == 1),
            "completion pushes a boundary that hands the incoming track \
             its own gain once the device actually reaches it"
        );

        let available = consumer.slots();
        assert!(
            available > 4_000,
            "mixed audio should have reached the ring, got {available} samples"
        );
        let mixed: Vec<f32> = std::iter::from_fn(|| consumer.pop().ok()).collect();
        assert!(
            mixed.iter().any(|&s| s.abs() > 0.01),
            "the mix must carry audible signal, not silence"
        );
        assert!(
            mixed.iter().all(|&s| s.is_finite() && s.abs() <= 1.5),
            "mixing two full-scale tones must not produce garbage samples"
        );
    }

    #[test]
    fn seeking_mid_crossfade_cancels_it_and_restores_the_cursor() {
        let (mut engine, _consumer, path0, path1) = crossfading_engine("janis-crossfade-seek-test");
        engine.pump();
        assert!(engine.transition.is_some(), "precondition: fading");
        assert_eq!(engine.queue.index(), 1);

        engine.seek(0.1);

        let _ = std::fs::remove_file(&path0);
        let _ = std::fs::remove_file(&path1);
        assert!(
            engine.transition.is_none(),
            "a seek acts on the track being looked at — it must cancel a fade"
        );
        assert_eq!(
            engine.queue.index(),
            0,
            "the cursor reverts to the track that was actually seeked"
        );
    }

    #[test]
    fn next_mid_crossfade_cancels_it_before_advancing() {
        let (mut engine, _consumer, path0, path1) = crossfading_engine("janis-crossfade-next-test");
        engine.pump();
        assert!(engine.transition.is_some(), "precondition: fading");

        engine.handle(EngineCommand::Next);

        let _ = std::fs::remove_file(&path0);
        let _ = std::fs::remove_file(&path1);
        assert!(
            engine.transition.is_none(),
            "Next must not leave a stale fade running underneath it"
        );
        // Cancel restores the cursor to 0, then Next's own advance moves it
        // forward one step — landing back on the same track index the fade
        // was already heading to, now reached by an ordinary flushed join.
        assert_eq!(engine.queue.index(), 1);
        assert!(engine.decoder.is_some());
    }

    #[test]
    fn toggling_shuffle_or_repeat_mid_crossfade_lets_the_fade_finish() {
        for command in [
            EngineCommand::SetShuffle(true),
            EngineCommand::SetRepeat(true),
        ] {
            let (mut engine, _consumer, path0, path1) =
                crossfading_engine("audio-stack-crossfade-queue-mode-test");
            engine.pump();
            assert!(engine.transition.is_some(), "precondition: fading");
            let elapsed = engine.transition.as_ref().unwrap().elapsed_frames;

            engine.handle(command);

            assert!(
                engine.transition.is_some(),
                "a queue-mode toggle must not cut the incoming track off"
            );
            assert_eq!(engine.queue.index(), 1, "the incoming track stays current");
            for _ in 0..500 {
                engine.pump();
                if engine.transition.is_none() {
                    break;
                }
            }
            let _ = std::fs::remove_file(&path0);
            let _ = std::fs::remove_file(&path1);
            assert!(engine.transition.is_none(), "the original fade completes");
            assert_eq!(engine.queue.index(), 1);
            assert!(
                engine.boundaries.iter().any(|boundary| boundary.index == 1),
                "the incoming track took over without restarting (elapsed {elapsed})"
            );
        }
    }

    #[test]
    fn turning_crossfade_off_cancels_a_fade_in_progress() {
        let (mut engine, _consumer, path0, path1) =
            crossfading_engine("janis-crossfade-toggle-test");
        engine.pump();
        assert!(engine.transition.is_some(), "precondition: fading");

        engine.handle(EngineCommand::SetCrossfade(false));

        let _ = std::fs::remove_file(&path0);
        let _ = std::fs::remove_file(&path1);
        assert!(!engine.crossfade);
        assert!(
            engine.transition.is_none(),
            "turning the switch off must stop a fade already under way"
        );
        assert_eq!(engine.queue.index(), 0, "the cursor is restored");
    }

    #[test]
    fn gapless_off_flushes_between_tracks_instead_of_joining_in_ring() {
        let dir = "janis-gapless-off-test";
        let (path, entry) = wav_track(1, dir);

        let mut engine = bare_engine();
        engine.gapless = false;
        engine.mode = Mode::Local;
        engine.frames_written = 12_345; // audio already went out for track 0
        engine
            .queue
            .load(vec![missing_entries(1).remove(0), entry], 0);

        engine.advance_or_stop();

        let _ = std::fs::remove_file(&path);
        assert_eq!(
            engine.frames_written, 0,
            "a non-gapless join flushes and restarts the frame counter, \
             exactly like install_decoder does for a track started fresh"
        );
        assert_eq!(
            engine.boundaries.front().map(|b| b.frame),
            Some(0),
            "the flushed join's boundary sits at the rebased origin"
        );
    }

    #[test]
    fn gapless_on_keeps_the_frame_counter_running_across_the_join() {
        let dir = "janis-gapless-on-test";
        let (path, entry) = wav_track(1, dir);

        let mut engine = bare_engine();
        assert!(engine.gapless, "on by default");
        engine.mode = Mode::Local;
        engine.frames_written = 12_345;
        engine
            .queue
            .load(vec![missing_entries(1).remove(0), entry], 0);

        engine.advance_or_stop();

        let _ = std::fs::remove_file(&path);
        assert!(
            engine.frames_written >= 12_345,
            "a gapless join keeps counting from where the ring already was, \
             it never rebases mid-queue — got {}",
            engine.frames_written
        );
    }

    #[test]
    fn pausing_defers_the_actual_stream_pause_until_the_fade_completes() {
        let mut engine = bare_engine();
        engine.mode = Mode::Local;
        engine.playing = true;

        engine.set_playing(false);

        assert!(!engine.playing, "the transport reports paused right away");
        assert!(
            engine.params.flush_pending(),
            "a fade must be requested instead of an instant cutoff"
        );
        assert!(
            engine.pending_pause,
            "the real stream pause is deferred until the fade is serviced"
        );

        // No real callback exists in this test to service the fade, so the
        // pause must not have happened yet.
        engine.service_pending_pause();
        assert!(
            engine.pending_pause,
            "servicing before the flag clears must not finish the pause early"
        );

        // Simulate the callback completing the fade, as `process_block`
        // does for a real device.
        engine.params.finish_flush();
        engine.service_pending_pause();
        assert!(
            !engine.pending_pause,
            "once the fade is serviced, the deferred pause resolves"
        );
    }

    #[test]
    fn stopping_defers_the_pause_the_same_way_as_pausing() {
        let mut engine = bare_engine();
        engine.mode = Mode::Local;
        engine.playing = true;

        engine.stop();

        assert!(engine.params.flush_pending(), "stop must also fade first");
        assert!(engine.pending_pause);
    }

    #[test]
    fn resuming_before_a_deferred_pause_resolves_cancels_it() {
        let mut engine = bare_engine();
        engine.mode = Mode::Local;
        engine.playing = true;
        // A decoder must already be installed, or `set_playing(true)` takes
        // the "queue ran out, start fresh" branch instead of a plain resume.
        let wav = fixtures::wav_bytes(44_100, 2, &fixtures::tone(44_100, 2, 44_100, 440.0));
        let mut hint = symphonia::core::formats::probe::Hint::new();
        hint.with_extension("wav");
        engine.decoder =
            Some(Decoder::open(Box::new(std::io::Cursor::new(wav)), hint).expect("wav decodes"));

        engine.set_playing(false);
        assert!(engine.pending_pause, "precondition: a pause is pending");

        // The listener changed their mind before the callback ever
        // serviced the fade — the stream was never actually paused, so
        // there is nothing left to finish.
        engine.set_playing(true);

        assert!(engine.playing);
        assert!(
            !engine.pending_pause,
            "resuming must not let a stale pending pause later freeze a \
             stream that is now playing new audio"
        );
    }

    #[test]
    fn starting_a_fresh_track_cancels_a_pause_still_fading_out() {
        let dir = "janis-pause-then-play-test";
        let (path, entry) = wav_track(1, dir);

        let mut engine = bare_engine();
        engine.mode = Mode::Local;
        engine.playing = true;
        engine.pending_pause = true; // a Stop/Pause moments earlier
        engine.queue.load(vec![entry], 0);

        engine.start_current();

        let _ = std::fs::remove_file(&path);
        assert!(
            !engine.pending_pause,
            "a fresh decoder means new audio is about to flow — a leftover \
             pending pause from before must not later freeze this stream \
             once its own flush happens to clear"
        );
    }

    #[test]
    fn a_boundary_is_announced_only_when_the_device_reaches_it() {
        let mut engine = bare_engine();
        engine.mode = Mode::Local;
        engine.current_duration = 10.0;
        engine.reported_index = Some(0);
        engine.boundaries.push_back(Boundary {
            frame: 0,
            index: 0,
            duration_secs: 10.0,
            gain_db: 0.0,
            offset_secs: 0.0,
            seeked: None,
        });
        engine.boundaries.push_back(Boundary {
            frame: 1000,
            index: 1,
            duration_secs: 20.0,
            gain_db: -3.0,
            offset_secs: 0.0,
            seeked: None,
        });

        engine.params.reset_frames(999);
        engine.emit_progress_now();
        assert_eq!(
            engine.reported_index,
            Some(0),
            "the join is still in the ring, not yet audible"
        );
        assert_eq!(engine.boundaries.len(), 2);

        engine.params.reset_frames(1000);
        engine.emit_progress_now();
        assert_eq!(
            engine.reported_index,
            Some(1),
            "the join reached the device"
        );
        assert_eq!(engine.current_duration, 20.0);
        assert_eq!(
            engine.boundaries.len(),
            1,
            "the crossed boundary becomes the position origin"
        );
        assert!(
            (engine.current_gain_db - -3.0).abs() < f64::EPSILON,
            "the incoming track's gain lands with its first audible frame"
        );
    }

    /// End-to-end through a real output device: decode a generated file,
    /// resample it, push it through the ring, and confirm the callback
    /// consumed frames.
    ///
    /// Ignored by default because it needs an audio device — `just test-rust`
    /// has to pass on a headless machine. Run it deliberately with
    /// `cargo test -- --ignored play_a_file_end_to_end --nocapture`.
    #[test]
    #[ignore = "requires a real audio output device"]
    fn play_a_file_end_to_end() {
        let dir = std::env::temp_dir().join("janis-engine-test");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("tone.wav");

        // Two seconds of a 440 Hz tone at 44.1 kHz — a rate that differs from
        // most devices' 48 kHz, so the resampler is on the path too.
        let wav = fixtures::wav_bytes(44_100, 2, &fixtures::tone(44_100, 2, 88_200, 440.0));

        std::fs::File::create(&path)
            .and_then(|mut f| f.write_all(&wav))
            .expect("write test tone");

        let engine = crate::init(no_store(), no_sink(), None);
        // Silent: the callback still runs and counts frames, and nobody in
        // the room has to listen to a test tone.
        engine.set_volume(0.0);
        engine.load_queue(
            vec![QueueEntry {
                track_id: 1,
                location: path.clone().into(),
                duration_secs: 2.0,
                gain_db: 0.0,
            }],
            0,
        );

        std::thread::sleep(Duration::from_millis(700));
        let played = engine.frames_played();
        engine.shutdown();
        let _ = std::fs::remove_file(&path);

        assert!(
            played > 0,
            "the output callback consumed no frames — nothing reached the device"
        );
        // Roughly half a second of audio should have gone out by now; allow a
        // wide margin for a slow device opening.
        assert!(
            played > 4_000,
            "only {played} frames played, expected the stream to be running"
        );
    }

    /// End-to-end with time-stretch on the path: at half tempo the playhead
    /// must advance at roughly half wall-clock speed — the timeline markers,
    /// not the device frame counter, are what report position.
    ///
    /// `cargo test --features stretch -- --ignored half_tempo --nocapture`.
    #[cfg(feature = "stretch")]
    #[test]
    #[ignore = "requires a real audio output device"]
    fn half_tempo_halves_the_reported_position_rate() {
        let dir = std::env::temp_dir().join("janis-engine-test");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("stretch-tone.wav");
        let wav = fixtures::wav_bytes(44_100, 2, &fixtures::tone(44_100, 2, 4 * 44_100, 440.0));
        std::fs::File::create(&path)
            .and_then(|mut f| f.write_all(&wav))
            .expect("write test tone");

        let sink = Arc::new(CaptureSink(std::sync::Mutex::new(Vec::new())));
        let engine = crate::init(no_store(), sink.clone(), None);
        engine.set_volume(0.0);
        engine.set_time_stretch(true, 0.5);
        engine.load_queue(
            vec![QueueEntry {
                track_id: 1,
                location: path.clone().into(),
                duration_secs: 4.0,
                gain_db: 0.0,
            }],
            0,
        );

        let wall = Duration::from_millis(1_500);
        std::thread::sleep(wall);
        let played = engine.frames_played();
        engine.shutdown();
        let _ = std::fs::remove_file(&path);

        let position = sink
            .0
            .lock()
            .unwrap()
            .iter()
            .rev()
            .find_map(|e| match e {
                EngineEvent::Position { position_secs, .. } => Some(*position_secs),
                _ => None,
            })
            .expect("position events were emitted");

        assert!(played > 4_000, "only {played} frames played");
        // At tempo 0.5 the playhead covers ~half the wall time. The device
        // open eats an unknown slice of the sleep, so only bound it above:
        // an unstretched playhead would read close to the full wall time.
        assert!(
            position < wall.as_secs_f64() * 0.75,
            "position {position:.2}s advanced too fast for half tempo over {:.2}s",
            wall.as_secs_f64()
        );
        assert!(position > 0.2, "position {position:.2}s never advanced");
    }

    /// Serves a short stream that ends, and checks the engine goes back for
    /// more instead of falling silent.
    ///
    /// Counting connections is the assertion: a station that ends is
    /// indistinguishable from one that dropped, so a second request means the
    /// reconnect path ran.
    ///
    /// `cargo test -- --ignored a_dropped_station_is_reconnected --nocapture`
    #[test]
    #[ignore = "requires an audio output device"]
    fn a_dropped_station_is_reconnected() {
        use std::io::{Read, Write};
        use std::sync::atomic::AtomicUsize;

        // Half a second of silence is enough to decode and end quickly.
        let wav = fixtures::wav_bytes(44_100, 1, &fixtures::silence(1, 22_050));

        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind a local server");
        let port = listener.local_addr().expect("addr").port();
        let connections = Arc::new(AtomicUsize::new(0));

        let served = Arc::clone(&connections);
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                served.fetch_add(1, Ordering::Relaxed);
                let mut stream = stream;
                // Read past the request headers, then answer.
                let mut scratch = [0u8; 1024];
                let _ = stream.read(&mut scratch);
                let _ = stream.write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nConnection: close\r\n\r\n",
                );
                let _ = stream.write_all(&wav);
                let _ = stream.flush();
            }
        });

        let url = format!("http://127.0.0.1:{port}/stream.wav");

        let engine = crate::init(no_store(), no_sink(), None);
        engine.set_volume(0.0);
        test_runtime()
            .block_on(engine.play_stream("local-test".to_string(), url.clone(), None))
            .expect("engine accepts the station");

        // Long enough for the stream to end and the first backoff (1s) to
        // elapse and reconnect.
        std::thread::sleep(Duration::from_millis(4000));
        let served = connections.load(Ordering::Relaxed);
        engine.shutdown();

        assert!(
            served >= 2,
            "the station was fetched {served} time(s); a dropped stream should be reopened"
        );
        println!("station fetched {served} times — reconnect ran");
    }

    /// The same path, fed by a live Icecast station instead of a file.
    ///
    /// Ignored by default: it needs both an audio device and the network.
    /// Run with `cargo test -- --ignored play_a_radio_station --nocapture`.
    #[test]
    #[ignore = "requires an audio device and network access"]
    fn play_a_radio_station() {
        // SomaFM is in the curated list, allows direct connections, and sends
        // ICY metadata — so this exercises the streaming path end to end.
        const URL: &str = "https://ice1.somafm.com/groovesalad-128-mp3";

        let engine = crate::init(no_store(), no_sink(), None);
        engine.set_volume(0.0);
        test_runtime()
            .block_on(engine.play_stream("soma-groovesalad".to_string(), URL.to_string(), None))
            .expect("engine accepts the station");

        std::thread::sleep(Duration::from_millis(2500));
        let played = engine.frames_played();
        engine.shutdown();

        assert!(
            played > 4_000,
            "only {played} frames played — the station did not reach the device"
        );
        println!("radio played {played} frames");
    }

    #[test]
    fn seeking_a_pending_remote_transition_seeks_the_native_decoder() {
        let input = fixtures::ramp(1, 16_000);
        let expected = f32::from(input[4_000]) / 32_768.0;
        let address = crate::remote::test_server::serve(fixtures::wav_bytes(8_000, 1, &input));
        let next = QueueEntry {
            track_id: 2,
            location: Location::Remote(crate::RemoteFile {
                url: format!("{address}/track.wav"),
                extension: Some("wav".to_owned()),
                bounded_range: None,
            }),
            duration_secs: 2.0,
            gain_db: 0.0,
        };
        let outgoing = QueueEntry {
            track_id: 1,
            location: std::path::PathBuf::from("finished.wav").into(),
            duration_secs: 1.0,
            gain_db: 0.0,
        };
        let mut engine = bare_engine();
        engine.queue.load(vec![outgoing, next], 0);
        engine.mode = Mode::Local;
        engine.current_duration = 1.0;
        engine.refresh_preload();
        engine.advance_or_stop();
        assert!(engine.pending_current.is_some());
        assert!(engine.seek(0.5));
        until(|| {
            engine.poll_current();
            engine.pending_current.is_none()
        });
        let decoder = engine
            .decoder
            .as_mut()
            .expect("the positioned source opens");
        assert!((decoder.position_secs() - 0.5).abs() < 0.01);
        let mut samples = [0.0; 64];
        assert!(decoder.read(&mut samples).unwrap() > 0);
        assert!(
            (samples[0] - expected).abs() < 0.001,
            "audio follows the requested seek: {} instead of {expected}",
            samples[0]
        );
        assert!((engine.position_secs() - 0.5).abs() < 0.01);
    }

    #[test]
    fn relative_seeks_while_opening_use_the_pending_tracks_duration() {
        let mut engine = bare_engine();
        let (resolver, calls) = controlled_resolver();
        engine.resolver = Some(resolver);
        let mut entry = deferred_entry(1, "long-track");
        entry.duration_secs = 100.0;
        engine.queue.load(vec![entry], 0);
        engine.mode = Mode::Local;
        engine.current_duration = 10.0;
        engine.start_current();
        let first = next_resolve(&calls);
        engine.seek_by(30.0, SeekDirection::Forward);
        let second = next_resolve(&calls);
        assert_cancelled(first.reply);
        assert_eq!(engine.position_secs(), 30.0);
        assert_eq!(engine.current_duration, 100.0);
        engine.seek_by(5.0, SeekDirection::Backward);
        let third = next_resolve(&calls);
        assert_cancelled(second.reply);
        assert_eq!(engine.position_secs(), 25.0);
        engine.stop();
        assert_cancelled(third.reply);
    }

    #[test]
    fn a_failed_speculative_preload_is_reopened_when_the_track_is_needed() {
        let (mut engine, sink) = captured_engine();
        let (resolver, calls) = controlled_resolver();
        engine.resolver = Some(resolver);
        engine
            .queue
            .load(vec![deferred_entry(1, "one"), deferred_entry(2, "two")], 0);
        engine.mode = Mode::Local;
        engine.current_duration = 1.0;
        engine.refresh_preload();
        next_resolve(&calls)
            .reply
            .send(Err("temporary failure".to_owned()))
            .unwrap();
        until(|| {
            engine.refresh_preload();
            matches!(engine.preloading.as_ref(), Some((_, PreloadState::Failed)))
        });
        engine.advance_or_stop();
        let foreground = next_resolve(&calls);
        assert_eq!(foreground.id, "two");
        let address = crate::remote::test_server::serve(fixtures::wav_bytes(
            8_000,
            1,
            &fixtures::ramp(1, 16_000),
        ));
        foreground
            .reply
            .send(Ok(crate::RemoteFile {
                url: format!("{address}/track.wav"),
                extension: Some("wav".to_owned()),
                bounded_range: None,
            }))
            .unwrap();
        until(|| {
            engine.poll_current();
            engine.pending_current.is_none()
        });
        let mut audio = [0.0; 64];
        assert!(
            engine
                .decoder
                .as_mut()
                .expect("foreground open recovered")
                .read(&mut audio)
                .unwrap()
                > 0
        );
        assert_eq!(engine.queue.index(), 1);
        assert!(!sink.0.lock().unwrap().iter().any(|event| {
            matches!(event, EngineEvent::Error { message } if message.contains("disconnected"))
        }));
    }

    #[test]
    #[ignore = "requires an audio output device"]
    fn a_failed_refresh_keeps_the_requested_position_with_an_active_output_clock() {
        let (mut engine, _) = captured_engine();
        engine.params.set_volume(0.0);
        engine
            .ensure_output()
            .expect("an output device is available");
        let (resolver, calls) = controlled_resolver();
        engine.resolver = Some(resolver);
        engine.queue.load(vec![deferred_entry(1, "one")], 0);
        engine.mode = Mode::Local;
        engine.reported_index = Some(0);
        let rate = engine.device_rate();
        assert!(rate > 0);
        engine.params.reset_frames(u64::from(rate));
        engine.frames_written = u64::from(rate);
        engine.handle(EngineCommand::RefreshCurrent {
            position_secs: 0.5,
            playing: false,
        });
        next_resolve(&calls)
            .reply
            .send(Err("source unavailable".to_owned()))
            .unwrap();
        until(|| {
            engine.poll_current();
            engine.pending_current.is_none()
        });
        assert!(!engine.playing);
        assert_eq!(engine.position_secs(), 0.5);
        assert_eq!(engine.queue.index(), 0);
    }

    fn seeked_events(sink: &CaptureSink) -> Vec<(i64, f64)> {
        sink.0
            .lock()
            .unwrap()
            .iter()
            .filter_map(|event| match event {
                EngineEvent::Seeked {
                    track_id,
                    position_secs,
                } => Some((*track_id, *position_secs)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn transport_state_reports_the_actual_shuffled_successor() {
        let (mut engine, sink) = captured_engine();
        engine.queue.load(missing_entries(5), 0);
        engine.queue.set_shuffle(true);
        engine.mode = Mode::Local;
        for _ in 0..engine.queue.len() - 1 {
            engine.emit_state();
            assert!(matches!(
                sink.0.lock().unwrap().last(),
                Some(EngineEvent::State { has_next: true, .. })
            ));
            engine.queue.advance().unwrap();
        }
        assert!(engine.queue.peek_next().is_none());
        engine.emit_state();
        assert!(matches!(
            sink.0.lock().unwrap().last(),
            Some(EngineEvent::State {
                has_next: false,
                ..
            })
        ));
        engine.queue.set_repeat(true);
        engine.emit_state();
        assert!(matches!(
            sink.0.lock().unwrap().last(),
            Some(EngineEvent::State { has_next: true, .. })
        ));
        for mode in [Mode::Radio, Mode::Idle] {
            engine.mode = mode;
            engine.emit_state();
            assert!(matches!(
                sink.0.lock().unwrap().last(),
                Some(EngineEvent::State {
                    has_next: false,
                    ..
                })
            ));
        }
    }

    #[test]
    fn seeks_report_only_confirmed_landings_with_the_queue_identity() {
        let (mut engine, sink) = captured_engine();
        engine.queue.load(missing_entries(1), 0);
        let tone = engine_with_tone(3);
        engine.mode = Mode::Local;
        engine.current_duration = tone.current_duration;
        engine.decoder = tone.decoder;
        engine.handle(EngineCommand::SeekTrack {
            track_id: 0,
            position_secs: 0.51234,
        });
        let landed = engine.decoder.as_ref().unwrap().position_secs();
        assert_eq!(seeked_events(&sink), vec![(0, landed)]);
        assert!((landed - 0.51234).abs() < 1.0 / 44_100.0);
        {
            let events = sink.0.lock().unwrap();
            let confirmation = events
                .iter()
                .position(|event| matches!(event, EngineEvent::Seeked { .. }))
                .unwrap();
            let progress = events
                .iter()
                .position(|event| matches!(event, EngineEvent::Position { .. }))
                .unwrap();
            assert!(confirmation < progress);
        }
        sink.0.lock().unwrap().clear();
        engine.seek_by(0.25, SeekDirection::Forward);
        let landed = engine.decoder.as_ref().unwrap().position_secs();
        assert_eq!(seeked_events(&sink), vec![(0, landed)]);
        sink.0.lock().unwrap().clear();
        assert!(!engine.seek(f64::INFINITY));
        engine.handle(EngineCommand::SeekTrack {
            track_id: 999,
            position_secs: 1.0,
        });
        engine.emit_progress_now();
        engine.emit_progress_now();
        assert!(seeked_events(&sink).is_empty());
        engine.mode = Mode::Radio;
        engine.handle(EngineCommand::SeekTrack {
            track_id: 0,
            position_secs: 1.0,
        });
        assert!(seeked_events(&sink).is_empty());
    }

    #[test]
    fn next_then_stale_set_position_does_not_seek_the_new_track() {
        let (mut engine, sink) = captured_engine();
        let (resolver, calls) = controlled_resolver();
        engine.resolver = Some(resolver);
        engine.queue.load(
            vec![deferred_entry(1, "first"), deferred_entry(2, "second")],
            0,
        );
        engine.mode = Mode::Local;
        engine.handle(EngineCommand::Next);
        let pending = next_resolve(&calls);
        assert_eq!(pending.id, "second");
        engine.handle(EngineCommand::SeekTrack {
            track_id: 1,
            position_secs: 0.5,
        });
        engine.handle(EngineCommand::RefreshTrack {
            track_id: 1,
            position_secs: 0.5,
            playing: false,
        });
        assert_eq!(engine.pending_current.as_ref().unwrap().position_secs, 0.0);
        assert!(
            calls.try_recv().is_err(),
            "stale seek must not reopen the source"
        );
        assert!(seeked_events(&sink).is_empty());
        engine.stop();
        assert_cancelled(pending.reply);
    }

    #[test]
    fn a_pending_open_confirms_a_seek_only_after_native_positioning() {
        let (path, entry) = wav_track(7, "audio-stack-seek-confirmation");
        let (mut engine, sink) = captured_engine();
        engine.mode = Mode::Local;
        engine.queue.load(vec![entry], 0);
        // Advance owns its output already; this exercises the real opener
        // and installer without constructing an audio device.
        engine.open_current(1, OpenPurpose::Advance, 0.1, false);
        engine.handle(EngineCommand::SeekTrack {
            track_id: 7,
            position_secs: 0.51234,
        });
        assert!(
            seeked_events(&sink).is_empty(),
            "accepting a target is not success"
        );
        until(|| {
            engine.poll_current();
            engine.pending_current.is_none()
        });
        let landed = engine.decoder.as_ref().unwrap().position_secs();
        assert!((landed - 0.51234).abs() < 1.0 / 44_100.0);
        assert_eq!(seeked_events(&sink), vec![(7, landed)]);
        engine.emit_progress_now();
        engine.emit_progress_now();
        assert_eq!(seeked_events(&sink), vec![(7, landed)]);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_cancelled_or_failed_deferred_seek_never_confirms() {
        let (mut engine, sink) = captured_engine();
        let (resolver, calls) = controlled_resolver();
        engine.resolver = Some(resolver);
        engine.mode = Mode::Local;
        engine.queue.load(vec![deferred_entry(8, "cancelled")], 0);
        engine.open_current(1, OpenPurpose::Advance, 0.0, false);
        let initial = next_resolve(&calls);
        engine.seek(0.5);
        let sought = next_resolve(&calls);
        assert_cancelled(initial.reply);
        assert!(seeked_events(&sink).is_empty());
        engine.stop();
        assert_cancelled(sought.reply);
        assert!(!engine.poll_current());
        assert!(seeked_events(&sink).is_empty());

        engine.mode = Mode::Local;
        engine.open_current(1, OpenPurpose::Advance, 0.0, false);
        let initial = next_resolve(&calls);
        engine.seek(0.5);
        let sought = next_resolve(&calls);
        assert_cancelled(initial.reply);
        sought.reply.send(Err("not playable".to_owned())).unwrap();
        until(|| {
            engine.poll_current();
            engine.pending_current.is_none()
        });
        assert_eq!(engine.mode, Mode::Idle);
        assert!(seeked_events(&sink).is_empty());
    }

    #[test]
    fn same_identity_restarts_confirm_once_and_ticks_do_not() {
        let (mut engine, sink) = captured_engine();
        engine.queue.load(missing_entries(1), 0);
        engine.mode = Mode::Local;
        let tone = engine_with_tone(3);
        engine.install_decoder(tone.decoder.unwrap(), 0.0);
        engine.emit_progress_now();
        assert!(
            seeked_events(&sink).is_empty(),
            "initial load is not a seek"
        );
        let tone = engine_with_tone(3);
        engine.install_decoder(tone.decoder.unwrap(), 0.0);
        assert!(
            seeked_events(&sink).is_empty(),
            "wait for the audible origin"
        );
        engine.emit_progress_now();
        engine.emit_progress_now();
        assert_eq!(seeked_events(&sink), vec![(0, 0.0)]);
        sink.0.lock().unwrap().clear();
        let tone = engine_with_tone(3);
        engine.install_decoder_at(tone.decoder.unwrap(), 0.0, 0.5, true);
        engine.emit_progress_now();
        assert!(
            seeked_events(&sink).is_empty(),
            "installation alone is not a seek"
        );
    }

    #[test]
    fn a_one_item_repeat_confirms_only_when_the_join_becomes_audible() {
        let (mut engine, sink) = captured_engine();
        let entry = missing_entries(1).remove(0);
        engine.queue.load(vec![entry.clone()], 0);
        engine.queue.set_repeat(true);
        engine.mode = Mode::Local;
        let tone = engine_with_tone(3);
        engine.install_decoder(tone.decoder.unwrap(), 0.0);
        engine.emit_progress_now();
        engine.frames_written = 128;
        let tone = engine_with_tone(3);
        engine.preloaded = tone.decoder;
        engine.preloaded_entry = Some(entry);
        engine.advance_or_stop();
        engine.params.reset_frames(127);
        engine.emit_progress_now();
        assert!(seeked_events(&sink).is_empty());
        engine.params.reset_frames(128);
        engine.emit_progress_now();
        engine.params.reset_frames(129);
        engine.emit_progress_now();
        assert_eq!(seeked_events(&sink), vec![(0, 0.0)]);
    }

    #[test]
    fn natural_advance_updates_has_next_only_at_the_audible_boundary() {
        let (mut engine, sink) = captured_engine();
        engine.queue.load(missing_entries(3), 0);
        engine.mode = Mode::Local;
        let tone = engine_with_tone(3);
        engine.install_decoder(tone.decoder.unwrap(), 0.0);
        engine.emit_progress_now();
        sink.0.lock().unwrap().clear();
        for frame in [128, 256] {
            engine.frames_written = frame;
            engine.preloaded = engine_with_tone(3).decoder;
            engine.preloaded_entry = engine.queue.peek_next().cloned();
            engine.advance_or_stop();
        }
        assert_eq!(engine.queue.index(), 2, "decode is two tracks ahead");
        engine.params.reset_frames(127);
        engine.emit_progress_now();
        assert!(!sink
            .0
            .lock()
            .unwrap()
            .iter()
            .any(|event| { matches!(event, EngineEvent::State { .. }) }));
        engine.params.reset_frames(128);
        engine.emit_progress_now();
        {
            let events = sink.0.lock().unwrap();
            let changed = events
                .iter()
                .position(|event| matches!(event, EngineEvent::TrackChanged { index: 1 }))
                .unwrap();
            assert!(matches!(
                events.get(changed + 1),
                Some(EngineEvent::State {
                    index: 1,
                    has_next: false,
                    ..
                })
            ));
        }
        sink.0.lock().unwrap().clear();
        engine.params.reset_frames(256);
        engine.emit_progress_now();
        let events = sink.0.lock().unwrap();
        let changed = events
            .iter()
            .position(|event| matches!(event, EngineEvent::TrackChanged { index: 2 }))
            .unwrap();
        assert!(matches!(
            events.get(changed + 1),
            Some(EngineEvent::State {
                index: 2,
                has_next: false,
                ..
            })
        ));
    }

    #[test]
    fn guarded_refresh_waits_for_success_and_never_confirms_failure_or_cancellation() {
        let (mut engine, sink) = captured_engine();
        let (resolver, calls) = controlled_resolver();
        engine.resolver = Some(resolver);
        engine.mode = Mode::Local;
        engine.queue.load(vec![deferred_entry(9, "refresh")], 0);
        engine.handle(EngineCommand::RefreshTrack {
            track_id: 9,
            position_secs: 0.5,
            playing: false,
        });
        let failed = next_resolve(&calls);
        assert!(engine.pending_current.as_ref().unwrap().notify_seek);
        assert!(seeked_events(&sink).is_empty());
        failed
            .reply
            .send(Err("refresh unavailable".to_owned()))
            .unwrap();
        until(|| {
            engine.poll_current();
            engine.pending_current.is_none()
        });
        assert!(!engine.playing);
        assert!(seeked_events(&sink).is_empty());
        engine.handle(EngineCommand::RefreshTrack {
            track_id: 9,
            position_secs: 0.75,
            playing: false,
        });
        let cancelled = next_resolve(&calls);
        engine.stop();
        assert_cancelled(cancelled.reply);
        assert!(!engine.poll_current());
        assert!(seeked_events(&sink).is_empty());
    }

    #[test]
    fn guarded_transport_checks_the_outgoing_crossfade_identity_before_cancelling() {
        let (mut engine, _consumer, path0, path1) =
            crossfading_engine("audio-stack-guarded-crossfade");
        let sink = Arc::new(CaptureSink(std::sync::Mutex::new(Vec::new())));
        engine.sink = sink.clone();
        engine.decoder_track_id = Some(0);
        assert!(engine.maybe_start_transition());
        assert_eq!(engine.queue.current().unwrap().track_id, 1);
        engine.handle(EngineCommand::SeekTrack {
            track_id: 1,
            position_secs: 0.25,
        });
        engine.handle(EngineCommand::RefreshTrack {
            track_id: 1,
            position_secs: 0.25,
            playing: false,
        });
        assert!(
            engine.transition.is_some(),
            "stale guards must not mutate playback"
        );
        assert!(seeked_events(&sink).is_empty());
        engine.handle(EngineCommand::SeekTrack {
            track_id: 0,
            position_secs: 0.25,
        });
        assert!(engine.transition.is_none());
        assert_eq!(engine.queue.current().unwrap().track_id, 0);
        let landed = engine.decoder.as_ref().unwrap().position_secs();
        assert_eq!(seeked_events(&sink), vec![(0, landed)]);
        assert!((landed - 0.25).abs() < 1.0 / 44_100.0);
        let _ = std::fs::remove_file(path0);
        let _ = std::fs::remove_file(path1);
    }

    #[test]
    fn cancelling_a_crossfade_republishes_the_restored_successor() {
        for command in [
            EngineCommand::SetCrossfade(false),
            EngineCommand::Seek(f64::INFINITY),
            EngineCommand::JumpTo(usize::MAX),
        ] {
            let (mut engine, _consumer, path0, path1) =
                crossfading_engine("audio-stack-cancelled-crossfade-state");
            let sink = Arc::new(CaptureSink(std::sync::Mutex::new(Vec::new())));
            engine.sink = sink.clone();
            engine.decoder_track_id = Some(0);
            assert!(engine.maybe_start_transition());
            engine.emit_state();
            assert!(matches!(
                sink.0.lock().unwrap().last(),
                Some(EngineEvent::State {
                    index: 1,
                    has_next: false,
                    ..
                })
            ));
            sink.0.lock().unwrap().clear();
            engine.handle(command);
            assert!(engine.transition.is_none());
            assert_eq!(engine.queue.index(), 0);
            assert!(sink.0.lock().unwrap().iter().any(|event| {
                matches!(
                    event,
                    EngineEvent::State {
                        index: 0,
                        has_next: true,
                        ..
                    }
                )
            }));
            assert!(seeked_events(&sink).is_empty());
            let _ = std::fs::remove_file(path0);
            let _ = std::fs::remove_file(path1);
        }
    }
}
