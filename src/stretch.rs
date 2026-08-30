//! Time-stretch: tempo without pitch, as a [`Chain`](super::chain) effect.
//!
//! Wraps the `timestretch` crate's real-time engine. The profile is
//! WideKeylock — the CDJ "wide Master Tempo" phase-vocoder — because it is
//! the only one that keylocks the full spectrum across its whole tempo
//! range: the narrower Keylock profile fades to plain varispeed (pitch
//! following tempo) beyond ±35%, which would break this effect's one
//! promise. WideKeylock's contract also bounds our public ratio range. The
//! backend is pull-based and fills silence when pulled past what was fed, so
//! the one rule of this wrapper is that every pull is gated on the backend's
//! own `demand_hint` — mid-stream underrun silence is impossible by
//! construction, not by tuning.
//!
//! Off is structural: a disabled `TimeStretch` sheds its backend at the next
//! chain reset and drops out of the chain entirely, restoring the
//! byte-identical no-effects path. Disabling mid-play ramps the backend to
//! unity first (the crate's own click-free retarget), because the backend is
//! not bit-transparent at 1.0 and cutting straight to bypass would both
//! click and drop its in-flight tail.
//!
//! Enabling mid-play starts the backend cold: any reported pipeline delay
//! is discarded from the first output to keep the playhead aligned
//! (WideKeylock reports zero — it buffers lookahead source-side instead),
//! and a short fade-in covers the splice.

use timestretch::engine::{
    Engine, EngineConfig, EngineController, EngineProcessor, EngineProfile, SourceProducer,
};

use super::chain::Effect;

/// Frames per gated pull from the backend, and the callback size it is told
/// to optimize for. Mirrors the resampler's 1024-frame chunking.
const PULL_FRAMES: usize = 1024;

/// Source-ring depth handed to the backend: comfortably more than one
/// `demand_hint` at the fastest tempo, so feeding and pulling never deadlock.
const SOURCE_CAPACITY_FRAMES: usize = PULL_FRAMES * 16;

/// Frames of linear fade-in applied to the first audible output after a cold
/// start, covering the enable splice. ~5 ms at 48 kHz.
const FADE_FRAMES: usize = 256;

/// The tempo range offered to hosts: the span the WideKeylock profile
/// documents full-spectrum pitch preservation over. The backend accepts up
/// to 4.0, but past 2.0 nothing guarantees the pitch stays put — and a
/// time-stretch that moves pitch is a broken one, so the range is the
/// guarantee, not the mechanism's limit.
pub const TEMPO_RATIO_RANGE: (f32, f32) = (0.25, 2.0);

/// The time-stretch effect: `enabled`/`ratio` are the user's setting, the
/// backend exists only while there is a device shape to build it for.
pub struct TimeStretch {
    enabled: bool,
    ratio: f32,
    rate: u32,
    channels: usize,
    backend: Option<Backend>,
}

struct Backend {
    controller: EngineController,
    processor: EngineProcessor,
    producer: SourceProducer,
    /// Interleaved scratch for one pull.
    scratch: Vec<f32>,
    /// The ratio the backend is currently targeting — 1.0 while ramping out
    /// after a disable, whatever the user set otherwise.
    target_ratio: f64,
    /// Cold-start output still to discard (the stage chain's pipeline
    /// delay), keeping the playhead aligned across an enable.
    discard_frames: usize,
    /// Fade-in still to apply to fresh output after a (re)start.
    fade_frames: usize,
    /// Whether any source has been fed since the last restart — a fresh
    /// backend has nothing in flight regardless of what its estimates say.
    primed: bool,
}

impl TimeStretch {
    pub fn new(ratio: f32) -> Self {
        Self {
            enabled: true,
            ratio,
            rate: 0,
            channels: 0,
            backend: None,
        }
    }

    /// Updates the user's setting. Disabling ramps the backend to unity —
    /// audibly neutral within its ~30 ms retarget — and leaves the teardown
    /// to the next chain reset, so the switch itself can never click.
    pub fn set(&mut self, enabled: bool, ratio: f32) {
        self.enabled = enabled;
        if enabled {
            self.ratio = ratio;
        }
        let target = if enabled { f64::from(ratio) } else { 1.0 };
        if let Some(backend) = self.backend.as_mut() {
            backend.target_ratio = target;
            backend.controller.set_tempo_rate(target);
        }
    }

    /// The current setting, for the event echo and `describe`.
    pub fn setting(&self) -> (bool, f32) {
        (self.enabled, self.ratio)
    }
}

impl Effect for TimeStretch {
    fn name(&self) -> &'static str {
        "time-stretch"
    }

    fn is_active(&self) -> bool {
        self.enabled || self.backend.is_some()
    }

    fn is_bypassed(&self) -> bool {
        self.backend.is_none()
    }

    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) -> Result<(), String> {
        let Some(backend) = self.backend.as_mut() else {
            output.extend_from_slice(input);
            return Ok(());
        };
        backend.feed(input, output, self.channels);
        Ok(())
    }

    fn drain(&mut self, output: &mut Vec<f32>) -> Result<(), String> {
        if let Some(backend) = self.backend.as_mut() {
            backend.drain(output, self.channels);
        }
        Ok(())
    }

    fn reset(&mut self) {
        if !self.enabled {
            // The pending disable completes here: backend gone, effect
            // inactive, the chain retires it.
            self.backend = None;
            return;
        }
        if let Some(backend) = self.backend.as_mut() {
            backend.restart();
        }
    }

    fn time_ratio(&self) -> f64 {
        self.backend.as_ref().map(|b| b.target_ratio).unwrap_or(1.0)
    }

    fn pending_output_frames(&self) -> u64 {
        let Some(backend) = self.backend.as_ref() else {
            return 0;
        };
        if !backend.primed {
            return 0;
        }
        let buffered = backend.producer.occupied_frames() as f64 / backend.target_ratio.max(0.25);
        buffered as u64 + backend.processor.pipeline_latency_frames() as u64
    }

    fn matches(&self, rate: u32, channels: usize) -> bool {
        self.rate == rate && self.channels == channels && (self.backend.is_some() || !self.enabled)
    }

    fn reconfigure(&mut self, rate: u32, channels: usize) -> Result<(), String> {
        self.rate = rate;
        self.channels = channels;
        self.backend = None;
        if self.enabled && rate > 0 && channels > 0 {
            self.backend = Some(Backend::build(rate, channels, self.ratio)?);
        }
        Ok(())
    }

    fn spawn_mirror(&self) -> Box<dyn Effect> {
        let backend = if self.enabled && self.rate > 0 && self.channels > 0 {
            match Backend::build(self.rate, self.channels, self.ratio) {
                Ok(backend) => Some(backend),
                Err(message) => {
                    // Non-fatal: the incoming track plays unstretched until
                    // the next reconfigure retries the build.
                    log::warn!("time-stretch mirror: {}", message);
                    None
                }
            }
        } else {
            None
        };
        Box::new(TimeStretch {
            enabled: self.enabled,
            ratio: self.ratio,
            rate: self.rate,
            channels: self.channels,
            backend,
        })
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Backend {
    fn build(rate: u32, channels: usize, ratio: f32) -> Result<Self, String> {
        let handles = Engine::build(EngineConfig {
            sample_rate: rate,
            channels,
            profile: EngineProfile::WideKeylock,
            initial_tempo_rate: f64::from(ratio),
            max_block_frames: PULL_FRAMES,
            source_capacity_frames: SOURCE_CAPACITY_FRAMES,
            pre_analysis: None,
        })
        .map_err(|e| format!("build time-stretch backend: {}", e))?;
        let discard = handles.processor.pipeline_latency_frames();
        Ok(Self {
            controller: handles.controller,
            processor: handles.processor,
            producer: handles.source,
            scratch: Vec::new(),
            target_ratio: f64::from(ratio),
            discard_frames: discard,
            fade_frames: FADE_FRAMES,
            primed: false,
        })
    }

    /// Feeds interleaved input and pulls whatever is safely ready. The push
    /// side loops on partial acceptance; a full source ring always leaves
    /// more than a `demand_hint` buffered, so pulling is what makes room.
    fn feed(&mut self, input: &[f32], output: &mut Vec<f32>, channels: usize) {
        let channels = channels.max(1);
        let mut offset = 0;
        while offset < input.len() {
            let accepted = self.producer.push(&input[offset..]);
            offset += accepted * channels;
            if accepted > 0 {
                self.primed = true;
                continue;
            }
            let before = output.len();
            self.pull_ready(output, channels);
            if output.len() == before {
                // Cannot push and cannot pull: unreachable with our ring
                // sizing, but an audible-latency bug beats a livelock.
                log::warn!("time-stretch feed stalled; dropping remainder");
                break;
            }
        }
        self.pull_ready(output, channels);
    }

    /// Pulls `PULL_FRAMES` blocks while the backend holds enough source to
    /// guarantee each one renders without underrun. The 10% pad covers the
    /// instantaneous rate overshooting the target while a retarget ramps.
    fn pull_ready(&mut self, output: &mut Vec<f32>, channels: usize) {
        loop {
            let need = self
                .producer
                .demand_hint(PULL_FRAMES, self.target_ratio * 1.1);
            if self.producer.occupied_frames() < need {
                return;
            }
            self.pull_block(PULL_FRAMES, output, channels);
        }
    }

    /// One unconditional pull, minus the cold-start discard and under the
    /// splice fade-in.
    fn pull_block(&mut self, frames: usize, output: &mut Vec<f32>, channels: usize) {
        self.scratch.resize(frames * channels, 0.0);
        self.processor.process(&mut self.scratch);

        let mut from = 0;
        if self.discard_frames > 0 {
            let discard = self.discard_frames.min(frames);
            self.discard_frames -= discard;
            from = discard * channels;
        }
        if from >= self.scratch.len() {
            return;
        }
        let fresh = &mut self.scratch[from..];
        if self.fade_frames > 0 {
            let frames_here = (fresh.len() / channels).min(self.fade_frames);
            let done = FADE_FRAMES - self.fade_frames;
            for frame in 0..frames_here {
                let gain = (done + frame + 1) as f32 / FADE_FRAMES as f32;
                for sample in &mut fresh[frame * channels..(frame + 1) * channels] {
                    *sample *= gain;
                }
            }
            self.fade_frames -= frames_here;
        }
        output.extend_from_slice(fresh);
    }

    /// End of stream: pad the varispeed lookahead out (`finish`), pull the
    /// buffered tail plus the stage chain's delay, then restart clean.
    fn drain(&mut self, output: &mut Vec<f32>, channels: usize) {
        if !self.primed {
            return;
        }
        let channels = channels.max(1);
        // `finish` needs ring room for its padding; pulling is what frees it.
        let mut guard = 64;
        while !self.producer.finish() && guard > 0 {
            self.pull_ready(output, channels);
            guard -= 1;
        }
        let mut remaining = (self.producer.occupied_frames() as f64 / self.target_ratio.max(0.25))
            .ceil() as usize
            + self.processor.pipeline_latency_frames();
        while remaining > 0 {
            let take = remaining.min(PULL_FRAMES);
            self.pull_block(take, output, channels);
            remaining -= take;
        }
        self.restart();
    }

    /// Hard restart for a discontinuity: the backend's own reset drops
    /// everything in flight (with its built-in declick release), and the
    /// next stream begins like a cold start.
    fn restart(&mut self) {
        self.processor.reset();
        self.discard_frames = self.processor.pipeline_latency_frames();
        self.fade_frames = FADE_FRAMES;
        self.primed = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;

    fn effect(ratio: f32) -> TimeStretch {
        let mut effect = TimeStretch::new(ratio);
        effect
            .reconfigure(RATE, 2)
            .expect("backend must build for a stereo 48 kHz device");
        effect
    }

    /// A stereo interleaved sine at `freq`, `frames` long.
    fn sine(freq: f32, frames: usize) -> Vec<f32> {
        let mut signal = Vec::with_capacity(frames * 2);
        for n in 0..frames {
            let sample = (2.0 * std::f32::consts::PI * freq * n as f32 / RATE as f32).sin() * 0.5;
            signal.push(sample);
            signal.push(sample);
        }
        signal
    }

    /// Processes `input` in pump-sized slices and drains, returning all
    /// output — the shape the engine feeds the chain in.
    fn run_through(effect: &mut TimeStretch, input: &[f32]) -> Vec<f32> {
        let mut output = Vec::new();
        for slice in input.chunks(2048 * 2) {
            effect.process(slice, &mut output).unwrap();
        }
        effect.drain(&mut output).unwrap();
        output
    }

    /// Dominant frequency of a mono view of interleaved stereo, via the same
    /// realfft the analyser uses.
    fn dominant_hz(interleaved: &[f32]) -> f32 {
        use realfft::RealFftPlanner;
        let mono: Vec<f32> = interleaved.chunks_exact(2).map(|f| f[0]).collect();
        // A power-of-two window from the middle, past any fade-in.
        let window = 16_384.min(mono.len() / 2);
        let start = (mono.len() - window) / 2;
        let mut buf: Vec<f32> = mono[start..start + window].to_vec();
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(window);
        let mut spectrum = fft.make_output_vec();
        fft.process(&mut buf, &mut spectrum).unwrap();
        let bin = spectrum
            .iter()
            .enumerate()
            .skip(1)
            .max_by(|a, b| a.1.norm().total_cmp(&b.1.norm()))
            .map(|(i, _)| i)
            .unwrap_or(0);
        bin as f32 * RATE as f32 / window as f32
    }

    #[test]
    fn time_stretch_changes_length_not_pitch() {
        let mut effect = effect(2.0);
        let input = sine(440.0, RATE as usize * 4);
        let output = run_through(&mut effect, &input);

        let ratio = (input.len() as f64) / (output.len() as f64);
        assert!(
            (ratio - 2.0).abs() < 0.15,
            "double tempo must roughly halve the frames: got {ratio:.3}x"
        );
        let hz = dominant_hz(&output);
        assert!(
            (hz - 440.0).abs() < 15.0,
            "pitch must not move with tempo: got {hz:.1} Hz"
        );
    }

    #[test]
    fn slowdown_produces_proportionally_more_frames() {
        let mut effect = effect(0.5);
        let input = sine(440.0, RATE as usize * 2);
        let output = run_through(&mut effect, &input);

        let ratio = (output.len() as f64) / (input.len() as f64);
        assert!(
            (ratio - 2.0).abs() < 0.15,
            "half tempo must roughly double the frames: got {ratio:.3}x"
        );
        let hz = dominant_hz(&output);
        assert!(
            (hz - 440.0).abs() < 15.0,
            "pitch must not move with tempo: got {hz:.1} Hz"
        );
    }

    #[test]
    fn unity_ratio_preserves_length() {
        let mut effect = effect(1.0);
        let input = sine(330.0, RATE as usize * 2);
        let output = run_through(&mut effect, &input);

        let ratio = (output.len() as f64) / (input.len() as f64);
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "unity tempo must keep the frame count: got {ratio:.3}x"
        );
    }

    #[test]
    fn live_ratio_change_has_no_discontinuity() {
        let mut effect = effect(1.0);
        let input = sine(220.0, RATE as usize * 2);
        let mut output = Vec::new();
        let half = input.len() / 2 / (2048 * 2) * (2048 * 2);
        effect.process(&input[..half], &mut output).unwrap();
        effect.set(true, 1.5);
        effect.process(&input[half..], &mut output).unwrap();
        effect.drain(&mut output).unwrap();

        // Past the enable fade-in, no adjacent samples of a smooth 220 Hz
        // tone may jump more than the tone's own steepest slope allows,
        // with headroom for the retarget window.
        let mono: Vec<f32> = output.chunks_exact(2).map(|f| f[0]).collect();
        let max_step = mono
            .windows(2)
            .skip(FADE_FRAMES)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_step < 0.05,
            "a live retarget must not click: max sample step {max_step:.4}"
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut effect = effect(1.0);
        let loud = sine(440.0, RATE as usize);
        let mut sink = Vec::new();
        effect.process(&loud, &mut sink).unwrap();
        effect.reset();

        let mut output = Vec::new();
        effect.process(&vec![0.0f32; 48_000], &mut output).unwrap();
        effect.drain(&mut output).unwrap();
        let peak = output.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(
            peak < 0.01,
            "audio from before a reset must not bleed after it: peak {peak:.4}"
        );
    }

    #[test]
    fn drain_flushes_the_tail_and_is_idempotent() {
        let mut effect = effect(1.0);
        let input = sine(440.0, 8192);
        let mut output = Vec::new();
        effect.process(&input, &mut output).unwrap();
        let held_back = input.len() - output.len();
        assert!(held_back > 0, "the backend buffers some tail internally");

        effect.drain(&mut output).unwrap();
        let shortfall = input.len() as i64 - output.len() as i64;
        assert!(
            shortfall.unsigned_abs() < 4096,
            "drain must recover the tail within a latency window: short {shortfall} samples"
        );

        let before = output.len();
        effect.drain(&mut output).unwrap();
        assert_eq!(output.len(), before, "a second drain must add nothing");
    }

    #[test]
    fn disabling_ramps_to_unity_and_reset_retires_the_backend() {
        let mut effect = effect(2.0);
        assert!(effect.is_active() && !effect.is_bypassed());

        effect.set(false, 2.0);
        assert!(
            effect.is_active(),
            "a pending disable keeps processing until the next reset"
        );
        assert_eq!(effect.time_ratio(), 1.0, "but already targets unity");

        effect.reset();
        assert!(!effect.is_active(), "reset completes the disable");
        assert!(effect.is_bypassed());
    }

    #[test]
    fn matches_tracks_the_built_shape() {
        let effect = effect(1.0);
        assert!(effect.matches(RATE, 2));
        assert!(!effect.matches(44_100, 2), "different rate");
        assert!(!effect.matches(RATE, 6), "different channel count");
    }

    #[test]
    fn a_bypassed_effect_passes_audio_through_untouched() {
        // Enabled but with no device shape yet: settings stored, audio
        // untouched until `reconfigure` builds the backend.
        let mut effect = TimeStretch::new(1.5);
        let input = sine(440.0, 512);
        let mut output = Vec::new();
        effect.process(&input, &mut output).unwrap();
        assert_eq!(output, input, "no backend must mean identical bytes");
        assert_eq!(effect.pending_output_frames(), 0);
    }
}
