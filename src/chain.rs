//! The decode-thread effect chain: pluggable DSP between the resampler and
//! the ring.
//!
//! Effects live here, not in the cpal callback, because they are allowed to
//! be heavy — the ring is precisely the buffer that decouples this work from
//! the realtime deadline. The chain owns the whole cross-cutting integration
//! once (bypass, crossfade mirroring, drain at end of stream, the playhead
//! timeline below), so adding an effect never touches the engine's pump,
//! seek, or transition code again.
//!
//! An empty or fully bypassed chain is byte-identical to not having one: the
//! resampler writes straight into the caller's buffer, exactly as the engine
//! did before the chain existed.
//!
//! The chain also owns the source-time **timeline**. `position_secs` rests on
//! "one device frame played == one source frame of musical time", and a
//! time-changing effect breaks that. As the chain produces audio it records
//! markers mapping ring positions (device frames written) to cumulative
//! source seconds, and the engine interpolates between them; with no markers
//! the engine's original division is used untouched.

use std::collections::VecDeque;

use super::resample::Resampler;

/// One pluggable DSP stage. Runs at device rate, post-resampler, on
/// interleaved f32 — never in the realtime callback.
pub trait Effect: Send {
    /// Stable name for error messages and logs.
    fn name(&self) -> &'static str;

    /// False once the effect is disabled and holds no audio; the chain drops
    /// inactive effects at the next reset.
    fn is_active(&self) -> bool;

    /// True while the effect is a byte-identical no-op, letting the chain
    /// skip it entirely. An effect that must keep consuming audio to stay
    /// warm (time-stretch ramped back to unity) returns false even when
    /// audibly neutral.
    fn is_bypassed(&self) -> bool;

    /// Processes interleaved `input`, appending to `output`. May produce
    /// more, fewer, or zero frames — internal buffering is the effect's own.
    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) -> Result<(), String>;

    /// Flushes whatever is still buffered inside into `output`. The effect is
    /// a fresh stream afterwards.
    fn drain(&mut self, output: &mut Vec<f32>) -> Result<(), String>;

    /// Drops audio state across a discontinuity (seek, flush, stop). Also
    /// where a disabled effect sheds its backend, going inactive.
    fn reset(&mut self);

    /// Source seconds represented by one second of output — the effect's
    /// current time ratio. 1.0 for every duration-preserving effect.
    fn time_ratio(&self) -> f64;

    /// Output-domain estimate of the frames currently buffered inside, so a
    /// gapless boundary can sit past them. 0 when nothing is in flight.
    fn pending_output_frames(&self) -> u64;

    /// Whether this instance is already built for this device shape.
    fn matches(&self, rate: u32, channels: usize) -> bool;

    /// Rebuilds for a new device shape, keeping the user's settings.
    fn reconfigure(&mut self, rate: u32, channels: usize) -> Result<(), String>;

    /// A fresh instance with the same settings and no audio state — the
    /// crossfade mirror for an incoming track.
    fn spawn_mirror(&self) -> Box<dyn Effect>;

    /// Typed access for the per-effect setters on [`Chain`]. Part of the
    /// plugin contract in every build, though only feature-gated setters
    /// call it — hence the allow.
    #[allow(dead_code)]
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// One point on the device-frame → source-seconds map. Audio written after
/// `device_frame` advances source time by `secs_per_frame` per frame, until
/// the next marker says otherwise.
#[derive(Clone, Copy, Debug)]
struct Marker {
    device_frame: u64,
    source_secs: f64,
    secs_per_frame: f64,
}

/// Markers recorded while a run at one slope still projects within this of
/// the exact accumulator are coalesced into the previous marker. Effect
/// buffering jitters the projection by well under this, so a steady ratio
/// keeps a single marker instead of one per pump.
const COALESCE_SECS: f64 = 0.002;

/// The device-frame → source-seconds map. Empty until a time-changing effect
/// produces audio, which is what keeps the engine's original position math
/// byte-for-byte in charge whenever the chain has never stretched time.
#[derive(Debug, Default)]
pub struct Timeline {
    markers: VecDeque<Marker>,
}

impl Timeline {
    fn clear(&mut self) {
        self.markers.clear();
    }

    /// Records that output written at `device_frame` sits at `source_secs`
    /// and advances at `secs_per_frame`. Regressions are clamped rather than
    /// honoured — the map must stay monotonic for `position_secs` to be.
    fn record(&mut self, device_frame: u64, source_secs: f64, secs_per_frame: f64) {
        if let Some(last) = self.markers.back_mut() {
            if device_frame < last.device_frame {
                return;
            }
            let source_secs = source_secs.max(last.source_secs);
            if device_frame == last.device_frame {
                last.source_secs = source_secs;
                last.secs_per_frame = secs_per_frame;
                return;
            }
            let projected =
                last.source_secs + (device_frame - last.device_frame) as f64 * last.secs_per_frame;
            if last.secs_per_frame == secs_per_frame
                && (projected - source_secs).abs() < COALESCE_SECS
            {
                return;
            }
            self.markers.push_back(Marker {
                device_frame,
                source_secs,
                secs_per_frame,
            });
            return;
        }
        self.markers.push_back(Marker {
            device_frame,
            source_secs,
            secs_per_frame,
        });
    }

    /// Cumulative source seconds at a device frame, or `None` when no marker
    /// covers it — the caller falls back to plain frames-over-rate, which is
    /// exact there by the seeding rule (see [`Chain::process`]).
    fn stream_secs(&self, device_frame: u64) -> Option<f64> {
        let marker = self
            .markers
            .iter()
            .rev()
            .find(|m| m.device_frame <= device_frame)?;
        Some(
            marker.source_secs
                + (device_frame - marker.device_frame) as f64 * marker.secs_per_frame,
        )
    }

    /// Drops markers fully behind `keep_from`, keeping the one in force.
    fn prune(&mut self, keep_from: u64) {
        while self.markers.len() > 1 && self.markers[1].device_frame <= keep_from {
            self.markers.pop_front();
        }
    }
}

/// The ordered effects between the resampler and the ring, plus the timeline
/// they imply. One lives on the engine for the playing track and one on each
/// crossfade [`Transition`](super::engine) for the incoming track.
pub struct Chain {
    effects: Vec<Box<dyn Effect>>,
    rate: u32,
    channels: usize,
    /// Ping-pong scratch between the resampler and the effects.
    scratch_a: Vec<f32>,
    scratch_b: Vec<f32>,
    /// Cumulative source seconds fed through the chain, advanced by exact
    /// input frames so ramp transients cannot accumulate drift. `None` until
    /// the first engaged run seeds it from the ring position.
    consumed_secs: Option<f64>,
    timeline: Timeline,
    /// A crossfade mirror records no markers: its output lands at ring
    /// positions only known at mix time. Flipped on when it is adopted.
    record_timeline: bool,
    /// Set by `drain`, cleared by `process`/`reset`, so the engine can call
    /// drain every idle turn of an exhausted track without re-flushing.
    drained: bool,
}

impl Default for Chain {
    fn default() -> Self {
        Self::new()
    }
}

impl Chain {
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
            rate: 0,
            channels: 0,
            scratch_a: Vec::new(),
            scratch_b: Vec::new(),
            consumed_secs: None,
            timeline: Timeline::default(),
            record_timeline: true,
            drained: false,
        }
    }

    /// Whether any effect is actually shaping audio right now.
    fn engaged(&self) -> bool {
        self.effects
            .iter()
            .any(|e| e.is_active() && !e.is_bypassed())
    }

    /// The chain's combined time ratio: source seconds per output second.
    /// 1.0 whenever nothing is engaged.
    pub fn time_ratio(&self) -> f64 {
        self.effects
            .iter()
            .filter(|e| e.is_active() && !e.is_bypassed())
            .map(|e| e.time_ratio())
            .product()
    }

    /// Output frames still buffered inside the engaged effects.
    pub fn pending_output_frames(&self) -> u64 {
        self.effects
            .iter()
            .filter(|e| e.is_active() && !e.is_bypassed())
            .map(|e| e.pending_output_frames())
            .sum()
    }

    /// The pump's one entry point: `input` through the resampler and the
    /// effects, appended to `out`. Disengaged, this is byte-identical to
    /// calling the resampler directly.
    ///
    /// `frames_written` is the ring position `out`'s existing content starts
    /// at, which is what pins the timeline markers: a sample appended here at
    /// offset `n` frames will be the ring's frame `frames_written + n`,
    /// because `out` drains into the ring strictly in order. Markers seed
    /// from `origin / rate` — exact, since everything before the first
    /// engaged run advanced one source frame per device frame.
    pub fn process(
        &mut self,
        resampler: Option<&mut Resampler>,
        input: &[f32],
        out: &mut Vec<f32>,
        frames_written: u64,
    ) -> Result<(), String> {
        let channels = self.channels.max(1);
        if !self.engaged() {
            let before = out.len();
            match resampler {
                Some(resampler) => resampler.process(input, out)?,
                None => out.extend_from_slice(input),
            }
            // Keep the map seamless for 1:1 audio that follows an engaged
            // run, so an earlier non-unity slope is not extrapolated over it.
            if self.record_timeline && self.consumed_secs.is_some() && self.rate > 0 {
                let produced = (out.len() - before) / channels;
                self.note_output(frames_written, before / channels, produced, produced, 1.0);
            }
            return Ok(());
        }

        self.drained = false;
        self.scratch_a.clear();
        match resampler {
            Some(resampler) => resampler.process(input, &mut self.scratch_a)?,
            None => self.scratch_a.extend_from_slice(input),
        }
        let consumed = self.scratch_a.len() / channels;
        let ratio = self.time_ratio();
        let before = out.len();
        self.run_effects(out)?;
        let produced = (out.len() - before) / channels;
        if self.record_timeline && self.rate > 0 {
            self.note_output(frames_written, before / channels, produced, consumed, ratio);
        }
        Ok(())
    }

    /// Runs `scratch_a` through every engaged effect and appends the result.
    fn run_effects(&mut self, out: &mut Vec<f32>) -> Result<(), String> {
        let Self {
            effects,
            scratch_a,
            scratch_b,
            ..
        } = self;
        let mut current: &mut Vec<f32> = scratch_a;
        let mut spare: &mut Vec<f32> = scratch_b;
        for effect in effects.iter_mut() {
            if !effect.is_active() || effect.is_bypassed() {
                continue;
            }
            spare.clear();
            effect
                .process(current, spare)
                .map_err(|e| format!("{}: {}", effect.name(), e))?;
            std::mem::swap(&mut current, &mut spare);
        }
        out.extend_from_slice(current);
        Ok(())
    }

    /// Records one timeline marker for a run of output and advances the
    /// source-time accumulator by the input that produced it.
    fn note_output(
        &mut self,
        frames_written: u64,
        out_frames_before: usize,
        produced_frames: usize,
        consumed_frames: usize,
        ratio: f64,
    ) {
        let rate = f64::from(self.rate);
        let origin = frames_written + out_frames_before as u64;
        let secs = *self
            .consumed_secs
            .get_or_insert_with(|| origin as f64 / rate);
        if produced_frames > 0 {
            self.timeline.record(origin, secs, ratio / rate);
        }
        self.consumed_secs = Some(secs + consumed_frames as f64 / rate);
    }

    /// Flushes every engaged effect's buffered tail into `out`, each drained
    /// tail continuing through the effects after it. Idempotent per stream.
    pub fn drain(&mut self, out: &mut Vec<f32>) -> Result<(), String> {
        if self.drained || !self.engaged() {
            return Ok(());
        }
        self.drained = true;
        let Self {
            effects,
            scratch_a,
            scratch_b,
            ..
        } = self;
        for split in 1..=effects.len() {
            let (head, rest) = effects.split_at_mut(split);
            let effect = head
                .last_mut()
                .expect("split_at_mut(1..) always leaves a head");
            if !effect.is_active() || effect.is_bypassed() {
                continue;
            }
            scratch_a.clear();
            effect
                .drain(scratch_a)
                .map_err(|e| format!("{}: {}", effect.name(), e))?;
            let mut current: &mut Vec<f32> = scratch_a;
            let mut spare: &mut Vec<f32> = scratch_b;
            for downstream in rest
                .iter_mut()
                .filter(|e| e.is_active() && !e.is_bypassed())
            {
                spare.clear();
                downstream
                    .process(current, spare)
                    .map_err(|e| format!("{}: {}", downstream.name(), e))?;
                std::mem::swap(&mut current, &mut spare);
            }
            out.extend_from_slice(current);
        }
        // The drained tail keeps the last marker's slope, which is what
        // produced it; no new marker is needed.
        Ok(())
    }

    /// A discontinuity: every effect drops its audio state, disabled effects
    /// drop out entirely, and the timeline starts over.
    pub fn reset(&mut self) {
        for effect in self.effects.iter_mut() {
            effect.reset();
        }
        self.effects.retain(|e| e.is_active());
        self.timeline.clear();
        self.consumed_secs = None;
        self.drained = false;
    }

    /// The device shape changed (or just became known): rebuild what needs
    /// rebuilding, then reset — the old shape's audio state is meaningless.
    pub fn reconfigure(&mut self, rate: u32, channels: usize) -> Result<(), String> {
        self.rate = rate;
        self.channels = channels;
        let mut error = None;
        for effect in self.effects.iter_mut() {
            if !effect.matches(rate, channels) {
                if let Err(message) = effect.reconfigure(rate, channels) {
                    error.get_or_insert(format!("{}: {}", effect.name(), message));
                }
            }
        }
        self.reset();
        match error {
            Some(message) => Err(message),
            None => Ok(()),
        }
    }

    /// A fresh chain with the same active effects and settings but no audio
    /// state — what a crossfade's incoming track processes through. It
    /// records no markers until [`Chain::adopt_timeline`] on completion.
    pub fn spawn_mirror(&self) -> Chain {
        Chain {
            effects: self
                .effects
                .iter()
                .filter(|e| e.is_active())
                .map(|e| e.spawn_mirror())
                .collect(),
            rate: self.rate,
            channels: self.channels,
            scratch_a: Vec::new(),
            scratch_b: Vec::new(),
            consumed_secs: None,
            timeline: Timeline::default(),
            record_timeline: false,
            drained: false,
        }
    }

    /// Promotes an adopted crossfade mirror to the chain of record: markers
    /// start recording, seeded fresh from the ring position — the same
    /// approximation the crossfade's position reporting already makes.
    pub fn adopt_timeline(&mut self) {
        self.record_timeline = true;
        self.consumed_secs = None;
        self.timeline.clear();
    }

    /// Timeline lookup for `position_secs`. `None` when no marker covers the
    /// frame — the engine's own division is exact there.
    pub fn stream_secs(&self, device_frame: u64) -> Option<f64> {
        self.timeline.stream_secs(device_frame)
    }

    /// Drops markers nothing can look up any more.
    pub fn prune_timeline(&mut self, keep_from: u64) {
        self.timeline.prune(keep_from);
    }

    /// Enables/disables time-stretch and sets its tempo ratio, creating or
    /// retiring the effect as needed. `ratio` must already be clamped.
    #[cfg(feature = "stretch")]
    pub fn set_time_stretch(&mut self, enabled: bool, ratio: f32) -> Result<(), String> {
        use super::stretch::TimeStretch;
        let existing = self
            .effects
            .iter_mut()
            .find_map(|e| e.as_any_mut().downcast_mut::<TimeStretch>());
        match existing {
            Some(effect) => {
                effect.set(enabled, ratio);
                if enabled && effect.is_bypassed() && self.rate > 0 && self.channels > 0 {
                    effect.reconfigure(self.rate, self.channels)?;
                }
            }
            None if enabled => {
                let mut effect = TimeStretch::new(ratio);
                if self.rate > 0 && self.channels > 0 {
                    effect.reconfigure(self.rate, self.channels)?;
                }
                self.effects.push(Box::new(effect));
            }
            None => {}
        }
        // Disabled with no backend still holding audio: gone right away.
        self.effects.retain(|e| e.is_active());
        Ok(())
    }

    /// The current time-stretch setting, for the event echo and `describe`.
    #[cfg(feature = "stretch")]
    pub fn time_stretch(&mut self) -> (bool, f32) {
        use super::stretch::TimeStretch;
        self.effects
            .iter_mut()
            .find_map(|e| e.as_any_mut().downcast_mut::<TimeStretch>())
            .map(|e| e.setting())
            .unwrap_or((false, 1.0))
    }

    /// Enables/disables the linear-phase FIR EQ, creating or retiring the
    /// effect as needed. It reads its band gains from `params`, the same block
    /// the callback EQ reads. Mirrors [`Chain::set_time_stretch`].
    #[cfg(feature = "fir-eq")]
    pub fn set_fir_eq(
        &mut self,
        enabled: bool,
        params: &std::sync::Arc<super::params::Params>,
    ) -> Result<(), String> {
        use super::fireq::FirEq;
        let existing = self
            .effects
            .iter_mut()
            .find_map(|e| e.as_any_mut().downcast_mut::<FirEq>());
        match existing {
            Some(effect) => {
                effect.set(enabled);
                if enabled && effect.is_bypassed() && self.rate > 0 && self.channels > 0 {
                    effect.reconfigure(self.rate, self.channels)?;
                }
            }
            None if enabled => {
                let mut effect = FirEq::new(std::sync::Arc::clone(params));
                effect.set(true);
                if self.rate > 0 && self.channels > 0 {
                    effect.reconfigure(self.rate, self.channels)?;
                }
                self.effects.push(Box::new(effect));
            }
            None => {}
        }
        self.effects.retain(|e| e.is_active());
        Ok(())
    }

    /// The current FIR EQ setting and its latency at `rate`, for the event echo
    /// and `describe`.
    #[cfg(feature = "fir-eq")]
    pub fn fir_eq(&mut self, rate: u32) -> (bool, f32) {
        use super::fireq::FirEq;
        self.effects
            .iter_mut()
            .find_map(|e| e.as_any_mut().downcast_mut::<FirEq>())
            .map(|e| (e.enabled(), e.latency_secs(rate)))
            .unwrap_or((false, 0.0))
    }

    /// Enables/disables the convolution effect and sets its IR and mix, creating
    /// or retiring the effect as needed. Loading a bad IR returns the error and
    /// leaves the effect bypassed. Mirrors [`Chain::set_time_stretch`].
    #[cfg(feature = "convolution")]
    pub fn set_convolution(
        &mut self,
        enabled: bool,
        ir_path: Option<std::path::PathBuf>,
        mix: f32,
    ) -> Result<(), String> {
        use super::convolution::Convolution;
        let existing = self
            .effects
            .iter_mut()
            .find_map(|e| e.as_any_mut().downcast_mut::<Convolution>());
        match existing {
            Some(effect) => {
                effect.set(enabled, ir_path, mix)?;
                if enabled && effect.is_bypassed() && self.rate > 0 && self.channels > 0 {
                    effect.reconfigure(self.rate, self.channels)?;
                }
            }
            None if enabled => {
                let mut effect = Convolution::new();
                effect.set(true, ir_path, mix)?;
                if self.rate > 0 && self.channels > 0 {
                    effect.reconfigure(self.rate, self.channels)?;
                }
                self.effects.push(Box::new(effect));
            }
            None => {}
        }
        self.effects.retain(|e| e.is_active());
        Ok(())
    }

    /// The current convolution setting, for the event echo and `describe`.
    #[cfg(feature = "convolution")]
    pub fn convolution(&mut self) -> (bool, f32) {
        use super::convolution::Convolution;
        self.effects
            .iter_mut()
            .find_map(|e| e.as_any_mut().downcast_mut::<Convolution>())
            .map(|e| e.setting())
            .unwrap_or((false, 0.0))
    }

    /// Enables/disables pitch-shift and sets its cents, creating or retiring
    /// the effect as needed. Pushed after any time-stretch, so the shift sits
    /// downstream of it (both duration-preserving — order is audio quality
    /// only). Mirrors [`Chain::set_time_stretch`].
    #[cfg(feature = "pitch")]
    pub fn set_pitch_shift(&mut self, enabled: bool, cents: f32) -> Result<(), String> {
        use super::pitch::PitchShift;
        let existing = self
            .effects
            .iter_mut()
            .find_map(|e| e.as_any_mut().downcast_mut::<PitchShift>());
        match existing {
            Some(effect) => {
                effect.set(enabled, cents);
                if enabled && effect.is_bypassed() && self.rate > 0 && self.channels > 0 {
                    effect.reconfigure(self.rate, self.channels)?;
                }
            }
            None if enabled => {
                let mut effect = PitchShift::new(cents);
                if self.rate > 0 && self.channels > 0 {
                    effect.reconfigure(self.rate, self.channels)?;
                }
                self.effects.push(Box::new(effect));
            }
            None => {}
        }
        self.effects.retain(|e| e.is_active());
        Ok(())
    }

    /// The current pitch-shift setting, for the event echo and `describe`.
    #[cfg(feature = "pitch")]
    pub fn pitch_shift(&mut self) -> (bool, f32) {
        use super::pitch::PitchShift;
        self.effects
            .iter_mut()
            .find_map(|e| e.as_any_mut().downcast_mut::<PitchShift>())
            .map(|e| e.setting())
            .unwrap_or((false, 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── timeline ────────────────────────────────────────────────────────

    #[test]
    fn empty_timeline_falls_back() {
        let timeline = Timeline::default();
        assert!(timeline.stream_secs(48_000).is_none());
    }

    #[test]
    fn constant_ratio_interpolates_linearly() {
        let mut timeline = Timeline::default();
        // Ratio 2.0 at 100 Hz: each device frame is 0.02 source seconds.
        timeline.record(1_000, 10.0, 0.02);
        assert_eq!(timeline.stream_secs(1_000), Some(10.0));
        let secs = timeline.stream_secs(1_050).unwrap();
        assert!((secs - 11.0).abs() < 1e-9, "expected 11.0, got {secs}");
    }

    #[test]
    fn ratio_change_mid_stream_keeps_position_exact() {
        let mut timeline = Timeline::default();
        timeline.record(0, 0.0, 0.02); // 2x for 100 frames
        timeline.record(100, 2.0, 0.01); // back to 1x
        let secs = timeline.stream_secs(150).unwrap();
        assert!((secs - 2.5).abs() < 1e-9, "expected 2.5, got {secs}");
    }

    #[test]
    fn position_is_monotonic_across_a_ratio_change() {
        let mut timeline = Timeline::default();
        timeline.record(0, 0.0, 0.02);
        timeline.record(100, 2.0, 0.005);
        timeline.record(300, 3.0, 0.02);
        let mut last = f64::MIN;
        for frame in 0..500 {
            let secs = timeline.stream_secs(frame).unwrap();
            assert!(
                secs >= last,
                "position went backwards at frame {frame}: {secs} < {last}"
            );
            last = secs;
        }
    }

    #[test]
    fn out_of_order_records_are_dropped_and_regressions_clamped() {
        let mut timeline = Timeline::default();
        timeline.record(100, 5.0, 0.01);
        timeline.record(50, 1.0, 0.01); // behind the last marker: dropped
        assert_eq!(timeline.stream_secs(100), Some(5.0));
        timeline.record(200, 1.0, 0.01); // source time regressed: clamped up
        let secs = timeline.stream_secs(200).unwrap();
        assert!(
            secs >= 5.0,
            "a clamped marker must not move source time backwards: {secs}"
        );
    }

    #[test]
    fn frames_before_the_first_marker_are_uncovered() {
        let mut timeline = Timeline::default();
        timeline.record(1_000, 10.0, 0.02);
        assert!(
            timeline.stream_secs(999).is_none(),
            "audio before the first marker was 1:1 and belongs to the fallback"
        );
    }

    #[test]
    fn pruning_keeps_the_marker_in_force() {
        let mut timeline = Timeline::default();
        timeline.record(0, 0.0, 0.02);
        timeline.record(100, 2.0, 0.01);
        timeline.record(200, 3.0, 0.04);
        timeline.prune(150);
        let secs = timeline.stream_secs(150).unwrap();
        assert!(
            (secs - 2.5).abs() < 1e-9,
            "the marker covering 150 must survive pruning: {secs}"
        );
        assert!(timeline.stream_secs(50).is_none(), "behind the kept marker");
    }

    #[test]
    fn steady_runs_coalesce_into_one_marker() {
        let mut timeline = Timeline::default();
        // Same slope, projections within tolerance: one pump after another
        // at a steady ratio.
        timeline.record(0, 0.0, 0.02);
        timeline.record(100, 2.0001, 0.02);
        timeline.record(200, 4.0002, 0.02);
        assert_eq!(timeline.markers.len(), 1, "steady state keeps one marker");
    }

    // ── chain ───────────────────────────────────────────────────────────

    /// Emits every `factor`-th frame: a toy time-changing effect, so the
    /// chain's plumbing is testable without any real DSP dependency.
    struct Decimate {
        factor: usize,
        channels: usize,
        active: bool,
    }

    impl Effect for Decimate {
        fn name(&self) -> &'static str {
            "decimate"
        }
        fn is_active(&self) -> bool {
            self.active
        }
        fn is_bypassed(&self) -> bool {
            false
        }
        fn process(&mut self, input: &[f32], output: &mut Vec<f32>) -> Result<(), String> {
            for frame in input.chunks_exact(self.channels).step_by(self.factor) {
                output.extend_from_slice(frame);
            }
            Ok(())
        }
        fn drain(&mut self, _output: &mut Vec<f32>) -> Result<(), String> {
            Ok(())
        }
        fn reset(&mut self) {}
        fn time_ratio(&self) -> f64 {
            self.factor as f64
        }
        fn pending_output_frames(&self) -> u64 {
            0
        }
        fn matches(&self, _rate: u32, _channels: usize) -> bool {
            true
        }
        fn reconfigure(&mut self, _rate: u32, _channels: usize) -> Result<(), String> {
            Ok(())
        }
        fn spawn_mirror(&self) -> Box<dyn Effect> {
            Box::new(Decimate {
                factor: self.factor,
                channels: self.channels,
                active: self.active,
            })
        }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }

    /// Holds everything back until drained — a stand-in for internal latency.
    struct HoldBack {
        held: Vec<f32>,
        channels: usize,
    }

    impl Effect for HoldBack {
        fn name(&self) -> &'static str {
            "hold-back"
        }
        fn is_active(&self) -> bool {
            true
        }
        fn is_bypassed(&self) -> bool {
            false
        }
        fn process(&mut self, input: &[f32], _output: &mut Vec<f32>) -> Result<(), String> {
            self.held.extend_from_slice(input);
            Ok(())
        }
        fn drain(&mut self, output: &mut Vec<f32>) -> Result<(), String> {
            output.append(&mut self.held);
            Ok(())
        }
        fn reset(&mut self) {
            self.held.clear();
        }
        fn time_ratio(&self) -> f64 {
            1.0
        }
        fn pending_output_frames(&self) -> u64 {
            (self.held.len() / self.channels) as u64
        }
        fn matches(&self, _rate: u32, _channels: usize) -> bool {
            true
        }
        fn reconfigure(&mut self, _rate: u32, _channels: usize) -> Result<(), String> {
            Ok(())
        }
        fn spawn_mirror(&self) -> Box<dyn Effect> {
            Box::new(HoldBack {
                held: Vec::new(),
                channels: self.channels,
            })
        }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }

    fn configured(rate: u32, channels: usize) -> Chain {
        let mut chain = Chain::new();
        chain.reconfigure(rate, channels).unwrap();
        chain
    }

    #[test]
    fn an_empty_chain_is_bit_transparent() {
        let mut chain = configured(48_000, 2);
        let input: Vec<f32> = (0..64).map(|n| n as f32 / 64.0).collect();
        let mut out = Vec::new();
        chain.process(None, &input, &mut out, 0).unwrap();
        assert_eq!(out, input, "no effects must mean identical bytes");
        assert!(chain.stream_secs(32).is_none(), "and no markers");
    }

    #[test]
    fn an_empty_chain_defers_to_the_resampler_untouched() {
        let input: Vec<f32> = (0..4096).map(|n| (n as f32 * 0.01).sin()).collect();

        let mut direct = Resampler::new(44_100, 48_000, 2).unwrap();
        let mut expected = Vec::new();
        direct.process(&input, &mut expected).unwrap();

        let mut through = Resampler::new(44_100, 48_000, 2).unwrap();
        let mut chain = configured(48_000, 2);
        let mut out = Vec::new();
        chain
            .process(Some(&mut through), &input, &mut out, 0)
            .unwrap();

        assert_eq!(out, expected, "the chain must not touch resampled audio");
    }

    #[test]
    fn an_engaged_effect_shapes_audio_and_records_a_marker() {
        let mut chain = configured(100, 1);
        chain.effects.push(Box::new(Decimate {
            factor: 2,
            channels: 1,
            active: true,
        }));

        let input = vec![0.5f32; 100]; // one source second at 100 Hz
        let mut out = Vec::new();
        chain.process(None, &input, &mut out, 0).unwrap();

        assert_eq!(out.len(), 50, "ratio 2 halves the frame count");
        let secs = chain.stream_secs(25).unwrap();
        assert!(
            (secs - 0.5).abs() < 1e-9,
            "25 device frames at 2x cover 0.5 source seconds, got {secs}"
        );
    }

    #[test]
    fn markers_anchor_to_the_ring_position() {
        let mut chain = configured(100, 1);
        chain.effects.push(Box::new(Decimate {
            factor: 2,
            channels: 1,
            active: true,
        }));

        // 300 frames already in the ring timeline, 10 more waiting in `out`.
        let mut out = vec![0.0f32; 10];
        chain.process(None, &[0.5f32; 100], &mut out, 300).unwrap();

        assert!(
            chain.stream_secs(309).is_none(),
            "audio queued before this run stays on the fallback"
        );
        let secs = chain.stream_secs(310).unwrap();
        assert!(
            (secs - 3.1).abs() < 1e-9,
            "the run starts at frame 310 = 3.1 source seconds, got {secs}"
        );
    }

    #[test]
    fn one_to_one_audio_after_an_engaged_run_corrects_the_slope() {
        let mut chain = configured(100, 1);
        chain.effects.push(Box::new(Decimate {
            factor: 2,
            channels: 1,
            active: true,
        }));
        let mut out = Vec::new();
        chain.process(None, &[0.5f32; 100], &mut out, 0).unwrap();

        // The effect goes away without a reset (hypothetical), and 1:1 audio
        // follows: its slope must be recorded, not extrapolated from 2x.
        chain.effects[0]
            .as_any_mut()
            .downcast_mut::<Decimate>()
            .unwrap()
            .active = false;
        let before = out.len() as u64;
        chain.process(None, &[0.5f32; 50], &mut out, 0).unwrap();

        let at_join = chain.stream_secs(before).unwrap();
        let later = chain.stream_secs(before + 50).unwrap();
        assert!(
            (later - at_join - 0.5).abs() < 1e-9,
            "50 one-to-one frames must advance 0.5 source seconds, got {}",
            later - at_join
        );
    }

    #[test]
    fn drain_cascades_through_downstream_effects() {
        let mut chain = configured(100, 1);
        chain.effects.push(Box::new(HoldBack {
            held: Vec::new(),
            channels: 1,
        }));
        chain.effects.push(Box::new(Decimate {
            factor: 2,
            channels: 1,
            active: true,
        }));

        let mut out = Vec::new();
        chain.process(None, &[0.5f32; 100], &mut out, 0).unwrap();
        assert!(out.is_empty(), "everything is held upstream");
        assert_eq!(chain.pending_output_frames(), 100);

        chain.drain(&mut out).unwrap();
        assert_eq!(
            out.len(),
            50,
            "the held tail must come out through the decimator"
        );

        let before = out.len();
        chain.drain(&mut out).unwrap();
        assert_eq!(out.len(), before, "a second drain is a no-op");
    }

    #[test]
    fn reset_drops_audio_state_and_inactive_effects() {
        let mut chain = configured(100, 1);
        chain.effects.push(Box::new(HoldBack {
            held: Vec::new(),
            channels: 1,
        }));
        chain.effects.push(Box::new(Decimate {
            factor: 2,
            channels: 1,
            active: false,
        }));

        let mut out = Vec::new();
        chain.process(None, &[0.5f32; 40], &mut out, 0).unwrap();
        chain.reset();

        assert_eq!(chain.effects.len(), 1, "the inactive effect is retired");
        assert_eq!(chain.pending_output_frames(), 0, "held audio is gone");
        assert!(chain.stream_secs(10).is_none(), "the timeline starts over");
    }

    #[test]
    fn a_mirror_carries_settings_but_records_no_markers() {
        let mut chain = configured(100, 1);
        chain.effects.push(Box::new(Decimate {
            factor: 2,
            channels: 1,
            active: true,
        }));

        let mut mirror = chain.spawn_mirror();
        let mut out = Vec::new();
        mirror.process(None, &[0.5f32; 100], &mut out, 0).unwrap();
        assert_eq!(out.len(), 50, "the mirror shapes audio like the original");
        assert!(mirror.stream_secs(25).is_none(), "but records nothing");

        mirror.adopt_timeline();
        let mut adopted_out = Vec::new();
        mirror
            .process(None, &[0.5f32; 100], &mut adopted_out, 1_000)
            .unwrap();
        assert!(
            mirror.stream_secs(1_020).is_some(),
            "adoption starts the timeline"
        );
    }

    #[test]
    fn time_ratio_multiplies_engaged_effects_only() {
        let mut chain = configured(100, 1);
        assert_eq!(chain.time_ratio(), 1.0);
        chain.effects.push(Box::new(Decimate {
            factor: 2,
            channels: 1,
            active: true,
        }));
        chain.effects.push(Box::new(Decimate {
            factor: 3,
            channels: 1,
            active: false,
        }));
        assert_eq!(
            chain.time_ratio(),
            2.0,
            "the inactive effect must not count"
        );
    }
}
