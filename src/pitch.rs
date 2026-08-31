//! Pitch-shift, in cents, as a [`Chain`](super::chain) effect.
//!
//! An owned phase vocoder over `realfft` — never the `timestretch` crate — so
//! the shift is parameterized in **cents** (the house unit; a semitone is 100
//! cents, an octave 1200). It is duration-preserving: `time_ratio` stays 1.0
//! and the playhead needs nothing. The same STFT core is the seed for an owned
//! time-stretcher, should the pinned `timestretch` dependency ever need
//! replacing.
//!
//! Method. A phase vocoder time-stretches the signal by the pitch ratio
//! `2^(cents/1200)` (analysis at a fixed hop, synthesis at a scaled hop, with
//! per-bin instantaneous-frequency phase accumulation), then a linear resample
//! by the same factor compresses it back to the original length — the net is a
//! pitch shift at constant duration. Latency is one FFT window.
//!
//! Off is structural, like [`TimeStretch`](super::stretch): a disabled shifter
//! ramps to 0 cents and stays warm until the next chain reset drops its
//! backend, restoring the byte-identical no-effect path.

use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use super::chain::Effect;

/// STFT size. 2048 at 48 kHz is ~43 ms — the vocoder's latency.
const FFT_SIZE: usize = 2048;
/// Analysis hop: 75% overlap.
const ANALYSIS_HOP: usize = FFT_SIZE / 4;
/// The pitch range offered, in cents: ±one octave.
pub const CENTS_LIMIT: f32 = 1200.0;
/// How fast a live cents change is chased, per analysis frame — enough to
/// glide a sweep without a click, slow enough to avoid warble.
const RATIO_SMOOTH: f32 = 0.1;

/// Wraps a phase to (−π, π].
fn princarg(x: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    x - tau * (x / tau).round()
}

/// The pitch ratio for `cents`: `2^(cents/1200)`.
fn ratio_of(cents: f32) -> f32 {
    2f32.powf(cents / 1200.0)
}

/// A streaming phase-vocoder pitch shifter for one channel.
struct Vocoder {
    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,
    scratch: Vec<Complex<f32>>,
    window: Vec<f32>,
    /// Expected phase advance per analysis hop, per bin.
    omega: Vec<f32>,
    /// The most recent `FFT_SIZE` input samples, oldest first.
    history: Vec<f32>,
    /// New input samples not yet consumed by an analysis hop.
    pending: Vec<f32>,
    fft_in: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    synth: Vec<Complex<f32>>,
    ifft_out: Vec<f32>,
    prev_phase: Vec<f32>,
    sum_phase: Vec<f32>,
    /// Overlap-add accumulators for the synthesized (stretched) signal and the
    /// window energy that normalizes it.
    sig_ola: Vec<f32>,
    norm_ola: Vec<f32>,
    /// The time-stretched signal awaiting resampling back to length.
    stretched: Vec<f32>,
    /// Fractional read position into `stretched` for the resample.
    read_pos: f32,
    /// The current stretch/resample factor (synthesis hop / analysis hop).
    factor: f32,
    /// The smoothed and target pitch ratios.
    ratio: f32,
    target_ratio: f32,
    inv_n: f32,
}

impl Vocoder {
    fn new(ratio: f32) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(FFT_SIZE);
        let c2r = planner.plan_fft_inverse(FFT_SIZE);
        let bins = FFT_SIZE / 2 + 1;
        let scratch_len = r2c.get_scratch_len().max(c2r.get_scratch_len());
        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|n| {
                let phase = std::f32::consts::TAU * n as f32 / FFT_SIZE as f32;
                0.5 - 0.5 * phase.cos()
            })
            .collect();
        let omega = (0..bins)
            .map(|k| std::f32::consts::TAU * k as f32 * ANALYSIS_HOP as f32 / FFT_SIZE as f32)
            .collect();
        Self {
            scratch: vec![Complex::default(); scratch_len],
            fft_in: r2c.make_input_vec(),
            spectrum: r2c.make_output_vec(),
            synth: c2r.make_input_vec(),
            ifft_out: c2r.make_output_vec(),
            window,
            omega,
            history: vec![0.0; FFT_SIZE],
            pending: Vec::with_capacity(ANALYSIS_HOP),
            prev_phase: vec![0.0; bins],
            sum_phase: vec![0.0; bins],
            sig_ola: Vec::new(),
            norm_ola: Vec::new(),
            stretched: Vec::new(),
            read_pos: 0.0,
            factor: 1.0,
            ratio,
            target_ratio: ratio,
            inv_n: 1.0 / FFT_SIZE as f32,
            r2c,
            c2r,
        }
    }

    fn set_ratio(&mut self, ratio: f32) {
        self.target_ratio = ratio;
    }

    /// Processes mono `input`, appending the pitch-shifted result to `output`.
    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        for &sample in input {
            self.pending.push(sample);
            if self.pending.len() == ANALYSIS_HOP {
                self.analyse_frame();
                self.pending.clear();
            }
        }
        self.resample(output);
    }

    /// Flushes the tail: push enough silence to carry the last real samples
    /// through the window, then drain the stretched remainder.
    fn drain(&mut self, output: &mut Vec<f32>) {
        // One window of silence flushes the analysis and OLA tails.
        let pad = FFT_SIZE + ANALYSIS_HOP;
        let silence = vec![0.0f32; pad];
        self.process(&silence, output);
    }

    fn reset(&mut self) {
        self.history.iter_mut().for_each(|s| *s = 0.0);
        self.pending.clear();
        self.prev_phase.iter_mut().for_each(|p| *p = 0.0);
        self.sum_phase.iter_mut().for_each(|p| *p = 0.0);
        self.sig_ola.clear();
        self.norm_ola.clear();
        self.stretched.clear();
        self.read_pos = 0.0;
        self.ratio = self.target_ratio;
        self.factor = 1.0;
    }

    fn analyse_frame(&mut self) {
        // Slide the analysis window forward one hop.
        self.history.rotate_left(ANALYSIS_HOP);
        self.history[FFT_SIZE - ANALYSIS_HOP..].copy_from_slice(&self.pending);

        // Glide toward the target ratio, then set the synthesis hop.
        self.ratio += (self.target_ratio - self.ratio) * RATIO_SMOOTH;
        let hs = (ANALYSIS_HOP as f32 * self.ratio).round().max(1.0) as usize;
        self.factor = hs as f32 / ANALYSIS_HOP as f32;

        for ((slot, sample), weight) in self
            .fft_in
            .iter_mut()
            .zip(self.history.iter())
            .zip(self.window.iter())
        {
            *slot = sample * weight;
        }
        if self
            .r2c
            .process_with_scratch(&mut self.fft_in, &mut self.spectrum, &mut self.scratch)
            .is_err()
        {
            return;
        }

        for k in 0..self.spectrum.len() {
            let bin = self.spectrum[k];
            let magnitude = bin.norm();
            let phase = bin.im.atan2(bin.re);
            let delta = princarg(phase - self.prev_phase[k] - self.omega[k]);
            // True phase advance per analysis hop, retimed to the synthesis hop.
            self.sum_phase[k] += (self.omega[k] + delta) * self.factor;
            self.prev_phase[k] = phase;
            let sp = self.sum_phase[k];
            self.synth[k] = Complex::new(magnitude * sp.cos(), magnitude * sp.sin());
        }
        // The inverse needs a real DC and Nyquist bin.
        self.synth[0].im = 0.0;
        if let Some(last) = self.synth.last_mut() {
            last.im = 0.0;
        }
        if self
            .c2r
            .process_with_scratch(&mut self.synth, &mut self.ifft_out, &mut self.scratch)
            .is_err()
        {
            return;
        }

        // Overlap-add the synthesized frame at the synthesis hop, tracking the
        // window energy so the sum can be normalized to unity gain.
        if self.sig_ola.len() < FFT_SIZE {
            self.sig_ola.resize(FFT_SIZE, 0.0);
            self.norm_ola.resize(FFT_SIZE, 0.0);
        }
        for i in 0..FFT_SIZE {
            let w = self.window[i];
            self.sig_ola[i] += self.ifft_out[i] * self.inv_n * w;
            self.norm_ola[i] += w * w;
        }
        // The first `hs` samples are now final: no later frame reaches them.
        for i in 0..hs {
            let energy = self.norm_ola[i];
            let value = if energy > 1e-6 {
                self.sig_ola[i] / energy
            } else {
                0.0
            };
            self.stretched.push(value);
        }
        self.sig_ola.drain(..hs);
        self.norm_ola.drain(..hs);
    }

    /// Linearly resamples the stretched signal by `factor` back to the original
    /// length, which is what turns the time-stretch into a pitch shift.
    fn resample(&mut self, output: &mut Vec<f32>) {
        let step = self.factor.max(1e-4);
        while self.read_pos + 1.0 < self.stretched.len() as f32 {
            let i = self.read_pos.floor() as usize;
            let frac = self.read_pos - i as f32;
            output.push(self.stretched[i] * (1.0 - frac) + self.stretched[i + 1] * frac);
            self.read_pos += step;
        }
        let keep = self.read_pos.floor() as usize;
        if keep > 0 {
            self.stretched.drain(..keep);
            self.read_pos -= keep as f32;
        }
    }
}

/// One vocoder per channel, plus deinterleave/interleave scratch.
struct Backend {
    channels: usize,
    vocoders: Vec<Vocoder>,
    chan_in: Vec<Vec<f32>>,
    chan_out: Vec<Vec<f32>>,
}

impl Backend {
    fn new(channels: usize, ratio: f32) -> Self {
        Self {
            channels,
            vocoders: (0..channels).map(|_| Vocoder::new(ratio)).collect(),
            chan_in: (0..channels).map(|_| Vec::new()).collect(),
            chan_out: (0..channels).map(|_| Vec::new()).collect(),
        }
    }

    fn set_ratio(&mut self, ratio: f32) {
        for vocoder in self.vocoders.iter_mut() {
            vocoder.set_ratio(ratio);
        }
    }

    /// Deinterleaves, shifts each channel, and reinterleaves. The channels stay
    /// sample-aligned: each processes the same frame count at the same factor.
    fn run(&mut self, input: &[f32], output: &mut Vec<f32>, drain: bool) {
        let channels = self.channels;
        for chan in self.chan_in.iter_mut() {
            chan.clear();
        }
        for frame in input.chunks_exact(channels) {
            for (chan, sample) in self.chan_in.iter_mut().zip(frame) {
                chan.push(*sample);
            }
        }
        for ((vocoder, signal), out) in self
            .vocoders
            .iter_mut()
            .zip(self.chan_in.iter())
            .zip(self.chan_out.iter_mut())
        {
            out.clear();
            vocoder.process(signal, out);
            if drain {
                vocoder.drain(out);
            }
        }
        let frames = self.chan_out.iter().map(|c| c.len()).min().unwrap_or(0);
        for i in 0..frames {
            for chan in self.chan_out.iter() {
                output.push(chan[i]);
            }
        }
    }

    fn reset(&mut self) {
        for vocoder in self.vocoders.iter_mut() {
            vocoder.reset();
        }
    }
}

/// The pitch-shift effect. `enabled`/`cents` are the user's setting; the
/// per-channel vocoders exist only while there is a device shape to build for.
pub struct PitchShift {
    enabled: bool,
    cents: f32,
    rate: u32,
    channels: usize,
    backend: Option<Backend>,
}

impl PitchShift {
    pub fn new(cents: f32) -> Self {
        Self {
            enabled: true,
            cents: cents.clamp(-CENTS_LIMIT, CENTS_LIMIT),
            rate: 0,
            channels: 0,
            backend: None,
        }
    }

    /// Updates the user's setting. Disabling ramps the shift to 0 cents and
    /// leaves the teardown to the next chain reset, so the switch cannot click.
    pub fn set(&mut self, enabled: bool, cents: f32) {
        self.enabled = enabled;
        if enabled {
            self.cents = cents.clamp(-CENTS_LIMIT, CENTS_LIMIT);
        }
        let ratio = if enabled { ratio_of(self.cents) } else { 1.0 };
        if let Some(backend) = self.backend.as_mut() {
            backend.set_ratio(ratio);
        }
    }

    /// The current setting, for the event echo and `describe`.
    pub fn setting(&self) -> (bool, f32) {
        (self.enabled, self.cents)
    }
}

impl Effect for PitchShift {
    fn name(&self) -> &'static str {
        "pitch-shift"
    }

    fn is_active(&self) -> bool {
        self.enabled || self.backend.is_some()
    }

    fn is_bypassed(&self) -> bool {
        self.backend.is_none()
    }

    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) -> Result<(), String> {
        match self.backend.as_mut() {
            Some(backend) => backend.run(input, output, false),
            None => output.extend_from_slice(input),
        }
        Ok(())
    }

    fn drain(&mut self, output: &mut Vec<f32>) -> Result<(), String> {
        if let Some(backend) = self.backend.as_mut() {
            backend.run(&[], output, true);
        }
        Ok(())
    }

    fn reset(&mut self) {
        if !self.enabled {
            self.backend = None;
            return;
        }
        if let Some(backend) = self.backend.as_mut() {
            backend.reset();
        }
    }

    fn time_ratio(&self) -> f64 {
        1.0
    }

    fn pending_output_frames(&self) -> u64 {
        if self.backend.is_some() {
            FFT_SIZE as u64
        } else {
            0
        }
    }

    fn matches(&self, rate: u32, channels: usize) -> bool {
        self.rate == rate && self.channels == channels && (self.backend.is_some() || !self.enabled)
    }

    fn reconfigure(&mut self, rate: u32, channels: usize) -> Result<(), String> {
        self.rate = rate;
        self.channels = channels;
        self.backend = None;
        if self.enabled && rate > 0 && channels > 0 {
            self.backend = Some(Backend::new(channels, ratio_of(self.cents)));
        }
        Ok(())
    }

    fn spawn_mirror(&self) -> Box<dyn Effect> {
        let backend = if self.enabled && self.rate > 0 && self.channels > 0 {
            Some(Backend::new(self.channels, ratio_of(self.cents)))
        } else {
            None
        };
        Box::new(PitchShift {
            enabled: self.enabled,
            cents: self.cents,
            rate: self.rate,
            channels: self.channels,
            backend,
        })
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral::test_support::dominant_hz;

    const RATE: u32 = 48_000;

    fn sine(hz: f32, frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|n| (std::f32::consts::TAU * hz * n as f32 / RATE as f32).sin() * 0.5)
            .collect()
    }

    fn shift(cents: f32, input: &[f32]) -> Vec<f32> {
        let mut vocoder = Vocoder::new(ratio_of(cents));
        let mut out = Vec::new();
        for chunk in input.chunks(1024) {
            vocoder.process(chunk, &mut out);
        }
        vocoder.drain(&mut out);
        out
    }

    #[test]
    fn an_octave_up_doubles_the_frequency() {
        let input = sine(440.0, RATE as usize * 2);
        let out = shift(1200.0, &input);
        let hz = dominant_hz(&out, RATE);
        assert!((hz - 880.0).abs() < 15.0, "expected ~880 Hz, got {hz:.1}");
    }

    #[test]
    fn an_octave_down_halves_the_frequency() {
        let input = sine(440.0, RATE as usize * 2);
        let out = shift(-1200.0, &input);
        let hz = dominant_hz(&out, RATE);
        assert!((hz - 220.0).abs() < 15.0, "expected ~220 Hz, got {hz:.1}");
    }

    #[test]
    fn a_semitone_up_matches_the_ratio() {
        let input = sine(440.0, RATE as usize * 2);
        let out = shift(100.0, &input);
        let hz = dominant_hz(&out, RATE);
        let expected = 440.0 * ratio_of(100.0);
        assert!(
            (hz - expected).abs() < 12.0,
            "expected ~{expected:.1} Hz, got {hz:.1}"
        );
    }

    #[test]
    fn the_length_is_preserved() {
        let input = sine(330.0, RATE as usize * 2);
        let out = shift(700.0, &input);
        let ratio = out.len() as f64 / input.len() as f64;
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "pitch shift must preserve length: got {ratio:.3}x"
        );
    }

    /// A stereo interleaved sine, both channels identical.
    fn stereo_sine(hz: f32, frames: usize) -> Vec<f32> {
        sine(hz, frames).iter().flat_map(|&s| [s, s]).collect()
    }

    fn run_effect(effect: &mut PitchShift, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::new();
        for chunk in input.chunks(2048 * 2) {
            effect.process(chunk, &mut out).unwrap();
        }
        effect.drain(&mut out).unwrap();
        out
    }

    #[test]
    fn the_effect_shifts_pitch_across_channels() {
        let mut effect = PitchShift::new(1200.0);
        effect.reconfigure(RATE, 2).unwrap();
        let out = run_effect(&mut effect, &stereo_sine(440.0, RATE as usize * 2));
        let left: Vec<f32> = out.chunks_exact(2).map(|f| f[0]).collect();
        let hz = dominant_hz(&left, RATE);
        assert!(
            (hz - 880.0).abs() < 15.0,
            "an octave up on both channels: {hz:.1}"
        );
    }

    #[test]
    fn a_live_cents_sweep_does_not_click() {
        let mut effect = PitchShift::new(0.0);
        effect.reconfigure(RATE, 1).unwrap();
        let tone = sine(220.0, RATE as usize * 2);
        let mut out = Vec::new();
        let quarter = tone.len() / 4;
        for (i, chunk) in tone.chunks(quarter).enumerate() {
            effect.set(true, i as f32 * 200.0); // 0 → 600 cents in steps
            effect.process(chunk, &mut out).unwrap();
        }
        effect.drain(&mut out).unwrap();

        // Past the warm-up, no second difference far above a swept sine's own.
        let max_curvature = out
            .windows(3)
            .skip(FFT_SIZE)
            .map(|w| (w[0] - 2.0 * w[1] + w[2]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_curvature < 0.1,
            "a live cents sweep must not click: max curvature {max_curvature}"
        );
    }

    #[test]
    fn reset_leaves_no_bleed() {
        let mut effect = PitchShift::new(500.0);
        effect.reconfigure(RATE, 1).unwrap();
        let mut sink = Vec::new();
        effect
            .process(&sine(440.0, RATE as usize), &mut sink)
            .unwrap();
        effect.reset();

        let mut out = Vec::new();
        effect
            .process(&vec![0.0f32; RATE as usize], &mut out)
            .unwrap();
        effect.drain(&mut out).unwrap();
        let peak = out.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(
            peak < 0.01,
            "audio before a reset must not bleed after: peak {peak:.4}"
        );
    }

    // Time-stretch keeps pitch, pitch-shift keeps duration; chained, each does
    // its own job. A 440 Hz tone at 0.5x tempo and +1200 cents must still read
    // ~880 Hz — the shift lands regardless of the stretch running before it.
    #[cfg(feature = "stretch")]
    #[test]
    fn composes_with_time_stretch() {
        use crate::chain::Chain;
        let mut chain = Chain::new();
        chain.reconfigure(RATE, 1).unwrap();
        chain.set_time_stretch(true, 0.5).unwrap();
        chain.set_pitch_shift(true, 1200.0).unwrap();

        let input = sine(440.0, RATE as usize * 3);
        let mut out = Vec::new();
        for chunk in input.chunks(2048) {
            chain.process(None, chunk, &mut out, 0).unwrap();
        }
        chain.drain(&mut out).unwrap();

        let hz = dominant_hz(&out, RATE);
        assert!(
            (hz - 880.0).abs() < 20.0,
            "the shift must still land through a 0.5x stretch: {hz:.1} Hz"
        );
    }
}
