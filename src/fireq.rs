//! Linear-phase FIR EQ: a mastering-style ten-band EQ as a [`Chain`](super::chain)
//! effect, with no inter-band phase distortion.
//!
//! This *complements* the realtime biquad EQ (`src/dsp.rs`) rather than
//! replacing it. The callback EQ stays the zero-latency default; this is the
//! opt-in high-quality mode. While it is enabled the engine flattens the
//! callback bank (via `Params::set_fir_eq_active`) so the two never stack, and
//! both read the same ten band gains — the FIR effect reads them straight from
//! [`Params`] and redesigns its kernel whenever a slider moves.
//!
//! **Latency.** A linear-phase FIR is symmetric, so it delays the signal by
//! half its length — ~43 ms at 48 kHz for the 4096-tap kernel here. That is the
//! price of constant group delay (no band smears in time relative to another),
//! and it is inherent, not tunable. While enabled, heard audio sits that
//! constant behind the reported position; the chain's `pending_output_frames`
//! keeps gapless boundaries honest across it. A gain change is not instant
//! either: it reaches the ear only once the ~0.5 s already buffered in the ring
//! has played.
//!
//! **Off is structural**, exactly like [`TimeStretch`](super::stretch): a
//! disabled `FirEq` swaps to a flat (pure-delay) kernel so it stays audibly
//! transparent while it drains, and sheds its backend at the next chain reset,
//! restoring the byte-identical no-effect path.

use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner};

use super::chain::Effect;
use super::params::{Params, CENTER_FREQS, EQ_BAND_COUNT};
use super::spectral::{hann_periodic, StereoConvolver};

/// FIR length. 4096 taps ≈ 85 ms window, so ~43 ms linear-phase group delay at
/// 48 kHz and ~12 Hz design resolution. A documented constant, not a knob, for
/// v1: longer sharpens the low bands at the cost of more latency.
const KERNEL_LEN: usize = 4096;

/// Group delay of the symmetric kernel — half its length. The effect's latency,
/// reported so gapless joins sit past the delayed tail.
const LATENCY_FRAMES: usize = KERNEL_LEN / 2;

/// Partition size for the convolver. Independent of latency (the convolver is
/// causal); a divisor of `KERNEL_LEN` keeps every partition full.
const BLOCK: usize = 512;

/// The linear-phase FIR EQ effect. `enabled` is the user's setting; the backend
/// (one convolver per channel) exists only while there is a device shape to
/// build it for, mirroring `TimeStretch`.
pub struct FirEq {
    enabled: bool,
    rate: u32,
    channels: usize,
    /// Read for the band gains and, via its epoch, to know when to redesign.
    params: Arc<Params>,
    /// The `eq_epoch` the current kernel was designed from; a mismatch on the
    /// next `process` triggers a click-free redesign. `None` when undesigned.
    design_epoch: Option<u64>,
    backend: Option<StereoConvolver>,
}

impl FirEq {
    /// A disabled effect bound to `params`; no backend until [`Self::reconfigure`]
    /// learns the device shape.
    pub fn new(params: Arc<Params>) -> Self {
        Self {
            enabled: false,
            rate: 0,
            channels: 0,
            params,
            design_epoch: None,
            backend: None,
        }
    }

    /// Updates the user's setting. Disabling swaps the backend to a flat
    /// (pure-delay) kernel so it fades out click-free and stays transparent
    /// until the next chain reset retires it — the `TimeStretch` disable shape.
    pub fn set(&mut self, enabled: bool) {
        self.enabled = enabled;
        if let Some(backend) = self.backend.as_mut() {
            if enabled {
                // Force a redesign from the live gains on the next `process`.
                self.design_epoch = None;
            } else {
                backend.set_kernel(&design_kernel(self.rate, &[0.0; EQ_BAND_COUNT]));
                self.design_epoch = None;
            }
        }
    }

    /// Whether the user has it on, for the event echo and `describe`.
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// The constant delay the effect adds while enabled, in seconds (0 off).
    pub fn latency_secs(&self, rate: u32) -> f32 {
        if self.enabled && rate > 0 {
            LATENCY_FRAMES as f32 / rate as f32
        } else {
            0.0
        }
    }

    /// Reads the ten band gains from the shared params.
    fn gains(&self) -> [f32; EQ_BAND_COUNT] {
        let mut gains = [0.0f32; EQ_BAND_COUNT];
        self.params.eq_gains(&mut gains);
        gains
    }

    /// Builds a backend for the current device shape and gains, recording the
    /// epoch it was designed from.
    fn build_backend(&mut self) {
        if self.rate > 0 && self.channels > 0 {
            let epoch = self.params.eq_epoch();
            let kernel = design_kernel(self.rate, &self.gains());
            self.backend = Some(StereoConvolver::new(&kernel, BLOCK, self.channels));
            self.design_epoch = Some(epoch);
        }
    }
}

impl Effect for FirEq {
    fn name(&self) -> &'static str {
        "fir-eq"
    }

    fn is_active(&self) -> bool {
        self.enabled || self.backend.is_some()
    }

    fn is_bypassed(&self) -> bool {
        // A live backend always delays by the group delay, so it is never a
        // byte-identical no-op — even disabled-and-flat, it stays warm.
        self.backend.is_none()
    }

    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) -> Result<(), String> {
        let Some(backend) = self.backend.as_mut() else {
            output.extend_from_slice(input);
            return Ok(());
        };
        // Pick up a slider move: redesign the kernel and swap it click-free.
        if self.enabled {
            let epoch = self.params.eq_epoch();
            if self.design_epoch != Some(epoch) {
                let mut gains = [0.0f32; EQ_BAND_COUNT];
                self.params.eq_gains(&mut gains);
                backend.set_kernel(&design_kernel(self.rate, &gains));
                self.design_epoch = Some(epoch);
            }
        }
        backend.process(input, output);
        Ok(())
    }

    fn drain(&mut self, output: &mut Vec<f32>) -> Result<(), String> {
        if let Some(backend) = self.backend.as_mut() {
            backend.drain(output);
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
            backend.reset();
        }
    }

    fn time_ratio(&self) -> f64 {
        1.0
    }

    fn pending_output_frames(&self) -> u64 {
        // The constant group delay, plus whatever sub-block input the convolver
        // has staged but not yet emitted — without the latter a gapless
        // boundary could land up to a block early.
        match self.backend.as_ref() {
            Some(backend) => (LATENCY_FRAMES + backend.staged_frames()) as u64,
            None => 0,
        }
    }

    fn matches(&self, rate: u32, channels: usize) -> bool {
        self.rate == rate && self.channels == channels && (self.backend.is_some() || !self.enabled)
    }

    fn reconfigure(&mut self, rate: u32, channels: usize) -> Result<(), String> {
        self.rate = rate;
        self.channels = channels;
        self.backend = None;
        self.design_epoch = None;
        if self.enabled {
            self.build_backend();
        }
        Ok(())
    }

    fn spawn_mirror(&self) -> Box<dyn Effect> {
        let mut mirror = FirEq {
            enabled: self.enabled,
            rate: self.rate,
            channels: self.channels,
            params: Arc::clone(&self.params),
            design_epoch: None,
            backend: None,
        };
        if self.enabled {
            mirror.build_backend();
        }
        Box::new(mirror)
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Designs a linear-phase FIR whose magnitude response tracks the ten band
/// gains, smoothly interpolated in log-frequency.
///
/// Frequency-sampling design: set the desired magnitude at every FFT bin,
/// attach a linear phase of exactly `KERNEL_LEN/2` samples' group delay, invert
/// to a symmetric impulse response, then taper with a periodic Hann window
/// (symmetric about the kernel centre, so linear phase survives the window).
fn design_kernel(rate: u32, gains: &[f32; EQ_BAND_COUNT]) -> Vec<f32> {
    let n = KERNEL_LEN;
    let mut planner = RealFftPlanner::<f32>::new();
    let c2r: Arc<dyn ComplexToReal<f32>> = planner.plan_fft_inverse(n);

    let mut spectrum = c2r.make_input_vec(); // n/2 + 1 complex bins
    let group_delay = LATENCY_FRAMES as f32;
    for (k, bin) in spectrum.iter_mut().enumerate() {
        let hz = k as f32 * rate as f32 / n as f32;
        let magnitude = 10f32.powf(interp_gain_db(hz, gains) / 20.0);
        // Linear phase e^{-j 2π k D / n}; with D = n/2 this is (-1)^k, so DC and
        // Nyquist stay real — exactly what the inverse transform requires.
        let theta = -std::f32::consts::TAU * k as f32 * group_delay / n as f32;
        *bin = Complex::new(magnitude * theta.cos(), magnitude * theta.sin());
    }
    spectrum[0].im = 0.0;
    if let Some(last) = spectrum.last_mut() {
        last.im = 0.0;
    }

    let mut time = c2r.make_output_vec();
    let mut scratch = vec![Complex::default(); c2r.get_scratch_len()];
    c2r.process_with_scratch(&mut spectrum, &mut time, &mut scratch)
        .expect("fir-eq design inverse fft");

    let window = hann_periodic(n);
    let inv_n = 1.0 / n as f32;
    time.iter()
        .zip(&window)
        .map(|(sample, weight)| sample * inv_n * weight)
        .collect()
}

/// The gain in dB at `hz`, piecewise-linear in log-frequency between band
/// centres and held flat beyond the outermost bands.
fn interp_gain_db(hz: f32, gains: &[f32; EQ_BAND_COUNT]) -> f32 {
    if hz <= CENTER_FREQS[0] {
        return gains[0];
    }
    if hz >= CENTER_FREQS[EQ_BAND_COUNT - 1] {
        return gains[EQ_BAND_COUNT - 1];
    }
    for band in 0..EQ_BAND_COUNT - 1 {
        let (low, high) = (CENTER_FREQS[band], CENTER_FREQS[band + 1]);
        if hz <= high {
            let t = (hz.ln() - low.ln()) / (high.ln() - low.ln());
            return gains[band] + t * (gains[band + 1] - gains[band]);
        }
    }
    gains[EQ_BAND_COUNT - 1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral::test_support::response_db;
    use crate::spectral::Convolver;

    const RATE: u32 = 48_000;

    fn gains_with(band: usize, db: f32) -> [f32; EQ_BAND_COUNT] {
        let mut gains = [0.0f32; EQ_BAND_COUNT];
        gains[band] = db;
        gains
    }

    /// Asserts a mono `hz`-tone output has no click across a mid-stream kernel
    /// swap. A click is a discontinuity, which spikes the second difference; a
    /// smooth sine of peak `A` keeps it at `A·ω²`, so the bar is that natural
    /// curvature with headroom — a dropout blows an order past it. A kernel
    /// length is trimmed off each end first: the effect's one-time cold-start
    /// onset and final drain tail ramp through the group delay by design.
    fn assert_no_click(out: &[f32], hz: f32) {
        let trim = KERNEL_LEN;
        assert!(out.len() > 2 * trim + 3, "need audio past the transients");
        let body = &out[trim..out.len() - trim];
        let peak = body.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        let omega = std::f32::consts::TAU * hz / RATE as f32;
        let natural = peak * omega * omega;
        let max_curvature = body
            .windows(3)
            .map(|w| (w[0] - 2.0 * w[1] + w[2]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_curvature < natural * 2.5 + 1e-4,
            "swap clicked at {hz} Hz: curvature {max_curvature:.4}, natural {natural:.4}"
        );
    }

    /// Runs a mono probe through a fresh convolver built on `kernel`.
    fn through(kernel: &[f32], input: &[f32]) -> Vec<f32> {
        let mut convolver = Convolver::new(kernel, BLOCK);
        let mut out = Vec::new();
        convolver.process(input, &mut out);
        convolver.drain(&mut out);
        out
    }

    #[test]
    fn each_band_center_tracks_its_requested_gain() {
        // A distinct gain per band; every band centre should read back close to
        // what was asked — the linear-phase design's core promise. The lowest
        // three bands (32/64/125 Hz) are resolution-limited: a 4096-tap FIR's
        // ~12 Hz bins and Hann main lobe are wider than the octave spacing down
        // there, so an adversarial notch smears by up to ~1.5 dB. Every band
        // from 250 Hz up holds the ±0.5 dB mastering target.
        let gains = [6.0, -4.0, 3.0, -6.0, 2.0, 5.0, -3.0, 4.0, -5.0, 6.0];
        let kernel = design_kernel(RATE, &gains);
        for (band, &center) in CENTER_FREQS.iter().enumerate() {
            if center < 200.0 {
                continue;
            }
            let measured = response_db(|input| through(&kernel, input), RATE, center);
            assert!(
                (measured - gains[band]).abs() < 0.5,
                "band {band} at {center} Hz: asked {} dB, got {measured:.2} dB",
                gains[band]
            );
        }
    }

    #[test]
    fn flat_gains_are_transparent() {
        // All bands at 0 dB: a pure delay, so the response is flat (unity) and
        // energy is preserved — not bit-exact (windowed FIR), but close.
        let kernel = design_kernel(RATE, &[0.0; EQ_BAND_COUNT]);
        for hz in [100.0, 500.0, 1_000.0, 4_000.0, 12_000.0] {
            let db = response_db(|input| through(&kernel, input), RATE, hz);
            assert!(
                db.abs() < 0.2,
                "flat EQ must pass {hz} Hz cleanly: {db:.3} dB"
            );
        }
    }

    #[test]
    fn the_kernel_is_symmetric_so_phase_is_linear() {
        // A linear-phase FIR is symmetric about its centre. Any real gain set
        // must keep that symmetry — that symmetry *is* the constant group delay.
        let gains = [6.0, -4.0, 3.0, -6.0, 2.0, 5.0, -3.0, 4.0, -5.0, 6.0];
        let kernel = design_kernel(RATE, &gains);
        let mid = KERNEL_LEN / 2;
        let peak = kernel.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        let max_asymmetry = (1..mid)
            .map(|k| (kernel[mid - k] - kernel[mid + k]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_asymmetry < peak * 1e-3,
            "kernel must be symmetric about its centre: max asymmetry {max_asymmetry:.2e}, peak {peak:.3}"
        );
    }

    #[test]
    fn a_gain_change_does_not_click() {
        // Redesign the kernel mid-stream via the convolver's click-free swap.
        let params = Arc::new(Params::default());
        params.set_eq_gains(&gains_with(5, 6.0)); // +6 dB at 1 kHz
        let mut effect = FirEq::new(Arc::clone(&params));
        effect.set(true);
        effect.reconfigure(RATE, 1).unwrap();

        let tone: Vec<f32> = (0..RATE)
            .map(|n| (std::f32::consts::TAU * 1_000.0 * n as f32 / RATE as f32).sin() * 0.5)
            .collect();
        let mut out = Vec::new();
        let half = tone.len() / 2;
        effect.process(&tone[..half], &mut out).unwrap();
        params.set_eq_gains(&gains_with(5, -6.0)); // swing to -6 dB
        effect.process(&tone[half..], &mut out).unwrap();
        effect.drain(&mut out).unwrap();

        assert_no_click(&out, 1_000.0);
    }

    #[test]
    fn disabling_swaps_to_a_transparent_kernel_without_clicking() {
        let params = Arc::new(Params::default());
        params.set_eq_gains(&gains_with(5, 8.0));
        let mut effect = FirEq::new(Arc::clone(&params));
        effect.set(true);
        effect.reconfigure(RATE, 1).unwrap();

        let tone: Vec<f32> = (0..RATE)
            .map(|n| (std::f32::consts::TAU * 1_000.0 * n as f32 / RATE as f32).sin() * 0.5)
            .collect();
        let mut out = Vec::new();
        let half = tone.len() / 2;
        effect.process(&tone[..half], &mut out).unwrap();
        effect.set(false); // disable: swap to a flat kernel, stay warm
        assert!(
            !effect.is_bypassed(),
            "disabled but warm still delays the tail"
        );
        effect.process(&tone[half..], &mut out).unwrap();
        effect.drain(&mut out).unwrap();

        assert_no_click(&out, 1_000.0);
    }

    #[test]
    fn reset_while_disabled_retires_the_effect() {
        let params = Arc::new(Params::default());
        let mut effect = FirEq::new(Arc::clone(&params));
        effect.set(true);
        effect.reconfigure(RATE, 2).unwrap();
        assert!(effect.is_active() && !effect.is_bypassed());

        effect.set(false);
        effect.reset();
        assert!(
            !effect.is_active(),
            "a disabled effect goes inactive at reset"
        );
        assert!(effect.is_bypassed(), "and drops its backend");
    }

    #[test]
    fn latency_is_the_group_delay() {
        let params = Arc::new(Params::default());
        let mut effect = FirEq::new(Arc::clone(&params));
        effect.set(true);
        effect.reconfigure(RATE, 2).unwrap();
        assert_eq!(effect.pending_output_frames(), LATENCY_FRAMES as u64);
        let expected = LATENCY_FRAMES as f32 / RATE as f32;
        assert!((effect.latency_secs(RATE) - expected).abs() < 1e-6);
    }

    #[test]
    fn pending_output_counts_the_staged_remainder_too() {
        // Sub-block input sits in the convolver's stage, accepted but not yet
        // emitted; a gapless boundary must sit past it as well as the group
        // delay, so both are reported.
        let params = Arc::new(Params::default());
        let mut effect = FirEq::new(Arc::clone(&params));
        effect.set(true);
        effect.reconfigure(RATE, 1).unwrap();

        let mut out = Vec::new();
        effect.process(&vec![0.1f32; 100], &mut out).unwrap();
        assert_eq!(
            effect.pending_output_frames(),
            LATENCY_FRAMES as u64 + 100,
            "group delay plus the 100 staged frames"
        );
    }
}
