//! Shared spectral-domain DSP: the foundation the later spectral phases build
//! on.
//!
//! Everything here runs on the decode thread or in tests — never in the cpal
//! callback. All the FFT planning and scratch allocation happens once, at
//! construction, so `process`/`drain` touch no allocator: the realtime
//! contract stays a property of the callback, and this stays comfortably on
//! the right side of the ring.
//!
//! The core primitive is [`Convolver`], a uniformly-partitioned
//! frequency-domain convolution (the overlap-save FDL — frequency-domain delay
//! line). It is *causal*: output sample `i` is exactly the linear convolution
//! at index `i`, so it reports zero latency. Any delay a linear-phase kernel
//! needs lives in the kernel's own design, not here (see the FIR-EQ phase).
//! [`StereoConvolver`] runs one per channel over interleaved audio and swaps
//! kernels click-free.
//!
//! This module is not feature-gated — it pulls in nothing `realfft` does not
//! already give the analyser — but its consumers arrive with the later phases.
//! Until then the public surface is dead by design, hence the module-wide
//! allow; every item gains a caller as the true-peak / FIR-EQ / convolution
//! phases land.
#![allow(dead_code)]

use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

/// Uniformly-partitioned frequency-domain convolution (overlap-save).
///
/// The kernel is split into `block`-sample partitions, each transformed once
/// at construction. Input is transformed a block at a time with a size-`2·block`
/// FFT; the block spectra are kept in a ring (the FDL) and multiply-accumulated
/// against the partitioned kernel, so a kernel of any length costs one forward
/// and one inverse FFT per block regardless of how many partitions it spans.
///
/// Streaming: `process` stages input that does not fill a whole block, exactly
/// as [`Resampler`](super::resample::Resampler) does, and emits whole blocks;
/// `drain` flushes the staged remainder and the `kernel_len − 1` tail, so a
/// stream's total output is `input_len + kernel_len − 1` frames — the length of
/// the full linear convolution.
pub struct Convolver {
    /// Partition size; the FFT is twice this.
    block: usize,
    fft_size: usize,
    /// `1 / fft_size` — realfft's forward/inverse pair is unnormalized, so the
    /// round trip scales by `fft_size` and the output is divided back down.
    inv_n: f32,
    kernel_len: usize,
    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,
    /// One spectrum per kernel partition, transformed once at construction.
    kernel: Vec<Vec<Complex<f32>>>,
    /// Frequency-domain delay line: the last `kernel.len()` input-block
    /// spectra, newest at `write`.
    fdl: Vec<Vec<Complex<f32>>>,
    write: usize,
    /// The previous input block, the left half of the overlap-save window.
    history: Vec<f32>,
    /// Preallocated scratch, reused every block.
    block_scratch: Vec<f32>,
    fft_in: Vec<f32>,
    fft_out: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    accum: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
    /// Input held until a whole block is available.
    staged: Vec<f32>,
    frames_in: usize,
    frames_out: usize,
}

impl Convolver {
    /// Builds a convolver for `kernel`, partitioned into `block`-sample blocks.
    ///
    /// Panics if `block` is zero or `kernel` is empty — both are programmer
    /// errors, not runtime conditions, on this decode-thread-only path.
    pub fn new(kernel: &[f32], block: usize) -> Self {
        assert!(block > 0, "convolver block size must be non-zero");
        assert!(!kernel.is_empty(), "convolver kernel must be non-empty");

        let fft_size = block * 2;
        let bins = fft_size / 2 + 1;
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(fft_size);
        let c2r = planner.plan_fft_inverse(fft_size);

        let scratch_len = r2c.get_scratch_len().max(c2r.get_scratch_len());
        let mut fft_scratch = vec![Complex::default(); scratch_len];

        // Transform each kernel partition once: partition `p` is
        // `kernel[p·block .. p·block+block]`, zero-padded into the FFT's left
        // half (the right half stays zero, the overlap-save convention).
        let partitions = kernel.len().div_ceil(block);
        let mut kernel_spectra = Vec::with_capacity(partitions);
        let mut pad = r2c.make_input_vec();
        for p in 0..partitions {
            let start = p * block;
            let end = (start + block).min(kernel.len());
            pad.iter_mut().for_each(|s| *s = 0.0);
            pad[..end - start].copy_from_slice(&kernel[start..end]);
            let mut spectrum = r2c.make_output_vec();
            r2c.process_with_scratch(&mut pad, &mut spectrum, &mut fft_scratch)
                .expect("kernel partition fft");
            kernel_spectra.push(spectrum);
        }

        Self {
            block,
            fft_size,
            inv_n: 1.0 / fft_size as f32,
            kernel_len: kernel.len(),
            fdl: (0..partitions)
                .map(|_| vec![Complex::default(); bins])
                .collect(),
            write: 0,
            history: vec![0.0; block],
            block_scratch: vec![0.0; block],
            fft_in: vec![0.0; fft_size],
            fft_out: vec![0.0; fft_size],
            spectrum: vec![Complex::default(); bins],
            accum: vec![Complex::default(); bins],
            fft_scratch,
            kernel: kernel_spectra,
            staged: Vec::with_capacity(block),
            frames_in: 0,
            frames_out: 0,
            r2c,
            c2r,
        }
    }

    /// Latency the convolver itself adds: none. It is causal — output `i` is
    /// the linear convolution at index `i`. A linear-phase kernel's own group
    /// delay is a property of the kernel, reported by its owner.
    pub fn latency_frames(&self) -> usize {
        0
    }

    /// Convolves `input`, appending whole blocks of result to `output`. Input
    /// that does not complete a block is staged for the next call.
    pub fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        self.frames_in += input.len();
        self.staged.extend_from_slice(input);
        while self.staged.len() >= self.block {
            self.block_scratch
                .copy_from_slice(&self.staged[..self.block]);
            self.process_block(output);
            self.staged.drain(..self.block);
        }
    }

    /// Flushes the staged remainder and the `kernel_len − 1` convolution tail,
    /// so total output over the stream is `input_len + kernel_len − 1` frames.
    /// The convolver is a fresh stream afterwards.
    pub fn drain(&mut self, output: &mut Vec<f32>) {
        if self.frames_in == 0 {
            return;
        }
        let target = self.frames_in + self.kernel_len - 1;
        let remaining = target - self.frames_out;
        let start = output.len();
        // Pad the staged partial and feed zero blocks until the whole tail has
        // been produced; the overshoot past `remaining` is exact silence.
        let blocks = remaining.div_ceil(self.block);
        self.staged.resize(blocks * self.block, 0.0);
        for _ in 0..blocks {
            self.block_scratch
                .copy_from_slice(&self.staged[..self.block]);
            self.process_block(output);
            self.staged.drain(..self.block);
        }
        output.truncate(start + remaining);
        self.reset();
    }

    /// Drops all audio state across a discontinuity, leaving a fresh stream.
    pub fn reset(&mut self) {
        for spectrum in self.fdl.iter_mut() {
            spectrum.iter_mut().for_each(|c| *c = Complex::default());
        }
        self.write = 0;
        self.history.iter_mut().for_each(|s| *s = 0.0);
        self.staged.clear();
        self.frames_in = 0;
        self.frames_out = 0;
    }

    /// Transforms one block (in `block_scratch`), multiply-accumulates it
    /// against the partitioned kernel, and appends the alias-free result.
    fn process_block(&mut self, output: &mut Vec<f32>) {
        // Overlap-save window: [previous block | current block].
        self.fft_in[..self.block].copy_from_slice(&self.history);
        self.fft_in[self.block..].copy_from_slice(&self.block_scratch);
        self.r2c
            .process_with_scratch(&mut self.fft_in, &mut self.spectrum, &mut self.fft_scratch)
            .expect("forward fft");

        // Store this block's spectrum as the newest FDL entry.
        self.fdl[self.write].copy_from_slice(&self.spectrum);

        // Y = Σ_p kernel[p] · X[block p ago].
        let partitions = self.kernel.len();
        self.accum.iter_mut().for_each(|c| *c = Complex::default());
        for p in 0..partitions {
            let past = (self.write + partitions - p) % partitions;
            let kernel = &self.kernel[p];
            let block = &self.fdl[past];
            for ((acc, k), x) in self.accum.iter_mut().zip(kernel).zip(block) {
                *acc += k * x;
            }
        }
        self.write = (self.write + 1) % partitions;

        // The inverse rejects a DC or Nyquist bin with a non-zero imaginary
        // part; products of real-signal spectra keep them real up to rounding,
        // so clear the residue rather than let realfft error on it.
        if let Some(first) = self.accum.first_mut() {
            first.im = 0.0;
        }
        if let Some(last) = self.accum.last_mut() {
            last.im = 0.0;
        }
        self.c2r
            .process_with_scratch(&mut self.accum, &mut self.fft_out, &mut self.fft_scratch)
            .expect("inverse fft");

        // Overlap-save keeps the second half — the first is circular-aliased.
        output.extend(self.fft_out[self.block..].iter().map(|s| s * self.inv_n));
        self.history.copy_from_slice(&self.block_scratch);
        self.frames_out += self.block;
    }
}

/// One [`Convolver`] per channel over interleaved audio, with a click-free
/// kernel swap: [`set_kernel`](Self::set_kernel) runs the old and new kernels
/// in parallel for one block and equal-power crossfades between them.
pub struct StereoConvolver {
    channels: usize,
    block: usize,
    active: Vec<Convolver>,
    /// The incoming kernel's convolvers while a swap crossfades; `None` at rest.
    next: Option<Vec<Convolver>>,
    /// Frames into the crossfade, `0..block`.
    fade_pos: usize,
    /// Deinterleaved input and per-channel outputs, reused every call.
    channel_in: Vec<Vec<f32>>,
    out_active: Vec<Vec<f32>>,
    out_next: Vec<Vec<f32>>,
}

impl StereoConvolver {
    /// Builds `channels` convolvers, one per interleaved channel, all sharing
    /// `kernel` and `block`.
    pub fn new(kernel: &[f32], block: usize, channels: usize) -> Self {
        assert!(channels > 0, "convolver needs at least one channel");
        Self {
            channels,
            block,
            active: (0..channels)
                .map(|_| Convolver::new(kernel, block))
                .collect(),
            next: None,
            fade_pos: 0,
            channel_in: (0..channels).map(|_| Vec::new()).collect(),
            out_active: (0..channels).map(|_| Vec::new()).collect(),
            out_next: (0..channels).map(|_| Vec::new()).collect(),
        }
    }

    /// Swaps in a new kernel with an equal-power crossfade over the next block,
    /// so a change mid-stream produces no click. A swap requested while another
    /// is still fading restarts the crossfade from the active kernel to this
    /// newest one, discarding the previous incoming kernel.
    pub fn set_kernel(&mut self, kernel: &[f32]) {
        self.next = Some(
            (0..self.channels)
                .map(|_| Convolver::new(kernel, self.block))
                .collect(),
        );
        self.fade_pos = 0;
    }

    /// Convolves interleaved `input`, appending interleaved result to `output`.
    pub fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        let channels = self.channels;
        for channel in self.channel_in.iter_mut() {
            channel.clear();
        }
        for frame in input.chunks_exact(channels) {
            for (channel, sample) in self.channel_in.iter_mut().zip(frame) {
                channel.push(*sample);
            }
        }

        for channel in 0..channels {
            self.out_active[channel].clear();
            self.active[channel].process(&self.channel_in[channel], &mut self.out_active[channel]);
        }
        let frames = self.out_active[0].len();

        if self.next.is_none() {
            for i in 0..frames {
                for channel in 0..channels {
                    output.push(self.out_active[channel][i]);
                }
            }
            return;
        }

        // A swap is fading: run the incoming convolvers over the same input,
        // then blend old → new equal-power across `block` frames.
        {
            let next = self.next.as_mut().expect("checked is_none above");
            for ((convolver, input), out) in next
                .iter_mut()
                .zip(self.channel_in.iter())
                .zip(self.out_next.iter_mut())
            {
                out.clear();
                convolver.process(input, out);
            }
        }
        for i in 0..frames {
            let t = ((self.fade_pos + i) as f32 / self.block as f32).min(1.0);
            let (fade_out, fade_in) = equal_power(t);
            for channel in 0..channels {
                output.push(
                    self.out_active[channel][i] * fade_out + self.out_next[channel][i] * fade_in,
                );
            }
        }
        self.fade_pos += frames;
        if self.fade_pos >= self.block {
            self.active = self.next.take().expect("checked is_none above");
            self.fade_pos = 0;
        }
    }

    /// Flushes the active kernel's tail, interleaved. A stream boundary: any
    /// crossfade still in flight is abandoned — a swap completes within one
    /// block, so this only bites a `set_kernel` with under a block of audio
    /// behind it — and the convolvers are a fresh stream afterwards.
    pub fn drain(&mut self, output: &mut Vec<f32>) {
        let channels = self.channels;
        let mut per_channel: Vec<Vec<f32>> = (0..channels).map(|_| Vec::new()).collect();
        for (convolver, out) in self.active.iter_mut().zip(per_channel.iter_mut()) {
            convolver.drain(out);
        }
        let frames = per_channel[0].len();
        for i in 0..frames {
            for channel in per_channel.iter() {
                output.push(channel[i]);
            }
        }
        self.next = None;
        self.fade_pos = 0;
    }

    /// Drops audio state across a discontinuity and abandons any pending swap.
    pub fn reset(&mut self) {
        for convolver in self.active.iter_mut() {
            convolver.reset();
        }
        self.next = None;
        self.fade_pos = 0;
    }

    /// Latency the convolvers add: none (see [`Convolver::latency_frames`]).
    pub fn latency_frames(&self) -> usize {
        0
    }
}

/// Equal-power crossfade weights at `t` (0..=1): `(fade_out, fade_in)`, whose
/// squares sum to a constant so a blend holds perceived level. The same shape
/// the engine's track crossfade uses.
fn equal_power(t: f32) -> (f32, f32) {
    let angle = t.clamp(0.0, 1.0) * std::f32::consts::FRAC_PI_2;
    (angle.cos(), angle.sin())
}

/// A symmetric Hann window of `n` points — zero at both ends, for FIR kernel
/// tapering where the endpoints should vanish.
pub fn hann(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    (0..n)
        .map(|i| {
            let phase = std::f32::consts::TAU * i as f32 / (n as f32 - 1.0);
            0.5 - 0.5 * phase.cos()
        })
        .collect()
}

/// A periodic (DFT-even) Hann window of `n` points — the analysis window for
/// overlap-add STFT work, where the periodic form is the one that sums flat.
pub fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let phase = std::f32::consts::TAU * i as f32 / n as f32;
            0.5 - 0.5 * phase.cos()
        })
        .collect()
}

/// Test-side spectral measurements, shared across the phases' test modules the
/// way [`fixtures`](super::fixtures) shares synthesis. Frequency-domain
/// assertions on real effects: what tone came out, and at what level.
#[cfg(test)]
pub mod test_support {
    use realfft::RealFftPlanner;

    /// The largest power of two not exceeding `n` (at least 1), so an FFT
    /// window can be taken from an arbitrary-length signal.
    fn window_len(n: usize) -> usize {
        assert!(n >= 2, "need at least two samples to measure");
        1usize << (usize::BITS - 1 - n.leading_zeros())
    }

    /// The strongest non-DC frequency in a mono `signal`, in Hz. A power-of-two
    /// window is taken from the middle, past any onset transient.
    pub fn dominant_hz(signal: &[f32], rate: u32) -> f32 {
        let window = window_len(signal.len());
        let start = (signal.len() - window) / 2;
        let mut buf = signal[start..start + window].to_vec();
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
        bin as f32 * rate as f32 / window as f32
    }

    /// The windowed magnitude of a mono `signal` at `hz` (nearest bin). Hann
    /// windowed to tame leakage, so it is a relative measure — meaningful in
    /// the ratio [`response_db`] takes, not as an absolute amplitude.
    pub fn magnitude_at(signal: &[f32], rate: u32, hz: f32) -> f32 {
        let window = window_len(signal.len());
        let start = (signal.len() - window) / 2;
        let mut buf = signal[start..start + window].to_vec();
        for (sample, weight) in buf.iter_mut().zip(super::hann_periodic(window)) {
            *sample *= weight;
        }
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(window);
        let mut spectrum = fft.make_output_vec();
        fft.process(&mut buf, &mut spectrum).unwrap();
        let bin = ((hz * window as f32 / rate as f32).round() as usize).min(spectrum.len() - 1);
        spectrum[bin].norm()
    }

    /// The frequency response of an effect at `hz`, in dB: drive it with a
    /// probe tone at `hz` and compare the output level to the input's. `process`
    /// takes an interleaved-free mono probe and returns the mono result.
    pub fn response_db<F>(mut process: F, rate: u32, hz: f32) -> f32
    where
        F: FnMut(&[f32]) -> Vec<f32>,
    {
        let frames = (rate as usize).max(8_192);
        let input: Vec<f32> = (0..frames)
            .map(|n| (std::f32::consts::TAU * hz * n as f32 / rate as f32).sin() * 0.5)
            .collect();
        let output = process(&input);
        let in_mag = magnitude_at(&input, rate, hz);
        let out_mag = magnitude_at(&output, rate, hz);
        20.0 * (out_mag / in_mag).log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: usize = 64;

    /// Naive O(n·m) linear convolution, the reference the FFT path must match.
    fn naive(signal: &[f32], kernel: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; signal.len() + kernel.len() - 1];
        for (i, &s) in signal.iter().enumerate() {
            for (j, &k) in kernel.iter().enumerate() {
                out[i + j] += s * k;
            }
        }
        out
    }

    /// Deterministic pseudo-random samples in −1..1, so tests need no rng dep.
    fn noise(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn convolve_all(kernel: &[f32], block: usize, input: &[f32]) -> Vec<f32> {
        let mut convolver = Convolver::new(kernel, block);
        let mut out = Vec::new();
        convolver.process(input, &mut out);
        convolver.drain(&mut out);
        out
    }

    #[test]
    fn an_identity_kernel_is_transparent() {
        let input = noise(BLOCK * 5, 1);
        let out = convolve_all(&[1.0], BLOCK, &input);
        assert_eq!(out.len(), input.len(), "identity conv keeps the length");
        let max = input
            .iter()
            .zip(&out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max < 1e-4,
            "identity must pass audio through: max err {max}"
        );
    }

    #[test]
    fn a_pure_delay_kernel_shifts_by_exactly_k() {
        const K: usize = 20;
        let mut kernel = vec![0.0f32; K + 1];
        kernel[K] = 1.0;
        let input = noise(BLOCK * 4, 2);
        let out = convolve_all(&kernel, BLOCK, &input);

        for value in &out[..K] {
            assert!(
                value.abs() < 1e-4,
                "the first k samples are pre-signal silence"
            );
        }
        let max = (0..input.len())
            .map(|i| (out[i + K] - input[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max < 1e-4,
            "delayed audio must equal the input: max err {max}"
        );
    }

    #[test]
    fn convolution_matches_the_naive_reference() {
        // A kernel spanning several partitions, an input spanning several
        // blocks: the multi-partition FDL path, not just a single block.
        let kernel = noise(150, 3);
        let input = noise(500, 4);
        let out = convolve_all(&kernel, BLOCK, &input);
        let reference = naive(&input, &kernel);

        assert_eq!(out.len(), reference.len(), "linear-convolution length");
        let max = out
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max < 1e-4,
            "FFT convolution must match naive: max err {max}"
        );
    }

    #[test]
    fn sub_block_feeds_are_staged_not_padded() {
        let kernel = noise(100, 5);
        let input = noise(400, 6);
        let whole = convolve_all(&kernel, BLOCK, &input);

        // Feed the same input in ragged little pieces; staging must make the
        // result identical to one big call.
        let mut convolver = Convolver::new(&kernel, BLOCK);
        let mut piecemeal = Vec::new();
        let sizes = [1usize, 7, 3, 40, 13, 100];
        let mut cursor = 0;
        let mut i = 0;
        while cursor < input.len() {
            let take = sizes[i % sizes.len()].min(input.len() - cursor);
            convolver.process(&input[cursor..cursor + take], &mut piecemeal);
            cursor += take;
            i += 1;
        }
        convolver.drain(&mut piecemeal);

        assert_eq!(whole.len(), piecemeal.len(), "same total output length");
        let max = whole
            .iter()
            .zip(&piecemeal)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max < 1e-6,
            "chunking must not change the result: max err {max}"
        );
    }

    #[test]
    fn drain_emits_input_plus_kernel_minus_one_frames() {
        let kernel = noise(90, 7);
        let input = noise(300, 8);
        let mut convolver = Convolver::new(&kernel, BLOCK);
        let mut out = Vec::new();
        convolver.process(&input, &mut out);
        assert_eq!(out.len(), 256, "process emits only whole blocks (4·64)");
        convolver.drain(&mut out);
        assert_eq!(
            out.len(),
            input.len() + kernel.len() - 1,
            "drain flushes the staged remainder and the tail"
        );
    }

    #[test]
    fn reset_leaves_no_bleed_from_before() {
        let kernel = noise(80, 9);
        let mut convolver = Convolver::new(&kernel, BLOCK);
        let mut sink = Vec::new();
        convolver.process(&noise(BLOCK * 3, 10), &mut sink);
        convolver.reset();

        // Silence after a reset must convolve to silence — no retained tail.
        let mut out = Vec::new();
        convolver.process(&vec![0.0f32; BLOCK * 3], &mut out);
        convolver.drain(&mut out);
        let peak = out.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(
            peak < 1e-6,
            "audio before a reset must not bleed after: peak {peak}"
        );
    }

    #[test]
    fn stereo_channels_do_not_leak() {
        // Left silent, right a tone; an identity kernel must keep them apart.
        let mut convolver = StereoConvolver::new(&[1.0], BLOCK, 2);
        let mut input = Vec::new();
        for n in 0..BLOCK * 4 {
            input.push(0.0);
            input.push((n as f32 * 0.1).sin() * 0.5);
        }
        let mut out = Vec::new();
        convolver.process(&input, &mut out);
        convolver.drain(&mut out);

        let left_peak = out.iter().step_by(2).fold(0.0f32, |m, s| m.max(s.abs()));
        let right_peak = out
            .iter()
            .skip(1)
            .step_by(2)
            .fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(
            left_peak < 1e-4,
            "the silent channel stayed silent: peak {left_peak}"
        );
        assert!(
            right_peak > 0.3,
            "the tone channel kept its level: peak {right_peak}"
        );
    }

    #[test]
    fn a_kernel_swap_does_not_click() {
        // Two different lowpass-ish kernels; a steady sine through the swap must
        // have no sample step beyond the tone's own slope with fade headroom.
        let rate = 48_000.0f32;
        let hz = 220.0f32;
        let kernel_a: Vec<f32> = hann(48).iter().map(|w| w / 24.0).collect();
        let kernel_b: Vec<f32> = hann(96).iter().map(|w| w / 48.0).collect();

        let mut convolver = StereoConvolver::new(&kernel_a, BLOCK, 1);
        let tone: Vec<f32> = (0..BLOCK * 8)
            .map(|n| (std::f32::consts::TAU * hz * n as f32 / rate).sin() * 0.5)
            .collect();

        let mut out = Vec::new();
        let half = BLOCK * 4;
        convolver.process(&tone[..half], &mut out);
        convolver.set_kernel(&kernel_b);
        convolver.process(&tone[half..], &mut out);
        convolver.drain(&mut out);

        // Skip the very start (kernel warm-up) and check adjacent steps.
        let max_step = out
            .windows(2)
            .skip(BLOCK)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_step < 0.05,
            "a kernel swap must not click: max step {max_step}"
        );
    }

    #[test]
    fn a_completed_swap_applies_the_new_kernel() {
        // Identity → a half-gain kernel: once the one-block fade is behind us,
        // the output must follow the new kernel exactly.
        let mut convolver = StereoConvolver::new(&[1.0], BLOCK, 1);
        let input = vec![1.0f32; BLOCK * 6];
        let mut out = Vec::new();
        convolver.process(&input[..BLOCK * 2], &mut out);
        convolver.set_kernel(&[0.5]);
        convolver.process(&input[BLOCK * 2..], &mut out);
        convolver.drain(&mut out);

        let tail = &out[out.len() - BLOCK..];
        let avg = tail.iter().sum::<f32>() / tail.len() as f32;
        assert!(
            (avg - 0.5).abs() < 1e-3,
            "past the fade the output follows the new 0.5 kernel: {avg}"
        );
    }

    #[test]
    fn a_lowpass_kernel_passes_lows_and_cuts_highs() {
        use test_support::response_db;
        let rate = 48_000u32;
        // A crude windowed-sinc lowpass at ~2 kHz, so the response helper has a
        // real shape to measure: lows near 0 dB, a high tone well down.
        let cutoff = 2_000.0f32;
        let taps = 257usize;
        let kernel = lowpass(taps, cutoff, rate);

        let low = response_db(|input| convolve_all(&kernel, 128, input), rate, 500.0);
        let high = response_db(|input| convolve_all(&kernel, 128, input), rate, 12_000.0);
        assert!(low.abs() < 1.0, "a low tone passes near unity: {low:.2} dB");
        assert!(high < -20.0, "a high tone is well attenuated: {high:.2} dB");
    }

    /// A windowed-sinc lowpass FIR, for the response test above.
    fn lowpass(taps: usize, cutoff: f32, rate: u32) -> Vec<f32> {
        let fc = cutoff / rate as f32; // normalized 0..0.5
        let mid = (taps - 1) as f32 / 2.0;
        let window = hann(taps);
        let mut kernel: Vec<f32> = (0..taps)
            .map(|i| {
                let x = i as f32 - mid;
                let sinc = if x == 0.0 {
                    2.0 * fc
                } else {
                    (std::f32::consts::TAU * fc * x).sin() / (std::f32::consts::PI * x)
                };
                sinc * window[i]
            })
            .collect();
        let sum: f32 = kernel.iter().sum();
        kernel.iter_mut().for_each(|k| *k /= sum); // unity DC gain
        kernel
    }

    #[test]
    fn hann_windows_have_the_expected_endpoints() {
        let symmetric = hann(8);
        assert!(
            symmetric[0].abs() < 1e-6,
            "symmetric Hann is zero at the start"
        );
        assert!(symmetric[7].abs() < 1e-6, "and zero at the end");

        let periodic = hann_periodic(8);
        assert!(periodic[0].abs() < 1e-6, "periodic Hann is zero at index 0");
        assert!(periodic[4] > 0.99, "and peaks at the centre");
    }
}
