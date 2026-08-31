//! Convolution reverb / correction: a generic impulse-response effect as a
//! [`Chain`](super::chain) effect.
//!
//! One effect covers reverb, room and headphone correction, and per-channel
//! filtering: the host supplies an impulse-response *file*, which is decoded
//! through the same [`Decoder`](super::decode) as any track and resampled to
//! the device rate at load. A mono IR is applied to every channel; a stereo IR
//! is a per-channel pair (left IR on the left, right IR on the right) — not a
//! full stereo matrix, so this is reverb and correction rather than crossfeed.
//!
//! It is causal (latency 0): a reverb's impulse starts with the direct sound at
//! t=0, so the wet output aligns with the dry input and no delay is introduced.
//! A wet/dry `mix` (0..=1, equal-power) blends the two; `mix` 0 is bit-identical
//! dry. The reverb tail rings out through the chain's `drain` at end of stream.
//!
//! Cost. Uniformly-partitioned convolution is O(kernel length) per sample, so a
//! long IR is real work — capped at [`MAX_IR_SECS`]. The transformed kernel is
//! ~8 bytes per sample per channel (partition spectra); a 10 s stereo IR at
//! 48 kHz is ~7.7 MB. Both live on the decode thread, never the callback. The
//! kernel and the decoded source IR are `Arc`-shared, so a crossfade mirror and
//! the per-channel convolvers re-transform nothing.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::chain::Effect;
use super::decode::Decoder;
use super::resample::Resampler;
use super::spectral::{Convolver, Kernel};

/// Partition size for the IR convolvers. Larger amortizes a long IR into fewer,
/// bigger FFTs; independent of latency (the convolver is causal).
const BLOCK: usize = 1024;

/// The longest impulse response accepted, in seconds at its source rate. Past
/// this the compute and memory of uniform partitioning stop being sensible.
pub const MAX_IR_SECS: usize = 10;

/// A decoded impulse response at its own sample rate, retained so the kernel
/// can be re-resampled if the device rate changes. `Arc`-shared across mirrors.
struct SourceIr {
    rate: u32,
    channels: usize,
    interleaved: Vec<f32>,
}

/// The built convolution: one convolver per audio channel over the shared
/// per-IR-channel kernels, plus wet/dry scratch. Staged to whole blocks so dry
/// and wet stay sample-aligned.
struct Backend {
    channels: usize,
    /// Transformed IR, one kernel per IR channel, `Arc`-shared with mirrors.
    ir_kernels: Vec<Arc<Kernel>>,
    convolvers: Vec<Convolver>,
    /// Interleaved input held until a whole block has arrived.
    stage: Vec<f32>,
    /// Deinterleaved whole-block dry input and its wet convolution, per channel.
    dry: Vec<Vec<f32>>,
    wet: Vec<Vec<f32>>,
}

/// The impulse-response effect. `enabled`/`mix`/`ir_path` are the user's
/// setting; the backend exists only while there is an IR and a device shape.
pub struct Convolution {
    enabled: bool,
    mix: f32,
    rate: u32,
    channels: usize,
    ir_path: Option<PathBuf>,
    source_ir: Option<Arc<SourceIr>>,
    backend: Option<Backend>,
}

impl Default for Convolution {
    fn default() -> Self {
        Self::new()
    }
}

impl Convolution {
    /// A disabled effect with no IR.
    pub fn new() -> Self {
        Self {
            enabled: false,
            mix: 0.5,
            rate: 0,
            channels: 0,
            ir_path: None,
            source_ir: None,
            backend: None,
        }
    }

    /// Applies the user's setting. Loading a new IR path decodes the file (and
    /// may fail — a bad file leaves the effect disabled and bypassed and
    /// returns the error, so the event echo never claims an IR that is not
    /// loaded); an unchanged path is not re-decoded. The backend is (re)built
    /// lazily by [`reconfigure`](Effect::reconfigure) once the device shape is
    /// known.
    pub fn set(&mut self, enabled: bool, ir_path: Option<PathBuf>, mix: f32) -> Result<(), String> {
        self.mix = mix.clamp(0.0, 1.0);
        if ir_path != self.ir_path {
            self.backend = None;
            self.source_ir = None; // bypassed until a load succeeds
            if let Some(path) = &ir_path {
                match load_source_ir(path) {
                    Ok(source) => self.source_ir = Some(Arc::new(source)),
                    Err(message) => {
                        // The old IR is already dropped (the user asked to
                        // replace it) and none was loaded: recording `enabled`
                        // here would echo "on" with nothing behind it. A failed
                        // load leaves the effect off, and the path unset so a
                        // retry of the same file is attempted, not skipped.
                        self.enabled = false;
                        self.ir_path = None;
                        return Err(message);
                    }
                }
            }
            self.ir_path = ir_path;
        }
        self.enabled = enabled;
        Ok(())
    }

    /// The current setting, for the event echo and `describe`.
    pub fn setting(&self) -> (bool, f32) {
        (self.enabled, self.mix)
    }

    /// The mix actually applied: the user's while enabled, 0 (dry) while
    /// disabled — so a disabled effect stays warm but transparent until the
    /// next reset retires it, the `TimeStretch`/`FirEq` disable shape.
    fn effective_mix(&self) -> f32 {
        if self.enabled {
            self.mix
        } else {
            0.0
        }
    }
}

impl Effect for Convolution {
    fn name(&self) -> &'static str {
        "convolution"
    }

    fn is_active(&self) -> bool {
        self.enabled || self.backend.is_some()
    }

    fn is_bypassed(&self) -> bool {
        self.backend.is_none()
    }

    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) -> Result<(), String> {
        let mix = self.effective_mix();
        match self.backend.as_mut() {
            Some(backend) => backend.process(input, output, mix),
            None => output.extend_from_slice(input),
        }
        Ok(())
    }

    fn drain(&mut self, output: &mut Vec<f32>) -> Result<(), String> {
        let mix = self.effective_mix();
        if let Some(backend) = self.backend.as_mut() {
            backend.drain(output, mix);
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
        // Causal — the wet aligns with the dry, no group delay — but input
        // staged awaiting a whole block has been accepted and not yet emitted,
        // and a gapless boundary must sit past it.
        self.backend
            .as_ref()
            .map_or(0, |backend| (backend.stage.len() / backend.channels) as u64)
    }

    fn matches(&self, rate: u32, channels: usize) -> bool {
        self.rate == rate && self.channels == channels && (self.backend.is_some() || !self.enabled)
    }

    fn reconfigure(&mut self, rate: u32, channels: usize) -> Result<(), String> {
        self.rate = rate;
        self.channels = channels;
        self.backend = None;
        if self.enabled && rate > 0 && channels > 0 {
            if let Some(source) = self.source_ir.clone() {
                self.backend = Some(Backend::build(&source, rate, channels)?);
            }
        }
        Ok(())
    }

    fn spawn_mirror(&self) -> Box<dyn Effect> {
        Box::new(Convolution {
            enabled: self.enabled,
            mix: self.mix,
            rate: self.rate,
            channels: self.channels,
            ir_path: self.ir_path.clone(),
            source_ir: self.source_ir.clone(),
            backend: self.backend.as_ref().map(|b| b.mirror()),
        })
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Backend {
    /// Resamples the source IR to the device rate, transforms each IR channel
    /// once, and wires one convolver per audio channel.
    fn build(source: &SourceIr, device_rate: u32, audio_channels: usize) -> Result<Self, String> {
        let resampled = if source.rate == device_rate {
            source.interleaved.clone()
        } else {
            let mut resampler = Resampler::new(source.rate, device_rate, source.channels)?;
            let mut out = Vec::new();
            resampler.process(&source.interleaved, &mut out)?;
            resampler.drain(&mut out)?;
            out
        };
        let ir_channels = source.channels;
        if resampled.len() / ir_channels == 0 {
            return Err("impulse response is empty".to_string());
        }

        let mut ir_kernels = Vec::with_capacity(ir_channels);
        for channel in 0..ir_channels {
            let samples: Vec<f32> = resampled
                .chunks_exact(ir_channels)
                .map(|frame| frame[channel])
                .collect();
            ir_kernels.push(Arc::new(Kernel::new(&samples, BLOCK)));
        }

        let convolvers = kernels_to_convolvers(&ir_kernels, audio_channels);
        Ok(Self {
            channels: audio_channels,
            ir_kernels,
            convolvers,
            stage: Vec::new(),
            dry: (0..audio_channels).map(|_| Vec::new()).collect(),
            wet: (0..audio_channels).map(|_| Vec::new()).collect(),
        })
    }

    /// A fresh backend sharing the same transformed kernels — the crossfade
    /// mirror, which re-resamples and re-transforms nothing.
    fn mirror(&self) -> Self {
        Self {
            channels: self.channels,
            convolvers: kernels_to_convolvers(&self.ir_kernels, self.channels),
            ir_kernels: self.ir_kernels.clone(),
            stage: Vec::new(),
            dry: (0..self.channels).map(|_| Vec::new()).collect(),
            wet: (0..self.channels).map(|_| Vec::new()).collect(),
        }
    }

    fn process(&mut self, input: &[f32], output: &mut Vec<f32>, mix: f32) {
        let channels = self.channels;
        self.stage.extend_from_slice(input);
        let whole = (self.stage.len() / channels / BLOCK) * BLOCK;
        if whole == 0 {
            return;
        }
        deinterleave(&mut self.dry, &self.stage[..whole * channels], channels);
        for ((convolver, wet), dry) in self
            .convolvers
            .iter_mut()
            .zip(self.wet.iter_mut())
            .zip(self.dry.iter())
        {
            wet.clear();
            convolver.process(dry, wet);
        }
        let (dry_gain, wet_gain) = mix_gains(mix);
        for i in 0..whole {
            for channel in 0..channels {
                output.push(dry_gain * self.dry[channel][i] + wet_gain * self.wet[channel][i]);
            }
        }
        self.stage.drain(..whole * channels);
    }

    /// Flushes the staged remainder and each convolver's tail — the reverb
    /// ring-out. The first `remainder` frames still carry dry; the tail is wet
    /// only (the input has ended).
    fn drain(&mut self, output: &mut Vec<f32>, mix: f32) {
        let channels = self.channels;
        let remainder = self.stage.len() / channels;
        if remainder > 0 {
            deinterleave(&mut self.dry, &self.stage, channels);
        }
        for ((convolver, wet), dry) in self
            .convolvers
            .iter_mut()
            .zip(self.wet.iter_mut())
            .zip(self.dry.iter())
        {
            wet.clear();
            if remainder > 0 {
                convolver.process(dry, wet);
            }
            convolver.drain(wet);
        }
        let (dry_gain, wet_gain) = mix_gains(mix);
        let tail = self.wet[0].len();
        for i in 0..tail {
            for channel in 0..channels {
                let dry_sample = if i < remainder {
                    self.dry[channel][i]
                } else {
                    0.0
                };
                output.push(dry_gain * dry_sample + wet_gain * self.wet[channel][i]);
            }
        }
        self.stage.clear();
    }

    fn reset(&mut self) {
        for convolver in self.convolvers.iter_mut() {
            convolver.reset();
        }
        self.stage.clear();
    }
}

/// One convolver per audio channel, each over the IR channel it maps to (a mono
/// IR feeds every channel; a stereo IR pairs channel-for-channel, clamped).
fn kernels_to_convolvers(ir_kernels: &[Arc<Kernel>], audio_channels: usize) -> Vec<Convolver> {
    let last = ir_kernels.len() - 1;
    (0..audio_channels)
        .map(|channel| Convolver::from_kernel(Arc::clone(&ir_kernels[channel.min(last)])))
        .collect()
}

/// Deinterleaves `src` into per-channel buffers.
fn deinterleave(dst: &mut [Vec<f32>], src: &[f32], channels: usize) {
    for channel in dst.iter_mut() {
        channel.clear();
    }
    for frame in src.chunks_exact(channels) {
        for (channel, sample) in dst.iter_mut().zip(frame) {
            channel.push(*sample);
        }
    }
}

/// Equal-power wet/dry gains at `mix` (0..=1): `(dry, wet)`, so a blend holds
/// perceived level. `mix` 0 is `(1, 0)` — bit-identical dry.
fn mix_gains(mix: f32) -> (f32, f32) {
    let angle = mix.clamp(0.0, 1.0) * std::f32::consts::FRAC_PI_2;
    (angle.cos(), angle.sin())
}

/// Decodes an impulse-response file to interleaved f32 at its own rate, capped
/// at [`MAX_IR_SECS`].
fn load_source_ir(path: &Path) -> Result<SourceIr, String> {
    let mut decoder = Decoder::open_file(path)?;
    let format = decoder.format();
    let rate = format.sample_rate;
    let channels = format.channels as usize;
    if rate == 0 || channels == 0 {
        return Err(format!("impulse response has no audio: {}", path.display()));
    }

    let max_samples = MAX_IR_SECS * rate as usize * channels;
    let mut interleaved = Vec::new();
    let mut buf = vec![0.0f32; 8_192];
    while interleaved.len() < max_samples && !decoder.is_exhausted() {
        let read = decoder.read(&mut buf)?;
        if read == 0 {
            break;
        }
        interleaved.extend_from_slice(&buf[..read]);
    }
    // Whole frames only, and no more than the cap.
    let frames = (interleaved.len() / channels).min(max_samples / channels);
    interleaved.truncate(frames * channels);
    if interleaved.is_empty() {
        return Err(format!(
            "impulse response decoded to nothing: {}",
            path.display()
        ));
    }
    Ok(SourceIr {
        rate,
        channels,
        interleaved,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;

    const RATE: u32 = 48_000;

    /// Writes an interleaved f32 IR to a temp WAV and returns the path.
    fn ir_file(name: &str, rate: u32, channels: u16, interleaved: &[f32]) -> PathBuf {
        let samples: Vec<i16> = interleaved
            .iter()
            .map(|s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .collect();
        let bytes = fixtures::wav_bytes(rate, channels, &samples);
        let path = std::env::temp_dir().join(format!("audio-stack-rs-conv-{name}.wav"));
        std::fs::write(&path, bytes).expect("write ir wav");
        path
    }

    fn enabled_with(path: PathBuf, mix: f32, channels: usize) -> Convolution {
        let mut effect = Convolution::new();
        effect.set(true, Some(path), mix).expect("load ir");
        effect.reconfigure(RATE, channels).expect("build backend");
        effect
    }

    fn run(effect: &mut Convolution, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::new();
        effect.process(input, &mut out).unwrap();
        effect.drain(&mut out).unwrap();
        out
    }

    /// Mono interleaved (1 channel) ramp-free noise for a probe.
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

    #[test]
    fn a_delta_ir_at_full_wet_is_transparent() {
        // A unit impulse convolves to identity, so full wet ≈ the input minus
        // FFT float noise.
        let path = ir_file("delta", RATE, 1, &[1.0]);
        let mut effect = enabled_with(path, 1.0, 1);
        let input = noise(BLOCK * 4, 1);
        let out = run(&mut effect, &input);

        let max = input
            .iter()
            .zip(&out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max < 1e-3,
            "delta IR at mix 1.0 must pass audio: max err {max}"
        );
    }

    #[test]
    fn mix_zero_is_bit_transparent() {
        let path = ir_file("delta0", RATE, 1, &[1.0]);
        let mut effect = enabled_with(path, 0.0, 1);
        let input = noise(BLOCK * 4, 2);
        let out = run(&mut effect, &input);
        for (a, b) in input.iter().zip(&out) {
            assert_eq!(*a, *b, "mix 0.0 must be bit-identical dry");
        }
    }

    #[test]
    fn a_two_tap_ir_produces_the_expected_echo() {
        // IR = impulse at 0 plus a half-scale tap at delay D: the output is the
        // input plus a half-scale copy delayed by D.
        const D: usize = 300;
        let mut ir = vec![0.0f32; D + 1];
        ir[0] = 1.0;
        ir[D] = 0.5;
        let path = ir_file("echo", RATE, 1, &ir);
        let mut effect = enabled_with(path, 1.0, 1);

        let mut input = vec![0.0f32; BLOCK * 2];
        input[10] = 1.0; // a single click
        let out = run(&mut effect, &input);

        assert!(
            (out[10] - 1.0).abs() < 1e-3,
            "direct click at 10: {}",
            out[10]
        );
        assert!(
            (out[10 + D] - 0.5).abs() < 1e-3,
            "echo tap at 10+D: {}",
            out[10 + D]
        );
    }

    #[test]
    fn a_bad_ir_path_errors_and_leaves_the_effect_bypassed() {
        let mut effect = Convolution::new();
        let missing = std::env::temp_dir().join("audio-stack-rs-conv-does-not-exist.wav");
        let result = effect.set(true, Some(missing), 0.5);
        assert!(result.is_err(), "a missing IR file must surface an error");
        assert!(effect.is_bypassed(), "and leave the effect bypassed");
        // A bypassed effect passes audio straight through.
        let input = noise(BLOCK, 3);
        let mut out = Vec::new();
        effect.process(&input, &mut out).unwrap();
        assert_eq!(out, input, "bypassed convolution is a pass-through");
    }

    #[test]
    fn a_stereo_ir_applies_each_channel_to_its_side() {
        // Left IR is a delta (pass), right IR is a half-scale delta (halve).
        let ir: Vec<f32> = [1.0f32, 0.5].to_vec(); // one interleaved frame: L=1.0, R=0.5
        let path = ir_file("stereo", RATE, 2, &ir);
        let mut effect = enabled_with(path, 1.0, 2);

        let mut input = vec![0.0f32; BLOCK * 2 * 2];
        input[20] = 1.0; // left click
        input[21] = 1.0; // right click, same frame
        let out = run(&mut effect, &input);
        assert!((out[20] - 1.0).abs() < 1e-3, "left passes: {}", out[20]);
        assert!((out[21] - 0.5).abs() < 1e-3, "right halves: {}", out[21]);
    }

    #[test]
    fn staged_input_counts_as_pending_output() {
        // Sub-block input is accepted but not yet emitted; a gapless boundary
        // must sit past it, so it has to be reported as pending.
        let path = ir_file("staged", RATE, 1, &[1.0]);
        let mut effect = enabled_with(path, 1.0, 1);
        assert_eq!(
            effect.pending_output_frames(),
            0,
            "fresh effect holds nothing"
        );

        let mut out = Vec::new();
        effect.process(&noise(BLOCK / 2, 4), &mut out).unwrap();
        assert_eq!(
            effect.pending_output_frames(),
            (BLOCK / 2) as u64,
            "half a block in, nothing out: all of it is pending"
        );

        effect.process(&noise(BLOCK / 2, 5), &mut out).unwrap();
        assert_eq!(
            effect.pending_output_frames(),
            0,
            "a whole block has been emitted, the stage is empty"
        );
    }

    #[test]
    fn a_failed_ir_swap_disables_the_effect() {
        // Replacing a working IR with a bad file drops the old IR, so claiming
        // "enabled" would echo an IR that is not loaded — the effect must come
        // back disabled and bypassed, with the error surfaced.
        let path = ir_file("swap-good", RATE, 1, &[1.0]);
        let mut effect = enabled_with(path, 0.8, 1);
        assert_eq!(effect.setting(), (true, 0.8));

        let missing = std::env::temp_dir().join("audio-stack-rs-conv-swap-missing.wav");
        let result = effect.set(true, Some(missing), 0.8);
        assert!(result.is_err(), "a missing replacement IR must error");
        assert!(
            !effect.setting().0,
            "a failed swap must not leave the effect claiming to be enabled"
        );
        assert!(effect.is_bypassed(), "and it is bypassed");
    }

    #[test]
    fn spawn_mirror_shares_the_kernel() {
        let path = ir_file("shared", RATE, 1, &noise(2_000, 9));
        let effect = enabled_with(path, 0.7, 2);
        let before = Arc::strong_count(&effect.backend.as_ref().unwrap().ir_kernels[0]);
        let _mirror = effect.spawn_mirror();
        let after = Arc::strong_count(&effect.backend.as_ref().unwrap().ir_kernels[0]);
        assert!(
            after > before,
            "the mirror must share the transformed kernel, not rebuild it: {before} -> {after}"
        );
    }
}
