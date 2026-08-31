//! Decode-thread music analysis: tempo (BPM) and musical key.
//!
//! The decode thread already sees every sample of a local track, so measuring
//! its tempo and key costs only the STFT — no extra I/O. Fed from the same
//! buffers playback uses (at the source rate, before resampling), exactly like
//! the loudness meter, and finalized the same way: a result is produced only
//! for a track heard end to end, so a seek or a skip reports nothing rather
//! than the wrong thing.
//!
//! - **Tempo** is the autocorrelation of an onset-strength envelope (spectral
//!   flux — the frame-to-frame rise in magnitude), whose strongest lag in the
//!   60–200 BPM range is the beat period.
//! - **Key** is Krumhansl–Schmuckler template matching: fold the STFT
//!   magnitudes into a twelve-bin chroma vector and correlate it, rotated to
//!   each tonic, against the major and minor key profiles.
//!
//! Nothing here is realtime-safe; it belongs on the decode thread.

use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

/// STFT window. 2048 at 44.1/48 kHz is ~45 ms — long enough for a stable
/// low-frequency chroma, short enough to localise an onset.
const FFT_SIZE: usize = 2048;
/// STFT hop. The onset envelope is sampled at `rate / HOP` (~86–94 Hz), fine
/// enough to resolve tempo well past 200 BPM.
const HOP: usize = 512;

/// Tempo search range, in BPM.
const BPM_MIN: f32 = 60.0;
const BPM_MAX: f32 = 200.0;

/// Pitch-class names, index 0 = C.
const NOTE_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// Krumhansl–Schmuckler major key profile (tonic-relative weights).
const MAJOR_PROFILE: [f64; 12] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];
/// Krumhansl–Schmuckler minor key profile.
const MINOR_PROFILE: [f64; 12] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

/// The frequency band folded into chroma: below ~55 Hz and above ~5 kHz is
/// mostly noise and inharmonic energy for key detection.
const CHROMA_MIN_HZ: f32 = 55.0;
const CHROMA_MAX_HZ: f32 = 5000.0;

/// What a completed analysis found. Either estimate may be `None` when the
/// track was too short or too featureless to judge.
#[derive(Clone, Debug, PartialEq)]
pub struct TrackAnalysis {
    pub bpm: Option<f32>,
    pub bpm_confidence: f32,
    pub key: Option<String>,
    pub key_confidence: f32,
}

/// Accumulates tempo and key evidence from a track as it decodes.
pub struct Analysis {
    channels: usize,
    rate: u32,
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    /// The most recent `FFT_SIZE` mono samples, oldest first.
    history: Vec<f32>,
    /// Mono samples not yet consumed by a hop.
    pending: Vec<f32>,
    /// Windowed FFT input and its spectrum, reused per frame.
    fft_in: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    magnitude: Vec<f32>,
    /// Previous frame's magnitude, for spectral flux.
    prev_magnitude: Vec<f32>,
    /// The onset-strength envelope, one value per hop.
    onset: Vec<f32>,
    /// Accumulated chroma over the track.
    chroma: [f64; 12],
    /// Which chroma bin each FFT bin folds into (or `None` outside the band).
    chroma_map: Vec<Option<usize>>,
    started: bool,
}

impl Analysis {
    pub fn new(sample_rate: u32, channels: u16) -> Option<Self> {
        if sample_rate == 0 || channels == 0 {
            return None;
        }
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let bins = FFT_SIZE / 2 + 1;

        // Precompute which pitch class each bin belongs to.
        let chroma_map = (0..bins)
            .map(|bin| {
                let hz = bin as f32 * sample_rate as f32 / FFT_SIZE as f32;
                if !(CHROMA_MIN_HZ..=CHROMA_MAX_HZ).contains(&hz) {
                    return None;
                }
                let midi = 69.0 + 12.0 * (hz / 440.0).log2();
                let class = (midi.round() as i32).rem_euclid(12) as usize;
                Some(class)
            })
            .collect();

        Some(Self {
            channels: usize::from(channels),
            rate: sample_rate,
            scratch: vec![Complex::default(); fft.get_scratch_len()],
            window: (0..FFT_SIZE)
                .map(|n| {
                    let phase = std::f32::consts::TAU * n as f32 / FFT_SIZE as f32;
                    0.5 - 0.5 * phase.cos()
                })
                .collect(),
            history: vec![0.0; FFT_SIZE],
            pending: Vec::with_capacity(HOP),
            fft_in: fft.make_input_vec(),
            spectrum: fft.make_output_vec(),
            magnitude: vec![0.0; bins],
            prev_magnitude: vec![0.0; bins],
            onset: Vec::new(),
            chroma: [0.0; 12],
            chroma_map,
            fft,
            started: false,
        })
    }

    /// Adds interleaved frames, downmixed to mono; a trailing partial frame is
    /// dropped rather than skewing channel alignment.
    pub fn feed(&mut self, interleaved: &[f32]) {
        let channels = self.channels;
        for frame in interleaved.chunks_exact(channels) {
            let mono = frame.iter().sum::<f32>() / channels as f32;
            self.pending.push(mono);
            if self.pending.len() == HOP {
                self.consume_hop();
            }
        }
    }

    /// Slides the analysis window forward one hop and folds the new frame into
    /// the onset envelope and chroma.
    fn consume_hop(&mut self) {
        self.history.rotate_left(HOP);
        self.history[FFT_SIZE - HOP..].copy_from_slice(&self.pending);
        self.pending.clear();

        for ((slot, sample), weight) in self
            .fft_in
            .iter_mut()
            .zip(self.history.iter())
            .zip(self.window.iter())
        {
            *slot = sample * weight;
        }
        if self
            .fft
            .process_with_scratch(&mut self.fft_in, &mut self.spectrum, &mut self.scratch)
            .is_err()
        {
            return;
        }

        // Magnitudes; spectral flux against the previous frame is the onset.
        let mut flux = 0.0f32;
        for ((mag, bin), prev) in self
            .magnitude
            .iter_mut()
            .zip(self.spectrum.iter())
            .zip(self.prev_magnitude.iter())
        {
            *mag = bin.norm();
            let rise = *mag - *prev;
            if rise > 0.0 {
                flux += rise;
            }
        }
        // The first frame has no predecessor, so its flux is a meaningless
        // whole-spectrum onset — skip it.
        if self.started {
            self.onset.push(flux);
        }
        self.started = true;
        self.prev_magnitude.copy_from_slice(&self.magnitude);

        // Fold this frame's magnitudes into chroma.
        for (mag, class) in self.magnitude.iter().zip(self.chroma_map.iter()) {
            if let Some(class) = class {
                self.chroma[*class] += f64::from(*mag);
            }
        }
    }

    /// The tempo and key found, or `None` fields where there was not enough to
    /// judge. Consumes nothing; call once at end of track.
    pub fn finish(&self) -> TrackAnalysis {
        let fps = self.rate as f32 / HOP as f32;
        let (bpm, bpm_confidence) = match estimate_bpm(&self.onset, fps) {
            Some((bpm, confidence)) => (Some(bpm), confidence),
            None => (None, 0.0),
        };
        let (key, key_confidence) = match estimate_key(&self.chroma) {
            Some((key, confidence)) => (Some(key), confidence),
            None => (None, 0.0),
        };
        TrackAnalysis {
            bpm,
            bpm_confidence,
            key,
            key_confidence,
        }
    }
}

/// Autocorrelation-based tempo: the strongest lag in the BPM range is the beat
/// period. A parabolic fit around the peak recovers sub-lag precision.
fn estimate_bpm(onset: &[f32], fps: f32) -> Option<(f32, f32)> {
    let min_lag = (fps * 60.0 / BPM_MAX).floor().max(1.0) as usize;
    let max_lag = (fps * 60.0 / BPM_MIN).ceil() as usize;
    // Need at least two full periods at the slowest tempo to trust a lag.
    if onset.len() < max_lag * 2 {
        return None;
    }

    let mean = onset.iter().sum::<f32>() / onset.len() as f32;
    let centered: Vec<f32> = onset.iter().map(|x| x - mean).collect();

    let mut autocorr = Vec::with_capacity(max_lag - min_lag + 1);
    for lag in min_lag..=max_lag {
        let mut sum = 0.0f32;
        for n in 0..centered.len() - lag {
            sum += centered[n] * centered[n + lag];
        }
        autocorr.push(sum);
    }

    let (peak_index, &peak) = autocorr
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))?;
    if peak <= 0.0 {
        return None;
    }

    // Parabolic interpolation on the three points around the peak.
    let refined = peak_index as f32
        + if peak_index > 0 && peak_index + 1 < autocorr.len() {
            let (left, right) = (autocorr[peak_index - 1], autocorr[peak_index + 1]);
            let denom = left - 2.0 * peak + right;
            if denom.abs() > 1e-9 {
                0.5 * (left - right) / denom
            } else {
                0.0
            }
        } else {
            0.0
        };
    let lag = min_lag as f32 + refined;
    let bpm = fps * 60.0 / lag;

    // Confidence: the peak against the mean autocorrelation in the range.
    let mean_ac = autocorr.iter().sum::<f32>() / autocorr.len() as f32;
    let confidence = if mean_ac.abs() > 1e-9 {
        ((peak / mean_ac - 1.0) / 4.0).clamp(0.0, 1.0)
    } else {
        0.0
    };
    Some((bpm, confidence))
}

/// Krumhansl–Schmuckler key estimation: the tonic and mode whose profile best
/// correlates with the accumulated chroma.
fn estimate_key(chroma: &[f64; 12]) -> Option<(String, f32)> {
    let total: f64 = chroma.iter().sum();
    if total < 1e-6 {
        return None;
    }

    let mut best: Option<(f64, usize, bool)> = None;
    for tonic in 0..12 {
        for (is_major, profile) in [(true, &MAJOR_PROFILE), (false, &MINOR_PROFILE)] {
            let rotated: Vec<f64> = (0..12).map(|i| profile[(i + 12 - tonic) % 12]).collect();
            let corr = pearson(chroma, &rotated);
            if best.is_none_or(|(b, ..)| corr > b) {
                best = Some((corr, tonic, is_major));
            }
        }
    }

    let (corr, tonic, is_major) = best?;
    let name = format!(
        "{} {}",
        NOTE_NAMES[tonic],
        if is_major { "major" } else { "minor" }
    );
    Some((name, corr.clamp(0.0, 1.0) as f32))
}

/// Pearson correlation between two length-12 vectors.
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (x, y) in a.iter().zip(b) {
        let da = x - mean_a;
        let db = y - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let denom = (var_a * var_b).sqrt();
    if denom > 1e-12 {
        cov / denom
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 44_100;

    /// A click track: a short impulse every `60/bpm` seconds, `secs` long.
    fn click_track(bpm: f32, secs: f32) -> Vec<f32> {
        let frames = (secs * RATE as f32) as usize;
        let period = (RATE as f32 * 60.0 / bpm) as usize;
        let mut signal = vec![0.0f32; frames];
        let mut n = 0;
        while n < frames {
            // A 5 ms decaying click, enough onset energy to flux on.
            for k in 0..(RATE as usize / 200) {
                if n + k >= frames {
                    break;
                }
                let env = (-(k as f32) / (RATE as f32 * 0.002)).exp();
                signal[n + k] += env * 0.8;
            }
            n += period;
        }
        signal
    }

    fn analyse_mono(signal: &[f32]) -> TrackAnalysis {
        let mut analysis = Analysis::new(RATE, 1).expect("analysis");
        // Feed in playback-sized slices, like the engine does.
        for chunk in signal.chunks(4096) {
            analysis.feed(chunk);
        }
        analysis.finish()
    }

    #[test]
    fn a_click_track_reports_its_tempo() {
        for bpm in [90.0, 120.0, 140.0] {
            let result = analyse_mono(&click_track(bpm, 12.0));
            let measured = result.bpm.expect("a click track has a tempo");
            assert!(
                (measured - bpm).abs() < 2.0,
                "expected ~{bpm} BPM, got {measured:.1}"
            );
        }
    }

    #[test]
    fn a_c_major_chord_reports_c_major() {
        // Sustained C–E–G sine triad: the chroma peaks on C, E, G, which the
        // major profile fits best at the C tonic.
        let secs = 6.0;
        let frames = (secs * RATE as f32) as usize;
        let freqs = [261.63f32, 329.63, 392.00]; // C4, E4, G4
        let signal: Vec<f32> = (0..frames)
            .map(|n| {
                let t = n as f32 / RATE as f32;
                freqs
                    .iter()
                    .map(|f| (std::f32::consts::TAU * f * t).sin())
                    .sum::<f32>()
                    * 0.3
            })
            .collect();

        let key = analyse_mono(&signal).key.expect("a chord has a key");
        assert_eq!(key, "C major", "C–E–G should read as C major");
    }

    #[test]
    fn a_short_listen_reports_no_tempo() {
        // Under two beats at the slowest tempo: nothing to autocorrelate.
        let result = analyse_mono(&click_track(120.0, 0.5));
        assert!(result.bpm.is_none(), "half a second is not a tempo");
    }

    #[test]
    fn silence_has_no_key() {
        let result = analyse_mono(&vec![0.0f32; RATE as usize * 3]);
        assert!(result.key.is_none(), "silence has no key");
    }

    #[test]
    fn a_stereo_downmix_matches_mono() {
        // The same tone on both channels must analyse like the mono version.
        let mono = click_track(120.0, 10.0);
        let stereo: Vec<f32> = mono.iter().flat_map(|&s| [s, s]).collect();
        let mut analysis = Analysis::new(RATE, 2).expect("analysis");
        for chunk in stereo.chunks(4096) {
            analysis.feed(chunk);
        }
        let bpm = analysis.finish().bpm.expect("tempo");
        assert!((bpm - 120.0).abs() < 2.0, "stereo downmix tempo: {bpm:.1}");
    }
}
