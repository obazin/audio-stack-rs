//! Hear the convolution effect, no GUI required.
//!
//! Synthesizes two things, both generated right here so there is nothing to
//! license: a short dry clip with clear transients (a plucked riff with claps,
//! which reverb flatters) and a ~1.6 s decaying-noise reverb impulse response.
//! It writes both to temp WAVs, plays the dry clip, and sweeps the wet/dry mix:
//!
//! - 3 s fully dry (mix 0.0)
//! - 3 s a touch of room (mix 0.25)
//! - 3 s half wet (mix 0.5)
//! - 3 s drenched (mix 0.9)
//!
//! The engine buffers roughly half a second of audio ahead of the device, so
//! each change is *heard* about that long after it is printed.
//!
//! Run with: `cargo run --example convolution --features convolution`
//!
//! (Prefer a real space? Point it at any IR file: pass its path as the first
//! argument and the synthesized reverb is skipped.)

use std::io::Write as _;
use std::sync::Arc;
use std::time::Duration;

use audio_stack_rs::{EngineEvent, EventSink, Measured, QueueEntry, Store};

const SAMPLE_RATE: u32 = 44_100;

fn main() {
    let dry_path = std::env::temp_dir().join("audio-stack-rs-convolution-dry.wav");
    std::fs::write(&dry_path, wav_bytes(&compose(), 2)).expect("write dry wav");

    // A real IR from the command line, or the synthesized reverb otherwise.
    let ir_path = match std::env::args().nth(1) {
        Some(path) => std::path::PathBuf::from(path),
        None => {
            let path = std::env::temp_dir().join("audio-stack-rs-convolution-ir.wav");
            std::fs::write(&path, wav_bytes(&reverb_ir(), 2)).expect("write ir wav");
            path
        }
    };

    let engine = audio_stack_rs::init(Arc::new(NoStore), Arc::new(PrintSink), None);
    engine.set_volume(0.6);
    engine.load_queue(
        vec![QueueEntry {
            track_id: 1,
            path: dry_path.clone(),
            duration_secs: 12.0,
            gain_db: 0.0,
        }],
        0,
    );

    // (wall seconds to hold, wet/dry mix, label)
    let scenario = [
        (3.0, 0.0, "fully dry"),
        (3.0, 0.25, "a touch of room"),
        (3.0, 0.5, "half wet"),
        (3.0, 0.9, "drenched"),
    ];

    for (hold, mix, label) in scenario {
        println!("\n▶ {label} (mix {mix})");
        engine.set_convolution(true, Some(ir_path.clone()), mix);
        std::thread::sleep(Duration::from_secs_f64(hold));
    }

    println!("\ndone");
    engine.stop();
    engine.shutdown();
    let _ = std::fs::remove_file(&dry_path);
}

// ── the dry clip ───────────────────────────────────────────────────────────────

/// Four bars of a plucked riff with a clap on the backbeats at 100 BPM — sparse
/// and transient-heavy, so the reverb tail between hits is easy to hear.
fn compose() -> Vec<f32> {
    let beat = 60.0 / 100.0;
    let bar = beat * 4.0;
    let total = (4.0 * bar * SAMPLE_RATE as f64) as usize;
    let mut left = vec![0.0f32; total];
    let mut right = vec![0.0f32; total];

    // A minor pentatonic riff (semitones from A4), one note per beat.
    let riff = [0, 3, 5, 7, 5, 3, 0, -5];
    for bar_index in 0..4 {
        let bar_start = bar_index as f64 * bar;
        for (step, &note) in riff.iter().enumerate() {
            let start = bar_start + step as f64 * beat / 2.0;
            let freq = 440.0 * f64::powf(2.0, note as f64 / 12.0);
            pluck(&mut left, start, freq, beat, 0.24, 5.0);
            pluck(&mut right, start, freq * 1.004, beat, 0.20, 5.0);
        }
        // Claps on beats 2 and 4.
        for beat_index in [1.0, 3.0] {
            let start = bar_start + beat_index * beat;
            clap(&mut left, start, 0.35);
            clap(&mut right, start, 0.35);
        }
    }

    interleave(&left, &right)
}

/// One exponentially decaying tone (fundamental plus a little second harmonic).
fn pluck(channel: &mut [f32], start_secs: f64, freq: f64, length_secs: f64, gain: f64, decay: f64) {
    let start = (start_secs * SAMPLE_RATE as f64) as usize;
    let frames = (length_secs * SAMPLE_RATE as f64) as usize;
    for n in 0..frames {
        let Some(sample) = channel.get_mut(start + n) else {
            return;
        };
        let t = n as f64 / SAMPLE_RATE as f64;
        let envelope = (-t * decay).exp();
        let phase = 2.0 * std::f64::consts::PI * freq * t;
        let tone = phase.sin() + 0.3 * (2.0 * phase).sin();
        *sample += (gain * envelope * tone) as f32;
    }
}

/// A short filtered-noise burst — a clap, the kind of transient reverb loves.
fn clap(channel: &mut [f32], start_secs: f64, gain: f64) {
    let start = (start_secs * SAMPLE_RATE as f64) as usize;
    let frames = (0.05 * SAMPLE_RATE as f64) as usize;
    let mut state = 0x1234_5678_9abc_def0u64 ^ start as u64;
    for n in 0..frames {
        let Some(sample) = channel.get_mut(start + n) else {
            return;
        };
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let noise = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
        let envelope = (-(n as f32 / frames as f32) * 6.0).exp();
        *sample += gain as f32 * envelope * noise;
    }
}

// ── the impulse response ─────────────────────────────────────────────────────────

/// A cheap but convincing reverb: an early direct spike, then ~1.6 s of
/// exponentially decaying stereo noise (decorrelated per channel for width).
fn reverb_ir() -> Vec<f32> {
    let frames = (1.6 * SAMPLE_RATE as f64) as usize;
    let mut left = vec![0.0f32; frames];
    let mut right = vec![0.0f32; frames];
    left[0] = 1.0;
    right[0] = 1.0;

    let mut state_l = 0xdead_beef_cafe_babeu64;
    let mut state_r = 0x0bad_f00d_1234_5678u64;
    for n in 1..frames {
        let t = n as f32 / SAMPLE_RATE as f32;
        // A short pre-delay before the tail builds, then exponential decay.
        let envelope = if t < 0.02 {
            0.0
        } else {
            (-t * 4.5).exp() * 0.5
        };
        left[n] = envelope * noise(&mut state_l);
        right[n] = envelope * noise(&mut state_r);
    }
    interleave(&left, &right)
}

fn noise(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f32 / u64::MAX as f32) * 2.0 - 1.0
}

// ── wav plumbing ─────────────────────────────────────────────────────────────────

fn interleave(left: &[f32], right: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(left.len() * 2);
    for (l, r) in left.iter().zip(right) {
        out.push(l.clamp(-1.0, 1.0));
        out.push(r.clamp(-1.0, 1.0));
    }
    out
}

/// Interleaved f32 → a 16-bit PCM WAV file, dependency-free.
fn wav_bytes(interleaved: &[f32], channels: u16) -> Vec<u8> {
    let data_len = (interleaved.len() * 2) as u32;
    let block_align = channels * 2;
    let mut bytes = Vec::with_capacity(44 + data_len as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes()); // PCM chunk size
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    bytes.extend_from_slice(&(SAMPLE_RATE * block_align as u32).to_le_bytes()); // byte rate
    bytes.extend_from_slice(&block_align.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for sample in interleaved {
        bytes.extend_from_slice(&((sample * i16::MAX as f32) as i16).to_le_bytes());
    }
    bytes
}

// ── facade plumbing ──────────────────────────────────────────────────────────────

/// This demo has no library database, so nothing needs loudness measured.
struct NoStore;
impl Store for NoStore {
    fn needs_measurement(&self, _track_id: i64) -> bool {
        false
    }
    fn record(&self, _track_id: i64, _measured: Measured) {}
}

/// Prints the playhead and the convolution echoes — the "UI" of this demo.
struct PrintSink;
impl EventSink for PrintSink {
    fn send_event(&self, event: EngineEvent) {
        match event {
            EngineEvent::Position { position_secs, .. } => {
                print!("\r  track position {position_secs:5.1} s ");
                let _ = std::io::stdout().flush();
            }
            EngineEvent::Convolution { enabled, mix } => {
                println!("\r  engine confirms: enabled={enabled} mix={mix}");
            }
            EngineEvent::Error { message } => eprintln!("\rerror: {message}"),
            _ => {}
        }
    }
    fn send_frame(&self, _frame: &[u8]) {}
}
