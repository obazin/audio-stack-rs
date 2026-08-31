//! Hear the pitch-shift effect, no GUI required.
//!
//! Synthesizes a short copyright-free clip (a plucked arpeggio over a bass
//! line — generated right here, so there is nothing to license), writes it to
//! a temp WAV, plays it through the default output device, and steps the pitch
//! in cents while the tempo stays put:
//!
//! - 3 s at normal pitch (0 cents)
//! - 2 s up a whole tone (+200 cents)
//! - 2 s up a fifth (+700 cents)
//! - 2 s down a fourth (−500 cents)
//! - 2 s up an octave (+1200 cents)
//!
//! Duration never changes — only the pitch moves. The engine buffers roughly
//! half a second of audio ahead of the device, so each change is *heard* about
//! that long after it is printed.
//!
//! Run with: `cargo run --example pitch_shift --features pitch`

use std::io::Write as _;
use std::sync::Arc;
use std::time::Duration;

use audio_stack_rs::{EngineEvent, EventSink, Measured, QueueEntry, Store};

const SAMPLE_RATE: u32 = 44_100;

fn main() {
    let path = std::env::temp_dir().join("audio-stack-rs-pitch-shift-demo.wav");
    std::fs::write(&path, wav_bytes(&compose())).expect("write demo wav");

    let engine = audio_stack_rs::init(Arc::new(NoStore), Arc::new(PrintSink), None);
    engine.set_volume(0.6);
    engine.load_queue(
        vec![QueueEntry {
            track_id: 1,
            path: path.clone(),
            duration_secs: 12.0,
            gain_db: 0.0,
        }],
        0,
    );

    // (wall seconds to hold, cents, label)
    let scenario = [
        (3.0, 0.0, "normal pitch"),
        (2.0, 200.0, "up a whole tone"),
        (2.0, 700.0, "up a fifth"),
        (2.0, -500.0, "down a fourth"),
        (2.0, 1200.0, "up an octave"),
    ];

    for (hold, cents, label) in scenario {
        println!("\n▶ {label} ({cents:+} cents)");
        engine.set_pitch_shift(true, cents);
        std::thread::sleep(Duration::from_secs_f64(hold));
    }

    println!("\ndone");
    engine.stop();
    engine.shutdown();
    let _ = std::fs::remove_file(&path);
}

// ── the music ────────────────────────────────────────────────────────────────

/// Six bars of an Am–F–C–G arpeggio over a bass line at 110 BPM: enough
/// texture (attacks, sustained bass, a moving line) to hear the pitch move and
/// the tempo *not*.
fn compose() -> Vec<f32> {
    let beat = 60.0 / 110.0;
    let bar = beat * 4.0;
    let total = (6.0 * bar * SAMPLE_RATE as f64) as usize;
    let mut left = vec![0.0f32; total];
    let mut right = vec![0.0f32; total];

    let chords: [(i32, [i32; 4]); 4] = [
        (-24, [-12, -9, -5, 0]),   // A minor
        (-28, [-16, -12, -9, -4]), // F major
        (-21, [-9, -5, -2, 3]),    // C major
        (-26, [-14, -10, -7, -2]), // G major
    ];
    let pattern = [0usize, 1, 2, 3, 2, 3, 1, 2];

    for bar_index in 0..6 {
        let (bass, notes) = chords[bar_index % chords.len()];
        let bar_start = bar_index as f64 * bar;
        for half in 0..2 {
            let start = bar_start + half as f64 * bar / 2.0;
            pluck(&mut left, start, hz(bass), bar / 2.0, 0.28, 3.0);
            pluck(&mut right, start, hz(bass), bar / 2.0, 0.28, 3.0);
        }
        for (step, &index) in pattern.iter().enumerate() {
            let start = bar_start + step as f64 * beat / 2.0;
            let freq = hz(notes[index]);
            pluck(&mut right, start, freq, beat, 0.22, 6.0);
            pluck(&mut left, start, freq * 1.003, beat, 0.12, 6.0);
        }
    }

    let mut interleaved = Vec::with_capacity(total * 2);
    for (l, r) in left.iter().zip(&right) {
        interleaved.push(l.clamp(-1.0, 1.0));
        interleaved.push(r.clamp(-1.0, 1.0));
    }
    interleaved
}

fn hz(semitones_from_a4: i32) -> f64 {
    440.0 * f64::powf(2.0, semitones_from_a4 as f64 / 12.0)
}

/// One exponentially decaying tone (a cheap pluck: fundamental plus a touch of
/// second harmonic).
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

/// Interleaved stereo f32 → a 16-bit PCM WAV file, dependency-free.
fn wav_bytes(interleaved: &[f32]) -> Vec<u8> {
    let data_len = (interleaved.len() * 2) as u32;
    let mut bytes = Vec::with_capacity(44 + data_len as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    bytes.extend_from_slice(&(SAMPLE_RATE * 2 * 2).to_le_bytes());
    bytes.extend_from_slice(&4u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for sample in interleaved {
        bytes.extend_from_slice(&((sample * i16::MAX as f32) as i16).to_le_bytes());
    }
    bytes
}

// ── facade plumbing ──────────────────────────────────────────────────────────

/// This demo has no library database, so nothing needs loudness measured.
struct NoStore;
impl Store for NoStore {
    fn needs_measurement(&self, _track_id: i64) -> bool {
        false
    }
    fn record(&self, _track_id: i64, _measured: Measured) {}
}

/// Prints the playhead and the pitch-shift echoes — the "UI" of this demo.
struct PrintSink;
impl EventSink for PrintSink {
    fn send_event(&self, event: EngineEvent) {
        match event {
            EngineEvent::Position { position_secs, .. } => {
                print!("\r  track position {position_secs:5.1} s ");
                let _ = std::io::stdout().flush();
            }
            EngineEvent::PitchShift { enabled, cents } => {
                println!("\r  engine confirms: enabled={enabled} cents={cents}");
            }
            EngineEvent::Error { message } => eprintln!("\rerror: {message}"),
            _ => {}
        }
    }
    fn send_frame(&self, _frame: &[u8]) {}
}
