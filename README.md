# audio-stack-rs

A self-contained, **host-agnostic** Rust audio stack: a complete local + web-radio playback backend behind one small facade, with **no UI, GUI-framework, or database dependency**. You inject a couple of traits and drive it with plain method calls.

It was extracted from the [Janis](https://github.com/obazin/janis) desktop player so it can be reused on its own, and is MIT-licensed.

## Features

- **Playback engine** — a dedicated engine thread decodes with [symphonia](https://crates.io/crates/symphonia) (plus a bundled libopus-backed Opus decoder), resamples with [rubato](https://crates.io/crates/rubato) only when the file and device rates differ, runs a 10-band biquad EQ, and feeds a lock-free ring drained by a [cpal](https://crates.io/crates/cpal) realtime callback. The callback never allocates, locks, or blocks.
- **Gapless, crossfade, or a deliberate gap** — track joins happen in-ring for true gapless play, or overlap with an equal-power crossfade, or flush to a clean brief silence — your choice at runtime.
- **Volume normalization** — EBU R128 ([ebur128](https://crates.io/crates/ebur128)) measured while a track plays, or ReplayGain tags where present. You provide a `Store` for persistence; the library fills the answer in as tracks are heard.
- **Web radio** — an HTTP stream buffered into the same decode path as a local file, with automatic reconnect/backoff, Icecast/Shoutcast ICY titles, and pluggable now-playing providers (SomaFM, Radio France, Radio Paradise) whose cover art is fetched over an https host allowlist.
- **Time-stretch** *(opt-in `stretch` feature)* — live tempo control (0.25×–2×) with pitch preserved, toggled and adjusted during playback without a click via an effect chain in the decode path. The playhead stays correct while stretched.
- **Linear-phase EQ** *(opt-in `fir-eq` feature)* — a mastering-style FIR EQ (the same ten bands as the realtime biquad EQ) with no inter-band phase distortion, run in the decode-path effect chain and toggled live without a click. Trades ~43 ms of latency for constant group delay; while on it takes over from the realtime EQ so the two never stack.
- **Convolution reverb** *(opt-in `convolution` feature)* — a generic impulse-response effect (reverb, room/headphone correction, per-channel filtering): the host supplies an IR file, decoded and resampled to the device rate, applied in the decode-path effect chain with an equal-power wet/dry mix. Uniformly-partitioned frequency-domain convolution, causal (no added latency), IRs capped at ten seconds.
- **Pitch-shift** *(opt-in `pitch` feature)* — an owned phase-vocoder pitch shifter parameterized in cents (±1200 = ±one octave), duration-preserving so the playhead is untouched, toggled and swept live without a click. Composes with time-stretch (each does its own job). Built on `realfft`, not the time-stretch crate.
- **Music analysis** *(opt-in `analysis` feature)* — estimates tempo (BPM) and musical key of each local track heard end to end, from the samples the decode thread already sees (spectral-flux onset autocorrelation for tempo, Krumhansl–Schmuckler chroma matching for key), and reports them to the host as a `TrackAnalysis` event. Nothing to call: it runs automatically for whole listens.
- **Live analyser** — a compact 170-byte visual frame (160 waveform points + 10 spectrum bands) pushed at ~60 Hz.
- **Metadata parsing** — tag, audio-property, and embedded-cover reading via [lofty](https://crates.io/crates/lofty), plus filename-based track-number recovery. Pure functions returning plain data; you own the database.

Cross-platform: CoreAudio, WASAPI, and ALSA via cpal.

## Installation

Not yet on crates.io — depend on it from git:

```toml
[dependencies]
audio-stack-rs = { git = "https://github.com/obazin/audio-stack-rs", tag = "v0.1.0" }
```

### Build requirements

- A C toolchain and **CMake** — only for the default `opus` feature, which builds libopus from source via `opusic-sys`. Building with `--no-default-features` drops both the requirement and `.opus` playback (see [Feature flags](#feature-flags)).
- On **Linux**, the **ALSA** development headers (`libasound2-dev` / `alsa-lib`) for cpal's backend. macOS reaches CoreAudio through the SDK and needs nothing extra.

### Feature flags

- **`opus`** *(default)* — the libopus-backed Opus decoder, so `.opus` files play and appear in `AUDIO_EXTENSIONS`. It is the only thing that pulls in `opusic-sys`/CMake, so `--no-default-features` gives a pure-Rust build that still decodes every other format. Depend on it that way with:

  ```toml
  audio-stack-rs = { git = "https://github.com/obazin/audio-stack-rs", tag = "v0.1.0", default-features = false }
  ```

- **`stretch`** *(opt-in)* — live time-stretch via `AudioEngine::set_time_stretch`: tempo 0.25×–2× with pitch preserved, adjustable during playback. Pure Rust (the [timestretch](https://crates.io/crates/timestretch) engine, pinned exactly), so it adds no C-toolchain requirement:

  ```toml
  audio-stack-rs = { git = "https://github.com/obazin/audio-stack-rs", tag = "v0.1.0", features = ["stretch"] }
  ```

  Hear it without writing a host: `cargo run --example time_stretch --features stretch` synthesizes a copyright-free clip and steps the tempo live through your speakers.

- **`fir-eq`** *(opt-in)* — a linear-phase FIR EQ via `AudioEngine::set_fir_eq`, the same ten bands as the realtime EQ but with no inter-band phase distortion. It runs in the decode-path effect chain and, while enabled, flattens the callback biquad EQ so the two do not stack. The cost is a constant ~43 ms latency (heard audio sits that far behind the reported position). Pure Rust — it reuses the `realfft` the analyser already links, so it adds no new dependency and feature-off builds are byte-identical:

  ```toml
  audio-stack-rs = { git = "https://github.com/obazin/audio-stack-rs", tag = "v0.1.0", features = ["fir-eq"] }
  ```

- **`convolution`** *(opt-in)* — a convolution (impulse-response) effect via `AudioEngine::set_convolution`: reverb, room/headphone correction, or per-channel filtering from any decodable IR file, with an equal-power wet/dry mix. The IR is decoded and resampled to the device rate on load; it is applied causally (no added latency) in the decode-path effect chain, and capped at ten seconds. Pure Rust, reusing the symphonia/rubato/realfft the stack already links, so it adds no new dependency:

  ```toml
  audio-stack-rs = { git = "https://github.com/obazin/audio-stack-rs", tag = "v0.1.0", features = ["convolution"] }
  ```

  Hear it without writing a host: `cargo run --example convolution --features convolution` synthesizes a dry clip and a reverb IR and sweeps the mix dry → wet (pass an IR file path to use a real space instead).

- **`analysis`** *(opt-in)* — decode-thread tempo (BPM) and key estimation, reported per fully-heard local track as `EngineEvent::TrackAnalysis { track_id, bpm, bpm_confidence, key, key_confidence }`. There is nothing to call — it runs automatically while a track plays and reports when the track finishes (a seek or skip reports nothing). Pure Rust, reusing the `realfft` the analyser already links, so it adds no new dependency:

  ```toml
  audio-stack-rs = { git = "https://github.com/obazin/audio-stack-rs", tag = "v0.1.0", features = ["analysis"] }
  ```

- **`pitch`** *(opt-in)* — a duration-preserving pitch-shift via `AudioEngine::set_pitch_shift`, in cents (100 = one semitone, clamped to ±1200 = ±one octave). An owned phase vocoder over `realfft` (not the time-stretch crate), run in the decode-path effect chain and swept live without a click; latency ~43 ms while enabled. Pure Rust, no new dependency:

  ```toml
  audio-stack-rs = { git = "https://github.com/obazin/audio-stack-rs", tag = "v0.1.0", features = ["pitch"] }
  ```

  Hear it without writing a host: `cargo run --example pitch_shift --features pitch` synthesizes a clip and steps the pitch in cents while the tempo stays put.

## Usage

Two traits keep the library agnostic, and one handle drives it:

- **`EventSink`** — you receive transport `EngineEvent`s and raw visual frames and forward them wherever you like (an IPC channel, a callback, an `mpsc`). Called from the engine thread; must not block it.
- **`Store`** — you persist and answer measured loudness (`needs_measurement` / `record`).

```rust
use std::sync::Arc;
use audio_stack_rs::{AudioEngine, EngineEvent, EventSink, Measured, QueueEntry, Store};

struct MySink;
impl EventSink for MySink {
    fn send_event(&self, event: EngineEvent) { /* forward to your UI */ }
    fn send_frame(&self, frame: &[u8])       { /* 170-byte visual frame */ }
}

struct MyStore;
impl Store for MyStore {
    fn needs_measurement(&self, track_id: i64) -> bool { true }
    fn record(&self, track_id: i64, measured: Measured) { /* persist LUFS + peak */ }
}

let engine = AudioEngine::init(Arc::new(MyStore), Arc::new(MySink), /* device_id */ None);

engine.load_queue(
    vec![QueueEntry {
        track_id: 1,
        path: "/music/track.flac".into(),
        duration_secs: 214.0,
        gain_db: 0.0,
    }],
    0,
);
engine.play();
engine.set_crossfade(true);
engine.set_volume(0.8);
```

Web radio is one async call, driven by your runtime:

```rust
engine.play_stream("groovesalad".into(), "https://ice.somafm.com/groovesalad-128-mp3".into(), None).await?;
```

Metadata parsing is standalone — no engine required:

```rust
let meta  = audio_stack_rs::read_metadata(std::path::Path::new("/music/track.flac"))?;
let cover = audio_stack_rs::read_cover("/music/track.flac"); // Option<CoverArt>, base64 data URL parts
```

`AudioEngine` methods (`load_queue`, `play`/`pause`/`toggle`/`stop`, `next`/`previous`/`jump_to`, `seek`, `set_shuffle`/`set_repeat`/`set_normalize`/`set_gapless`/`set_crossfade`, `set_device`, `set_volume`/`set_eq`, `set_time_stretch` with the `stretch` feature, `set_fir_eq` with the `fir-eq` feature, `set_convolution` with the `convolution` feature, `set_pitch_shift` with the `pitch` feature, `play_stream`, `describe`, `devices`, `shutdown`) are the whole control surface. The engine owns a small tokio runtime for its detached network tasks; everything else is synchronous message-passing to the engine thread.

## Architecture notes

- The engine thread owns the `cpal::Stream`, the decoders, and the queue, reached only by command over a `crossbeam-channel`, so the `AudioEngine` handle stays `Send + Sync`.
- PCM never leaves the process boundary you build on top: the host sees only `EngineEvent`s and the 170-byte frame.
- Position is reported from what the callback has actually played, not what the decoder ran ahead to, so a UI playhead matches what the listener hears.

## Development

```sh
cargo test                                   # 175 unit tests (188 with --features stretch, 230 with --all-features); device/network tests are #[ignore]d
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

The ignored tests need a real output device and/or network; run them deliberately with `cargo test -- --ignored`.

## License

MIT © Olivier Bazin. See [LICENSE](LICENSE).
