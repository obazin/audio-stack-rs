# audio-stack-rs

A self-contained, **host-agnostic** Rust audio stack: a complete local + web-radio playback backend behind one small facade, with **no UI, GUI-framework, or database dependency**. You inject a couple of traits and drive it with plain method calls.

It was extracted from the [Janis](https://github.com/obazin/janis) desktop player so it can be reused on its own, and is MIT-licensed.

## Features

- **Playback engine** — a dedicated engine thread decodes with [symphonia](https://crates.io/crates/symphonia) (plus a bundled libopus-backed Opus decoder), resamples with [rubato](https://crates.io/crates/rubato) only when the file and device rates differ, runs a 10-band biquad EQ, and feeds a lock-free ring drained by a [cpal](https://crates.io/crates/cpal) realtime callback. The callback never allocates, locks, or blocks.
- **Gapless, crossfade, or a deliberate gap** — track joins happen in-ring for true gapless play, or overlap with an equal-power crossfade, or flush to a clean brief silence — your choice at runtime.
- **Volume normalization** — EBU R128 ([ebur128](https://crates.io/crates/ebur128)) measured while a track plays, or ReplayGain tags where present. You provide a `Store` for persistence; the library fills the answer in as tracks are heard.
- **Web radio** — an HTTP stream buffered into the same decode path as a local file, with automatic reconnect/backoff, Icecast/Shoutcast ICY titles, and pluggable now-playing providers (SomaFM, Radio France, Radio Paradise) whose cover art is fetched over an https host allowlist.
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

`AudioEngine` methods (`load_queue`, `play`/`pause`/`toggle`/`stop`, `next`/`previous`/`jump_to`, `seek`, `set_shuffle`/`set_repeat`/`set_normalize`/`set_gapless`/`set_crossfade`, `set_device`, `set_volume`/`set_eq`, `play_stream`, `describe`, `devices`, `shutdown`) are the whole control surface. The engine owns a small tokio runtime for its detached network tasks; everything else is synchronous message-passing to the engine thread.

## Architecture notes

- The engine thread owns the `cpal::Stream`, the decoders, and the queue, reached only by command over a `crossbeam-channel`, so the `AudioEngine` handle stays `Send + Sync`.
- PCM never leaves the process boundary you build on top: the host sees only `EngineEvent`s and the 170-byte frame.
- Position is reported from what the callback has actually played, not what the decoder ran ahead to, so a UI playhead matches what the listener hears.

## Development

```sh
cargo test                                   # 145 unit tests (5 device/network tests are #[ignore]d)
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

The ignored tests need a real output device and/or network; run them deliberately with `cargo test -- --ignored`.

## License

MIT © Olivier Bazin. See [LICENSE](LICENSE).
