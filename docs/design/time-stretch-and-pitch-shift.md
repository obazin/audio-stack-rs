# Design: time-stretch and pitch-shift

**Status:** Superseded — both effects implemented via an effect-chain architecture (`src/chain.rs`), not the direct engine wiring proposed below. Time-stretch (2026-08-30, feature `stretch`, `src/stretch.rs`) wraps `timestretch`; pitch-shift (2026-08-31, feature `pitch`, `src/pitch.rs`) is an **owned phase vocoder over `realfft`**, **cents-based (±1200), not semitones** and not the `timestretch` crate — per the spectral DSP roadmap's Phase 6, so it is duration-preserving and needs no playhead rework. Two findings drove pitch off the crate: `timestretch` 0.14.0's real-time engine has no independent pitch-shift control (semitone shifting exists only in its offline API), and its `Keylock` profile only preserves pitch within ±20% — the time-stretch implementation uses the `WideKeylock` profile and clamps the ratio to 0.25–2.0, the range that profile guarantees full-spectrum pitch preservation over. The `SetPitchShift`/facade signatures below therefore take **cents**, not the semitones this proposal sketched. · **Date:** 2026-08-30 (updated 2026-08-31) · **Author:** Olivier Bazin

## Summary

Add two independent, runtime-toggleable DSP effects to the playback path:

- **Time-stretch** — change tempo (playback speed) without changing pitch.
- **Pitch-shift** — change pitch without changing tempo or duration.

Both must be switchable on/off while a track plays and adjustable continuously during playback, with the off-state being bit-transparent (zero cost, no quality loss). The two effects are independent: a time-stretch must not move the pitch, and a pitch-shift must not move the tempo or the track's duration.

The recommendation is to add these as an **opt-in Cargo feature** built on the pure-Rust [`timestretch`](https://crates.io/crates/timestretch) crate, inserted as a new stage in the engine's decode thread between the resampler and the ring, plus a small rework of the playhead accounting so position stays correct under time-stretch.

## Goals

- Independent time-stretch and pitch-shift, each with a runtime on/off and a continuously adjustable amount.
- Live changes take effect during playback without a click or a restart.
- Off-state is a true passthrough — identical bytes to today, no measurable cost — mirroring `Resampler::is_passthrough` (`src/resample.rs:73`).
- The playback position the host sees stays correct and monotonic under time-stretch.
- The library stays pure-Rust in its default build. Anything pulling a C/C++ toolchain back in (which the recent `opus` feature work deliberately removed from the default build) is opt-in only.
- Effects apply to both local files and web radio, since both flow through the same decode path.

## Non-goals

- Formant preservation / correction (keep it simple; can be a later addition).
- Beat detection, automatic tempo sync, or BPM matching.
- Multi-band or per-stem processing.
- Applying the effect inside the realtime cpal callback (see "Placement" — it belongs in the decode thread).

## Requirements recap (from the request)

1. A time-stretch DSP, gated behind an on/off in the audio chain, activatable and adjustable during playback, that does **not** affect pitch.
2. A pitch-shift DSP, same on/off + live-adjust principle, that does **not** affect time/duration.

## Background: the current audio pipeline

Understanding where the effect slots in requires the existing data flow. Two threads matter (`src/lib.rs:9`):

- **Engine (decode) thread** — owns the decoders, the queue, and the `rtrb` ring producer. Not realtime-constrained: it may allocate and block.
- **cpal callback** — the only realtime context. It drains the ring, runs the EQ and gain, taps a mono copy for the analyser, and hands samples to the device. It must never allocate, lock, or block (`src/output.rs:1`).

The decode thread's `pump()` (`src/engine.rs:914`) runs, per turn, for the current decoder:

```
decode packet ─▶ remap_channels ─▶ resampler.process ─▶ pending_out ─▶ push_pending ─▶ ring
   (decode.rs)      (dsp.rs:128)     (resample.rs:89)                   (engine.rs:1007)
```

- `PUMP_FRAMES = 2048` frames per turn (`src/engine.rs:35`).
- The resampler converts source rate → device rate and is a bit-transparent passthrough when they already match (`src/resample.rs:38`). A crossfade runs a mirror of this loop for the incoming track in `advance_transition` (`src/engine.rs:1055`).
- The ring is sized `ring.max(PUMP_FRAMES * 4)` (`src/engine.rs:1319`).

The cpal callback's `process_block` (`src/output.rs:251`) then drains the ring, applies EQ (`src/dsp.rs:67`) and gain (`src/dsp.rs:176`), and calls `params.advance_frames(played)` (`src/output.rs:236`) — where `played` is the count of frames actually drained from the ring.

### The playhead invariant

Playback position is derived from what the callback has actually played, not from what the decoder ran ahead to. `position_secs()` (`src/engine.rs:1390`) computes:

```
position = (frames_played − boundary_frame) / device_rate
```

where `frames_played` is the callback's counter and `boundary_frame` comes from the `Boundary` list (`src/engine.rs:119`), which records where each track begins **measured in device frames written to the ring** (`frames_written`, `src/engine.rs:225`). This rests on a hidden invariant:

> **One device frame played == one source frame of elapsed musical time.**

At 1× that always holds, because resampling changes the frame *count* but not the mapping between wall-clock playback and musical position. **Time-stretch breaks this invariant** — see "The playhead rework" below. Pitch-shift does not (it preserves duration).

## Where the effect goes: the decode thread, not the callback

The stretch/pitch stage is inserted in `pump()` (and the crossfade mirror) **between `resampler.process` and `push_pending`**, exactly where the resampler already sits in the chain. Rationale:

- Time-stretch and pitch-shift are heavy (overlap-add / FFT with internal lookahead buffers). Even a crate whose `process()` is allocation-free is the wrong thing to run in the cpal callback here, because the effect must happen **before** the ring — the ring is precisely the buffer that decouples decode from the realtime callback.
- The decode thread is already the home of the analogous rate-conversion stage, so the code shape, the passthrough pattern, and the "top up on each pump" loop all already exist and are tested.

Ordering with the existing resampler (an open question to confirm in a spike — see Risks): the effect operates at a single rate and the device wants device-rate audio, so the natural placement is **resample to device rate first, then stretch/pitch at device rate**. `timestretch` performs its own internal varispeed; we must make sure we are not double-resampling. The simplest mental model that avoids that: rubato owns *rate matching* (source→device), the stretch stage owns *tempo/pitch* and runs at device rate, feeding the ring at device rate.

## Recommended implementation: the `timestretch` crate, behind a feature

### Crate survey

Searched crates.io / lib.rs / GitHub. Hard metadata (as of 2026-08-30):

| Crate | Version / last release | License | Pure Rust? | Effects | Real-time + live change | Assessment |
|---|---|---|---|---|---|---|
| [`timestretch`](https://crates.io/crates/timestretch) | 0.14.0 · 2026-08-26 | MIT | ✅ (rustfft, ebur128, serde, serde_json, arc-swap) | time **and** pitch | ✅ lock-free `set_tempo_rate`, allocation-free `process()`, CI-gated on click-freeness & spectral similarity vs Rubber Band | **Best fit.** Native both-effects, real-time, click-free live changes. Risk: pre-1.0, rapid API churn (5 releases in Aug 2026). Deps overlap ours (ebur128, serde/json already direct deps; rustfft already present via realfft). |
| [`pitch_shift`](https://crates.io/crates/pitch_shift) | 2.1.0 · 2026-04-20 | MIT | ✅ (phase vocoder) | time **and** pitch | per-call block params, alloc-free state | Settled version, but low-level 128-sample-block API you plumb yourself; phase-vocoder artifacts on transients; stereo = two instances. Solid fallback. |
| [`ssstretch`](https://github.com/bmisiak/ssstretch) (Signalsmith) | 0.1.0 · 2025-03-01 | MIT | ❌ C++ via `cxx` | both | yes | Highest quality (Signalsmith "Four Ways to Write a Pitch-Shifter"), but **requires a C++ toolchain** — reintroduces the build dependency the `opus` feature just made optional. v0.1, stale binding. |
| [`bungee-rs`](https://github.com/emuell/bungee-rs) | FFI bindings | — | ❌ C++ | both | yes | Same C++ build-dependency problem. |
| [Rubber Band](https://breakfastquay.com/rubberband/) | — | GPL / commercial | ❌ C++ | both | yes | Reference quality, but no maintained official Rust binding and GPL/commercial licensing — a poor fit for an MIT library. |
| [`tdpsola`](https://crates.io/crates/tdpsola) | pure Rust | — | ✅ | both (TD-PSOLA) | — | Formant-preserving, but PSOLA needs pitch marks — oriented to monophonic/voice, not arbitrary music. |
| [`rocoder`](https://github.com/ajyoon/rocoder) | — | — | ✅ | both | — | A live-coding application, not a clean embeddable library. |

### Recommendation

**Primary: `timestretch`, gated behind an opt-in `stretch` Cargo feature.** It is the only pure-Rust option that natively provides *both* effects with a real-time, lock-free, click-free live-change API — a direct match for the requirements — and its dependency set barely grows our tree (`ebur128`, `serde`, `serde_json` are already direct dependencies, and `rustfft` is already present transitively via `realfft`; the only genuinely new crate is `arc-swap`). Gating it behind a feature means users who don't want it pay nothing, and its immaturity/API churn is contained behind a pinned exact version.

**Fallback: `pitch_shift`.** If `timestretch`'s churn proves too disruptive, `pitch_shift` is more settled and pure-Rust and does both effects, at the cost of a lower-level integration and weaker transient quality.

**Rejected for the default build: the C++ options (`ssstretch`, `bungee-rs`, Rubber Band).** They would undo the CMake-optional work. If reference-grade quality is ever wanted, the clean path is a *second, non-default* feature (e.g. `stretch-signalsmith`) that opts back into a C++ build — added only on demand.

**Hand-rolled WSOLA** (a compact pure-Rust overlap-add stretcher) remains viable and fits the crate's DIY ethos (it already hand-writes its EQ, gain, and Opus decoder), but it is a few hundred lines of DSP to own and test for no quality advantage over `timestretch`. Kept as a documented alternative, not the primary plan.

## Detailed design

### New module: `src/stretch.rs`

A thin wrapper around the chosen backend, exposing the same shape the resampler does so `pump()` treats it uniformly:

- Constructed for a given `(rate, channels)` with an initial tempo ratio and pitch (semitones).
- `is_passthrough()` — true when tempo ratio == 1.0 and pitch == 0 semitones; in that state `process` just appends its input to the output (bit-transparent), so the effect is free when off.
- `process(&mut self, input: &[f32], output: &mut Vec<f32>)` — interleaved in, interleaved out, appending like `Resampler::process`. Internally feeds the backend and pulls whatever it produces.
- `set_tempo(ratio)` / `set_pitch(semitones)` — update targets; the backend's own lock-free control handles click-free application.
- `reset()` — drop internal state after a flush/seek so stale audio can't bleed across a discontinuity (parallels `Eq::reset` at `src/dsp.rs:85` and `Resampler` staging).
- `matches(rate, channels)` — so a gapless join can keep the instance instead of rebuilding, like `Resampler::matches` (`src/resample.rs:80`).

Pitch in semitones maps to a frequency ratio `2^(semitones / 12)`; the wrapper converts and hands the backend whatever primitive (time factor + pitch factor) it wants. Pitch-shift = internally stretch by `r` then resample by `1/r`, which the backend does for us; it preserves duration and therefore does **not** touch the playhead.

### Engine wiring

- New fields on `Engine`: `stretch: Option<Stretch>`, plus the current `tempo_ratio: f32` and `pitch_semitones: f32` targets (engine-thread state, following `gapless`/`crossfade`/`normalize` at `src/engine.rs:216`). The stage is engine-thread state, **not** part of `Params` (`src/params.rs`), because `Params` is specifically the block the *callback* reads, and this effect runs in the decode thread.
- `pump()` gains one line between resample and `push_pending`: run `pending_out` through `stretch` into a second scratch buffer, then push that. When `stretch.is_passthrough()` the buffer passes straight through.
- `advance_transition()` (`src/engine.rs:1055`) gets the mirror change for the incoming crossfade track.
- The stretch instance is (re)built alongside the resampler in `rebuild_resampler` / `open_resampler` (`src/engine.rs:1363`, `:1519`) and on device change, since it is `(device_rate, channels)`-shaped.

### Control plane and public API

Following the existing command pattern (facade method → `EngineCommand` → engine field):

```rust
// Facade (src/lib.rs)
impl AudioEngine {
    /// Enable/disable time-stretch and set the tempo ratio (1.0 = normal, 2.0 = double speed).
    /// Pitch is unaffected. Takes effect within one ring's worth of buffered audio.
    pub fn set_time_stretch(&self, enabled: bool, ratio: f32);

    /// Enable/disable pitch-shift and set the shift in semitones (0.0 = unchanged).
    /// Tempo and duration are unaffected.
    pub fn set_pitch_shift(&self, enabled: bool, semitones: f32);
}
```

New `EngineCommand::SetTimeStretch { enabled, ratio }` and `EngineCommand::SetPitchShift { enabled, semitones }`. The engine clamps to sane ranges (e.g. tempo 0.25×–4×, pitch ±12 semitones — final ranges TBD) and updates the stage. Values are also surfaced in `describe()` / the relevant `EngineEvent` so a reloaded UI recovers the current settings, consistent with how EQ/volume are echoed.

Latency of a live change ≈ the audio already buffered in the ring (`PUMP_FRAMES * 4` frames plus the device buffer — tens to low-hundreds of ms). If instant response is ever required, an optional in-place flush (reusing the existing flush handshake, `src/params.rs:114`) could re-derive the buffer at the new setting; not planned for v1.

### The playhead rework (the one hard part)

Time-stretch breaks "one device frame played == one source frame elapsed": at tempo ratio `r`, `M` played device-frames now represent `M · r` source-frames of music, and `r` can change mid-track while the ring still holds audio produced at the previous `r`.

Plan: generalize the existing `Boundary` list (`src/engine.rs:119`) into a **source-time timeline**. Today a `Boundary` records `frame` (device-frame position where a track starts). We extend the engine to record lightweight markers as it pushes to the ring: `(device_frame_written, source_seconds_covered_by_that_push)`. Because each push knows how many source-frames it consumed and how many device-frames it produced (the stretch ratio in force for that push), the marker is exact. `position_secs()` then finds the last marker at/below `frames_played` and interpolates the remainder using that marker's local ratio, instead of dividing by `device_rate` alone.

This keeps the "position = what the listener actually heard" property intact, contains the change to the reporting path, and leaves seek untouched (seek still targets source time and rebases via `reset_frames`, `src/params.rs:142`). It needs its own focused tests (see Testing).

Pitch-shift needs none of this — duration is preserved, so the existing accounting stays correct.

### Interactions checked

- **Crossfade / gapless joins** operate in device frames / wall-clock (`Transition::total_frames`, `elapsed_frames`, `src/engine.rs:149`), so a stretched crossfade still spans the intended wall-clock time. The incoming track's stretch stage mirrors the outgoing one.
- **Seek** is unaffected: it seeks by source time and rebases the counter.
- **Analyser** taps the post-processing signal in the callback (`src/output.rs:325`), so it visualizes stretched/shifted audio correctly with no change.
- **Web radio** flows through the same `pump()`; the effect applies to radio for free. (Whether that's desirable UX is a product choice, easily gated.)
- **Normalization / EQ / gain** are downstream in the callback and unaffected.

## Feature gating (Cargo)

Mirrors the `opus` feature added previously:

```toml
[features]
default = ["opus"]              # stretch is opt-in, not default
opus = ["dep:opus", "dep:symphonia-common"]
stretch = ["dep:timestretch"]

[dependencies]
timestretch = { version = "=0.14.0", optional = true }   # pinned exactly; pre-1.0 churn
```

All new code (`src/stretch.rs`, the engine fields/commands, the facade methods, the timeline extension) is `#[cfg(feature = "stretch")]`-gated, exactly as the Opus decoder is. When the feature is off, the codebase is byte-for-byte the current behavior. The README's "Feature flags" section documents it next to `opus`.

## Testing plan

- **Passthrough is bit-transparent** — ratio 1.0 / 0 semitones yields output identical to input (like `matching_rates_pass_through_untouched`, `src/resample.rs:162`).
- **Time-stretch changes length, preserves pitch** — feed a known sine; assert output frame count tracks `1/ratio` and the dominant frequency (via the existing `realfft` path) is unchanged within tolerance.
- **Pitch-shift changes pitch, preserves length** — assert output length equals input length and the dominant frequency scales by `2^(semitones/12)`.
- **Playhead timeline** — unit-test the marker/interpolation math directly: constant ratio, a ratio change mid-buffer, and monotonicity of reported position across a change.
- **Live change is click-free** — process a continuous signal across a ratio change and assert no sample-to-sample discontinuity above a threshold (the backend claims CI-gated click-freeness; we verify our integration).
- **Reset clears state** — after `reset()`, no energy bleeds from pre-reset input into post-reset output.
- **Feature-off build** — `cargo test --no-default-features` and default both green; clippy `-D warnings` and fmt clean in both, as enforced for the `opus` feature.

## Risks and open questions

- **`timestretch` maturity** — pre-1.0 with rapid API churn. Mitigation: pin the exact version; the feature gate isolates blast radius; `pitch_shift` is a fallback.
- **Resampler ↔ stretch ordering / double-resampling** — `timestretch` does internal varispeed. The main thing to validate in a phase-1 spike is that rubato (rate matching) and the stretch stage (tempo/pitch) compose without resampling twice or fighting over rate. This is the highest-uncertainty item.
- **Exact latency and stereo/interleaved plumbing** per `timestretch` profile (Keylock ~12.7 ms vs Tape/WideKeylock 0 ms output delay) need to be pinned during integration.
- **Range/units for the public API** (tempo as ratio vs percentage vs BPM; pitch in semitones vs cents) — proposed ratio + semitones; confirm.

## Phasing

1. **Spike** — wire `timestretch` into `pump()` behind the `stretch` feature, time-stretch only, and confirm the resampler ordering and audio quality on real files. De-risks the biggest unknown.
2. **Playhead timeline** — generalize `Boundary` into source-time markers; add tests.
3. **Pitch-shift** — add the pitch path and the second facade method.
4. **Crossfade path + polish** — mirror the stage into `advance_transition`, handle edge cases (device change, gapless join keeping/rebuilding the instance), README docs, and the full test matrix.

## Alternatives considered

- **Hand-rolled WSOLA** — fits the DIY ethos, no dependency, but ~hundreds of lines of DSP to own for no quality edge over `timestretch`. Reasonable if avoiding the dependency is valued over development time.
- **`pitch_shift`** — the conservative pure-Rust fallback if `timestretch` proves too unstable.
- **C++ backends (Signalsmith / Rubber Band / Bungee)** — best quality, rejected for the default build on the build-dependency and (for Rubber Band) licensing grounds; viable only as an opt-in non-default feature.

## References

- Current pipeline: `src/engine.rs` (`pump` at :914, `Boundary` at :119, `position_secs` at :1390), `src/resample.rs`, `src/output.rs`, `src/dsp.rs`, `src/params.rs`.
- Crates: [timestretch](https://crates.io/crates/timestretch) ([docs](https://docs.rs/timestretch/latest/timestretch/), [repo](https://github.com/robmorgan/timestretch-rs)) · [pitch_shift](https://crates.io/crates/pitch_shift) ([docs](https://docs.rs/pitch_shift)) · [ssstretch](https://github.com/bmisiak/ssstretch) · [bungee-rs](https://github.com/emuell/bungee-rs) · [tdpsola](https://crates.io/crates/tdpsola) · [Rubber Band](https://breakfastquay.com/rubberband/).
