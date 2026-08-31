# Implementation plan: spectral DSP roadmap (direct rustfft/realfft use)

**Status:** Proposal (no code written yet) · **Date:** 2026-08-31 · **Drafted by:** Claude Code, for review by Olivier Bazin

## Summary

`rustfft` is already in this crate's dependency tree three times over — directly as `realfft` (the analyser's spectrum), and transitively via `rubato`'s FFT resampler and the `timestretch` engine — so using it directly costs nothing in dependency weight. This plan sequences the capabilities that unlocks into six phases, ordered by value over risk: a shared spectral foundation (a partitioned convolution engine and FFT test assertions), true-peak loudness per BS.1770-4, a linear-phase EQ as a chain effect, a generic convolution effect (impulse responses: reverb, room/headphone correction, crossfeed), decode-thread music analysis (silence trim for gapless, BPM/key), and — the long game — an owned phase-vocoder pitch-shifter parameterized in **cents**, which is also the exit path from the pinned pre-1.0 `timestretch` dependency if it ever becomes a liability.

Every effect phase plugs into the existing `Chain` (`src/chain.rs`) and therefore touches **zero** engine code beyond the mechanical command/event/facade additions the chain architecture was built for. The two non-effect phases (true-peak, analysis) name their engine touchpoints explicitly and keep them small.

## Ground rules (house constraints, non-negotiable)

- Effects are `Chain` plugins implementing `Effect` (`src/chain.rs`): appending `process`, `drain`, `reset`, `time_ratio` (1.0 for everything in this plan — all duration-preserving), `pending_output_frames` for latency, `spawn_mirror` for crossfades. No new engine touchpoints for effects; control follows the `SetTimeStretch` template (one command variant + one additive `EngineEvent` + one facade one-liner + one `Chain` setter).
- Nothing spectral ever runs in the cpal callback: FFT plans allocate at construction, so all of this lives on the decode thread (chain, meter tap) or the analyser. The callback keeps its no-alloc/no-lock contract untouched.
- Pure Rust, MIT/Apache-compatible only. New capabilities are opt-in Cargo features mirroring `opus`/`stretch` (`dep:`-gated, rationale comments, README "Feature flags" bullets); feature-off builds stay byte-identical.
- Prefer `realfft` (already a direct dependency) as the interface everywhere — all signals here are real-valued and it halves the FFT work. Direct `rustfft` types only where complex spectra are unavoidable (phase vocoder).
- Tests are inline `#[cfg(test)]` modules, sentence-style names, interpolated assertion messages; every phase passes the full matrix: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`, each in default, `--no-default-features`, and all-new-features configurations.
- One phase = one reviewable branch/commit, each independently shippable. Later phases depend only on Phase 1.

## Phase 1 — Spectral foundation: `src/spectral.rs` (core, no feature gate)

**Goal:** the shared machinery every later phase consumes: a uniform-partition FFT convolver, window helpers, and test-side response assertions. Core (not feature-gated) because it adds no dependencies (`realfft` is already direct) and dead-code cost is zero until a consumer exists — but every public item must have a consumer by the end of Phase 2/3, so land it together with one of them if `-D warnings` would otherwise flag unused items.

**Files:** `src/spectral.rs` (new), `src/lib.rs` (module registration).

**Tasks:**

1. `Convolver` — uniform-partition frequency-domain convolution (FDL): `new(kernel: &[f32], block: usize)` precomputes the kernel's partitioned spectra with a `realfft` plan; `process(&mut self, input: &[f32], output: &mut Vec<f32>)` appends, staging sub-block input like `Resampler::process` does; `drain(&mut self, output)` flushes the tail (input length + kernel − 1 total output); `latency_frames()` reports 0 (FDL convolution is causal — kernel *design* owns any linear-phase delay, see Phase 3); allocation-free after construction (preplanned scratch, reused spectra buffers).
2. Multi-channel wrapper `StereoConvolver` (N mono convolvers over interleaved f32, matching the chain's device layout), with `set_kernel(&[f32])` performing a click-free swap: run old and new convolvers in parallel for one block and equal-power crossfade (reuse the `equal_power` shape from `src/engine.rs:1538`).
3. Window helpers: Hann (and periodic variant) as plain `fn hann(n: usize) -> Vec<f32>` — no windowing framework.
4. Test-side spectral assertions, shared via a `#[cfg(test)]`-only submodule (the `src/fixtures.rs` pattern): `dominant_hz(signal, rate)` (promote the copy living in `src/stretch.rs` tests), `magnitude_at(signal, rate, hz)`, and `response_db(process_fn, rate, hz)` for measuring any effect's frequency response with a probe tone.
5. Tests: identity kernel is bit-transparent; a pure-delay kernel delays by exactly k frames; convolution of short random signals matches a naive O(n·m) reference within 1e-5; sub-block feeds are staged not padded; `drain` emits input+kernel−1 frames; kernel swap produces no sample step above threshold on a steady sine; stereo channels do not leak (the `output_stays_interleaved_across_channels` idiom from `src/resample.rs:208`).

**Acceptance:** convolver output matches the naive reference; all existing tests untouched and green.

## Phase 2 — True-peak loudness (BS.1770-4)

> **Status update (2026-08-31, superseded): premise was false; landed as a regression test + docs, not a hand-rolled FIR.** Phase 2's goal below rests on "`Measured` currently records sample peak" and calls for hand-rolling a BS.1770-4 4× polyphase FIR true-peak estimator. That is false against the code: `src/loudness.rs` has, since the initial commit, built the meter with `Mode::I | Mode::TRUE_PEAK` and filled `Measured.peak` from `ebur128`'s `true_peak()` — which is itself a polyphase FIR oversampling 4× (2× above 96 kHz) per BS.1770-4. The measured path was already true peak, and the gain-clamp already uses it. Hand-rolling the FIR would duplicate a tested dependency on the one path where a wrong gain silently clips the DAC. So Phase 2 was implemented as "skip the FIR, add the gap":
>
> - The canonical inter-sample-peak regression test (`true_peak_catches_an_inter_sample_peak_the_samples_miss`): a full-scale Fs/4 sine phased so every sample sits at ±0.707 (−3 dBFS) while the reconstructed crest reaches full scale. The meter reads 1.012 (> 0 dBTP) — proving the inter-sample overshoot a bare sample read would miss is caught. This is exactly the task-5 fixture below, now green.
> - Doc clarifications on `Measured::peak`, `finish`, and `parse_peak` — spelling out that measured peak is BS.1770-4 true peak while ReplayGain tag peaks are sample peak, so the two aren't directly comparable.
> - A project memory recording that Phase 2's premise is superseded, so a future session doesn't try the FIR again.
>
> The remaining tasks below (own FIR, `Measured` field addition + semver decision) are therefore **not** to be implemented; they are kept for the record only.

**Goal:** close the correctness gap in normalization: `Measured` currently records sample peak, which under-reads inter-sample peaks by up to ~3 dB on hot masters, so a "safe" normalization gain can still clip the DAC.

**Files:** `src/loudness.rs`, `src/spectral.rs` (or a small FIR inline in loudness.rs), README.

**Tasks:**

1. Implement the BS.1770-4 Annex 2 true-peak estimator: 4× oversampling via the spec's polyphase FIR interpolator (a fixed small tap set — this is a ~30-line time-domain FIR, designed per the spec; rustfft's role here is *verification*, asserting the interpolator's passband/stopband in tests, not runtime).
2. Feed it from the existing `Loudness` meter path (`meter.feed` in `pump()` at the source rate — already the right tap point; no engine change, the meter internals grow a stage).
3. Extend `Measured` with `true_peak` alongside the existing peak. **Breaking-change decision to confirm at review:** adding a field breaks `Store` implementors' construction of `Measured`; either bump semver and document, or default it via `#[non_exhaustive]`/constructor — pick one, consistently with how `parse_peak` tag parsing feeds in.
4. Use true peak wherever sample peak currently caps the normalization gain clamp; parse `REPLAYGAIN_*_PEAK`-style tags unchanged (they are sample peak; document the discrepancy).
5. Tests: a full-scale sine sampled at a phase offset that hides its crest between samples must read > 0 dBTP while sample peak reads < 0 dBFS (the canonical inter-sample-peak fixture); silence reads −inf; the interpolator's frequency response is flat within spec in the passband (FFT assertion from Phase 1); existing loudness tests unchanged.

**Engine touchpoints:** none (the meter is already fed). **Acceptance:** the fixture above passes; `cargo test` matrix green.

## Phase 3 — Linear-phase EQ as a chain effect (feature `fir-eq`)

**Goal:** a mastering-style EQ with no inter-band phase distortion, as the first consumer proving the Phase 1 convolver. This *complements* the realtime biquad EQ rather than replacing it: the callback EQ stays the zero-latency default; the chain effect is the opt-in high-quality mode.

**Files:** `src/fireq.rs` (new, gated), `src/chain.rs` (one gated setter pair), `src/engine.rs` (one gated command arm + describe echo), `src/events.rs` (one gated variant), `src/lib.rs` (module + facade), `Cargo.toml`, README.

**Tasks:**

1. Kernel design: build the target magnitude response from the same ten band gains as the callback EQ (`CENTER_FREQS`, `EQ_BAND_COUNT` in `src/params.rs`), interpolated smoothly in log-frequency; frequency-sampling design → IFFT → shift to linear phase → Hann window to kernel length. Kernel length ~4096 at 48 kHz (≈ 43 ms latency, ~5 Hz resolution); make it a documented constant, not a knob, for v1.
2. `FirEq` implementing `Effect`: wraps a `StereoConvolver`; `time_ratio` 1.0; `pending_output_frames` = kernel latency; gain changes redesign the kernel on the decode thread and swap via the Phase 1 click-free crossfade; disabled = structural bypass exactly like `TimeStretch` (backend dropped at the next chain reset).
3. Coordination with the callback EQ: when `FirEq` is enabled the engine zeroes the callback EQ gains (via the existing `params.set_eq_gains` atomics) and routes the host's `set_eq` values to the effect instead; disabling restores them. Task: decide and document where that routing lives (recommendation: the facade keeps one `set_eq` entry point and the engine forwards based on the effect's enabled state, so hosts change nothing).
4. Control plane per the template: `EngineCommand::SetFirEq { enabled: bool }`, `EngineEvent::FirEq { enabled: bool, latency_secs: f32 }` echoed on change and from `describe()`, facade `set_fir_eq(&self, enabled: bool)` documenting the latency and the ~0.5 s ring delay for changes.
5. Playhead note: a constant ~43 ms delay shifts heard audio behind the reported position by that amount while enabled. Bounded, constant, and smaller than the ring; document it in the module header and accept for v1 (the chain's `pending_output_frames` already keeps gapless boundaries honest).
6. Tests: response at each band center tracks the requested gain within ±0.5 dB (Phase 1 `response_db` helper); flat gains are audibly transparent (energy delta below threshold; *not* bit-exact — windowed FIR); phase linearity (group delay constant within a tolerance across bands, computable from FFT phase); enable/disable and gain-change click tests on a steady sine; drain recovers the tail; the `Chain` bit-transparency tests still hold with the feature compiled but the effect off.

**Engine touchpoints:** the mechanical command arm + the `set_eq` routing decision (task 3) — the only non-boilerplate line. **Acceptance:** response conformance test passes; latency reported correctly.

## Phase 4 — Convolution effect: impulse responses (feature `convolution`)

**Goal:** one generic IR effect covering reverb, room/headphone correction, and crossfeed — the host supplies or selects the IR.

**Files:** `src/convolution.rs` (new, gated), the usual one-liner touchpoints, `Cargo.toml`, README.

**Tasks:**

1. IR loading through the existing `Decoder` (`src/decode.rs`) so any supported format works as an IR file; resample the IR to device rate with `Resampler` at load (offline, decode thread); cap IR length (e.g. 10 s) and document memory cost (partition spectra ≈ 8 bytes/sample/channel).
2. `Convolution` implementing `Effect`: wet/dry mix parameter (0..=1, equal-power), true/false-stereo handling (mono IR applied per channel vs stereo IR pair); latency 0 (causal); `spawn_mirror` shares the immutable partitioned kernel via `Arc` so a crossfade doesn't re-FFT the IR.
3. Control plane: `SetConvolution { enabled, ir_path: Option<PathBuf>, mix: f32 }` + echo event + facade; IR load errors surface as `EngineEvent::Error` and leave the effect bypassed.
4. Device-rate change (`reconfigure`) re-resamples the kernel from the retained source-rate copy.
5. Tests: delta IR at mix 1.0 is transparent minus float noise; a known two-tap echo IR produces the expected echo; mix 0.0 is bit-transparent bypass; kernel Arc-sharing keeps `spawn_mirror` allocation bounded; load-failure path emits an error and stays bypassed.

**Engine touchpoints:** mechanical only. **Acceptance:** echo-IR test passes; a real reverb IR plays audibly (manual listen, example extended or a new `examples/convolution.rs` following `examples/time_stretch.rs`).

## Phase 5 — Decode-thread music analysis (feature `analysis`)

> **Status update (2026-08-31): BPM/key (tasks 2–4) landed; silence/padding trim (task 1) deferred.** The tempo and key analysis is implemented behind the `analysis` feature — the STFT tap, the two estimators, and the `EngineEvent::TrackAnalysis` event, on the loudness meter's lifecycle (events only, no `Store` hook). The gapless silence/padding trim (task 1) was **deferred** at review: its stated trim point is wrong. `advance_or_stop` runs only when `pending_out.is_empty()` (`exhausted = decoder.is_exhausted() && pending_out.is_empty()`), so the outgoing silent tail has already been pushed to the ring by then — there is nothing to trim there. Doing it correctly needs a *silence hold-back* in the core pump loop (a trailing-silence run spans several pump buffers; the earlier ones are ringed before exhaustion is known — so silent runs must be held and dropped only into a gapless join), which is surgery on the delicate, heavily-tested pump/ring/`frames_written`/timeline path, and the incoming lead-in skip is similarly entangled (no seek to an unknown offset — decode-then-detect, shifting the boundary/position math). It also partly overlaps symphonia's own encoder-delay/padding trimming for formats that carry gapless metadata. Left as a tracked follow-up rather than risking the core playback loop; see the project memory.

**Goal:** put the decode thread's free access to every sample to work: trim encoder padding for truly gapless joins, and surface BPM/key to hosts.

**Files:** `src/analysis.rs` (new, gated), `src/engine.rs` (two small non-chain touchpoints, named below), `src/events.rs`, README.

**Tasks:**

1. Silence/padding detection: track trailing energy during decode (time-domain RMS window; cheap, always-on within the feature); at a gapless join, shorten the outgoing tail and the incoming lead-in below a threshold (~−70 dBFS, a documented constant). **Engine touchpoints (the honest list):** the `advance_or_stop` gapless branch (trim `pending_out`'s silent tail before the boundary is placed) and `install_decoder`/preload (skip the incoming track's silent lead-in via the existing seek machinery). Both small; both feature-gated; both tested against the existing boundary tests.
2. Tempo estimation: onset strength via spectral flux (realfft STFT over a mono downmix tap, hop ~512), autocorrelation over the onset envelope, report BPM with a confidence; runs incrementally while the track plays, finalized like the loudness meter (`finish_measuring` pattern).
3. Key estimation: chroma folding of the same STFT magnitudes, Krumhansl-style template correlation, report key + confidence.
4. New `EngineEvent::TrackAnalysis { track_id, bpm: Option<f32>, key: Option<String>, confidence... }` emitted once per fully-heard track (the `finish_measuring(complete: true)` condition), plus an optional `Store`-style persistence hook **only if** review decides hosts should cache it — otherwise events only, no trait change.
5. Tests: click-track fixtures at known BPMs (generate with `fixtures::tone`-style synthesis) within ±2 BPM; a chord fixture detects its key; silence trim: a track padded with 300 ms of digital silence joins gaplessly with the boundary moved by exactly the padded amount (extend the existing boundary tests); partial listens report nothing.

**Engine touchpoints:** the two named in task 1, plus the meter-style tap. **Acceptance:** BPM/key fixtures pass; padding-trim boundary test passes; feature off = byte-identical.

## Phase 6 — Owned pitch-shift, in cents (feature `pitch`)

> **Status update (2026-08-31): implemented.** `src/pitch.rs` behind the `pitch` feature: a streaming phase vocoder over `realfft` (2048/512, instantaneous-frequency phase accumulation) that time-stretches by the pitch ratio and linearly resamples back to length — duration-preserving, `time_ratio` 1.0. `PitchShift` is a `Chain` effect parameterized in cents (±1200), with the `SetTimeStretch` control-plane template (`SetPitchShift`/`EngineEvent::PitchShift`/`set_pitch_shift`), disable-ramps-to-unity semantics, and an `examples/pitch_shift.rs` walkthrough. One deviation from the tasks below: **identity/peak phase locking (task 2) was not implemented** — the basic instantaneous-frequency vocoder passes every acceptance test (frequency ratios, length preservation, no-click sweep, composition with time-stretch), and phase-locking is a phasiness-quality refinement, not a correctness requirement; it (and the task-7 `pitch_shift`-crate fallback) remain available as quality follow-ups if real music warrants. The stereo path runs an independent vocoder per channel (a documented v1 simplification; a shared-phase or mid/side variant would tighten the stereo image).

**Goal:** the deferred pitch-shift effect, parameterized in **cents** (never semitones — house decision), built on rustfft directly as a phase vocoder with identity-locked phases; duration-preserving (`time_ratio` 1.0), so the playhead needs nothing. Also the strategic hedge: the same STFT core is the seed for an owned time-stretcher should the pinned `timestretch` dependency need replacing.

**Files:** `src/pitch.rs` (new, gated), usual touchpoints, `Cargo.toml`, README, and an update to this doc + the time-stretch design doc.

**Tasks:**

1. STFT engine over `realfft`: analysis/synthesis windows (Hann, 75% overlap, FFT size 2048 at 48 kHz), preallocated plans and scratch, streaming in/out staging like `Resampler`.
2. Phase-vocoder pitch shift by ratio `2^(cents/1200)`: spectral peak identity phase locking (Laroche/Dolson) to avoid phasiness; resample-then-stretch inside the vocoder so duration is exactly preserved; latency = one FFT window, reported via `pending_output_frames`.
3. `PitchShift` implementing `Effect`: `cents: f32` clamped ±1200 (one octave, documented constant), smooth live changes by interpolating the ratio per hop; disable semantics copied from `TimeStretch` (ramp to 0 cents, structural bypass at next reset).
4. Control plane: `SetPitchShift { enabled: bool, cents: f32 }` + `EngineEvent::PitchShift` + facade `set_pitch_shift(&self, enabled: bool, cents: f32)`; ordering in the chain after `TimeStretch` (both duration-preserving from the chain's view, so order only affects audio quality — validate by ear in the spike).
5. Tests: +1200 cents doubles the dominant frequency, −1200 halves it, +100 cents ≈ ×1.0595, all length-preserving after drain (the `stretch.rs` test idioms, via the Phase 1 helpers); live cents sweep has no discontinuity; reset has no bleed; combined with `TimeStretch` at 0.5×, pitch shift still lands within tolerance (the two effects compose).
6. Extend `examples/time_stretch.rs` or add `examples/pitch_shift.rs` for an audible no-GUI walkthrough.
7. Quality bar and fallback: if the phase vocoder's transient smearing is unacceptable on real music (manual listen), the documented fallback is wrapping the `pitch_shift` 2.1 crate (per the 2026-08-30 research: per-channel 128-sample `Shifter`, semitones f32 — cents just divide by 100 at the boundary) behind the same `PitchShift` effect API, keeping the public surface identical.

**Engine touchpoints:** mechanical only. **Acceptance:** frequency-ratio tests pass; composition-with-stretch test passes; audible walkthrough sounds clean on music.

## Sequencing, risks, out of scope

- **Order:** 1 → 2 and 3 in either order (2 is smallest-first if a quick win is wanted; 3 proves the convolver) → 4 → 5 → 6. Each phase lands with the full verification matrix and a README update; the "N unit tests" counter in README updates per phase.
- **Risks:** FIR EQ latency (~43 ms) is inherent to linear phase — documented, not fixable; phase-vocoder quality on transients is the classic weakness — Phase 6 task 7 is the pre-agreed fallback; `Measured` field addition is a semver decision to settle at Phase 2 review; analysis phase touches the engine (only phase that does beyond boilerplate) — its two touchpoints are named up front and guarded by the existing boundary tests.
- **Out of scope, tracked separately:** upgrading the *callback* biquad EQ to Vicanek matched (decramped) coefficients — time-domain, no rustfft involvement, vendorable from the MIT `phosphor-dsp` reference per the 2026-08-31 EQ research; multi-band dynamics; visualizer constant-Q upgrade (pure analyser change, can ride along with Phase 5 if wanted).
