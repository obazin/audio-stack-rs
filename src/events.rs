//! What the engine tells the frontend.
//!
//! The Svelte `PlayerStore` is a mirror of this: Rust owns the queue and the
//! transport, and every change arrives here rather than being computed twice.
//!
//! Visualiser frames do **not** travel as one of these. They go over a second
//! channel as raw bytes — see [`super::analyser::FRAME_BYTES`] — because at
//! 60 Hz the JSON of 170 numbers would cost far more than the 170 bytes do.

use serde::{Deserialize, Serialize};

/// Which way a relative seek moves the playhead. See
/// [`AudioEngine::seek_by`](crate::AudioEngine::seek_by).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SeekDirection {
    /// Later in the track.
    Forward,
    /// Earlier in the track.
    Backward,
}

/// What the engine is playing, if anything.
#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Nothing is loaded or playing.
    Idle,
    /// Playing a local queue entry.
    Local,
    /// Playing a web-radio stream.
    Radio,
}

/// Note the two levels of `rename_all`: the one on the enum renames the
/// *variant tags*, and each variant needs its own to camel-case its fields.
#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase", tag = "event", content = "data")]
pub enum EngineEvent {
    /// Transport state. Sent on every change, and once on subscribe so a
    /// reloaded webview catches up with audio that never stopped.
    #[serde(rename_all = "camelCase")]
    State {
        /// Whether audio is currently flowing (as opposed to paused/stopped).
        playing: bool,
        /// What kind of source is loaded, if any.
        mode: Mode,
        /// Index of the current entry in the queue.
        index: usize,
        /// Number of entries in the queue.
        queue_len: usize,
        /// Whether shuffled queue order is on.
        shuffle: bool,
        /// Whether queue repeat is on.
        repeat: bool,
        /// The station id, when `mode` is [`Mode::Radio`].
        station_id: Option<String>,
    },
    /// The queue's track ids, in queue order. Sent on subscribe alongside
    /// `State`: a reloaded webview can rebuild its `Track` list from the
    /// library with these, where `State`'s bare index and length cannot —
    /// without them the mirror renders "nothing playing" over live audio.
    #[serde(rename_all = "camelCase")]
    Queue {
        /// Track ids in queue order.
        track_ids: Vec<i64>,
    },
    /// Roughly 10 Hz. The frontend interpolates between these with
    /// `performance.now()`, so the playhead stays smooth without paying for
    /// 60 Hz of IPC.
    #[serde(rename_all = "camelCase")]
    Position {
        /// Current playhead position, in seconds.
        position_secs: f64,
        /// Zero for radio, which has no end.
        duration_secs: f64,
    },
    /// The queue moved on — including a gapless transition, which is emitted
    /// when the boundary actually reaches the device rather than when the
    /// decoder crossed it.
    #[serde(rename_all = "camelCase")]
    TrackChanged {
        /// The new index in the queue.
        index: usize,
    },
    /// The format of the source now playing, for the Now Playing badges.
    #[serde(rename_all = "camelCase")]
    Format {
        /// Sample rate in Hz, as decoded from the source.
        sample_rate: u32,
        /// Channel count, as decoded from the source.
        channels: u16,
        /// Human-readable codec name (e.g. `"flac"`).
        codec: String,
    },
    /// What a station is currently playing. Every field is optional: ICY
    /// carries one free-form string, and what can be pulled out of it varies
    /// by station. All-`None` means the station said nothing useful.
    #[serde(rename_all = "camelCase")]
    StreamMetadata {
        /// Track title, when the station sent one.
        title: Option<String>,
        /// Track artist, when the station sent one.
        artist: Option<String>,
        /// Album/station name, when the station sent one.
        album: Option<String>,
        /// Cover art as a `data:` URL. Fetched and encoded in Rust, because
        /// the webview's CSP allows no remote images.
        cover: Option<String>,
    },
    /// The output device actually in use — what the Settings screen shows
    /// instead of the hard-coded "System default" it used to claim.
    #[serde(rename_all = "camelCase")]
    Device {
        /// Device name as reported by the OS.
        name: String,
        /// The device's active sample rate, in Hz.
        sample_rate: u32,
        /// The device's active channel count.
        channels: u16,
    },
    /// The A–B loop, echoed on every change — set, cleared, each pass
    /// completed — and on subscribe so a reloaded UI recovers it. Cleared
    /// (with `enabled: false`) whenever the track changes.
    #[serde(rename_all = "camelCase")]
    Loop {
        /// Whether a loop region is active.
        enabled: bool,
        /// Start of the region, in seconds; 0 when disabled.
        start_secs: f64,
        /// End of the region, in seconds; 0 when disabled.
        end_secs: f64,
        /// Passes still to be repeated once the current one reaches
        /// `end_secs`; `None` means the loop repeats until cleared.
        repeats_left: Option<u32>,
    },
    /// The time-stretch setting, echoed on every change and on subscribe so
    /// a reloaded UI recovers it. `ratio` 1.0 is normal speed; pitch is
    /// never affected.
    #[cfg(feature = "stretch")]
    #[serde(rename_all = "camelCase")]
    TimeStretch {
        /// Whether time-stretch is currently applied.
        enabled: bool,
        /// Tempo ratio (1.0 = normal, 2.0 = double speed).
        ratio: f32,
    },
    /// The linear-phase FIR EQ setting, echoed on every change and on subscribe
    /// so a reloaded UI recovers it. `latency_secs` is the constant delay the
    /// effect adds while enabled (0 when off), which a UI can surface.
    #[cfg(feature = "fir-eq")]
    #[serde(rename_all = "camelCase")]
    FirEq {
        /// Whether the FIR EQ is currently applied.
        enabled: bool,
        /// Constant added latency, in seconds (0 when disabled).
        latency_secs: f32,
    },
    /// The convolution effect setting, echoed on every change and on subscribe
    /// so a reloaded UI recovers it. `mix` is the wet/dry blend (0 dry … 1 wet).
    #[cfg(feature = "convolution")]
    #[serde(rename_all = "camelCase")]
    Convolution {
        /// Whether the convolution effect is currently applied.
        enabled: bool,
        /// Wet/dry blend, 0.0 (dry) to 1.0 (fully wet).
        mix: f32,
    },
    /// Tempo and key of a track heard end to end, emitted once when it
    /// finishes. Either estimate is `None` when the track was too short or too
    /// featureless to judge; the confidences are 0..1.
    #[cfg(feature = "analysis")]
    #[serde(rename_all = "camelCase")]
    TrackAnalysis {
        /// The track that finished.
        track_id: i64,
        /// Estimated tempo, in beats per minute.
        bpm: Option<f32>,
        /// Confidence of the tempo estimate, `0.0..=1.0`.
        bpm_confidence: f32,
        /// Estimated musical key (e.g. `"C major"`).
        key: Option<String>,
        /// Confidence of the key estimate, `0.0..=1.0`.
        key_confidence: f32,
    },
    /// The pitch-shift setting, echoed on every change and on subscribe so a
    /// reloaded UI recovers it. `cents` 0 is normal pitch; ±1200 is an octave.
    #[cfg(feature = "pitch")]
    #[serde(rename_all = "camelCase")]
    PitchShift {
        /// Whether pitch-shift is currently applied.
        enabled: bool,
        /// Shift amount in cents (100 cents = one semitone).
        cents: f32,
    },
    /// Playback failed. Non-fatal: the engine stays alive and idle.
    #[serde(rename_all = "camelCase")]
    Error {
        /// Human-readable failure description.
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json(event: &EngineEvent) -> String {
        serde_json::to_string(event).expect("engine events must serialise")
    }

    #[test]
    fn events_are_tagged_and_camel_cased() {
        let event = EngineEvent::Position {
            position_secs: 12.5,
            duration_secs: 300.0,
        };
        let encoded = json(&event);
        assert!(encoded.contains(r#""event":"position""#), "{encoded}");
        assert!(encoded.contains(r#""positionSecs":12.5"#), "{encoded}");
        assert!(encoded.contains(r#""durationSecs":300.0"#), "{encoded}");
    }

    #[test]
    fn state_fields_are_camel_cased_too() {
        // The enum-level rename_all only touches variant tags, so a missing
        // per-variant attribute would silently ship snake_case to the UI.
        let encoded = json(&EngineEvent::State {
            playing: true,
            mode: Mode::Local,
            index: 3,
            queue_len: 10,
            shuffle: false,
            repeat: true,
            station_id: None,
        });
        assert!(encoded.contains(r#""queueLen":10"#), "{encoded}");
        assert!(encoded.contains(r#""stationId":null"#), "{encoded}");
        assert!(encoded.contains(r#""mode":"local""#), "{encoded}");
    }

    #[cfg(feature = "stretch")]
    #[test]
    fn time_stretch_event_is_tagged_and_camel_cased() {
        let encoded = json(&EngineEvent::TimeStretch {
            enabled: true,
            ratio: 1.5,
        });
        assert!(encoded.contains(r#""event":"timeStretch""#), "{encoded}");
        assert!(encoded.contains(r#""enabled":true"#), "{encoded}");
        assert!(encoded.contains(r#""ratio":1.5"#), "{encoded}");
    }

    #[cfg(feature = "fir-eq")]
    #[test]
    fn fir_eq_event_is_tagged_and_camel_cased() {
        let encoded = json(&EngineEvent::FirEq {
            enabled: true,
            latency_secs: 0.0427,
        });
        assert!(encoded.contains(r#""event":"firEq""#), "{encoded}");
        assert!(encoded.contains(r#""enabled":true"#), "{encoded}");
        assert!(encoded.contains(r#""latencySecs":0.0427"#), "{encoded}");
    }

    #[cfg(feature = "convolution")]
    #[test]
    fn convolution_event_is_tagged_and_camel_cased() {
        let encoded = json(&EngineEvent::Convolution {
            enabled: true,
            mix: 0.5,
        });
        assert!(encoded.contains(r#""event":"convolution""#), "{encoded}");
        assert!(encoded.contains(r#""enabled":true"#), "{encoded}");
        assert!(encoded.contains(r#""mix":0.5"#), "{encoded}");
    }

    #[cfg(feature = "analysis")]
    #[test]
    fn track_analysis_event_is_tagged_and_camel_cased() {
        let encoded = json(&EngineEvent::TrackAnalysis {
            track_id: 42,
            bpm: Some(120.0),
            bpm_confidence: 0.8,
            key: Some("C major".into()),
            key_confidence: 0.6,
        });
        assert!(encoded.contains(r#""event":"trackAnalysis""#), "{encoded}");
        assert!(encoded.contains(r#""trackId":42"#), "{encoded}");
        assert!(encoded.contains(r#""bpm":120.0"#), "{encoded}");
        assert!(encoded.contains(r#""bpmConfidence":0.8"#), "{encoded}");
        assert!(encoded.contains(r#""key":"C major""#), "{encoded}");
    }

    #[cfg(feature = "pitch")]
    #[test]
    fn pitch_shift_event_is_tagged_and_camel_cased() {
        let encoded = json(&EngineEvent::PitchShift {
            enabled: true,
            cents: -700.0,
        });
        assert!(encoded.contains(r#""event":"pitchShift""#), "{encoded}");
        assert!(encoded.contains(r#""enabled":true"#), "{encoded}");
        assert!(encoded.contains(r#""cents":-700.0"#), "{encoded}");
    }

    #[test]
    fn every_variant_carries_its_tag() {
        let events = [
            EngineEvent::TrackChanged { index: 1 },
            EngineEvent::Format {
                sample_rate: 44_100,
                channels: 2,
                codec: "flac".into(),
            },
            EngineEvent::Device {
                name: "Speakers".into(),
                sample_rate: 48_000,
                channels: 2,
            },
            EngineEvent::Error {
                message: "boom".into(),
            },
        ];
        for event in &events {
            let encoded = json(event);
            assert!(encoded.contains(r#""event":"#), "{encoded}");
            assert!(encoded.contains(r#""data":"#), "{encoded}");
        }
    }
}
