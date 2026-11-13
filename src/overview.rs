//! A whole track's shape at a glance, as a DAW draws an audio clip: the
//! track decoded from end to end, as fast as the processor and the network
//! allow, and reduced to one slice per hundredth of a second holding the
//! mono signal's lowest and highest sample and its RMS.
//!
//! The host draws the slices however it likes; nothing here knows about
//! pixels. Slices are handed over in batches while the decode runs, so a
//! slow source (a media server's file) can be drawn as it arrives, and the
//! host can stop the read at any batch.

use std::ops::ControlFlow;

use crate::decode::Decoder;
use crate::queue::Location;

/// Slices per second of audio: each covers ten milliseconds, fine enough for
/// a ten-minute track across a 4K screen, and 1 200 bytes per second.
pub const OVERVIEW_SLICES_PER_SECOND: u32 = 100;

/// How many interleaved samples are decoded between two batches: about a
/// tenth of a second of stereo at 48 kHz, so memory stays flat on any length.
const READ_CHUNK: usize = 8_192;

/// Ten milliseconds of the track, mixed down to mono.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OverviewSlice {
    /// The lowest sample, in −1..=0.
    pub minimum: f32,
    /// The highest sample, in 0..=1.
    pub maximum: f32,
    /// The root mean square of the slice's samples.
    pub rms: f32,
}

/// Cuts interleaved samples into slices, carrying a partial slice and a
/// partial frame from one chunk to the next.
struct SliceBuilder {
    frames_in_slice: usize,
    minimum: f32,
    maximum: f32,
    square_sum: f64,
    /// The channels of a frame cut by the end of a chunk, summed so far.
    partial_frame_sum: f32,
    partial_frame_channels: usize,
}

impl SliceBuilder {
    fn new() -> Self {
        Self {
            frames_in_slice: 0,
            minimum: 0.0,
            maximum: 0.0,
            square_sum: 0.0,
            partial_frame_sum: 0.0,
            partial_frame_channels: 0,
        }
    }

    /// Adds interleaved samples of `channels` channels at `sample_rate`, and
    /// pushes every slice they complete onto `slices`.
    fn feed(
        &mut self,
        samples: &[f32],
        channels: usize,
        sample_rate: u32,
        slices: &mut Vec<OverviewSlice>,
    ) {
        let channels = channels.max(1);
        let frames_per_slice = (sample_rate / OVERVIEW_SLICES_PER_SECOND).max(1) as usize;
        for sample in samples {
            self.partial_frame_sum += sample;
            self.partial_frame_channels += 1;
            if self.partial_frame_channels < channels {
                continue;
            }
            let mono = self.partial_frame_sum / channels as f32;
            self.partial_frame_sum = 0.0;
            self.partial_frame_channels = 0;
            self.minimum = self.minimum.min(mono);
            self.maximum = self.maximum.max(mono);
            self.square_sum += f64::from(mono) * f64::from(mono);
            self.frames_in_slice += 1;
            if self.frames_in_slice >= frames_per_slice {
                slices.push(self.close());
            }
        }
    }

    /// The slice cut short by the end of the track, if it holds anything.
    fn finish(&mut self) -> Option<OverviewSlice> {
        (self.frames_in_slice > 0).then(|| self.close())
    }

    fn close(&mut self) -> OverviewSlice {
        let slice = OverviewSlice {
            minimum: self.minimum.clamp(-1.0, 0.0),
            maximum: self.maximum.clamp(0.0, 1.0),
            rms: ((self.square_sum / self.frames_in_slice.max(1) as f64).sqrt() as f32)
                .clamp(0.0, 1.0),
        };
        self.frames_in_slice = 0;
        self.minimum = 0.0;
        self.maximum = 0.0;
        self.square_sum = 0.0;
        slice
    }
}

/// Decodes `decoder` to its end, handing each batch of new slices to
/// `on_slices`. Returns `Ok(true)` when the track was read to its end and
/// `Ok(false)` when `on_slices` asked to stop.
fn read_decoder(
    decoder: &mut Decoder,
    on_slices: &mut dyn FnMut(&[OverviewSlice]) -> ControlFlow<()>,
) -> Result<bool, String> {
    let mut builder = SliceBuilder::new();
    let mut buffer = vec![0.0f32; READ_CHUNK];
    let mut slices = Vec::new();
    loop {
        let read = decoder.read(&mut buffer)?;
        let format = decoder.format();
        builder.feed(
            &buffer[..read],
            usize::from(format.channels),
            format.sample_rate,
            &mut slices,
        );
        let ended = read < buffer.len();
        if ended {
            slices.extend(builder.finish());
        }
        if !slices.is_empty() {
            if on_slices(&slices).is_break() {
                return Ok(false);
            }
            slices.clear();
        }
        if ended {
            return Ok(true);
        }
    }
}

/// Opens `location` and reads its overview; see
/// [`AudioEngine::read_overview`](crate::AudioEngine::read_overview).
pub(crate) fn read(
    location: &Location,
    runtime: &tokio::runtime::Handle,
    on_slices: &mut dyn FnMut(&[OverviewSlice]) -> ControlFlow<()>,
) -> Result<bool, String> {
    let mut decoder = match location {
        Location::File(path) => Decoder::open_file(path)?,
        Location::Remote(remote) => crate::remote::open_blocking(remote, runtime)?,
    };
    read_decoder(&mut decoder, on_slices).map_err(|error| format!("{location}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{silence, tone, wav_bytes};
    use crate::queue::RemoteFile;

    fn runtime() -> &'static tokio::runtime::Runtime {
        use std::sync::OnceLock;
        static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
        RUNTIME.get_or_init(|| tokio::runtime::Runtime::new().expect("test runtime"))
    }

    /// Writes a fixture WAV into its own temporary directory.
    fn fixture_file(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "audio-stack-overview-{}-{}",
            name,
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).expect("fixture directory");
        let path = directory.join(format!("{name}.wav"));
        std::fs::write(&path, bytes).expect("fixture file");
        path
    }

    fn read_all(location: &Location) -> Vec<OverviewSlice> {
        let mut slices = Vec::new();
        let finished = read(location, runtime().handle(), &mut |batch| {
            slices.extend_from_slice(batch);
            ControlFlow::Continue(())
        })
        .expect("the fixture decodes");
        assert!(finished, "nothing asked the read to stop");
        slices
    }

    #[test]
    fn a_slice_keeps_the_extremes_and_the_rms_of_its_frames() {
        let mut builder = SliceBuilder::new();
        let mut slices = Vec::new();
        // At 400 Hz a slice is four frames.
        builder.feed(&[0.5, -0.25, 0.5, -0.25], 1, 400, &mut slices);
        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].maximum, 0.5);
        assert_eq!(slices[0].minimum, -0.25);
        let expected_rms = ((0.25 + 0.0625) / 2.0f32).sqrt();
        assert!((slices[0].rms - expected_rms).abs() < 1e-6);
    }

    #[test]
    fn channels_are_mixed_and_a_frame_split_across_chunks_stays_whole() {
        let mut builder = SliceBuilder::new();
        let mut slices = Vec::new();
        // Two stereo frames whose channels cancel, then two at full scale, the
        // third frame cut in half by the end of the first chunk.
        builder.feed(&[1.0, -1.0, 1.0, -1.0, 1.0], 2, 400, &mut slices);
        assert!(slices.is_empty());
        builder.feed(&[1.0, 1.0, 1.0], 2, 400, &mut slices);
        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].maximum, 1.0);
        assert_eq!(slices[0].minimum, 0.0);
    }

    #[test]
    fn the_last_partial_slice_is_kept_and_an_empty_one_is_not() {
        let mut builder = SliceBuilder::new();
        let mut slices = Vec::new();
        builder.feed(&[0.5; 6], 1, 400, &mut slices);
        assert_eq!(slices.len(), 1);
        assert_eq!(builder.finish().map(|slice| slice.maximum), Some(0.5));
        assert_eq!(builder.finish(), None);
    }

    #[test]
    fn a_file_reads_as_a_hundred_slices_a_second_loud_then_silent() {
        let rate = 8_000;
        let mut samples = tone(rate, 2, rate as usize, 440.0);
        samples.extend(silence(2, rate as usize));
        let path = fixture_file("tone-then-silence", &wav_bytes(rate, 2, &samples));

        let slices = read_all(&Location::File(path));

        assert_eq!(slices.len(), 2 * OVERVIEW_SLICES_PER_SECOND as usize);
        let (loud, silent) = slices.split_at(OVERVIEW_SLICES_PER_SECOND as usize);
        // The fixture tone peaks at 12 000 of 32 768.
        assert!(loud
            .iter()
            .all(|slice| slice.maximum > 0.3 && slice.minimum < -0.3 && slice.rms > 0.2));
        assert!(silent
            .iter()
            .all(|slice| slice.maximum == 0.0 && slice.minimum == 0.0 && slice.rms == 0.0));
    }

    #[test]
    fn a_read_stops_when_asked() {
        let rate = 48_000;
        let path = fixture_file(
            "long-tone",
            &wav_bytes(rate, 2, &tone(rate, 2, rate as usize * 5, 440.0)),
        );
        let mut batches = 0;
        let finished = read(&Location::File(path), runtime().handle(), &mut |_| {
            batches += 1;
            ControlFlow::Break(())
        })
        .expect("the fixture decodes");
        assert!(!finished);
        assert_eq!(batches, 1);
    }

    #[test]
    fn a_server_file_reads_like_the_same_file_on_disk() {
        let rate = 8_000;
        let bytes = wav_bytes(rate, 1, &tone(rate, 1, rate as usize * 2, 220.0));
        let path = fixture_file("server-copy", &bytes);
        let address = crate::remote::test_server::serve(bytes);
        let remote = Location::Remote(RemoteFile {
            url: format!("{address}/track.wav?api_key=secret"),
            extension: Some("wav".to_owned()),
        });

        assert_eq!(read_all(&remote), read_all(&Location::File(path)));
    }

    #[test]
    fn a_missing_file_is_an_error() {
        let missing = std::env::temp_dir().join("audio-stack-overview-does-not-exist.wav");
        let result = read(&Location::File(missing), runtime().handle(), &mut |_| {
            ControlFlow::Continue(())
        });
        assert!(result.is_err());
    }
}
