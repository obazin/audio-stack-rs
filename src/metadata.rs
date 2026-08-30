//! Audio-file metadata: tags, audio properties, and embedded cover art.
//!
//! Pure functions over a path — no database, no host framework. Reads tags +
//! properties via `lofty` (which normalises every container onto the same
//! keys) and returns plain data the host maps into its own storage. A missing
//! track number is recovered from the filename.

use base64::Engine;
use lofty::picture::PictureType;
use lofty::prelude::*;
use lofty::probe::Probe;
use serde::Serialize;
use std::path::Path;

use crate::loudness;

/// Extensions the scanner picks up. This list tracks what the engine can
/// decode, not what any browser accepts — so `.opus` is present only when the
/// `opus` feature is on. (`.ogg` stays either way: it also carries Vorbis.)
#[cfg(feature = "opus")]
pub const AUDIO_EXTENSIONS: &[&str] = &[
    "mp3", "flac", "wav", "m4a", "aac", "ogg", "opus", "aif", "aiff",
];
#[cfg(not(feature = "opus"))]
pub const AUDIO_EXTENSIONS: &[&str] = &["mp3", "flac", "wav", "m4a", "aac", "ogg", "aif", "aiff"];

/// Container formats that are always lossless. `m4a` is deliberately absent:
/// it can hold either AAC (lossy) or ALAC (lossless) and we don't inspect the
/// codec, so it is reported lossy — the conservative claim.
pub const LOSSLESS_EXTENSIONS: &[&str] = &["flac", "wav", "aif", "aiff"];

/// Cover art as a base64 payload, ready for a `data:` URL.
#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct CoverArt {
    pub mime: String,
    pub data_base64: String,
}

/// Everything one audio file's tags + properties say about it. The host maps
/// this into its own track storage.
#[derive(Clone, Debug)]
pub struct Metadata {
    pub title: String,
    pub artist: Option<String>,
    pub album: Option<String>,
    pub composer: Option<String>,
    pub duration_secs: f64,
    pub format: String,
    pub sample_rate: Option<u32>,
    pub bit_depth: Option<u8>,
    pub channels: Option<u8>,
    pub lossless: bool,
    pub track_number: Option<u32>,
    pub track_total: Option<u32>,
    pub disc_number: Option<u32>,
    pub disc_total: Option<u32>,
    pub year: Option<u32>,
    pub genre: Option<String>,
    pub album_artist: Option<String>,
    pub rg_track_gain_db: Option<f64>,
    pub rg_track_peak: Option<f64>,
    pub rg_album_gain_db: Option<f64>,
    pub rg_album_peak: Option<f64>,
}

/// The file's lowercased extension when it is one the engine can decode.
pub fn audio_extension(path: &Path) -> Option<String> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    AUDIO_EXTENSIONS.contains(&ext.as_str()).then_some(ext)
}

/// Reads tags + properties for one file. Tag-less files still scan: the
/// filename stem becomes the title, properties come from the decoder.
pub fn read_metadata(path: &Path) -> Result<Metadata, String> {
    let ext = audio_extension(path).ok_or_else(|| "not an audio file".to_string())?;
    let tagged = Probe::open(path)
        .map_err(|e| format!("probe {}: {}", path.display(), e))?
        .read()
        .map_err(|e| format!("read {}: {}", path.display(), e))?;

    let props = tagged.properties();
    let tag = tagged.primary_tag().or_else(|| tagged.first_tag());

    let stem_title = || {
        path.file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Unknown".to_string())
    };
    let non_empty = |v: Option<String>| v.filter(|s| !s.trim().is_empty());

    let (title, artist, album, composer) = match tag {
        Some(tag) => (
            non_empty(tag.title().map(|c| c.into_owned())).unwrap_or_else(stem_title),
            non_empty(tag.artist().map(|c| c.into_owned())),
            non_empty(tag.album().map(|c| c.into_owned())),
            non_empty(tag.get_string(&ItemKey::Composer).map(|s| s.to_string())),
        ),
        None => (stem_title(), None, None, None),
    };

    // `get_string` reaches the same item whatever the underlying format wrote
    // it as — ID3v2.2/2.3/2.4 frames, Vorbis comments, MP4 atoms or APE keys
    // all normalise onto these keys, so this needs no per-format branching.
    let number = |key: ItemKey| -> Option<u32> {
        tag.and_then(|t| t.get_string(&key))
            // ID3 writes "3/12" into a single frame; lofty usually splits it,
            // but a tagger that wrote the pair verbatim must not parse to None.
            .and_then(|raw| {
                raw.split('/')
                    .next()
                    .map(str::trim)
                    .and_then(|n| n.parse().ok())
            })
    };
    let text = |key: ItemKey| non_empty(tag.and_then(|t| t.get_string(&key)).map(str::to_string));
    // ReplayGain values are stored as free-form text whatever the container,
    // so the parsing lives in one tolerant place.
    let gain = |key: ItemKey| {
        tag.and_then(|t| t.get_string(&key))
            .and_then(loudness::parse_gain_db)
    };
    let peak = |key: ItemKey| {
        tag.and_then(|t| t.get_string(&key))
            .and_then(loudness::parse_peak)
    };

    // A file with no track number still belongs somewhere in the album, and
    // the position is usually sitting in its filename.
    let from_name = track_number_from_filename(path);
    let track_number = number(ItemKey::TrackNumber).or(from_name.track);
    let disc_number = number(ItemKey::DiscNumber).or(from_name.disc);

    Ok(Metadata {
        title,
        artist,
        album,
        composer,
        duration_secs: props.duration().as_secs_f64(),
        format: ext.to_ascii_uppercase(),
        sample_rate: props.sample_rate(),
        bit_depth: props.bit_depth(),
        channels: props.channels(),
        lossless: LOSSLESS_EXTENSIONS.contains(&ext.as_str()),
        track_number,
        track_total: number(ItemKey::TrackTotal),
        disc_number,
        disc_total: number(ItemKey::DiscTotal),
        year: number(ItemKey::Year).or_else(|| {
            // Vorbis and MP4 usually carry a full date; the leading four
            // digits are the year.
            text(ItemKey::RecordingDate).and_then(|d| d.get(..4).and_then(|y| y.parse().ok()))
        }),
        genre: text(ItemKey::Genre),
        album_artist: text(ItemKey::AlbumArtist),
        rg_track_gain_db: gain(ItemKey::ReplayGainTrackGain),
        rg_track_peak: peak(ItemKey::ReplayGainTrackPeak),
        rg_album_gain_db: gain(ItemKey::ReplayGainAlbumGain),
        rg_album_peak: peak(ItemKey::ReplayGainAlbumPeak),
    })
}

/// A disc and track position read out of a filename.
#[derive(Debug, Default, PartialEq, Eq)]
struct FilenamePosition {
    disc: Option<u32>,
    track: Option<u32>,
}

/// Recovers a track's position from its filename when the tags do not carry
/// one — `07 - Alive.flac`, `2-03 Reprise.mp3`, `104 Title.m4a`.
///
/// Only leading digits count, and only when something separates them from the
/// title. That is what keeps `1984 - Track.mp3` from reading as track 198 and
/// `2001 A Space Odyssey.mp3` from reading as a track at all.
fn track_number_from_filename(path: &Path) -> FilenamePosition {
    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
        return FilenamePosition::default();
    };
    let stem = stem.trim();

    let digits: String = stem.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() || digits.len() > 3 {
        return FilenamePosition::default();
    }
    let rest = &stem[digits.len()..];

    // `2-03 Title` / `2_03 Title`: a single leading digit, a separator, then
    // the real track number. Checked before the plain form so the disc is not
    // mistaken for the track.
    if digits.len() == 1 {
        if let Some(after) = rest.strip_prefix(['-', '_']) {
            let track_digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
            let tail = &after[track_digits.len()..];
            if (1..=3).contains(&track_digits.len()) && starts_with_separator(tail) {
                return FilenamePosition {
                    disc: digits.parse().ok().filter(|d| *d > 0),
                    track: track_digits.parse().ok().filter(|t| *t > 0),
                };
            }
        }
    }

    if !starts_with_separator(rest) {
        return FilenamePosition::default();
    }
    FilenamePosition {
        disc: None,
        track: digits.parse().ok().filter(|t| *t > 0),
    }
}

/// Whether what follows the digits marks them as a standalone number rather
/// than the first part of a word or a longer figure.
fn starts_with_separator(rest: &str) -> bool {
    match rest.chars().next() {
        // Nothing after the digits at all — the whole stem is a number.
        None => true,
        Some(c) => c.is_whitespace() || matches!(c, '-' | '.' | '_'),
    }
}

/// Embedded pictures larger than this are treated as absent. Real sleeve
/// scans run a few MB; the ID3 size field allows up to 256 MB, and the
/// base64 payload is ~1.33× the picture — so without a cap one crafted (or
/// absurdly ripped) file balloons the host by hundreds of MB.
const MAX_EMBEDDED_COVER_BYTES: usize = 10 * 1024 * 1024;

/// The first embedded picture in a file, preferring the front cover.
///
/// Searches every tag block, not just the primary one: a file can carry both
/// ID3v2 and APE, and the artwork is not always in the tag lofty considers
/// primary.
pub fn read_cover(path: &str) -> Option<CoverArt> {
    let tagged = Probe::open(path).ok()?.read().ok()?;
    let pictures = || tagged.tags().iter().flat_map(|tag| tag.pictures());

    let picture = pictures()
        .find(|p| p.pic_type() == PictureType::CoverFront)
        .or_else(|| pictures().next())?;

    if picture.data().len() > MAX_EMBEDDED_COVER_BYTES {
        log::warn!(
            "embedded cover in {} is {} bytes; skipped",
            path,
            picture.data().len()
        );
        return None;
    }

    Some(CoverArt {
        mime: picture
            .mime_type()
            .map(|m| m.to_string())
            .unwrap_or_else(|| "image/jpeg".to_string()),
        data_base64: base64::engine::general_purpose::STANDARD.encode(picture.data()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_extension_accepts_known_formats_case_insensitively() {
        assert_eq!(
            audio_extension(Path::new("/m/a.FLAC")).as_deref(),
            Some("flac")
        );
        assert_eq!(
            audio_extension(Path::new("/m/b.mp3")).as_deref(),
            Some("mp3")
        );
        assert_eq!(audio_extension(Path::new("/m/c.txt")), None);
        assert_eq!(audio_extension(Path::new("/m/noext")), None);
    }

    #[test]
    fn lossless_extensions_are_a_subset_of_audio_extensions() {
        for ext in LOSSLESS_EXTENSIONS {
            assert!(
                AUDIO_EXTENSIONS.contains(ext),
                "{ext} missing from AUDIO_EXTENSIONS"
            );
        }
        // m4a stays out deliberately: AAC-or-ALAC, reported lossy.
        assert!(!LOSSLESS_EXTENSIONS.contains(&"m4a"));
    }

    fn position(name: &str) -> FilenamePosition {
        track_number_from_filename(Path::new(name))
    }

    #[test]
    fn reads_a_leading_track_number() {
        for name in [
            "07 - Alive.flac",
            "07 Alive.flac",
            "07. Alive.flac",
            "07.Alive.flac",
            "07_Alive.flac",
        ] {
            assert_eq!(position(name).track, Some(7), "{name} should give track 7");
        }
    }

    #[test]
    fn reads_a_disc_and_track_pair() {
        assert_eq!(
            position("2-03 Reprise.mp3"),
            FilenamePosition {
                disc: Some(2),
                track: Some(3)
            }
        );
        assert_eq!(
            position("1_11 Closing Time.mp3"),
            FilenamePosition {
                disc: Some(1),
                track: Some(11)
            }
        );
    }

    #[test]
    fn a_three_digit_number_is_still_a_track() {
        // Some rips number across discs: 104 = disc 1, track 4.
        assert_eq!(position("104 - Title.m4a").track, Some(104));
    }

    #[test]
    fn a_year_is_not_a_track_number() {
        // The guard that matters: four digits are never a position, so this
        // must not read as track 198.
        assert_eq!(position("1984 - Track.mp3"), FilenamePosition::default());
        assert_eq!(
            position("2001 A Space Odyssey.mp3"),
            FilenamePosition::default()
        );
    }

    #[test]
    fn digits_running_into_the_title_are_not_a_track_number() {
        assert_eq!(position("07Alive.flac"), FilenamePosition::default());
        assert_eq!(position("99Luftballons.mp3"), FilenamePosition::default());
    }

    #[test]
    fn a_filename_with_no_leading_digits_yields_nothing() {
        assert_eq!(position("Alive.flac"), FilenamePosition::default());
        assert_eq!(
            position("Pearl Jam - Alive.flac"),
            FilenamePosition::default()
        );
    }

    #[test]
    fn track_zero_is_treated_as_absent() {
        // "00 - Intro" is a hidden-track convention, not position zero.
        assert_eq!(position("00 - Intro.mp3").track, None);
    }

    #[test]
    fn a_bare_number_is_the_whole_name() {
        assert_eq!(position("03.mp3").track, Some(3));
    }

    /// A minimal untagged WAV, so parsing can be exercised without shipping a
    /// binary fixture.
    fn write_wav(path: &Path) {
        use std::io::Write;
        let (rate, channels, frames) = (44_100u32, 1u16, 100usize);
        let block_align = channels * 2;
        let data_len = frames as u32 * block_align as u32;
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&channels.to_le_bytes());
        wav.extend_from_slice(&rate.to_le_bytes());
        wav.extend_from_slice(&(rate * block_align as u32).to_le_bytes());
        wav.extend_from_slice(&block_align.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        wav.extend_from_slice(&vec![0u8; data_len as usize]);
        std::fs::create_dir_all(path.parent().expect("has a parent")).expect("temp dir");
        std::fs::File::create(path)
            .and_then(|mut f| f.write_all(&wav))
            .expect("write test wav");
    }

    #[test]
    fn an_untagged_file_still_gets_its_position_from_its_name() {
        // The end-to-end path: no tags at all, so everything shown about this
        // track has to come from the filename and the decoder.
        let dir = std::env::temp_dir().join("audio-stack-scan-test");
        let path = dir.join("04 - Blue in Green.wav");
        write_wav(&path);

        let meta = read_metadata(&path).expect("an untagged wav should still scan");

        assert_eq!(meta.track_number, Some(4), "recovered from the filename");
        assert_eq!(meta.title, "04 - Blue in Green", "the stem is the title");
        assert_eq!(meta.disc_number, None);
        assert_eq!(meta.year, None);
        assert_eq!(meta.format, "WAV");
        assert!(meta.lossless);

        let _ = std::fs::remove_file(&path);
    }

    /// Writes a cover picture into a file's ID3v2 tag.
    fn tag_with_cover(path: &Path, pic_type: PictureType) {
        use lofty::config::WriteOptions;
        use lofty::picture::{MimeType, Picture};
        use lofty::tag::{Tag, TagType};

        let mut tag = Tag::new(TagType::Id3v2);
        tag.push_picture(Picture::new_unchecked(
            pic_type,
            Some(MimeType::Jpeg),
            None,
            // Not a real JPEG; `read_cover` never decodes it, it only
            // re-encodes the bytes.
            vec![0xFF, 0xD8, 0xFF, 0xE0, 1, 2, 3, 4],
        ));
        tag.save_to_path(path, WriteOptions::default())
            .expect("write cover tag");
    }

    #[test]
    fn cover_art_is_read_back_from_the_file() {
        let dir = std::env::temp_dir().join("audio-stack-cover-test");
        let path = dir.join("with-art.wav");
        write_wav(&path);
        tag_with_cover(&path, PictureType::CoverFront);

        let cover = read_cover(path.to_str().expect("utf-8 path"))
            .expect("the picture just written should come back");
        assert_eq!(cover.mime, "image/jpeg");
        assert!(!cover.data_base64.is_empty());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_file_with_no_picture_yields_none() {
        let dir = std::env::temp_dir().join("audio-stack-cover-test");
        let path = dir.join("no-art.wav");
        write_wav(&path);

        assert!(read_cover(path.to_str().expect("utf-8 path")).is_none());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_back_cover_is_used_when_there_is_no_front() {
        // Better the back of the sleeve than the generated gradient.
        let dir = std::env::temp_dir().join("audio-stack-cover-test");
        let path = dir.join("back-art.wav");
        write_wav(&path);
        tag_with_cover(&path, PictureType::CoverBack);

        assert!(read_cover(path.to_str().expect("utf-8 path")).is_some());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_absurdly_large_embedded_cover_is_skipped() {
        use lofty::config::WriteOptions;
        use lofty::picture::{MimeType, Picture};
        use lofty::tag::{Tag, TagType};

        let dir = std::env::temp_dir().join("audio-stack-cover-test");
        let path = dir.join("huge-art.wav");
        write_wav(&path);

        let mut tag = Tag::new(TagType::Id3v2);
        tag.push_picture(Picture::new_unchecked(
            PictureType::CoverFront,
            Some(MimeType::Jpeg),
            None,
            vec![0u8; MAX_EMBEDDED_COVER_BYTES + 1],
        ));
        tag.save_to_path(&path, WriteOptions::default())
            .expect("write oversized cover");

        assert!(
            read_cover(path.to_str().expect("utf-8 path")).is_none(),
            "a picture past the cap must not be ballooned into a payload"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_disc_prefixed_name_survives_parsing() {
        let dir = std::env::temp_dir().join("audio-stack-scan-test");
        let path = dir.join("2-11 Reprise.wav");
        write_wav(&path);

        let meta = read_metadata(&path).expect("should scan");
        assert_eq!(meta.disc_number, Some(2));
        assert_eq!(meta.track_number, Some(11));

        let _ = std::fs::remove_file(&path);
    }
}
