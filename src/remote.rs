//! A whole audio file on a server, as a seekable media source.
//!
//! Unlike a station, a media server's file has an end and a length. It is
//! downloaded into a temporary file as it plays, and a seek past what has
//! arrived becomes an HTTP range request, so the decoder sees something that
//! behaves like a file on disk: the track keeps its duration, its seek bar,
//! its gapless join and its loudness measurement.
//!
//! Connecting is async because the HTTP fetch is; the engine thread is not.
//! [`open_blocking`] drives the connect on the engine's own runtime, for the
//! places that need a decoder now (the listener pressed play), and
//! [`open_in_background`] does the same on a thread of its own, so preloading
//! the next track never stalls the audio already playing.

use std::io::{Read, Seek, SeekFrom};
use std::sync::mpsc;
use std::time::Duration;

use stream_download::http::reqwest::{self, Client};
use stream_download::http::{HttpStream, HttpStreamError};
use stream_download::storage::temp::TempStorageProvider;
use stream_download::{Settings, StreamDownload};
use symphonia::core::formats::probe::Hint;
use symphonia::core::io::MediaSource;

use super::decode::Decoder;
use super::queue::RemoteFile;

/// Bytes downloaded before decoding starts: enough for any container's
/// header and the first seconds of audio, small enough to start at once.
const PREFETCH_BYTES: u64 = 256 * 1024;
/// How long to wait for the server to answer at all.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// How long a read may stall before it counts as a lost connection, which
/// ends the track with an error instead of parking the engine thread.
const READ_TIMEOUT: Duration = Duration::from_secs(20);

/// A downloaded-as-it-plays file. Seekable when the server said how long it
/// is, which is what lets Symphonia jump to a timestamp.
struct RemoteSource<R> {
    reader: R,
    length: Option<u64>,
}

impl<R: Read> Read for RemoteSource<R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.reader.read(buffer)
    }
}

impl<R: Seek> Seek for RemoteSource<R> {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.reader.seek(position)
    }
}

impl<R: Read + Seek + Send + Sync> MediaSource for RemoteSource<R> {
    fn is_seekable(&self) -> bool {
        self.length.is_some()
    }

    fn byte_len(&self) -> Option<u64> {
        self.length
    }
}

/// Connects to the file and buffers its start.
async fn connect(remote: &RemoteFile) -> Result<(Box<dyn MediaSource>, Hint), String> {
    let shown = super::queue::Location::Remote(remote.clone()).to_string();
    let parsed: reqwest::Url = remote
        .url
        .parse()
        .map_err(|error| format!("bad remote address {shown}: {error}"))?;
    // The HTTP client names the full address in its errors, query included,
    // and the query may hold the server's credentials.
    let full_addresses = [remote.url.clone(), parsed.to_string()];
    let scrub = |message: String| {
        full_addresses.iter().fold(message, |message, address| {
            message.replace(address.as_str(), &shown)
        })
    };

    let client = Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(READ_TIMEOUT)
        .build()
        .map_err(|error| format!("build http client: {error}"))?;

    // An HTTP error status (a missing item, a revoked token) fails here, so it
    // reaches the listener as an unplayable track rather than as silence.
    let stream = HttpStream::new(client, parsed).await.map_err(|error| {
        let reason = match error {
            HttpStreamError::FetchFailure(error) => reason(&error),
            HttpStreamError::ResponseFailure(error) => reason(error.source()),
        };
        scrub(format!("{shown}: {reason}"))
    })?;
    let content_type = stream
        .content_type()
        .as_ref()
        .map(|content_type| format!("{}/{}", content_type.r#type, content_type.subtype));

    let reader = StreamDownload::from_stream(
        stream,
        TempStorageProvider::new(),
        Settings::default()
            .prefetch_bytes(PREFETCH_BYTES)
            // Skipping a track stops its download instead of leaving it to
            // fill a temporary file nobody will read.
            .cancel_on_drop(true),
    )
    .await
    .map_err(|error| scrub(format!("buffer {shown}: {error:?}")))?;
    let length = reader.content_length();

    let mut hint = Hint::new();
    if let Some(extension) = remote.extension.as_deref() {
        hint.with_extension(extension);
    }
    if let Some(mime) = content_type {
        hint.mime_type(&mime);
    }

    Ok((Box::new(RemoteSource { reader, length }), hint))
}

/// Why a request failed, in a few words a listener can read: the transport
/// error's own text names the address again, query included.
fn reason(error: &reqwest::Error) -> String {
    if let Some(status) = error.status() {
        format!("the server answered {status}")
    } else if error.is_connect() {
        "the server does not answer".to_owned()
    } else if error.is_timeout() {
        "the server took too long to answer".to_owned()
    } else {
        error.to_string()
    }
}

/// Opens a decoder over a remote file, waiting for the connect. Called from
/// the engine thread, which is not a runtime thread, so blocking on the
/// runtime is allowed; the download itself keeps running on the runtime.
pub fn open_blocking(
    remote: &RemoteFile,
    runtime: &tokio::runtime::Handle,
) -> Result<Decoder, String> {
    let (source, hint) = runtime.block_on(connect(remote))?;
    Decoder::open(source, hint)
}

/// Opens a decoder over a remote file on a thread of its own. The answer
/// arrives on the returned channel; dropping the receiver abandons it.
pub fn open_in_background(
    remote: RemoteFile,
    runtime: tokio::runtime::Handle,
) -> mpsc::Receiver<Result<Decoder, String>> {
    let (sender, receiver) = mpsc::channel();
    let spawned = std::thread::Builder::new()
        .name("remote track opener".to_owned())
        .spawn(move || {
            let _ = sender.send(open_blocking(&remote, &runtime));
        });
    if let Err(error) = spawned {
        log::warn!("start the remote track opener: {error}");
    }
    receiver
}

#[cfg(test)]
pub mod test_server {
    //! A tiny HTTP server for the tests: serves one body, honours single
    //! `Range: bytes=start-` requests, and answers 404 on any other path.

    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::sync::Arc;

    /// Serves `body` at `/track.wav` until the process ends. Returns the
    /// server's base address.
    pub fn serve(body: Vec<u8>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind a local port");
        let address = format!("http://{}", listener.local_addr().expect("local address"));
        let body = Arc::new(body);
        std::thread::spawn(move || {
            for connection in listener.incoming() {
                let Ok(mut connection) = connection else {
                    continue;
                };
                let body = Arc::clone(&body);
                std::thread::spawn(move || {
                    let mut reader = BufReader::new(connection.try_clone().expect("clone"));
                    let mut request_line = String::new();
                    if reader.read_line(&mut request_line).is_err() {
                        return;
                    }
                    let mut range_start: Option<u64> = None;
                    loop {
                        let mut line = String::new();
                        if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                            break;
                        }
                        let lower = line.to_ascii_lowercase();
                        if let Some(value) = lower.strip_prefix("range: bytes=") {
                            range_start = value
                                .trim()
                                .split('-')
                                .next()
                                .and_then(|start| start.parse().ok());
                        }
                    }
                    let path = request_line.split_whitespace().nth(1).unwrap_or_default();
                    if !path.starts_with("/track.wav") {
                        let _ = connection.write_all(
                            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                        );
                        return;
                    }
                    let total = body.len() as u64;
                    let start = range_start.unwrap_or(0).min(total);
                    let slice = &body[start as usize..];
                    let head = match range_start {
                        Some(_) => format!(
                            "HTTP/1.1 206 Partial Content\r\nContent-Type: audio/wav\r\nAccept-Ranges: bytes\r\nContent-Range: bytes {}-{}/{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            start,
                            total.saturating_sub(1),
                            total,
                            slice.len()
                        ),
                        None => format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nAccept-Ranges: bytes\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            slice.len()
                        ),
                    };
                    let _ = connection.write_all(head.as_bytes());
                    let _ = connection.write_all(slice);
                });
            }
        });
        address
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{ramp, wav_bytes};

    fn runtime() -> &'static tokio::runtime::Runtime {
        use std::sync::OnceLock;
        static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
        RUNTIME.get_or_init(|| tokio::runtime::Runtime::new().expect("test runtime"))
    }

    fn remote(address: &str, path: &str) -> RemoteFile {
        RemoteFile {
            url: format!("{address}{path}?api_key=secret"),
            extension: Some("wav".to_owned()),
        }
    }

    #[test]
    fn a_remote_file_decodes_with_its_length_and_seeks() {
        let sample_rate = 8_000;
        let frames = sample_rate as usize * 4;
        let address = test_server::serve(wav_bytes(sample_rate, 1, &ramp(1, frames)));

        let mut decoder = open_blocking(&remote(&address, "/track.wav"), runtime().handle())
            .expect("the served file opens");
        let duration = decoder
            .format()
            .duration_secs
            .expect("a finite file has a duration");
        assert!((duration - 4.0).abs() < 0.01, "duration was {duration}");

        decoder
            .seek(3.0)
            .expect("a seek past the prefetch is served");
        let mut buffer = vec![0.0f32; 256];
        let read = decoder.read(&mut buffer).expect("audio follows the seek");
        assert!(read > 0, "audio follows the seek");
        // The ramp rises with time: three quarters in, the sample is high.
        assert!(
            buffer[0] > 0.6,
            "the seek landed late in the file: {}",
            buffer[0]
        );
    }

    #[test]
    fn a_missing_remote_file_is_an_error_that_hides_the_credentials() {
        let address = test_server::serve(Vec::new());
        let error = open_blocking(&remote(&address, "/missing.flac"), runtime().handle())
            .err()
            .expect("a 404 does not open");
        assert_eq!(
            error,
            format!("{address}/missing.flac: the server answered 404 Not Found")
        );
    }

    #[test]
    fn a_server_that_is_not_there_says_so_briefly() {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .and_then(|listener| listener.local_addr())
            .expect("a free port")
            .port();
        let address = format!("http://127.0.0.1:{port}");
        let error = open_blocking(&remote(&address, "/track.wav"), runtime().handle())
            .err()
            .expect("nothing listens there");
        assert_eq!(
            error,
            format!("{address}/track.wav: the server does not answer")
        );
    }

    #[test]
    fn a_background_open_delivers_its_decoder() {
        let address = test_server::serve(wav_bytes(8_000, 1, &ramp(1, 8_000)));
        let receiver =
            open_in_background(remote(&address, "/track.wav"), runtime().handle().clone());
        let decoder = receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("the opener answers");
        assert!(decoder.is_ok());
    }
}
