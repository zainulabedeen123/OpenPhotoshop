//! The single HTTP call the generative backends need.
//!
//! The trait keeps the client testable without a network: the engine's tests point it at a
//! stub, and an integration test points it at a loopback server. Only native targets get a
//! real implementation; wasm keeps [`Disabled`], so the crate compiles everywhere.

use std::time::Duration;

use super::GenError;

/// One HTTP round trip: method, url, headers, body, timeout in; status and body bytes out.
///
/// Implementations must never include credentials in an error or a log.
pub trait Transport: Send + Sync + 'static {
    /// Send the request and return the status and the raw body.
    ///
    /// # Errors
    /// [`GenError::Transport`] for connection, TLS and timeout failures; the HTTP status is
    /// returned as data instead, so callers can read error bodies.
    fn post(&self, url: &str, headers: &[(&str, &str)], body: &[u8], timeout: Duration) -> Result<(u16, Vec<u8>), GenError>;

    /// Fetch a URL (used when the endpoint answers with a link instead of inline bytes).
    ///
    /// # Errors
    /// [`GenError::Transport`] for connection, TLS and timeout failures.
    fn get(&self, url: &str, headers: &[(&str, &str)], timeout: Duration) -> Result<(u16, Vec<u8>), GenError>;
}

/// The transport every non-native build uses: it answers honestly that there is no network.
#[derive(Debug, Default, Clone, Copy)]
pub struct Disabled;

impl Transport for Disabled {
    fn post(&self, _: &str, _: &[(&str, &str)], _: &[u8], _: Duration) -> Result<(u16, Vec<u8>), GenError> {
        Err(GenError::Unsupported)
    }

    fn get(&self, _: &str, _: &[(&str, &str)], _: Duration) -> Result<(u16, Vec<u8>), GenError> {
        Err(GenError::Unsupported)
    }
}

/// The transport the desktop app and CLI use: [`ureq`], pure Rust, TLS through rustls.
/// Wasm keeps [`Disabled`], so this exists on native targets only.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Default, Clone, Copy)]
pub struct UreqTransport;

#[cfg(not(target_arch = "wasm32"))]
impl Transport for UreqTransport {
    fn post(&self, url: &str, headers: &[(&str, &str)], body: &[u8], timeout: Duration) -> Result<(u16, Vec<u8>), GenError> {
        let agent = agent(timeout);
        let mut request = agent.post(url);
        for (key, value) in headers {
            request = request.header(*key, *value);
        }
        let mut response = request.send(body).map_err(transport_error)?;
        let status = response.status().as_u16();
        let raw = response.body_mut().read_to_vec().map_err(transport_error)?;
        Ok((status, raw))
    }

    fn get(&self, url: &str, headers: &[(&str, &str)], timeout: Duration) -> Result<(u16, Vec<u8>), GenError> {
        let agent = agent(timeout);
        let mut request = agent.get(url);
        for (key, value) in headers {
            request = request.header(*key, *value);
        }
        let mut response = request.call().map_err(transport_error)?;
        let status = response.status().as_u16();
        let raw = response.body_mut().read_to_vec().map_err(transport_error)?;
        Ok((status, raw))
    }
}

/// A short-lived agent. Generative calls are one-shot, so connection pooling buys nothing and
/// a fresh agent per call cannot leak state between commands.
#[cfg(not(target_arch = "wasm32"))]
fn agent(timeout: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        // A non-2xx status is data: the error body explains what the API refused.
        .http_status_as_error(false)
        .build()
        .into()
}

/// The transport a build gets by default.
pub fn default_transport() -> impl Transport {
    #[cfg(not(target_arch = "wasm32"))]
    {
        UreqTransport
    }
    #[cfg(target_arch = "wasm32")]
    {
        Disabled
    }
}

/// Render a transport failure without leaking the request (and so the key).
#[cfg(not(target_arch = "wasm32"))]
fn transport_error(error: ureq::Error) -> GenError {
    let text = error.to_string();
    GenError::Transport(if text.contains("Authorization") { "rejected by the transport".to_string() } else { text })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_disabled_transport_says_so_instead_of_failing_oddly() {
        let e = Disabled.post("https://example.com", &[], b"{}", Duration::from_secs(1)).unwrap_err();
        assert_eq!(e, GenError::Unsupported);
        let e = Disabled.get("https://example.com", &[], Duration::from_secs(1)).unwrap_err();
        assert_eq!(e, GenError::Unsupported);
    }

    #[cfg(not(target_arch = "wasm32"))]
    mod native {
        use super::*;
        use std::io::{Read, Write};
        use std::net::{TcpListener, TcpStream};

        /// A loopback server answering with a JSON body of the caller's choosing.
        fn serve_json(status_line: &str, body: &str) -> String {
            let reply =
                format!("HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            std::thread::spawn(move || {
                if let Ok((mut stream, _)) = listener.accept() {
                    stream.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
                    let request = read_request(&mut stream);
                    let len = header(&request, "content-length");
                    assert!(len > 0, "stub server got no body: {}", String::from_utf8_lossy(&request));
                    let expected = request.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4 + len);
                    assert_eq!(Some(request.len()), expected, "stub server did not read the whole request");
                    stream.write_all(reply.as_bytes()).unwrap();
                    stream.flush().unwrap();
                }
            });
            format!("http://{addr}")
        }

        fn read_request(stream: &mut TcpStream) -> Vec<u8> {
            let mut seen: Vec<u8> = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let want = match seen.windows(4).position(|w| w == b"\r\n\r\n") {
                    Some(end) => end + 4 + header(&seen, "content-length"),
                    None => seen.len() + 1,
                };
                if seen.len() >= want {
                    break;
                }
                match stream.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => seen.extend_from_slice(&buf[..n]),
                }
            }
            seen
        }

        fn header(request: &[u8], name: &str) -> usize {
            let head = String::from_utf8_lossy(request);
            head.lines()
                .filter_map(|l| l.split_once(':'))
                .find(|(k, _)| k.trim().eq_ignore_ascii_case(name))
                .and_then(|(_, v)| v.trim().parse().ok())
                .unwrap_or(0)
        }

        #[test]
        fn posts_json_and_reads_the_reply() {
            let base = serve_json("200 OK", "{}");
            let (status, body) =
                UreqTransport.post(&format!("{base}/api/v1/images"), &[("Content-Type", "application/json")], b"{\"n\":1}", Duration::from_secs(20)).unwrap();
            assert_eq!(status, 200);
            assert_eq!(body, b"{}");
        }

        #[test]
        fn an_error_status_is_data_with_its_body() {
            let base = serve_json("402 Payment Required", r#"{"error":{"code":402,"message":"no credits"}}"#);
            let (status, body) = UreqTransport.post(&format!("{base}/api/v1/images"), &[], b"{}", Duration::from_secs(20)).unwrap();
            assert_eq!(status, 402);
            assert!(String::from_utf8_lossy(&body).contains("no credits"), "{:?}", String::from_utf8_lossy(&body));
        }

        #[test]
        fn a_closed_socket_is_a_transport_error_not_a_panic() {
            let e = UreqTransport.post("http://127.0.0.1:1/api/v1/images", &[], b"{}", Duration::from_secs(5)).unwrap_err();
            assert!(matches!(e, GenError::Transport(_)), "{e:?}");
        }
    }
}
