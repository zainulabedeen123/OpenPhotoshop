//! # photocraft-gen
//!
//! Generative AI backends. The first one is OpenRouter's image API, which serves
//! `bytedance-seed/seedream-5-0-flash` (ByteDance Seedream 5.0 Flash) for the engine's
//! `edit.generativeFill` command.
//!
//! The crate is deliberately free of PhotoCraft dependencies and of any image type: it moves
//! encoded bytes in, encoded bytes out, so it compiles for every target including
//! `wasm32-unknown-unknown`. Network access only exists on native targets; on wasm the client
//! returns [`GenError::Unsupported`].
//!
//! The API key is never written to disk, never logged and never part of a document. It is read
//! from the environment (`PHOTOCRAFT_OPENROUTER_API_KEY`, falling back to `OPENROUTER_API_KEY`).
//!
//! ```
//! use photocraft_gen::{Client, GenError, ImageRequest, ReferenceImage, Transport};
//! use std::time::Duration;
//!
//! /// A stand-in for the real endpoint: one canned reply, no network.
//! struct Stub;
//! impl Transport for Stub {
//!     fn post(&self, _: &str, _: &[(&str, &str)], _: &[u8], _: Duration) -> Result<(u16, Vec<u8>), GenError> {
//!         Ok((200, br#"{"data":[{"b64_json":"","media_type":"image/png"}]}"#.to_vec()))
//!     }
//!     fn get(&self, _: &str, _: &[(&str, &str)], _: Duration) -> Result<(u16, Vec<u8>), GenError> {
//!         Ok((200, Vec::new()))
//!     }
//! }
//!
//! // The real client reads its key from the environment: Client::from_env()
//! let client = Client::with_transport(std::sync::Arc::new(Stub)).with_base_url("http://localhost");
//! let reply = client.generate(&ImageRequest {
//!     prompt: photocraft_gen::fill_prompt("a red sailboat"),
//!     references: vec![ReferenceImage::png(Vec::new())],
//!     resolution: Some("1K"),
//!     aspect_ratio: Some("4:3"),
//!     seed: None,
//! })?;
//! assert_eq!(reply.media_type, "image/png");
//! # Ok::<(), photocraft_gen::GenError>(())
//! ```
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

mod openrouter;
mod transport;

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

pub use openrouter::{ApiError, ImagesResponse};
#[cfg(not(target_arch = "wasm32"))]
pub use transport::UreqTransport;
pub use transport::{Disabled, Transport};

/// The model the generative-fill backend uses by default.
pub const DEFAULT_MODEL: &str = "bytedance-seed/seedream-5-0-flash";
/// OpenRouter's API root. Override with `PHOTOCRAFT_OPENROUTER_BASE_URL` (tests, proxies).
pub const DEFAULT_BASE_URL: &str = "https://openrouter.ai";
/// The OpenRouter images endpoint.
pub const IMAGES_PATH: &str = "/api/v1/images";
/// Environment variable the desktop app and CLI read the OpenRouter key from.
pub const API_KEY_ENV: &str = "PHOTOCRAFT_OPENROUTER_API_KEY";
/// Generic fallback, the name OpenRouter's own documentation uses.
pub const API_KEY_ENV_FALLBACK: &str = "OPENROUTER_API_KEY";
/// Environment variable that overrides the API root.
pub const BASE_URL_ENV: &str = "PHOTOCRAFT_OPENROUTER_BASE_URL";
/// Environment variable that overrides the model id.
pub const MODEL_ENV: &str = "PHOTOCRAFT_OPENROUTER_MODEL";

/// One image sent to the model as a reference (an https URL or an inline data URL).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceImage {
    /// Encoded bytes (PNG or JPEG).
    pub bytes: Vec<u8>,
    /// Media type matching `bytes`, such as `image/png`.
    pub media_type: String,
}

impl ReferenceImage {
    /// A PNG reference.
    pub fn png(bytes: impl Into<Vec<u8>>) -> Self {
        Self { bytes: bytes.into(), media_type: "image/png".to_string() }
    }

    /// A JPEG reference.
    pub fn jpeg(bytes: impl Into<Vec<u8>>) -> Self {
        Self { bytes: bytes.into(), media_type: "image/jpeg".to_string() }
    }

    fn data_url(&self) -> String {
        format!("data:{};base64,{}", self.media_type, base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &self.bytes))
    }
}

/// A single image-generation request. Only `prompt` is required; everything else has a
/// provider-independent default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageRequest {
    /// The text describing what to draw.
    pub prompt: String,
    /// Reference images for image-to-image generation (the model accepts up to 14).
    pub references: Vec<ReferenceImage>,
    /// Resolution tier: `1K` or `2K`. Concrete pixel sizes are chosen by the provider.
    pub resolution: Option<&'static str>,
    /// Aspect ratio of the generated frame, e.g. `4:3`.
    pub aspect_ratio: Option<&'static str>,
    /// Sample the same image again; determinism is not guaranteed by every provider.
    pub seed: Option<u64>,
}

impl ImageRequest {
    /// A text-to-image request with no references.
    pub fn text(prompt: impl Into<String>) -> Self {
        Self { prompt: prompt.into(), references: Vec::new(), resolution: Some("1K"), aspect_ratio: Some("1:1"), seed: None }
    }
}

/// One generated image plus what the call cost.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedImage {
    /// Encoded image bytes.
    pub bytes: Vec<u8>,
    /// Media type of `bytes`, such as `image/png`.
    pub media_type: String,
    /// USD charged for the call, when the provider reported it.
    pub cost_usd: Option<f64>,
}

/// Everything that can go wrong talking to a generative backend.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GenError {
    /// No API key in the environment.
    #[error("no API key: set {API_KEY_ENV} or {API_KEY_ENV_FALLBACK}")]
    NoApiKey,
    /// The endpoint answered with a non-2xx status.
    #[error("the image API refused the request ({status}): {message}")]
    Api {
        /// HTTP status.
        status: u16,
        /// Truncated, sanitized body of the error response.
        message: String,
    },
    /// A 2xx answer that is not a usable image (empty payload, unparsable JSON, bad base64).
    #[error("could not use the image API's reply: {0}")]
    BadReply(String),
    /// DNS, TLS, socket or timeout failure. Never contains credentials.
    #[error("could not reach the image API: {0}")]
    Transport(String),
    /// This build has no network stack (the web build).
    #[error("generative fill needs the desktop app or the CLI; this build has no network access")]
    Unsupported,
    /// The reply carried a URL instead of inline bytes and downloading it failed.
    #[error("could not download the generated image: {0}")]
    Download(String),
}

/// Wraps the user's prompt for a generative fill.
///
/// The reference image is the document composite with the selected area blanked to a soft
/// placeholder, so the instruction has to say what to fill and what to leave alone; that is
/// what keeps the generated patch from repainting the whole frame.
pub fn fill_prompt(user_prompt: &str) -> String {
    let ask = user_prompt.trim();
    if ask.is_empty() {
        return "Fill only the soft, uniform placeholder rectangle in the centre of the reference \
                image with new, photorealistic content that continues the surrounding scene, \
                matching its lighting, colour temperature, perspective, scale, texture, grain and \
                depth of field exactly. Leave every other part of the image untouched: no seam, \
                edge, frame, watermark or text at the boundary of the filled area."
            .to_string();
    }
    format!(
        "{ask}. Fill only the soft, uniform placeholder rectangle in the centre of the reference \
        image with this new content, photorealistic and seamless, matching the surrounding \
        lighting, colour temperature, perspective, scale, texture, grain and depth of field \
        exactly. Leave every other part of the image untouched: no seam, edge, frame, watermark \
        or text at the boundary of the filled area."
    )
}

/// The aspect ratios the model accepts. Used to pick the closest one for the selection.
pub const ASPECT_RATIOS: &[(&str, f64)] = &[
    ("1:1", 1.0),
    ("4:5", 0.8),
    ("5:4", 1.25),
    ("3:4", 0.75),
    ("4:3", 4.0 / 3.0),
    ("2:3", 2.0 / 3.0),
    ("3:2", 3.0 / 2.0),
    ("9:16", 9.0 / 16.0),
    ("16:9", 16.0 / 9.0),
];

/// The closest supported aspect ratio for a `w x h` region, in log distance.
///
/// A degenerate region (either side 0 or non-finite) falls back to `1:1`.
pub fn closest_aspect_ratio(w: u32, h: u32) -> &'static str {
    let target = if w == 0 || h == 0 {
        return "1:1";
    } else {
        w as f64 / h as f64
    };
    if !target.is_finite() {
        return "1:1";
    }
    ASPECT_RATIOS
        .iter()
        .map(|(name, r)| ((target / r).ln().abs(), *name))
        .fold(None, |best: Option<(f64, &'static str)>, cand| match best {
            Some((d, _)) if d <= cand.0 => best,
            _ => Some(cand),
        })
        .map(|(_, n)| n)
        .unwrap_or("1:1")
}

/// A client for one image endpoint. `Clone` is cheap; the API key is never printed.
#[derive(Clone)]
pub struct Client {
    transport: Arc<dyn Transport>,
    base_url: String,
    model: String,
    api_key: String,
    timeout: Duration,
}

impl fmt::Debug for Client {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Client")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("api_key", &"<redacted>")
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl Client {
    /// A client for the default endpoint with the key from the environment.
    ///
    /// # Errors
    /// [`GenError::NoApiKey`] when neither `PHOTOCRAFT_OPENROUTER_API_KEY` nor
    /// `OPENROUTER_API_KEY` is set.
    pub fn from_env() -> Result<Self, GenError> {
        Self::from_env_getter(|name| std::env::var(name).ok())
    }

    /// [`Client::from_env`] with an injectable environment lookup (used by tests).
    pub fn from_env_getter(get: impl Fn(&str) -> Option<String>) -> Result<Self, GenError> {
        let key = get(API_KEY_ENV).or_else(|| get(API_KEY_ENV_FALLBACK)).map(|k| k.trim().to_string()).filter(|k| !k.is_empty()).ok_or(GenError::NoApiKey)?;
        let mut client = Self::new(key);
        if let Some(url) = get(BASE_URL_ENV).filter(|u| !u.trim().is_empty()) {
            client = client.with_base_url(url.trim());
        }
        if let Some(model) = get(MODEL_ENV).filter(|m| !m.trim().is_empty()) {
            client = client.with_model(model.trim());
        }
        Ok(client)
    }

    /// A client for the default endpoint (OpenRouter) with an explicit key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            transport: Arc::new(transport::default_transport()),
            base_url: DEFAULT_BASE_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
            api_key: api_key.into().trim().to_string(),
            timeout: Duration::from_secs(180),
        }
    }

    /// Swap the transport (tests point the client at a stub or a local server).
    pub fn with_transport(transport: Arc<dyn Transport>) -> Self {
        Self {
            transport,
            base_url: DEFAULT_BASE_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
            api_key: "test-key".to_string(),
            timeout: Duration::from_secs(30),
        }
    }

    /// Point the client at another API root (a proxy or a test server).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    /// Use another model id.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// How long a single call may take before it is abandoned.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// The model this client asks for.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Run one image-generation request.
    ///
    /// # Errors
    /// [`GenError::Api`] for a rejected request, [`GenError::Transport`] for a connection or
    /// timeout failure, [`GenError::BadReply`] for an unusable answer, [`GenError::Unsupported`]
    /// in builds without a network stack.
    pub fn generate(&self, req: &ImageRequest) -> Result<GeneratedImage, GenError> {
        let body = openrouter::images_request(&self.model, req);
        let url = format!("{}{}", self.base_url, IMAGES_PATH);
        let auth = format!("Bearer {}", self.api_key);
        let headers: Vec<(&str, &str)> = vec![
            ("Content-Type", "application/json"),
            ("Authorization", auth.as_str()),
            ("HTTP-Referer", "https://github.com/storytold/photocraft"),
            ("X-Title", "PhotoCraft"),
        ];
        let (status, raw) = self.transport.post(&url, &headers, &body, self.timeout)?;
        if !(200..300).contains(&status) {
            return Err(openrouter::api_error(status, &raw));
        }
        let reply: ImagesResponse = serde_json::from_slice(&raw).map_err(|e| GenError::BadReply(format!("invalid JSON: {e}")))?;
        let image = reply.first().ok_or_else(|| GenError::BadReply("the reply contained no image".to_string()))?;
        let bytes = match &image.b64_json {
            Some(b64) => base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64)
                .map_err(|e| GenError::BadReply(format!("invalid base64 image data: {e}")))?,
            None => match image.url.as_deref() {
                Some(url) => self.download(url)?,
                None => return Err(GenError::BadReply("the image had neither base64 data nor a url".to_string())),
            },
        };
        let media_type = image.media_type.clone().unwrap_or_else(|| "image/png".to_string());
        Ok(GeneratedImage { bytes, media_type, cost_usd: reply.cost_usd() })
    }

    /// Fetch an image the endpoint pointed at instead of inlining.
    fn download(&self, url: &str) -> Result<Vec<u8>, GenError> {
        let (status, raw) = self.transport.get(url, &[], self.timeout)?;
        if !(200..300).contains(&status) {
            return Err(GenError::Download(format!("HTTP {status}")));
        }
        Ok(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_prompt_names_the_placeholder_and_asks_for_a_seamless_match() {
        let p = fill_prompt("a koi pond");
        assert!(p.starts_with("a koi pond"), "{p}");
        assert!(p.contains("placeholder"));
        assert!(p.contains("seamless"));
    }

    #[test]
    fn fill_prompt_survives_an_empty_ask() {
        assert!(fill_prompt("   ").contains("placeholder"));
    }

    #[test]
    fn closest_aspect_ratio_matches_shape() {
        assert_eq!(closest_aspect_ratio(100, 100), "1:1");
        assert_eq!(closest_aspect_ratio(1600, 900), "16:9");
        assert_eq!(closest_aspect_ratio(300, 300), "1:1");
        assert_eq!(closest_aspect_ratio(400, 300), "4:3");
        assert_eq!(closest_aspect_ratio(0, 50), "1:1");
    }

    #[test]
    fn reference_images_become_data_urls() {
        let url = ReferenceImage::png(vec![1u8, 2, 3]).data_url();
        assert!(url.starts_with("data:image/png;base64,"), "{url}");
        assert_eq!(&url[url.len() - 4..], "AQID");
    }

    #[test]
    fn the_scoped_key_variable_wins_over_the_generic_one() {
        let c = Client::from_env_getter(|n| match n {
            API_KEY_ENV => Some("scoped-secret".into()),
            API_KEY_ENV_FALLBACK => Some("generic-secret".into()),
            _ => None,
        })
        .unwrap();
        let shown = format!("{c:?}");
        assert!(shown.contains("<redacted>"), "{shown}");
        assert!(!shown.contains("scoped-secret") && !shown.contains("generic-secret"), "{shown}");
    }

    #[test]
    fn missing_key_is_an_error_not_a_panic() {
        let e = Client::from_env_getter(|_| None).unwrap_err();
        assert_eq!(e, GenError::NoApiKey);
    }

    #[test]
    fn blank_environment_entries_are_ignored() {
        let e = Client::from_env_getter(|_| Some("   ".into())).unwrap_err();
        assert_eq!(e, GenError::NoApiKey);
    }
}
