//! OpenRouter's image API, as JSON.
//!
//! `POST {base}/api/v1/images` with a model, a prompt and up to 14 reference images; the reply
//! carries base64 image data and what the call cost. See
//! <https://openrouter.ai/docs/guides/overview/multimodal/image-generation>.

use serde::{Deserialize, Serialize};

use super::{GenError, ImageRequest, ReferenceImage};

/// How many characters of an error body to keep. Enough for a useful message, small enough
/// that a huge upstream failure cannot flood a log or a dialog.
const ERROR_BODY_LIMIT: usize = 600;

#[derive(Serialize)]
struct ImagesRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    n: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolution: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    input_references: Vec<InputReference>,
}

#[derive(Serialize)]
struct InputReference {
    #[serde(rename = "type")]
    kind: &'static str,
    image_url: ImageUrl,
}

#[derive(Serialize)]
struct ImageUrl {
    url: String,
}

impl InputReference {
    fn from_image(image: &ReferenceImage) -> Self {
        Self { kind: "image_url", image_url: ImageUrl { url: image.data_url() } }
    }
}

/// The request body for one generation.
pub(crate) fn images_request(model: &str, req: &ImageRequest) -> Vec<u8> {
    let body = ImagesRequest {
        model,
        prompt: &req.prompt,
        n: 1,
        resolution: req.resolution,
        aspect_ratio: req.aspect_ratio,
        seed: req.seed,
        input_references: req.references.iter().map(InputReference::from_image).collect(),
    };
    serde_json::to_vec(&body).unwrap_or_default()
}

/// A `POST /api/v1/images` reply.
#[derive(Debug, Default, Deserialize)]
pub struct ImagesResponse {
    /// The generated images; the first one is used.
    #[serde(default)]
    pub data: Vec<ImageDatum>,
    /// Token counts and cost, when the provider reported them.
    #[serde(default)]
    pub usage: Option<Usage>,
}

impl ImagesResponse {
    /// The first image, if any.
    pub fn first(&self) -> Option<&ImageDatum> {
        self.data.first()
    }

    /// What the call cost in USD, when the provider reported it.
    pub fn cost_usd(&self) -> Option<f64> {
        self.usage.as_ref().and_then(|u| u.cost).filter(|c| c.is_finite())
    }
}

/// One image in the reply: inline base64 bytes, or a URL to fetch.
#[derive(Debug, Default, Deserialize)]
pub struct ImageDatum {
    #[serde(default)]
    pub b64_json: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub media_type: Option<String>,
}

/// Token counts and cost. Only `cost` matters here.
#[derive(Debug, Default, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub cost: Option<f64>,
}

/// OpenRouter's error envelope.
#[derive(Debug, Deserialize)]
pub struct ApiError {
    /// Nested `{ "error": { "code": .., "message": .. } }`.
    pub error: ApiErrorBody,
}

/// The body of an [`ApiError`].
#[derive(Debug, Default, Deserialize)]
pub struct ApiErrorBody {
    #[serde(default)]
    pub code: Option<i64>,
    #[serde(default)]
    pub message: Option<String>,
}

/// Turn a non-2xx reply into a [`GenError::Api`], keeping only safe text.
pub(crate) fn api_error(status: u16, raw: &[u8]) -> GenError {
    let message = match serde_json::from_slice::<ApiError>(raw) {
        Ok(e) => {
            let mut text = e.error.message.unwrap_or_default();
            if text.trim().is_empty() {
                text = match e.error.code {
                    Some(code) => format!("code {code}"),
                    None => "no message".to_string(),
                };
            }
            text
        }
        Err(_) => "no detail".to_string(),
    };
    GenError::Api { status, message: truncate(message.trim(), ERROR_BODY_LIMIT) }
}

/// Clip a message at a char boundary and mark what was dropped.
fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let clipped: String = text.chars().take(limit).collect();
    format!("{clipped}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_carries_the_model_prompt_and_reference() {
        let req = ImageRequest {
            prompt: "a koi pond".into(),
            references: vec![ReferenceImage::png(vec![1u8, 2, 3])],
            resolution: Some("2K"),
            aspect_ratio: Some("4:3"),
            seed: Some(7),
        };
        let body: serde_json::Value = serde_json::from_slice(&images_request("m", &req)).unwrap();
        assert_eq!(body["model"], "m");
        assert_eq!(body["prompt"], "a koi pond");
        assert_eq!(body["n"], 1);
        assert_eq!(body["resolution"], "2K");
        assert_eq!(body["aspect_ratio"], "4:3");
        assert_eq!(body["seed"], 7);
        let url = body["input_references"][0]["image_url"]["url"].as_str().unwrap();
        assert!(url.starts_with("data:image/png;base64,"), "{url}");
    }

    #[test]
    fn text_to_image_request_omits_references_and_seed() {
        let body: serde_json::Value = serde_json::from_slice(&images_request("m", &ImageRequest::text("hi"))).unwrap();
        assert!(body.get("input_references").is_none());
        assert!(body.get("seed").is_none());
    }

    #[test]
    fn reply_yields_image_and_cost() {
        let reply: ImagesResponse = serde_json::from_str(r#"{"data":[{"b64_json":"AAAA","media_type":"image/png"}],"usage":{"cost":0.04}}"#).unwrap();
        let image = reply.first().unwrap();
        assert_eq!(image.b64_json.as_deref(), Some("AAAA"));
        assert_eq!(image.media_type.as_deref(), Some("image/png"));
        assert_eq!(reply.cost_usd(), Some(0.04));
    }

    #[test]
    fn reply_without_a_cost_is_not_an_error() {
        let reply: ImagesResponse = serde_json::from_str(r#"{"data":[{"url":"https://x/y.png"}]}"#).unwrap();
        assert_eq!(reply.cost_usd(), None);
        assert_eq!(reply.first().and_then(|d| d.url.as_deref()), Some("https://x/y.png"));
    }

    #[test]
    fn empty_reply_is_detectable() {
        let reply: ImagesResponse = serde_json::from_str("{}").unwrap();
        assert!(reply.first().is_none());
    }

    #[test]
    fn error_body_becomes_a_short_message() {
        let e = api_error(402, br#"{"error":{"code":402,"message":"insufficient credits"}}"#);
        match e {
            GenError::Api { status, message } => {
                assert_eq!(status, 402);
                assert!(message.contains("insufficient credits"), "{message}");
            }
            other => panic!("wrong error: {other}"),
        }
    }

    #[test]
    fn unparsable_error_body_still_reports_the_status() {
        let e = api_error(502, b"<html>upstream</html>");
        assert!(matches!(e, GenError::Api { status: 502, .. }));
    }

    #[test]
    fn long_error_bodies_are_clipped() {
        let long = format!(r#"{{"error":{{"message":"{}"}}}}"#, "x".repeat(5000));
        let e = api_error(400, long.as_bytes());
        match e {
            GenError::Api { message, .. } => assert!(message.chars().count() <= 601, "{}", message.chars().count()),
            other => panic!("wrong error: {other}"),
        }
    }
}
