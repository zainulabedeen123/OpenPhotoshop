//! Generative Fill (#41): send the active selection to an image model, then composite what comes
//! back on a new layer that is masked to the selection.
//!
//! The round trip mirrors Photoshop's Generative Fill: the user selects an area, types a prompt,
//! and the app fills that area with generated content that matches the surrounding image. The
//! result is a new layer above the active one, masked by the (optionally feathered) selection,
//! so nothing else in the document is touched and the step is one undo.
//!
//! The model is `photocraft-gen`'s OpenRouter image API (ByteDance Seedream 5.0 Flash). The
//! reference image it receives is the composite of a window around the selection with the
//! selected area blanked to the mean colour of its surroundings, so the model is asked to fill a
//! hole rather than repaint the frame; only the pixels inside the selection are kept.

use std::sync::Arc;

use photocraft_color::PixelFormat;
use photocraft_doc::{Layer, LayerId, LayerMask};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

/// The command id, used by the UI, the CLI, the control channel and MCP.
pub const FILL: &str = "edit.generativeFill";
/// History and layer label. Photoshop names the result a generative layer.
const LABEL: &str = "Generative Fill";
/// Longest prompt sent (refused before it can cost anything).
const MAX_PROMPT_CHARS: usize = 4_000;
/// Longest side of the image sent to the model, and the bounds a caller may ask for.
const MIN_REFERENCE_SIDE: u32 = 256;
const DEFAULT_REFERENCE_SIDE: u32 = 1_536;
const MAX_REFERENCE_SIDE: u32 = 4_096;
/// Largest mask feather accepted, in pixels.
const MAX_FEATHER_PX: f64 = 250.0;
/// Context ring around the selection that the reference image carries.
const CONTEXT_MARGIN: i32 = 96;
/// Largest area rendered for a reference image (32 MP, about 512 MB of RGBA f32).
const MAX_REGION_PIXELS: u64 = 32 << 20;
/// Largest image accepted back, per side and in total (a 1K or 2K reply is far below this).
const MAX_IMAGE_SIDE: u32 = 16_384;

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: FILL.into(), msg: msg.into() }
}

fn gen_error(e: photocraft_gen::GenError) -> EngineError {
    EngineError::Other(e.to_string())
}

/// A block of pixels plus the rectangle it covers, in its own coordinate space.
///
/// The rect travels with the surface because a surface's tile bounds are rounded outwards and its
/// content bounds shrink past transparent edges, so neither describes what a caller wrote.
struct Pixels {
    surface: Surface,
    rect: Rect,
}

impl Pixels {
    fn width(&self) -> u32 {
        self.rect.width()
    }

    fn height(&self) -> u32 {
        self.rect.height()
    }

    fn pixel_count(&self) -> usize {
        self.width() as usize * self.height() as usize
    }

    fn resized(&self, sx: f64, sy: f64) -> Self {
        Self {
            surface: photocraft_algo::resample::resize_surface(&self.surface, sx, sy, photocraft_algo::resample::Resample::Bicubic),
            rect: photocraft_algo::resample::scaled_rect(self.rect, sx, sy),
        }
    }

    /// The same pixels in the document's format, converted when the model's sRGB RGBA does not
    /// match it (a CMYK document takes an RGB image the same way a paste does).
    fn to_target(&self, target: PixelFormat) -> Self {
        let surface = if self.surface.format() == target { self.surface.clone() } else { self.surface.convert(target) };
        Self { surface, rect: self.rect }
    }

    /// Encoded PNG bytes: what the API receives as a reference image.
    fn to_png(&self) -> Result<Vec<u8>> {
        if self.pixel_count() == 0 {
            return Err(bad("the image is empty"));
        }
        let mut rgba = crate::allocation::filled(self.pixel_count(), [0u8; 4], "sending the image to the model")?;
        self.surface.read_rgba8_into(self.rect, &mut rgba);
        let flat: Vec<u8> = rgba.into_iter().flatten().collect();
        let image = photocraft_codecs::Image::from_u8(self.width(), self.height(), photocraft_codecs::ChannelLayout::Rgba, flat)
            .map_err(|e| bad(format!("could not pack the image ({e})")))?;
        photocraft_codecs::encode(&image, photocraft_codecs::Format::Png, &photocraft_codecs::EncodeOptions::default())
            .map_err(|e| bad(format!("could not encode the image ({e})")))
    }
}

/// The prompt, mask feather, reference size, resolution tier and seed.
struct Options {
    prompt: String,
    feather: f32,
    reference_side: u32,
    resolution: &'static str,
    seed: Option<u64>,
}

fn options(p: &Value) -> Result<Options> {
    if !p.is_object() {
        return Err(bad("params must be an object"));
    }
    let prompt = p
        .get("prompt")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| bad("`prompt` is required: describe what should fill the selection"))?;
    if prompt.chars().count() > MAX_PROMPT_CHARS {
        return Err(bad(format!("`prompt` is longer than {MAX_PROMPT_CHARS} characters")));
    }
    let feather = number(p, "feather", MAX_FEATHER_PX)?.unwrap_or(0.0);
    let reference_side = match p.get("maxReferenceSide") {
        None | Some(Value::Null) => DEFAULT_REFERENCE_SIDE,
        Some(v) => v
            .as_f64()
            .filter(|n| n.is_finite())
            .map(|n| (n.round() as u32).clamp(MIN_REFERENCE_SIDE, MAX_REFERENCE_SIDE))
            .ok_or_else(|| bad("`maxReferenceSide` must be a number, clamped to 256..=4096"))?,
    };
    let resolution = match p.get("resolution") {
        None | Some(Value::Null) => "1K",
        Some(Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "1k" | "" => "1K",
            "2k" => "2K",
            other => return Err(bad(format!("`resolution` must be 1K or 2K, got `{other}`"))),
        },
        Some(_) => return Err(bad("`resolution` must be \"1K\" or \"2K\"")),
    };
    let seed = match p.get("seed") {
        None | Some(Value::Null) => None,
        Some(Value::Number(n)) if n.is_u64() => n.as_u64(),
        Some(_) => return Err(bad("`seed` must be a whole number")),
    };
    Ok(Options { prompt: prompt.to_string(), feather, reference_side, resolution, seed })
}

/// A clamped non-negative number: absent, or a finite value within 0..=max.
fn number(p: &Value, name: &str, max: f64) -> Result<Option<f32>> {
    p.get(name)
        .map(|v| {
            v.as_f64()
                .filter(|n| n.is_finite() && *n >= 0.0 && *n <= max)
                .map(|n| n as f32)
                .ok_or_else(|| bad(format!("`{name}` must be a finite number within 0..={max}")))
        })
        .transpose()
}

/// The reference frame: the selection plus a context ring, inside the canvas.
fn reference_frame(sel: &Surface, canvas: Rect) -> Rect {
    sel.content_bounds().intersect(&canvas).inflate(CONTEXT_MARGIN).intersect(&canvas)
}

/// The composite of `region` with the selected area blanked to the mean colour of its
/// surroundings, scaled so its longest side is at most `max_side`.
///
/// Blanking is what makes the request an inpainting one: the model sees a hole and is asked to
/// fill it, instead of being invited to repaint the parts of the frame that would be discarded.
fn reference_image(doc: &photocraft_doc::Document, sel: &Surface, region: Rect, max_side: u32) -> Result<Vec<u8>> {
    let (w, h) = (region.width(), region.height());
    if w == 0 || h == 0 {
        return Err(bad("the selected area is empty"));
    }
    let pixels = u64::from(w) * u64::from(h);
    if pixels > MAX_REGION_PIXELS {
        return Err(bad(format!("the selected area covers {pixels} megapixels, which is too much to send to the image API; fill a smaller area")));
    }
    crate::allocation::checkpoint("sending the image to the model")?;
    let composite = photocraft_compose::render(doc, region);
    let mut px = composite.px;
    if px.len() != usize::try_from(pixels).unwrap_or(usize::MAX) {
        return Err(bad("could not read the document where the selection is"));
    }
    let mask = crate::allocation::read_region(sel, region, "reading the selection")?;
    let mk = sel.format().channels().max(1);
    let coverage = |i: usize| mask.get(i * mk).copied().unwrap_or(0.0);
    // The placeholder colour: what the surroundings are, so the hole reads as "more of the same".
    let (mut sum, mut seen) = ([0.0f64; 3], 0u64);
    for (i, p) in px.iter().enumerate() {
        if coverage(i) < 0.5 && p[3] > 0.0 {
            for c in 0..3 {
                sum[c] += f64::from(p[c]);
            }
            seen += 1;
        }
    }
    let fill = if seen == 0 { [0.5, 0.5, 0.5] } else { [sum[0] / seen as f64, sum[1] / seen as f64, sum[2] / seen as f64] };
    for (i, p) in px.iter_mut().enumerate() {
        let cover = coverage(i).clamp(0.0, 1.0);
        if cover > 0.0 {
            let keep = 1.0 - cover;
            for c in 0..3 {
                p[c] = p[c] * keep + (fill[c] as f32) * cover;
            }
            p[3] = p[3] * keep + cover;
        }
    }
    let mut flat = crate::allocation::filled(px.len() * 4, 0.0, "sending the image to the model")?;
    for (i, p) in px.iter().enumerate() {
        flat.get_mut(i * 4..i * 4 + 4).ok_or_else(|| bad("could not pack the image"))?.copy_from_slice(p);
    }
    let mut surface = Surface::new(PixelFormat::RGBA32F);
    surface.try_write_region(Rect::from_xywh(0, 0, w, h), &flat).map_err(|e| bad(format!("not enough memory for the image ({e})")))?;
    let mut block = Pixels { surface, rect: Rect::from_xywh(0, 0, w, h) };
    let scale = (f64::from(max_side) / f64::from(w.max(h))).min(1.0);
    if scale < 1.0 {
        block = block.resized(scale, scale);
    }
    block.to_png()
}

/// The model's reply, as pixels at the origin.
fn reply_pixels(bytes: &[u8]) -> Result<Pixels> {
    let image = photocraft_codecs::decode(bytes).map_err(|e| bad(format!("the model returned an image PhotoCraft could not decode ({e})")))?;
    let (w, h) = image.dimensions();
    if w == 0 || h == 0 || w > MAX_IMAGE_SIDE || h > MAX_IMAGE_SIDE {
        return Err(bad(format!("the model returned an image of {w}x{h}, which PhotoCraft will not use")));
    }
    if u64::from(w) * u64::from(h) > MAX_REGION_PIXELS {
        return Err(bad(format!("the model returned an image of {w}x{h}, which is larger than this command accepts")));
    }
    let rgba = image.to_rgba_f32();
    let rect = Rect::from_xywh(0, 0, w, h);
    let mut surface = Surface::new(PixelFormat::RGBA32F);
    surface.try_write_region(rect, &rgba).map_err(|e| bad(format!("not enough memory for the model's image ({e})")))?;
    Ok(Pixels { surface, rect })
}

/// Fit the generated frame over the region.
///
/// The model is asked for the closest aspect ratio it supports, so this corrects a small mismatch
/// rather than distorting the patch.
fn fit(block: Pixels, region: Rect) -> Result<Pixels> {
    let (w, h) = (block.width(), block.height());
    if w == 0 || h == 0 {
        return Err(bad("the model returned an empty image"));
    }
    let (rw, rh) = (region.width(), region.height());
    if rw == 0 || rh == 0 {
        return Err(bad("the selected area is empty"));
    }
    let (sx, sy) = (f64::from(rw) / f64::from(w), f64::from(rh) / f64::from(h));
    if !sx.is_finite() || !sy.is_finite() || sx <= 0.0 || sy <= 0.0 {
        return Err(bad("the generated image could not be fitted to the selection"));
    }
    if (sx - 1.0).abs() < 1e-6 && (sy - 1.0).abs() < 1e-6 {
        return Ok(block);
    }
    Ok(block.resized(sx, sy))
}

/// The key a fill runs with: the environment wins (agents, CI, the CLI), then the key the app
/// stored in the session from its Settings dialog. Environment lookups are passed in so tests
/// never depend on the machine's.
fn key_with(get: impl Fn(&str) -> Option<String>, stored: Option<String>) -> Option<String> {
    let usable = |k: Option<String>| k.map(|k| k.trim().to_string()).filter(|k| !k.is_empty());
    usable(get(photocraft_gen::API_KEY_ENV).or_else(|| get(photocraft_gen::API_KEY_ENV_FALLBACK))).or(usable(stored))
}

/// The client Generative Fill runs with: resolution like [`key_with`], with the base-url and
/// model environment overrides applied on top.
fn client_for(s: &Session) -> std::result::Result<photocraft_gen::Client, photocraft_gen::GenError> {
    let key = key_with(|name| std::env::var(name).ok(), s.ai_key().map(str::to_string)).ok_or(photocraft_gen::GenError::NoApiKey)?;
    let client = photocraft_gen::Client::new(key);
    let override_env = |name: &str| std::env::var(name).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
    let client = match override_env(photocraft_gen::BASE_URL_ENV) {
        Some(url) => client.with_base_url(url),
        None => client,
    };
    Ok(match override_env(photocraft_gen::MODEL_ENV) {
        Some(model) => client.with_model(model),
        None => client,
    })
}

/// Generative Fill: document, selection, prompt and client in; one new layer out.
///
/// Everything the call can refuse without a key is refused first, so a bad request never
/// resolves a key at all.
fn generative_fill(s: &mut Session, p: &Value) -> Result<Value> {
    let prep = prepare(s, p)?;
    let client = client_for(s).map_err(gen_error)?;
    run_prepared(s, client, prep)
}

/// [`generative_fill`] with a caller-supplied client, so tests never touch the network: the
/// tests below inject a stub transport, and no other caller constructs a client by hand.
#[cfg(test)]
pub(crate) fn generative_fill_with(s: &mut Session, client: &photocraft_gen::Client, p: &Value) -> Result<Value> {
    let prep = prepare(s, p)?;
    run_prepared(s, client.clone(), prep)
}

/// What a fill needs before any key is resolved: parsed options, the layer above which the
/// result lands, the selection and the reference window.
struct Prepared {
    prompt: String,
    feather: f32,
    reference_side: u32,
    resolution: &'static str,
    seed: Option<u64>,
    id: LayerId,
    doc: Arc<photocraft_doc::Document>,
    sel: Surface,
    region: Rect,
    doc_format: PixelFormat,
}

fn prepare(s: &mut Session, p: &Value) -> Result<Prepared> {
    let Options { prompt, feather, reference_side, resolution, seed } = options(p)?;
    let id = crate::commands::layer_param(s, p)?;
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let doc = Arc::clone(&st.doc);
    let sel = doc.selection.clone().ok_or_else(|| bad("there is no selection: select the area to fill first"))?;
    let canvas = doc.bounds();
    if sel.content_bounds().intersect(&canvas).is_empty() {
        return Err(bad("the selection is outside the canvas"));
    }
    let region = reference_frame(&sel, canvas);
    if region.is_empty() {
        return Err(bad("the selected area is outside the canvas"));
    }
    Ok(Prepared { prompt, feather, reference_side, resolution, seed, id, doc_format: doc.pixel_format(), sel, region, doc })
}

fn run_prepared(s: &mut Session, client: photocraft_gen::Client, prep: Prepared) -> Result<Value> {
    let Prepared { prompt, feather, reference_side, resolution, seed, id, doc, sel, region, doc_format } = prep;
    let work_doc = Arc::clone(&doc);
    crate::jobs::run(
        s,
        LABEL,
        true,
        move |ctx| {
            ctx.progress(0.02, "Reading the selection");
            let reference = reference_image(&work_doc, &sel, region, reference_side)?;
            ctx.check()?;
            ctx.progress(0.15, "Generating");
            let reply = client
                .generate(&photocraft_gen::ImageRequest {
                    prompt: photocraft_gen::fill_prompt(&prompt),
                    references: vec![photocraft_gen::ReferenceImage::png(reference)],
                    resolution: Some(resolution),
                    aspect_ratio: Some(photocraft_gen::closest_aspect_ratio(region.width(), region.height())),
                    seed,
                })
                .map_err(gen_error)?;
            ctx.check()?;
            ctx.progress(0.85, "Compositing the result");
            let block = fit(reply_pixels(&reply.bytes)?, region)?;
            Ok((block, reply.cost_usd))
        },
        move |s, (block, cost)| apply_fill(s, id, doc_format, region, feather, block, cost),
    )
}

/// Write the generated frame onto a new layer masked by the selection, as one undo step.
fn apply_fill(s: &mut Session, above: LayerId, doc_format: PixelFormat, region: Rect, feather: f32, block: Pixels, cost: Option<f64>) -> Result<Value> {
    let before = s.active().map(|st| st.doc.clone());
    let target = PixelFormat::new(doc_format.mode, doc_format.sample, true);
    let (nid, bounds) = s.edit(LABEL, |doc, active| {
        let sel = doc.selection.clone().ok_or_else(|| EngineError::Other("the selection disappeared before the result was applied".into()))?;
        let pixels = block.to_target(target);
        let dst = Rect::from_xywh(region.x0, region.y0, block.width(), block.height());
        let mut layer = Layer::raster(doc.next_layer_name(LABEL), target);
        {
            let surface = crate::pixels_mut(&mut layer)?;
            let data = pixels.surface.read_region(pixels.rect);
            surface
                .try_write_region(dst, &data)
                .map_err(|e| EngineError::Other(format!("not enough memory for the generated image ({e}); the document was not changed")))?;
            surface.prune();
        }
        // The mask keeps the fill inside the selection and blends its edge; the selection itself
        // stays, so the user can keep working with it (Photoshop keeps it too).
        layer.mask = Some(LayerMask { surface: sel, feather, ..LayerMask::reveal_all() });
        let nid = doc.insert_above(Some(above), layer);
        *active = Some(nid);
        Ok((nid, dst))
    })?;
    if let Some(before) = before {
        crate::layer_multi_cmds::note_insert(s, &before);
    }
    Ok(json!({
        "layer": nid.0,
        "bounds": [bounds.x0, bounds.y0, bounds.width(), bounds.height()],
        "costUsd": cost,
    }))
}

/// Photoshop only offers Generative Fill with something selected.
fn enabled(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    if d.doc.selection.is_none() {
        return Err("no selection: select the area to fill".into());
    }
    Ok(())
}

pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: FILL,
        label: "Generative Fill…",
        menu: &["Edit"],
        shortcut: None,
        params: r##"{"prompt":text (required; what should fill the selection),"layer":id? (default the active layer),"feather":0..250? (pixels; the new layer's mask feather),"maxReferenceSide":256..4096=1536 (longest side of the image sent to the model),"resolution":"1K"|"2K"="1K","seed":int?} Ã¢â€ â€™ {layer,bounds,costUsd}; needs a selection and an API key in PHOTOCRAFT_OPENROUTER_API_KEY (or OPENROUTER_API_KEY)"##,
        enabled,
        run: generative_fill,
        journal: true,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_gen::{GenError, Transport};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    /// Stands in for OpenRouter: answers with `png`, checks the request carried a reference image
    /// and the wrapped prompt, and records what it was sent.
    struct Stub {
        png: Vec<u8>,
        sent: Arc<Mutex<Vec<u8>>>,
    }

    impl Transport for Stub {
        fn post(&self, _: &str, _: &[(&str, &str)], body: &[u8], _: Duration) -> std::result::Result<(u16, Vec<u8>), GenError> {
            let text = String::from_utf8_lossy(body).to_string();
            assert!(text.contains("data:image/png;base64,"), "the request should carry the reference image");
            assert!(text.contains("placeholder rectangle"), "the request should wrap the prompt: {text}");
            if let Ok(mut sent) = self.sent.lock() {
                *sent = body.to_vec();
            }
            let reply = json!({"data": [{"b64_json": base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &self.png), "media_type": "image/png"}], "usage": {"cost": 0.04}});
            Ok((200, serde_json::to_vec(&reply).unwrap()))
        }

        fn get(&self, _: &str, _: &[(&str, &str)], _: Duration) -> std::result::Result<(u16, Vec<u8>), GenError> {
            Ok((200, Vec::new()))
        }
    }

    fn client(png: &[u8]) -> (photocraft_gen::Client, Arc<Mutex<Vec<u8>>>) {
        let sent = Arc::new(Mutex::new(Vec::new()));
        let stub = Stub { png: png.to_vec(), sent: Arc::clone(&sent) };
        (photocraft_gen::Client::with_transport(Arc::new(stub) as Arc<dyn Transport>), sent)
    }

    fn solid_png(w: u32, h: u32, rgb: [u8; 3]) -> Vec<u8> {
        let mut data = Vec::with_capacity(w as usize * h as usize * 4);
        for _ in 0..w * h {
            data.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
        }
        let image = photocraft_codecs::Image::from_u8(w, h, photocraft_codecs::ChannelLayout::Rgba, data).unwrap();
        photocraft_codecs::encode(&image, photocraft_codecs::Format::Png, &photocraft_codecs::EncodeOptions::default()).unwrap()
    }

    fn document() -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 48, "height": 36})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s
    }

    /// A 24x18 selection in the middle of a flat blue-grey document.
    fn selected() -> Session {
        let mut s = document();
        s.execute("edit.fill", json!({"contents": "color", "color": "#446699"})).unwrap();
        s.execute("select.rect", json!({"x": 12, "y": 9, "width": 24, "height": 18, "mode": "replace"})).unwrap();
        s
    }

    fn composite(s: &Session) -> Vec<[f32; 4]> {
        let d = s.active().unwrap();
        photocraft_compose::render(&d.doc, d.doc.bounds()).px
    }

    /// The reference image a stub was sent, decoded.
    fn sent_reference(sent: &Mutex<Vec<u8>>) -> (u32, u32) {
        let body = sent.lock().unwrap();
        let text = String::from_utf8_lossy(&body).to_string();
        let at = text.find("data:image/png;base64,").unwrap() + "data:image/png;base64,".len();
        let end = text[at..].find('"').unwrap() + at;
        let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &text[at..end]).unwrap();
        let image = photocraft_codecs::decode(&bytes).unwrap();
        image.dimensions()
    }

    #[test]
    fn a_generation_lands_on_a_new_masked_layer() {
        let mut s = selected();
        let before = composite(&s);
        let (client, sent) = client(&solid_png(24, 18, [220, 40, 40]));
        let r = generative_fill_with(&mut s, &client, &json!({"prompt": "a red sailboat"})).unwrap();
        let layer = r.get("layer").and_then(Value::as_u64).unwrap();
        assert_eq!(r.get("costUsd").and_then(Value::as_f64), Some(0.04));
        // The reference image is the selection plus context, downscaled to the cap.
        assert_eq!(sent_reference(&sent), (48, 36));
        let d = s.active().unwrap();
        assert_eq!(d.doc.layers.len(), 3, "background, the layer under it, and the fill");
        let made = d.doc.layer(LayerId(layer)).unwrap();
        assert!(made.mask.is_some(), "the result layer must be masked by the selection");
        assert_eq!(made.mask.as_ref().map(|m| m.feather), Some(0.0));
        assert_eq!(d.active_layer, Some(LayerId(layer)));
        let after = composite(&s);
        let mut changed = 0;
        for (i, (a, b)) in before.iter().zip(after.iter()).enumerate() {
            let (x, y) = ((i % 48) as i32, (i / 48) as i32);
            let inside = (12..36).contains(&x) && (9..27).contains(&y);
            if a != b {
                changed += 1;
                assert!(inside, "pixel {x},{y} outside the selection changed");
            }
        }
        assert!(changed >= 24 * 18 - 8, "the selection should be filled ({changed} px)");
    }

    #[test]
    fn the_mask_feather_is_recorded_on_the_layer() {
        let mut s = selected();
        let (client, _) = client(&solid_png(24, 18, [10, 200, 10]));
        let r = generative_fill_with(&mut s, &client, &json!({"prompt": "grass", "feather": 3})).unwrap();
        let d = s.active().unwrap();
        let made = d.doc.layer(LayerId(r.get("layer").and_then(Value::as_u64).unwrap())).unwrap();
        assert_eq!(made.mask.as_ref().map(|m| m.feather), Some(3.0));
    }

    #[test]
    fn the_fill_is_one_undo_step() {
        let mut s = selected();
        let before = composite(&s);
        let steps = s.active().unwrap().history.past_len();
        let (client, _) = client(&solid_png(24, 18, [1, 2, 3]));
        generative_fill_with(&mut s, &client, &json!({"prompt": "anything"})).unwrap();
        assert_eq!(s.active().unwrap().history.past_len(), steps + 1, "the fill journals as one step");
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(composite(&s), before);
        assert_eq!(s.active().unwrap().doc.layers.len(), 2, "undo removes the generated layer");
    }

    #[test]
    fn a_reference_larger_than_the_cap_is_downscaled() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 600, "height": 400})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("edit.fill", json!({"contents": "color", "color": "#204080"})).unwrap();
        s.execute("select.rect", json!({"x": 50, "y": 50, "width": 500, "height": 300})).unwrap();
        let (client, sent) = client(&solid_png(500, 300, [200, 200, 200]));
        let r = generative_fill_with(&mut s, &client, &json!({"prompt": "hills", "maxReferenceSide": 256})).unwrap();
        assert!(r.get("layer").is_some());
        let (w, h) = sent_reference(&sent);
        assert!(w <= 256 && h <= 256, "the reference must be downscaled: {w}x{h}");
        assert!(w < 500 && h < 300, "the reference must be downscaled: {w}x{h}");
    }

    #[test]
    fn the_result_covers_the_selection_window() {
        // The generated frame is the selection plus its context ring, clipped to the canvas; the
        // layer mask then limits what shows to the selection itself.
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 200, "height": 120})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("edit.fill", json!({"contents": "color", "color": "#808080"})).unwrap();
        s.execute("select.rect", json!({"x": 20, "y": 30, "width": 60, "height": 40})).unwrap();
        let (client, _) = client(&solid_png(4, 3, [255, 255, 0]));
        let r = generative_fill_with(&mut s, &client, &json!({"prompt": "sun"})).unwrap();
        assert_eq!(r["bounds"], json!([0, 0, 176, 120]), "{r}");
    }

    #[test]
    fn the_key_order_is_environment_then_stored() {
        use photocraft_gen::{API_KEY_ENV, API_KEY_ENV_FALLBACK};
        // Nothing anywhere.
        assert_eq!(key_with(|_| None, None), None);
        // Only the generic environment variable.
        assert_eq!(key_with(|n| (n == API_KEY_ENV_FALLBACK).then(|| "generic".into()), None).as_deref(), Some("generic"));
        // The scoped variable beats the generic one, and both beat a stored key.
        assert_eq!(key_with(|n| (n == API_KEY_ENV || n == API_KEY_ENV_FALLBACK).then(|| "scoped".into()), Some("stored".into())).as_deref(), Some("scoped"));
        assert_eq!(key_with(|n| (n == API_KEY_ENV_FALLBACK).then(|| "generic".into()), Some("stored".into())).as_deref(), Some("generic"));
        // The stored (Settings dialog) key is the last resort, and blanks never win.
        assert_eq!(key_with(|_| None, Some("  stored  ".into())).as_deref(), Some("stored"));
        assert_eq!(key_with(|n| (n == API_KEY_ENV).then(|| "   ".into()), Some("stored".into())).as_deref(), Some("stored"));
    }

    #[test]
    fn stored_session_keys_are_trimmed_and_blanks_mean_none() {
        let mut s = Session::new();
        assert_eq!(s.ai_key(), None);
        s.set_ai_key(Some("  sk-or-v1-test  ".into()));
        assert_eq!(s.ai_key(), Some("sk-or-v1-test"));
        s.set_ai_key(Some("   ".into()));
        assert_eq!(s.ai_key(), None);
        s.set_ai_key(None);
        assert_eq!(s.ai_key(), None);
    }

    #[test]
    fn no_prompt_is_an_error() {
        let mut s = selected();
        let (client, _) = client(&solid_png(2, 2, [0, 0, 0]));
        let e = generative_fill_with(&mut s, &client, &json!({"prompt": "   "})).unwrap_err();
        assert!(e.to_string().contains("prompt"), "{e}");
    }

    #[test]
    fn no_selection_is_an_error() {
        let mut s = document();
        let (client, _) = client(&solid_png(2, 2, [0, 0, 0]));
        let e = generative_fill_with(&mut s, &client, &json!({"prompt": "a lake"})).unwrap_err();
        assert!(e.to_string().contains("no selection"), "{e}");
    }

    #[test]
    fn a_selection_outside_the_canvas_is_an_error() {
        let mut s = document();
        s.execute("select.rect", json!({"x": -40, "y": 20, "width": 4, "height": 4})).unwrap();
        let (client, _) = client(&solid_png(2, 2, [0, 0, 0]));
        let e = generative_fill_with(&mut s, &client, &json!({"prompt": "a lake"})).unwrap_err();
        assert!(e.to_string().contains("selection"), "{e}");
        assert_eq!(s.active().unwrap().doc.layers.len(), 2, "nothing was added");
    }

    #[test]
    fn hostile_params_are_rejected_without_touching_the_model() {
        let mut s = selected();
        let before = composite(&s);
        let (client, sent) = client(&solid_png(2, 2, [0, 0, 0]));
        for p in [
            json!({"prompt": "x", "feather": -3}),
            json!({"prompt": "x", "feather": "wide"}),
            json!({"prompt": "x", "feather": 1e30}),
            json!({"prompt": "x", "resolution": "8K"}),
            json!({"prompt": "x", "resolution": 2}),
            json!({"prompt": "x", "seed": -5}),
            json!({"prompt": "x", "seed": "seven"}),
            json!({"prompt": 12}),
            json!({"prompt": {"text": "x"}}),
            json!({"prompt": "x", "maxReferenceSide": "big"}),
            json!({}),
            json!([]),
        ] {
            assert!(generative_fill_with(&mut s, &client, &p).is_err(), "{p} should not be accepted");
            assert_eq!(composite(&s), before, "{p} must not change the document");
        }
        assert!(sent.lock().unwrap().is_empty(), "no refused call should reach the model");
    }

    #[test]
    fn an_unparsable_layer_param_falls_back_to_the_active_layer() {
        // The engine's convention for layer params (engine/src/commands.rs:layer_param): a value
        // that is not a layer id means "the active layer", so it is not an error.
        let mut s = selected();
        let (client, _) = client(&solid_png(2, 2, [0, 0, 0]));
        for p in [&json!({"prompt": "x", "layer": -1}), &json!({"prompt": "x", "layer": 4242})] {
            let r = generative_fill_with(&mut s, &client, p).unwrap();
            assert!(r.get("layer").is_some(), "{p} should fall back to the active layer");
        }
        assert_eq!(s.active().unwrap().doc.layers.len(), 4, "one fill per call");
    }

    #[test]
    fn an_absurd_max_reference_side_is_clamped_not_crashing() {
        let mut s = selected();
        let (client, _) = client(&solid_png(4, 3, [9, 9, 9]));
        let r = generative_fill_with(&mut s, &client, &json!({"prompt": "x", "maxReferenceSide": 1e30})).unwrap();
        assert!(r.get("layer").is_some());
    }

    #[test]
    fn an_unparsable_reply_is_an_error_not_a_panic() {
        struct Garbage;
        impl Transport for Garbage {
            fn post(&self, _: &str, _: &[(&str, &str)], _: &[u8], _: Duration) -> std::result::Result<(u16, Vec<u8>), GenError> {
                Ok((200, b"not json".to_vec()))
            }

            fn get(&self, _: &str, _: &[(&str, &str)], _: Duration) -> std::result::Result<(u16, Vec<u8>), GenError> {
                Ok((200, Vec::new()))
            }
        }
        let mut s = selected();
        let before = composite(&s);
        let e = generative_fill_with(&mut s, &photocraft_gen::Client::with_transport(Arc::new(Garbage)), &json!({"prompt": "x"})).unwrap_err();
        assert!(e.to_string().contains("decode") || e.to_string().contains("JSON"), "{e}");
        assert_eq!(composite(&s), before);
    }

    #[test]
    fn a_refused_request_keeps_the_document() {
        struct Refused;
        impl Transport for Refused {
            fn post(&self, _: &str, _: &[(&str, &str)], _: &[u8], _: Duration) -> std::result::Result<(u16, Vec<u8>), GenError> {
                Ok((402, br#"{"error":{"code":402,"message":"insufficient credits"}}"#.to_vec()))
            }

            fn get(&self, _: &str, _: &[(&str, &str)], _: Duration) -> std::result::Result<(u16, Vec<u8>), GenError> {
                Ok((200, Vec::new()))
            }
        }
        let mut s = selected();
        let before = composite(&s);
        let e = generative_fill_with(&mut s, &photocraft_gen::Client::with_transport(Arc::new(Refused)), &json!({"prompt": "x"})).unwrap_err();
        assert!(e.to_string().contains("insufficient credits"), "{e}");
        assert_eq!(composite(&s), before);
    }

    #[test]
    fn the_command_is_registered_and_disabled_without_a_selection() {
        let spec = crate::command_specs().iter().find(|s| s.id == FILL).unwrap();
        assert!(matches!(spec.menu, ["Edit"]));
        assert!(spec.label.contains("Generative Fill"));
        assert!(spec.params.contains("prompt"));
        let mut s = Session::new();
        assert!((spec.enabled)(&s).is_err(), "no document");
        s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
        assert!((spec.enabled)(&s).is_err(), "no selection");
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("select.rect", json!({"x": 0, "y": 0, "width": 4, "height": 4})).unwrap();
        assert!((spec.enabled)(&s).is_ok());
    }
}
