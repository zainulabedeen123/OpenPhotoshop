//! Edit › Generative Fill… : Photoshop's Generative Fill in front of `edit.generativeFill`.
//!
//! The dialog is a prompt and two buttons, like Photoshop's: describe what should fill the
//! selection and press Generate (the menu item and OK are greyed out without a selection, as in
//! Photoshop). The command runs as a background job, so the window keeps drawing while the model
//! works and the status bar shows its progress; the result lands on a new layer masked to the
//! selection, and the status line reports what the model time cost.
//!
//! Nothing is remembered on disk: the prompt is the user's words about their own image, so it is
//! not written to the preferences the way the Fill dialog's options are.

use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::state::DialogKind;
use crate::theme::Tokens;

/// The engine command this dialog drives.
pub const COMMAND: &str = "edit.generativeFill";
/// Dialog field marking Generative Fill.
const MARK: &str = "__genFill";
/// The prompt limit, the same one the engine refuses above.
pub const MAX_PROMPT: usize = 4_000;
/// The feather slider's range (the engine accepts up to 250).
const MAX_FEATHER: usize = 64;

/// Is this dialog Generative Fill?
pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.contains_key(MARK)
}

/// The trimmed prompt, when there is one.
pub fn prompt_of(fields: &Map<String, Value>) -> Option<String> {
    fields.get("prompt").and_then(Value::as_str).map(str::trim).filter(|p| !p.is_empty()).map(str::to_string)
}

/// The feather the dialog holds: a light blur of the mask, so a hard-edged selection does not
/// leave a seam where the generated content ends. Anything that is not a finite number keeps the
/// default; anything else clamps into the slider's range.
fn feather_of(fields: &Map<String, Value>) -> f64 {
    fields.get("feather").and_then(Value::as_f64).filter(|v| v.is_finite()).map_or(2.0, |v| v.clamp(0.0, MAX_FEATHER as f64))
}

/// Open Generative Fill; an open one is reused. `None` without a document to fill into.
pub fn open(app: &mut PhotocraftApp) -> Option<u64> {
    app.session.active()?;
    if let Some(d) = app.ui.dialogs.iter().find(|d| owns(&d.fields)) {
        return Some(d.id);
    }
    let fields = json!({
        MARK: true,
        "__command": COMMAND,
        "__label": "Generative Fill",
        "prompt": "",
        "feather": 2.0,
    });
    Some(app.ui.open_dialog(DialogKind::Command, fields.as_object().cloned().unwrap_or_default()))
}

/// Whether Generate (and Enter) may run: a prompt, a selection to fill, and a key to fill with
/// ([`crate::settings_ui::key_ready`], which the session knows from the environment or Settings).
pub fn can_confirm(app: &PhotocraftApp, fields: &Map<String, Value>) -> bool {
    if !owns(fields) {
        return true;
    }
    prompt_of(fields).is_some() && app.session.active().is_some_and(|st| st.doc.selection.is_some()) && crate::settings_ui::key_ready(app)
}

/// The `edit.generativeFill` params of the dialog's fields. Without a prompt it returns params
/// the engine refuses, so a hand-edited field set can never reach the model.
pub fn params(fields: &Map<String, Value>) -> Value {
    let Some(prompt) = prompt_of(fields) else { return json!({"prompt": ""}) };
    let feather = feather_of(fields);
    let mut p = json!({ "prompt": prompt });
    if feather > 0.0 {
        p["feather"] = json!(feather);
    }
    p
}

/// OK: generate. The command runs as a background job and replies with its job id.
pub fn confirm(app: &mut PhotocraftApp, fields: &Map<String, Value>) -> Result<Value, String> {
    if prompt_of(fields).is_none() {
        return Err(tl!("Describe what should fill the selection").to_string());
    }
    app.run(COMMAND, params(fields))
}

/// The dialog body: the prompt, the mask feather, and what will happen.
pub fn body(app: &PhotocraftApp, ui: &mut egui::Ui, fields: &mut Map<String, Value>) {
    ui.spacing_mut().item_spacing.y = 6.0;
    let mut prompt = fields.get("prompt").and_then(Value::as_str).unwrap_or_default().to_string();
    if prompt.chars().count() > MAX_PROMPT {
        prompt = prompt.chars().take(MAX_PROMPT).collect();
    }
    ui.horizontal_top(|ui| {
        ui.label(tl!("Prompt:"));
        let r = ui.add(
            egui::TextEdit::multiline(&mut prompt)
                .id_salt("gen-fill-prompt")
                .hint_text(tl!("a red sailboat on calm water"))
                .desired_width(300.0)
                .desired_rows(3)
                .char_limit(MAX_PROMPT),
        );
        // Photoshop focuses the prompt so typing replaces the placeholder.
        let focused = ui.id().with("gen-fill-focused");
        if !ui.ctx().data(|d| d.get_temp::<bool>(focused).unwrap_or(false)) && app.ui.dialogs.len() == 1 {
            r.request_focus();
            ui.ctx().data_mut(|d| d.insert_temp(focused, true));
        }
    });
    fields.insert("prompt".into(), json!(prompt));
    let mut feather = feather_of(fields);
    ui.horizontal(|ui| {
        ui.label(tl!("Mask feather:"));
        let slider = egui::Slider::new(&mut feather, 0.0..=MAX_FEATHER as f64)
            .suffix(" px")
            .clamping(egui::SliderClamping::Always)
            .max_decimals(0)
            .custom_formatter(|v, _| format!("{}", v.round() as i64));
        if ui.add(slider).changed() {
            fields.insert("feather".into(), json!(feather.round()));
        }
    });
    ui.horizontal_wrapped(|ui| {
        let selected = app.session.active().is_some_and(|st| st.doc.selection.is_some());
        let keyed = crate::settings_ui::key_ready(app);
        let note = match (selected, keyed) {
            (true, true) => tl!("The selected area is sent to the image model and the result is placed on a new layer, masked to the selection."),
            (true, false) => tl!("Save an OpenRouter API key in Settings (the gear at the top right) before generating."),
            (false, true) => tl!("Select the area to fill first: with no selection there is nothing for the model to fill."),
            (false, false) => tl!("Select the area to fill, and save an OpenRouter API key in Settings (the gear at the top right), before generating."),
        };
        ui.label(egui::RichText::new(note).color(Tokens::get(ui.ctx()).text_faint).small());
    });
}

#[cfg(test)]
#[path = "gen_fill_ui_tests.rs"]
mod tests;
