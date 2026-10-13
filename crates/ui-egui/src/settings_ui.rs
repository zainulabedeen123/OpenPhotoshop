//! The Settings dialog, opened from the gear at the right end of the menu bar.
//!
//! Photoshop's options live in Edit ▸ Preferences; this dialog holds what cannot safely go there:
//! the OpenRouter API key that powers Generative Fill. The key is stored through the app's
//! [`AiKeyStore`](crate::AiKeyStore) in the per-user settings directory — never in the
//! preferences file, never in a document, never in the log — and is handed to the engine as the
//! key `edit.generativeFill` uses when the environment provides none. The field is masked, the
//! stored key is never echoed back into it, and saving an empty field removes the key: a stale
//! key leaves the machine, not just the session.
//!
//! The engine still reads `PHOTOCRAFT_OPENROUTER_API_KEY` (then `OPENROUTER_API_KEY`) first, so
//! agents, the CLI and CI keep overriding whatever a user typed here.

use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::state::DialogKind;
use crate::theme::Tokens;

/// Dialog field marking Settings.
const MARK: &str = "__settings";
/// A key longer than this is refused before it is stored anywhere.
pub const MAX_KEY_CHARS: usize = 4096;
/// Every OpenRouter key starts with this; a different shape is accepted but warned about.
const KEY_PREFIX: &str = "sk-or-";

/// Is this dialog Settings?
pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.contains_key(MARK)
}

/// The key Save writes: the masked field's text, trimmed.
fn field_key(fields: &Map<String, Value>) -> Option<String> {
    fields.get("apiKey").and_then(Value::as_str).map(str::trim).filter(|k| !k.is_empty()).map(str::to_string)
}

/// Whether the session carries a key (set from the environment or the store at startup, and by
/// every save below). The dialog's status line and the Generative Fill prompt read this, so it
/// must reflect what the engine will do, not what the environment of some other process says.
pub fn key_ready(app: &PhotocraftApp) -> bool {
    app.session.ai_key().is_some()
}

/// Open Settings, reusing an open one. Works without a document; the web build opens it too, and
/// the body explains that Generative Fill is desktop-only.
pub fn open(app: &mut PhotocraftApp) -> u64 {
    if let Some(d) = app.ui.dialogs.iter().find(|d| owns(&d.fields)) {
        return d.id;
    }
    let fields = json!({
        MARK: true,
        "__label": "Settings",
        "apiKey": "",
    });
    app.ui.open_dialog(DialogKind::Command, fields.as_object().cloned().unwrap_or_default())
}

/// Whether a usable key store is attached (the desktop app; `None` on the web).
fn store(app: &PhotocraftApp) -> Option<&std::sync::Arc<dyn crate::AiKeyStore>> {
    app.services.ai_key.as_ref()
}

/// OK: store the typed key (an empty field removes it) and hand it to the engine.
pub fn confirm(app: &mut PhotocraftApp, fields: &Map<String, Value>) -> Result<Value, String> {
    let typed = field_key(fields);
    if let Some(key) = &typed && key.chars().count() > MAX_KEY_CHARS {
        return Err(format!("the key is longer than {MAX_KEY_CHARS} characters"));
    }
    if let Some(service) = store(app) {
        service.store(typed.as_deref())?;
    }
    app.session.set_ai_key(typed.clone());
    app.ui.status = match (&typed, store(app).is_some()) {
        (Some(_), _) => "Settings: OpenRouter key saved (used by Edit › Generative Fill)".into(),
        (None, true) => "Settings: store key removed".into(),
        (None, false) => "Settings: key cleared for this session".into(),
    };
    app.ui.status_error = false;
    Ok(json!({"saved": true, "stored": typed.is_some()}))
}

/// The dialog body: the key status, the masked field, and what saving does to it.
pub fn body(app: &PhotocraftApp, ui: &mut egui::Ui, fields: &mut Map<String, Value>) {
    ui.spacing_mut().item_spacing.y = 6.0;
    let t = Tokens::get(ui.ctx());
    ui.heading(tl!("Generative Fill (OpenRouter)"));
    ui.add_space(2.0);
    let ready = key_ready(app);
    ui.label(
        egui::RichText::new(if ready {
            tl!("Key in use: Generative Fill will run with it.")
        } else {
            tl!("No OpenRouter key yet: Generative Fill needs one before it can run.")
        })
        .color(if ready { t.accent_text } else { t.text_faint })
        .small(),
    );
    let mut key = fields.get("apiKey").and_then(Value::as_str).unwrap_or_default().to_string();
    if key.chars().count() > MAX_KEY_CHARS {
        key = key.chars().take(MAX_KEY_CHARS).collect();
    }
    ui.horizontal(|ui| {
        ui.label(tl!("API key:"));
        let edit = egui::TextEdit::singleline(&mut key)
            .id_salt("settings-api-key")
            .hint_text("sk-or-v1-…")
            .desired_width(330.0)
            .char_limit(MAX_KEY_CHARS)
            .password(true);
        let r = ui.add(edit);
        if !ready && r.lost_focus() {
            r.request_focus();
        }
        let malform = field_key(fields).is_some_and(|k| !k.starts_with(KEY_PREFIX));
        if malform {
            ui.label(egui::RichText::new(tl!("OpenRouter keys usually start with sk-or-")).color(t.warning).small());
        }
    });
    fields.insert("apiKey".into(), json!(key));
    ui.horizontal_wrapped(|ui| {
        ui.label(
            egui::RichText::new(tl!("Save an OpenRouter API key and Edit › Generative Fill can generate. Saving an empty field removes the key."))
                .color(t.text_faint)
                .small(),
        );
    });
    ui.horizontal_wrapped(|ui| {
        let path = app.services.ai_key.as_ref().and_then(|s| s.location());
        let note = match (app.services.ai_key.is_some(), path) {
            (true, Some(path)) => format!("{} {}", tl!("Stored (unencrypted) in:"), path.display()),
            (true, None) => tl!("Stored in the app's settings folder.").to_string(),
            (false, _) => tl!("This build has no key storage; the key applies to this session only. Generative Fill itself needs the desktop app.").to_string(),
        };
        ui.label(egui::RichText::new(note).color(t.text_faint).small());
    });
    ui.horizontal_wrapped(|ui| {
        let note = tl!("The key is a secret: it is sent only to OpenRouter's image request. PHOTOCRAFT_OPENROUTER_API_KEY and OPENROUTER_API_KEY in the environment win over it.");
        ui.label(egui::RichText::new(note).color(t.text_faint).small());
    });
}

#[cfg(test)]
#[path = "settings_ui_tests.rs"]
mod tests;
