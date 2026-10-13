use egui_kittest::Harness;
use serde_json::{Value, json};

use super::*;

fn app() -> PhotocraftApp {
    PhotocraftApp::new(photocraft_engine::Session::new(), Default::default())
}

fn app_with_document() -> PhotocraftApp {
    let mut app = app();
    app.run("file.new", json!({"width": 40, "height": 30})).unwrap();
    app.session.set_ai_key(Some("sk-or-v1-test".into()));
    app
}

fn fields_of(app: &PhotocraftApp) -> serde_json::Map<String, Value> {
    app.ui.dialogs.iter().find(|d| owns(&d.fields)).map(|d| d.fields.clone()).unwrap_or_default()
}

#[test]
fn the_menu_item_opens_the_dialog_and_one_dialog_only() {
    let mut app = app_with_document();
    let ctx = egui::Context::default();
    let r = crate::menus::invoke(&mut app, &ctx, COMMAND, json!({})).unwrap();
    let id = r["dialog"].as_u64().unwrap();
    let f = fields_of(&app);
    assert_eq!((f.get("__command").and_then(Value::as_str), f.get("__label").and_then(Value::as_str)), (Some(COMMAND), Some("Generative Fill")));
    assert_eq!(f.get("prompt").and_then(Value::as_str), Some(""));
    // A second click reuses it.
    let r = crate::menus::invoke(&mut app, &ctx, COMMAND, json!({})).unwrap();
    assert_eq!(r["dialog"].as_u64(), Some(id));
    assert_eq!(app.ui.dialogs.len(), 1);
    // With params (agents, actions) the engine runs directly: no dialog.
    let r = crate::menus::invoke(&mut app, &ctx, COMMAND, json!({"prompt": "x"}));
    assert!(r.is_err(), "no API key in the test environment: a clear error, and that is all");
    assert_eq!(app.ui.dialogs.len(), 1, "the dialog is untouched");
    assert_eq!(app.session.active().unwrap().doc.layers.len(), 1, "nothing was generated");
}

#[test]
fn no_document_opens_nothing() {
    let mut app = app();
    let ctx = egui::Context::default();
    assert!(crate::menus::invoke(&mut app, &ctx, COMMAND, json!({})).is_err());
    assert!(app.ui.dialogs.is_empty());
}

#[test]
fn generate_needs_a_prompt_and_a_selection() {
    let mut app = app_with_document();
    let open = |app: &mut PhotocraftApp| {
        let ctx = egui::Context::default();
        crate::menus::invoke(app, &ctx, COMMAND, json!({})).unwrap()["dialog"].as_u64().unwrap()
    };
    let id = open(&mut app);
    assert!(!can_confirm(&app, &fields_of(&app)), "an empty prompt");
    for blank in [json!("   "), json!("a selection needs words"), json!(null)] {
        if blank.is_null() {
            app.ui.dialog_mut(id).unwrap().fields.remove("prompt");
        } else {
            app.ui.dialog_mut(id).unwrap().fields.insert("prompt".into(), blank.clone());
        }
        assert!(!can_confirm(&app, &fields_of(&app)), "{blank}");
    }
    app.ui.dialog_mut(id).unwrap().fields.insert("prompt".into(), json!("a red sailboat"));
    assert!(!can_confirm(&app, &fields_of(&app)), "no selection yet");
    app.run("select.rect", json!({"x": 4, "y": 4, "width": 12, "height": 10})).unwrap();
    assert!(can_confirm(&app, &fields_of(&app)));
}

#[test]
fn generate_stays_disabled_until_an_api_key_is_saved() {
    let mut app = app_with_document();
    app.session.set_ai_key(None);
    let ctx = egui::Context::default();
    crate::menus::invoke(&mut app, &ctx, COMMAND, json!({})).unwrap();
    app.ui.dialogs.iter_mut().find(|d| owns(&d.fields)).unwrap().fields.insert("prompt".into(), json!("a red sailboat"));
    app.run("select.rect", json!({"x": 4, "y": 4, "width": 12, "height": 10})).unwrap();
    let fields = fields_of(&app);
    assert!(!can_confirm(&app, &fields), "prompt and selection are not enough without a key");
    // Saving one in Settings makes the same dialog ready: the session carries it.
    app.ui.status = String::new();
    crate::settings_ui::open(&mut app);
    let settings = app.ui.dialogs.iter().find(|d| crate::settings_ui::owns(&d.fields)).unwrap().id;
    app.ui.dialog_mut(settings).unwrap().fields.insert("apiKey".into(), json!("sk-or-v1-test"));
    crate::dialogs::confirm(&mut app, settings).unwrap();
    assert_eq!(app.session.ai_key(), Some("sk-or-v1-test"));
    assert!(can_confirm(&app, &fields), "the saved key completes the gate");
}

#[test]
fn the_fields_map_to_the_command_params() {
    let mut f = serde_json::Map::new();
    f.insert("prompt".into(), json!("  a koi pond  "));
    f.insert("feather".into(), json!(6.0));
    assert_eq!(params(&f), json!({"prompt": "a koi pond", "feather": 6.0}));
    // A hard feather is left to the engine's default (none).
    let mut g = f.clone();
    g.insert("feather".into(), json!(0.0));
    assert_eq!(params(&g), json!({"prompt": "a koi pond"}));
    // Hostile feather values clamp to the slider's range instead of failing.
    for bad in [json!(1e9), json!(-4), json!(0), json!("wide"), json!(null), json!(true)] {
        let mut h = f.clone();
        h.insert("feather".into(), bad.clone());
        let want = match bad {
            Value::Null | Value::String(_) | Value::Bool(_) => json!(2.0),
            _ if bad.as_f64().is_some_and(|v| v <= 0.0) => Value::Null,
            _ => json!(64.0),
        };
        assert_eq!(params(&h).get("feather").cloned().unwrap_or(Value::Null), want, "{bad}");
    }
    // No prompt: no prompt in the params either (the command refuses them).
    let mut empty = serde_json::Map::new();
    empty.insert("prompt".into(), json!("  "));
    assert_eq!(params(&empty), json!({"prompt": ""}));
}

#[test]
fn confirm_without_a_prompt_is_refused() {
    let mut app = app_with_document();
    let ctx = egui::Context::default();
    let id = crate::menus::invoke(&mut app, &ctx, COMMAND, json!({})).unwrap()["dialog"].as_u64().unwrap();
    assert!(crate::dialogs::confirm(&mut app, id).is_err(), "OK without a prompt explains why");
    assert!(app.ui.dialogs.is_empty(), "the dialog closed anyway");
    assert_eq!(app.session.active().unwrap().doc.layers.len(), 1, "nothing was generated");
}

/// The dialog body: the prompt field, the feather slider and the note that says what happens.
#[test]
fn the_body_shows_the_prompt_and_notes_what_else_is_needed() {
    use egui_kittest::kittest::{NodeT, Queryable};
    let mut h = Harness::builder().with_size(egui::vec2(1200.0, 800.0)).with_max_steps(64).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        let mut app = app_with_document();
        let ctx = egui::Context::default();
        crate::menus::invoke(&mut app, &ctx, COMMAND, json!({})).unwrap();
        app
    });
    h.run_steps(6);
    assert!(h.query_by_label("Prompt:").is_some());
    assert!(h.query_by_label("Mask feather:").is_some());
    assert!(h.query_by_label_contains("Select the area to fill first").is_some());
    h.state_mut().run("select.rect", json!({"x": 4, "y": 4, "width": 12, "height": 10})).unwrap();
    h.run_steps(3);
    // The note now describes what the command will do.
    assert!(h.query_by_label_contains("sent to the image model").is_some());
    // A selection and a prompt make Generate available.
    {
        let d = h.state_mut().ui.dialogs.iter_mut().find(|d| owns(&d.fields)).unwrap();
        d.fields.insert("prompt".into(), json!("a red sailboat"));
    }
    h.run_steps(3);
    assert!(!h.get_by_label("Generate").accesskit_node().is_disabled());
}
