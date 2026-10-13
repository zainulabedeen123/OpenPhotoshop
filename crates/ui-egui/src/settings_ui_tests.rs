use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use super::*;
use crate::AiKeyStore;

/// An in-memory key store: what was stored, and where it says it lives.
#[derive(Default)]
struct MemoryStore {
    stored: Mutex<Vec<Option<String>>>,
    location: Option<std::path::PathBuf>,
}

impl AiKeyStore for MemoryStore {
    fn load(&self) -> Result<Option<String>, String> {
        Ok(self.stored.lock().unwrap().last().cloned().flatten())
    }

    fn store(&self, key: Option<&str>) -> Result<(), String> {
        self.stored.lock().unwrap().push(key.map(str::to_string));
        Ok(())
    }

    fn location(&self) -> Option<std::path::PathBuf> {
        self.location.clone()
    }
}

fn services_with_store() -> (crate::Services, Arc<MemoryStore>) {
    let store = Arc::new(MemoryStore::default());
    let services = crate::Services { ai_key: Some(Arc::clone(&store) as Arc<dyn AiKeyStore>), ..Default::default() };
    (services, store)
}

fn app() -> PhotocraftApp {
    PhotocraftApp::new(photocraft_engine::Session::new(), Default::default())
}

fn open_settings(app: &mut PhotocraftApp) -> u64 {
    open(app)
}

fn with_key(app: &mut PhotocraftApp, key: &str) {
    app.session.set_ai_key(Some(key.into()));
}

#[test]
fn settings_reuses_one_dialog_and_needs_no_document() {
    let mut app = app();
    assert!(app.ui.dialogs.is_empty());
    let a = open_settings(&mut app);
    let b = open_settings(&mut app);
    assert_eq!(a, b, "a second open reuses the dialog");
    let fields = app.ui.dialogs.iter().find(|d| owns(&d.fields)).map(|d| d.fields.clone()).unwrap();
    assert_eq!(fields.get("__label").and_then(Value::as_str), Some("Settings"));
    assert_eq!(fields.get("apiKey").and_then(Value::as_str), Some(""));
}

#[test]
fn saving_a_key_stores_it_trimmed_and_hands_it_to_the_engine() {
    let (services, store) = services_with_store();
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
    let id = open_settings(&mut app);
    app.ui.dialog_mut(id).unwrap().fields.insert("apiKey".into(), json!("  sk-or-v1-shhh  "));
    let r = crate::dialogs::confirm(&mut app, id).unwrap();
    assert_eq!(r["stored"], json!(true), "{r}");
    assert_eq!(store.stored.lock().unwrap().last().cloned().flatten().as_deref(), Some("sk-or-v1-shhh"), "the typed key is stored trimmed");
    assert_eq!(app.session.ai_key(), Some("sk-or-v1-shhh"), "the engine can use it");
    assert!(app.ui.dialogs.is_empty(), "the dialog closes on save");
    assert!(app.ui.status.contains("saved"), "{}", app.ui.status);
    assert!(!app.ui.status.contains("shhh"), "the status line never shows the key");
}

#[test]
fn saving_an_empty_field_removes_the_stored_key() {
    let (services, store) = services_with_store();
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
    with_key(&mut app, "sk-or-v1-old");
    let id = open_settings(&mut app);
    app.ui.dialog_mut(id).unwrap().fields.insert("apiKey".into(), json!("   "));
    crate::dialogs::confirm(&mut app, id).unwrap();
    assert_eq!(store.stored.lock().unwrap().last().cloned().flatten(), None, "the stored key is removed");
    assert_eq!(app.session.ai_key(), None, "the session forgets it too");
    assert!(app.ui.status.contains("removed"), "{}", app.ui.status);
}

#[test]
fn an_absurd_key_is_refused_without_storing_anything() {
    let (services, store) = services_with_store();
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
    with_key(&mut app, "sk-or-v1-old");
    let id = open_settings(&mut app);
    let long = "k".repeat(MAX_KEY_CHARS + 1);
    app.ui.dialog_mut(id).unwrap().fields.insert("apiKey".into(), json!(long));
    // The dialog closes before its confirm runs (always; every dialog works that way), so the
    // refusal is the error the caller sees and the old key stays everywhere it was.
    let e = crate::dialogs::confirm(&mut app, id).unwrap_err();
    assert!(e.contains("longer than"), "{e}");
    assert!(store.stored.lock().unwrap().is_empty(), "nothing reached the store");
    assert_eq!(app.session.ai_key(), Some("sk-or-v1-old"), "the old key survives");
}

#[test]
fn without_a_store_the_key_is_session_only() {
    let mut app = app();
    let id = open_settings(&mut app);
    app.ui.dialog_mut(id).unwrap().fields.insert("apiKey".into(), json!("sk-or-v1-session"));
    crate::dialogs::confirm(&mut app, id).unwrap();
    assert_eq!(app.session.ai_key(), Some("sk-or-v1-session"));
    assert!(app.ui.status.contains("saved"), "{}", app.ui.status);
}

#[test]
fn a_failed_store_refuses_the_session_key_too() {
    struct Denied;
    impl AiKeyStore for Denied {
        fn load(&self) -> Result<Option<String>, String> {
            Ok(None)
        }

        fn store(&self, _: Option<&str>) -> Result<(), String> {
            Err("no settings directory".to_string())
        }
    }
    let services = crate::Services { ai_key: Some(Arc::new(Denied) as Arc<dyn AiKeyStore>), ..Default::default() };
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
    let id = open_settings(&mut app);
    app.ui.dialog_mut(id).unwrap().fields.insert("apiKey".into(), json!("sk-or-v1-shhh"));
    // The store refused: the caller sees why, and no half-applied key reaches the engine.
    let e = crate::dialogs::confirm(&mut app, id).unwrap_err();
    assert_eq!(e, "no settings directory");
    assert_eq!(app.session.ai_key(), None, "a refused store refuses the session key too");
    assert!(!app.ui.status.contains("shhh"), "the status line never shows the key");
}

#[test]
fn the_dialog_notes_the_state_with_and_without_a_key() {
    use egui_kittest::kittest::Queryable;
    let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(1000.0, 700.0)).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        let mut app = app();
        open_settings(&mut app);
        app
    });
    h.run_steps(4);
    assert!(h.query_by_label_contains("No OpenRouter key yet").is_some(), "no key yet: the note says what is missing");
    assert!(h.query_by_label_contains("no key storage").is_some(), "no store service: the note says the key is session-only");
    h.state_mut().session.set_ai_key(Some("sk-or-v1-x".into()));
    h.run_steps(2);
    assert!(h.query_by_label_contains("Key in use").is_some(), "with a key: the status line confirms it");
}
