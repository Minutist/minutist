//! Bindings-generation harness.
//!
//! Writes `ui/src/ipc/bindings.ts` from the tauri-specta builder.
//!
//! Run from the workspace root:
//!
//! ```sh
//! cargo run -p minutist --bin generate-bindings
//! ```
//!
//! Or via the cargo alias (see `.cargo/config.toml`):
//!
//! ```sh
//! cargo gen-bindings
//! ```
//!
//! This binary does NOT require a running Tauri application — it only calls
//! `bindings_builder().export(...)` which exercises the specta type collector
//! without starting the GUI.

fn main() {
    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("ui")
        .join("src")
        .join("ipc")
        .join("bindings.ts");

    // Ensure the parent directory exists.
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).expect("failed to create output directory");
    }

    ipc_bridge::bindings_builder()
        .export(
            specta_typescript::Typescript::default()
                .bigint(specta_typescript::BigIntExportBehavior::Number),
            &out_path,
        )
        .expect("failed to export TypeScript bindings");

    println!("TypeScript bindings written to: {}", out_path.display());
}
