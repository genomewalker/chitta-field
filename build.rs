// Generates the embedding-identity constants (EMBED_DIM / EMBED_MODEL_ID /
// EMBED_MODEL_ID_CSTR / TEXT_FORMAT_VERSION) from build-time env so the public release
// (CI, no env set) matches its bundled nomic-embed-text-v1.5 GGUF (768-dim), while a
// personal/local build overrides via CHITTA_EMBED_DIM / CHITTA_EMBED_MODEL_ID /
// CHITTA_TEXT_FORMAT_VERSION. The C++ CMake reads the same env vars, so one environment
// drives both layers and the on-disk format never drifts between them.
use std::{env, fs, path::Path};

fn main() {
    println!("cargo:rerun-if-env-changed=CHITTA_EMBED_DIM");
    println!("cargo:rerun-if-env-changed=CHITTA_EMBED_MODEL_ID");
    println!("cargo:rerun-if-env-changed=CHITTA_TEXT_FORMAT_VERSION");

    let dim: usize = env::var("CHITTA_EMBED_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(768);
    assert!(
        dim % 64 == 0,
        "CHITTA_EMBED_DIM ({dim}) must be a multiple of 64 (binary codes are packed 64/word)"
    );

    let model_id =
        env::var("CHITTA_EMBED_MODEL_ID").unwrap_or_else(|_| "nomic-embed-text-v1.5".to_string());
    assert!(
        !model_id.contains('"') && !model_id.contains('\\'),
        "CHITTA_EMBED_MODEL_ID must not contain quotes or backslashes"
    );

    let tfv: u32 = env::var("CHITTA_TEXT_FORMAT_VERSION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    // NUL-terminated byte array so the FFI can hand C++ a borrowed static C string
    // without allocating.
    let mut cstr_bytes = model_id.clone().into_bytes();
    cstr_bytes.push(0);
    let cstr_lit: String = cstr_bytes.iter().map(|b| format!("{b}u8,")).collect();

    let dest = Path::new(&env::var("OUT_DIR").unwrap()).join("embed_config.rs");
    fs::write(
        &dest,
        format!(
            "pub const EMBED_DIM: usize = {dim};\n\
             pub const EMBED_MODEL_ID: &str = \"{model_id}\";\n\
             pub const EMBED_MODEL_ID_CSTR: &[u8] = &[{cstr_lit}];\n\
             pub const TEXT_FORMAT_VERSION: u32 = {tfv};\n"
        ),
    )
    .unwrap();
}
