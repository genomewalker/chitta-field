//! Train the lite encoder from existing memories and save to disk.
//! Usage: ./build.sh run --bin train_lite --release -- --field-dir ~/.claude/mind/chitta-field

use std::path::PathBuf;
use chitta_field::field::ChittaField;

fn expand_home(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        let home = std::env::var("HOME").unwrap_or_default();
        PathBuf::from(home).join(rest)
    } else {
        PathBuf::from(p)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let field_dir = args.windows(2)
        .find(|w| w[0] == "--field-dir")
        .map(|w| expand_home(&w[1]))
        .unwrap_or_else(|| {
            eprintln!("Usage: train_lite --field-dir <path>");
            std::process::exit(1);
        });

    eprintln!("Opening chitta-field at {:?}", field_dir);
    let field = ChittaField::open(field_dir).expect("failed to open chitta-field");

    let total = field.memory_count();
    eprintln!("Total memories: {}  Cortical index: {}", total, field.cortical_count());

    eprintln!("Training lite encoder...");
    match field.train_lite_encoder() {
        Ok(n) => {
            eprintln!("Trained on {} examples.", n);
            field.save_lite_encoder().expect("failed to save lite encoder");
            eprintln!("Lite encoder saved.");
            eprintln!("Ready: {}", field.lite_encoder_ready());
        }
        Err(e) => {
            eprintln!("Training failed: {}", e);
            std::process::exit(1);
        }
    }
}
