//! #14 fine-tune: offline read-only dump of a store's memory graph for
//! training-pair mining. Open a COPY of the snapshot family — never the live
//! store dir (open replays WAL and takes the store lock).
//!
//! Usage: dump_graph <FIELD_DIR> <OUT_DIR>

use chitta_field::field::ChittaField;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: dump_graph <FIELD_DIR> <OUT_DIR>");
        std::process::exit(2);
    }
    let field_dir = PathBuf::from(&args[1]);
    let out_dir = PathBuf::from(&args[2]);

    let t0 = Instant::now();
    let field = ChittaField::open(field_dir).expect("open field store");
    eprintln!("[dump_graph] store open in {:.1?}", t0.elapsed());

    let t1 = Instant::now();
    let (nodes, edges) = field.dump_training_graph(&out_dir).expect("dump");
    eprintln!(
        "[dump_graph] wrote {nodes} nodes, {edges} edges to {} in {:.1?}",
        out_dir.display(),
        t1.elapsed()
    );
}
