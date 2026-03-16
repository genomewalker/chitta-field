use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::Path;
use std::io::{BufWriter, BufReader, Write, Read};
use crate::ids::{MemoryId, ArtifactId};
use crate::payload::MemoryPayload;
use crate::state::MemoryState;
use crate::field::AssocEdge;
use crate::hnsw::SemanticIndex;
use crate::organ::temporal::TemporalIndex;
use crate::organ::keyword::KeywordIndex;
use crate::organ::artifact::ArtifactIndex;
use crate::organ::triplet::TripletStore;
use crate::organ::symbol::SymbolIndex;
use crate::organ::callgraph::CallGraph;
use crate::organ::codefile::CodeFileIndex;
use crate::error::{FieldError, Result};

const FULL_SNAPSHOT_MAGIC: u64 = 0xF011_5741_7E00_0003;

#[derive(Serialize, Deserialize)]
pub struct FullSnapshot {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, MemoryPayload>,
    pub states: HashMap<MemoryId, MemoryState>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
}

impl FullSnapshot {
    pub fn save(&self, path: &Path) -> Result<()> {
        let tmp = path.with_extension("snapshot.tmp");
        {
            let f = std::fs::File::create(&tmp)?;
            let mut w = BufWriter::new(f);
            w.write_all(&FULL_SNAPSHOT_MAGIC.to_le_bytes())?;
            bincode::serialize_into(&mut w, self)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            w.flush()?;
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let f = std::fs::File::open(path)?;
        let mut r = BufReader::new(f);
        let mut magic_buf = [0u8; 8];
        r.read_exact(&mut magic_buf)?;
        let magic = u64::from_le_bytes(magic_buf);
        if magic != FULL_SNAPSHOT_MAGIC {
            return Err(FieldError::Manifest("invalid full snapshot magic".to_string()));
        }
        bincode::deserialize_from(r)
            .map_err(|e| FieldError::Serialization(e.to_string()))
    }
}
