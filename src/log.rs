use crate::error::{FieldError, Result};
use crate::ids::InstanceId;
use crate::ops::Op;
use crc32fast::Hasher as CrcHasher;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

pub const SEGMENT_MAGIC: &[u8; 8] = b"CFLOG001";
pub const SEGMENT_MAGIC_V2: &[u8; 8] = b"CFLOG002";
pub const MAX_SEGMENT_SIZE: u64 = 256 * 1024 * 1024;

/// SHA256 hash of a record, used for hash-chaining.
pub type ChainHash = [u8; 32];
pub const ZERO_HASH: ChainHash = [0u8; 32];

/// Compute the hash that links this record into the chain.
/// H = SHA256(seqno || op_type || prev_hash || payload)
fn compute_record_hash(seqno: u64, op_type: u8, prev_hash: &ChainHash, payload: &[u8]) -> ChainHash {
    let mut hasher = Sha256::new();
    hasher.update(seqno.to_be_bytes());
    hasher.update([op_type]);
    hasher.update(prev_hash);
    hasher.update(payload);
    hasher.finalize().into()
}

fn hex(h: &ChainHash) -> String {
    h.iter().map(|b| format!("{:02x}", b)).collect::<String>()
}

const OP_PUT_PAYLOAD: u8 = 0;
const OP_UPDATE_STATE: u8 = 1;
const OP_DELETE_MEMORY: u8 = 2;
const OP_ADD_ASSOC_EDGE: u8 = 3;
const OP_UPSERT_ARTIFACT: u8 = 4;
const OP_ADD_TRIPLET: u8 = 5;
const OP_INVALIDATE_TRIPLET: u8 = 6;
const OP_UPSERT_SYMBOL: u8 = 7;
const OP_REMOVE_SYMBOL: u8 = 8;
const OP_ADD_SYM_CALL_EDGE: u8 = 9;
const OP_REMOVE_SYM_CALL_EDGE: u8 = 10;
const OP_UPSERT_CODE_FILE: u8 = 11;
const OP_UPDATE_SPARSE_CODE: u8 = 12;
const OP_DEMOTE_MEMORY: u8 = 13;
const OP_TRAIN_PQ: u8 = 14;
const OP_UPDATE_RESIDUAL_PQ: u8 = 15;
use crate::ops::{
    OP_AGENT_DISABLE, OP_AGENT_UPSERT, OP_ANALYTICS_EVENT, OP_CLEAR_PROJECT, OP_MSG_EVENT,
    OP_SESSION_EVENT, OP_SKILL_DEPRECATE, OP_SKILL_UPLOAD, OP_TASK_EVENT, OP_THEME_EVENT,
    OP_RECORD_RECALL_BATCH, OP_STRENGTHEN_ASSOC_EDGE, OP_TRANSCRIPT_EVENT,
    OP_UPDATE_MEMORY_CONTENT, OP_UPDATE_SYMBOL_DESCRIPTION, OP_USER_MODEL_EVENT,
};

fn op_type_byte(op: &Op) -> u8 {
    match op {
        Op::PutPayload(_) => OP_PUT_PAYLOAD,
        Op::UpdateState(_) => OP_UPDATE_STATE,
        Op::DeleteMemory(_) => OP_DELETE_MEMORY,
        Op::AddAssocEdge(_) => OP_ADD_ASSOC_EDGE,
        Op::UpsertArtifact(_) => OP_UPSERT_ARTIFACT,
        Op::AddTriplet(_) => OP_ADD_TRIPLET,
        Op::InvalidateTriplet(_) => OP_INVALIDATE_TRIPLET,
        Op::UpsertSymbol(_) => OP_UPSERT_SYMBOL,
        Op::RemoveSymbol(_) => OP_REMOVE_SYMBOL,
        Op::AddSymCallEdge(_) => OP_ADD_SYM_CALL_EDGE,
        Op::RemoveSymCallEdge(_) => OP_REMOVE_SYM_CALL_EDGE,
        Op::UpsertCodeFile(_) => OP_UPSERT_CODE_FILE,
        Op::UpdateSparseCode(_) => OP_UPDATE_SPARSE_CODE,
        Op::DemoteMemory(_) => OP_DEMOTE_MEMORY,
        Op::TrainPQ(_) => OP_TRAIN_PQ,
        Op::UpdateResidualPQ(_) => OP_UPDATE_RESIDUAL_PQ,
        Op::SessionEvent(_) => OP_SESSION_EVENT,
        Op::TranscriptEvent(_) => OP_TRANSCRIPT_EVENT,
        Op::TaskEvent(_) => OP_TASK_EVENT,
        Op::UserModelEvent(_) => OP_USER_MODEL_EVENT,
        Op::ThemeEvent(_) => OP_THEME_EVENT,
        Op::AnalyticsEvent(_) => OP_ANALYTICS_EVENT,
        Op::ClearProject(_) => OP_CLEAR_PROJECT,
        Op::UpdateSymbolDescription(_) => OP_UPDATE_SYMBOL_DESCRIPTION,
        Op::UpdateMemoryContent(_) => OP_UPDATE_MEMORY_CONTENT,
        Op::RecordRecallBatch(_) => OP_RECORD_RECALL_BATCH,
        Op::StrengthenAssocEdge(_) => OP_STRENGTHEN_ASSOC_EDGE,
        Op::MsgEvent(_) => OP_MSG_EVENT,
        Op::SkillUpload(_) => OP_SKILL_UPLOAD,
        Op::SkillDeprecate(_) => OP_SKILL_DEPRECATE,
        Op::AgentUpsert(_) => OP_AGENT_UPSERT,
        Op::AgentDisable(_) => OP_AGENT_DISABLE,
    }
}

pub struct OpLog {
    data_dir: PathBuf,
    instance_id: InstanceId,
    current_segment: BufWriter<File>,
    current_segment_path: PathBuf,
    current_segment_size: u64,
    next_seqno: u64,
    ops_since_sync: u32,  // for batch fsync: sync_data() every N appends
    /// Hash-chain tip: SHA256 of the most recent V2 record.
    /// Zero for genesis or when only V1 segments exist.
    chain_head: ChainHash,
}

impl OpLog {
    /// Open the write log for a specific instance.
    /// Finds this instance's last segment to continue appending, or creates a new one.
    pub fn open(data_dir: &Path, instance_id: InstanceId, next_seqno: u64) -> Result<Self> {
        Self::open_with_chain(data_dir, instance_id, next_seqno, ZERO_HASH)
    }

    /// Open with a known chain_head (set by caller after replay).
    pub fn open_with_chain(
        data_dir: &Path,
        instance_id: InstanceId,
        next_seqno: u64,
        chain_head: ChainHash,
    ) -> Result<Self> {
        let seg_dir = data_dir.join("segments");
        fs::create_dir_all(&seg_dir)?;

        let existing = collect_instance_segments(&seg_dir, instance_id)?;

        if let Some(last_path) = existing.last() {
            let size = last_path.metadata()?.len();
            // Only continue appending to V2 segments; V1 segments force a new V2 segment
            if size < MAX_SEGMENT_SIZE && is_v2_segment(last_path) {
                let f = OpenOptions::new()
                    .write(true)
                    .append(true)
                    .open(last_path)?;
                return Ok(Self {
                    data_dir: data_dir.to_path_buf(),
                    instance_id,
                    current_segment: BufWriter::new(f),
                    current_segment_path: last_path.clone(),
                    current_segment_size: size,
                    next_seqno,
                    ops_since_sync: 0,
                    chain_head,
                });
            }
        }

        let path = segment_path(data_dir, instance_id, next_seqno);
        let f = create_segment_v2(&path, next_seqno, &chain_head)?;
        let header_size = V2_HEADER_SIZE as u64;
        Ok(Self {
            data_dir: data_dir.to_path_buf(),
            instance_id,
            current_segment: BufWriter::new(f),
            current_segment_path: path,
            current_segment_size: header_size,
            next_seqno,
            ops_since_sync: 0,
            chain_head,
        })
    }

    /// Append an op, returning its assigned seqno.
    /// V2 format: [payload_len:4][seqno:8][op_type:1][prev_hash:32][payload:N][crc32:4]
    pub fn append(&mut self, op: &Op) -> Result<u64> {
        self.rotate_if_needed()?;

        let seqno = self.next_seqno;
        let op_type = op_type_byte(op);
        let payload =
            rmp_serde::to_vec(op).map_err(|e| FieldError::Serialization(e.to_string()))?;

        let payload_len = payload.len() as u32;
        let seqno_bytes = seqno.to_be_bytes();
        let op_type_bytes = [op_type];
        let prev_hash = self.chain_head;

        // CRC covers seqno + op_type + prev_hash + payload
        let mut hasher = CrcHasher::new();
        hasher.update(&seqno_bytes);
        hasher.update(&op_type_bytes);
        hasher.update(&prev_hash);
        hasher.update(&payload);
        let crc = hasher.finalize();

        // Write V2 record
        self.current_segment.write_all(&payload_len.to_be_bytes())?;
        self.current_segment.write_all(&seqno_bytes)?;
        self.current_segment.write_all(&op_type_bytes)?;
        self.current_segment.write_all(&prev_hash)?;
        self.current_segment.write_all(&payload)?;
        self.current_segment.write_all(&crc.to_be_bytes())?;
        self.current_segment.flush()?;

        // Advance chain head
        self.chain_head = compute_record_hash(seqno, op_type, &prev_hash, &payload);

        // Batch fsync: sync_data() every 32 appends (fdatasync, cheaper than sync_all)
        self.ops_since_sync += 1;
        if self.ops_since_sync >= 32 {
            let _ = self.current_segment.get_ref().sync_data();
            self.ops_since_sync = 0;
        }

        // V2 entry: 4 + 8 + 1 + 32 + payload_len + 4
        let entry_size = 4 + 8 + 1 + 32 + payload.len() as u64 + 4;
        self.current_segment_size += entry_size;
        self.next_seqno += 1;

        Ok(seqno)
    }

    /// Force fsync — call after critical mutations (put_memory, forget, etc.)
    pub fn sync(&mut self) -> Result<()> {
        self.current_segment.flush()?;
        self.current_segment.get_ref().sync_data()?;
        self.ops_since_sync = 0;
        Ok(())
    }

    /// Replay ALL segment files in data_dir/segments/ (from all instances).
    /// Segments are sorted alphabetically — instance_id prefix ensures consistent ordering.
    /// Updates chain_head as V2 records are replayed.
    pub fn replay<F>(&mut self, _start_seqno: u64, mut f: F) -> Result<()>
    where
        F: FnMut(u64, Op) -> Result<()>,
    {
        let seg_dir = self.data_dir.join("segments");
        let segments = collect_all_segments(&seg_dir)?;
        let mut chain_head = ZERO_HASH;
        for seg_path in &segments {
            chain_head = replay_segment_chained(seg_path, 0, chain_head, &mut f)?;
        }
        self.chain_head = chain_head;
        Ok(())
    }

    /// Current chain tip hash. Zero if only V1 data exists.
    pub fn chain_head(&self) -> ChainHash {
        self.chain_head
    }

    /// Set chain_head externally (e.g. after caller-driven replay).
    pub fn set_chain_head(&mut self, h: ChainHash) {
        self.chain_head = h;
    }

    /// Flush the write buffer to the OS.
    pub fn flush_buf(&mut self) -> Result<()> {
        self.current_segment.flush()?;
        Ok(())
    }

    /// Return the seqno of the last op appended (0 if nothing written yet).
    pub fn last_seqno(&self) -> u64 {
        self.next_seqno.saturating_sub(1)
    }

    fn rotate_if_needed(&mut self) -> Result<()> {
        if self.current_segment_size < MAX_SEGMENT_SIZE {
            return Ok(());
        }
        self.current_segment.flush()?;
        let _ = self.current_segment.get_ref().sync_data(); // ensure old segment is durable before rotation
        let new_path = segment_path(&self.data_dir, self.instance_id, self.next_seqno);
        let f = create_segment_v2(&new_path, self.next_seqno, &self.chain_head)?;
        self.current_segment = BufWriter::new(f);
        self.current_segment_path = new_path;
        self.current_segment_size = V2_HEADER_SIZE as u64;
        Ok(())
    }
}

fn segment_path(data_dir: &Path, instance_id: InstanceId, first_seqno: u64) -> PathBuf {
    data_dir
        .join("segments")
        .join(format!("{:08x}_{:012}.seg", instance_id, first_seqno))
}

/// V1 header: magic(8) + first_seqno(8) = 16
const V1_HEADER_SIZE: usize = 16;
/// V2 header: magic(8) + first_seqno(8) + chain_head(32) = 48
const V2_HEADER_SIZE: usize = 48;

#[allow(dead_code)] // Retained for V1 backward compatibility
fn create_segment(path: &Path, first_seqno: u64) -> Result<File> {
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    f.write_all(SEGMENT_MAGIC)?;
    f.write_all(&first_seqno.to_be_bytes())?;
    f.sync_all()?;
    Ok(f)
}

fn create_segment_v2(path: &Path, first_seqno: u64, chain_head: &ChainHash) -> Result<File> {
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    f.write_all(SEGMENT_MAGIC_V2)?;
    f.write_all(&first_seqno.to_be_bytes())?;
    f.write_all(chain_head)?;
    f.sync_all()?;
    Ok(f)
}

/// Check if a segment file uses V2 format by reading the first 8 bytes.
fn is_v2_segment(path: &Path) -> bool {
    let mut f = match File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let mut magic = [0u8; 8];
    if f.read_exact(&mut magic).is_err() {
        return false;
    }
    &magic == SEGMENT_MAGIC_V2
}

/// Read the segment magic and return the version (1 or 2).
fn segment_version(magic: &[u8; 8]) -> Option<u8> {
    if magic == SEGMENT_MAGIC { Some(1) }
    else if magic == SEGMENT_MAGIC_V2 { Some(2) }
    else { None }
}

fn is_valid_segment_name(name: &str) -> bool {
    let Some(stem) = name.strip_suffix(".seg") else { return false; };
    let Some((instance_hex, seqno)) = stem.split_once('_') else { return false; };
    instance_hex.len() == 8
        && instance_hex.chars().all(|c| c.is_ascii_hexdigit())
        && seqno.len() == 12
        && seqno.chars().all(|c| c.is_ascii_digit())
}

/// All .seg files in directory matching the canonical naming pattern, sorted.
fn collect_all_segments(seg_dir: &Path) -> Result<Vec<PathBuf>> {
    if !seg_dir.exists() {
        return Ok(Vec::new());
    }
    let mut paths: Vec<PathBuf> = fs::read_dir(seg_dir)?
        .filter_map(|e| {
            let e = e.ok()?;
            let p = e.path();
            let name = p.file_name().and_then(|n| n.to_str())?;
            if is_valid_segment_name(name) {
                Some(p)
            } else {
                None
            }
        })
        .collect();
    paths.sort();
    Ok(paths)
}

/// Only .seg files belonging to this instance (by filename prefix).
fn collect_instance_segments(seg_dir: &Path, instance_id: InstanceId) -> Result<Vec<PathBuf>> {
    let prefix = format!("{:08x}_", instance_id);
    let all = collect_all_segments(seg_dir)?;
    Ok(all
        .into_iter()
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with(&prefix))
                .unwrap_or(false)
        })
        .collect())
}

/// Collect all segment files in `data_dir/segments/` not owned by `own_instance_id`.
pub fn collect_foreign_segments(
    data_dir: &Path,
    own_instance_id: InstanceId,
) -> Result<Vec<PathBuf>> {
    let seg_dir = data_dir.join("segments");
    let own_prefix = format!("{:08x}_", own_instance_id);
    let all = collect_all_segments(&seg_dir)?;
    Ok(all
        .into_iter()
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| !n.starts_with(&own_prefix))
                .unwrap_or(false)
        })
        .collect())
}

/// Replay a single segment file starting from `byte_offset`.
/// `byte_offset=0` reads from the start (parses the 16-byte header).
/// Non-zero offset seeks directly to that position (header already consumed).
/// Returns the byte offset past the last successfully decoded entry.
/// CRC mismatches or truncated reads at end-of-file are treated as
/// in-progress writes by a concurrent peer and simply stop the scan.
pub fn replay_from_offset<F>(path: &Path, byte_offset: u64, mut f: F) -> Result<u64>
where
    F: FnMut(u64, Op) -> Result<()>,
{
    use std::io::Seek;

    let mut file = std::fs::File::open(path)?;

    // Determine version from magic
    let (version, header_len) = if byte_offset == 0 {
        let mut magic = [0u8; 8];
        if file.read_exact(&mut magic).is_err() {
            return Ok(0);
        }
        let ver = segment_version(&magic).ok_or_else(|| {
            FieldError::Manifest(format!("bad segment magic in {}", path.display()))
        })?;
        let mut _buf = [0u8; 8];
        let _ = file.read_exact(&mut _buf);
        if ver == 2 {
            // Skip chain_head in V2 header
            let mut _chain = [0u8; 32];
            let _ = file.read_exact(&mut _chain);
            (ver, V2_HEADER_SIZE as u64)
        } else {
            (ver, V1_HEADER_SIZE as u64)
        }
    } else {
        // When resuming from offset, peek magic to know version
        let mut magic = [0u8; 8];
        if file.read_exact(&mut magic).is_err() {
            return Ok(byte_offset);
        }
        let ver = segment_version(&magic).unwrap_or(1);
        file.seek(std::io::SeekFrom::Start(byte_offset))?;
        (ver, byte_offset)
    };

    let mut cursor = if byte_offset == 0 { header_len } else { byte_offset };

    loop {
        let mut len_buf = [0u8; 4];
        match file.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(FieldError::Io(e)),
        }
        let payload_len = u32::from_be_bytes(len_buf) as usize;
        let prev_hash_size: u64 = if version == 2 { 32 } else { 0 };
        let entry_size = 4u64 + 8 + 1 + prev_hash_size + payload_len as u64 + 4;

        let mut seqno_buf = [0u8; 8];
        if file.read_exact(&mut seqno_buf).is_err() {
            break;
        }
        let seqno = u64::from_be_bytes(seqno_buf);

        let mut op_type_buf = [0u8; 1];
        if file.read_exact(&mut op_type_buf).is_err() {
            break;
        }

        // V2: read prev_hash
        let prev_hash = if version == 2 {
            let mut h = [0u8; 32];
            if file.read_exact(&mut h).is_err() { break; }
            h
        } else {
            ZERO_HASH
        };

        let mut payload = vec![0u8; payload_len];
        if file.read_exact(&mut payload).is_err() {
            break;
        }

        let mut crc_buf = [0u8; 4];
        if file.read_exact(&mut crc_buf).is_err() {
            break;
        }
        let stored_crc = u32::from_be_bytes(crc_buf);

        let mut hasher = CrcHasher::new();
        hasher.update(&seqno_buf);
        hasher.update(&op_type_buf);
        if version == 2 {
            hasher.update(&prev_hash);
        }
        hasher.update(&payload);
        if hasher.finalize() != stored_crc {
            // Partial write from a concurrent peer — stop here.
            break;
        }

        let op: Op = match rmp_serde::from_slice(&payload) {
            Ok(op) => op,
            Err(_) => break,
        };

        cursor += entry_size;
        f(seqno, op)?;
    }

    Ok(cursor)
}

/// Replay a segment, tracking chain hashes for V2 segments.
/// Returns the chain_head after replaying all records.
fn replay_segment_chained<F>(
    path: &Path,
    start_seqno: u64,
    mut chain_head: ChainHash,
    f: &mut F,
) -> Result<ChainHash>
where
    F: FnMut(u64, Op) -> Result<()>,
{
    let mut file = File::open(path)?;

    let mut magic = [0u8; 8];
    if file.read_exact(&mut magic).is_err() {
        return Ok(chain_head);
    }
    let version = segment_version(&magic).ok_or_else(|| {
        FieldError::Manifest(format!("bad segment magic in {}", path.display()))
    })?;

    let mut _first_seqno_buf = [0u8; 8];
    if file.read_exact(&mut _first_seqno_buf).is_err() {
        return Ok(chain_head);
    }

    // V2: read stored chain_head from header and verify continuity
    if version == 2 {
        let mut stored_head = [0u8; 32];
        if file.read_exact(&mut stored_head).is_err() {
            return Ok(chain_head);
        }
        // Verify segment continuity: stored head must match incoming chain_head
        if stored_head != chain_head {
            eprintln!(
                "[chitta-field] chain continuity warning in {}: expected {}, segment has {}",
                path.display(),
                &hex(&chain_head)[..16],
                &hex(&stored_head)[..16],
            );
        }
    }

    loop {
        let mut len_buf = [0u8; 4];
        match file.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(FieldError::Io(e)),
        }
        let payload_len = u32::from_be_bytes(len_buf) as usize;

        let mut seqno_buf = [0u8; 8];
        match file.read_exact(&mut seqno_buf) {
            Ok(()) => {}
            Err(_) => return Err(FieldError::TruncatedEntry { seqno: 0 }),
        }
        let seqno = u64::from_be_bytes(seqno_buf);

        let mut op_type_buf = [0u8; 1];
        if file.read_exact(&mut op_type_buf).is_err() {
            return Err(FieldError::TruncatedEntry { seqno });
        }
        let op_type = op_type_buf[0];

        // V2: read prev_hash from record
        let prev_hash = if version == 2 {
            let mut h = [0u8; 32];
            if file.read_exact(&mut h).is_err() {
                return Err(FieldError::TruncatedEntry { seqno });
            }
            h
        } else {
            ZERO_HASH
        };

        let mut payload = vec![0u8; payload_len];
        if file.read_exact(&mut payload).is_err() {
            return Err(FieldError::TruncatedEntry { seqno });
        }

        let mut crc_buf = [0u8; 4];
        if file.read_exact(&mut crc_buf).is_err() {
            return Err(FieldError::TruncatedEntry { seqno });
        }
        let stored_crc = u32::from_be_bytes(crc_buf);

        // CRC verification (V2 includes prev_hash in CRC)
        let mut hasher = CrcHasher::new();
        hasher.update(&seqno_buf);
        hasher.update(&op_type_buf);
        if version == 2 {
            hasher.update(&prev_hash);
        }
        hasher.update(&payload);
        let computed_crc = hasher.finalize();

        if computed_crc != stored_crc {
            return Err(FieldError::CrcMismatch {
                seqno,
                expected: stored_crc,
                actual: computed_crc,
            });
        }

        // V2: verify chain integrity
        if version == 2 {
            if prev_hash != chain_head {
                return Err(FieldError::ChainMismatch {
                    seqno,
                    expected: hex(&chain_head),
                    actual: hex(&prev_hash),
                });
            }
            chain_head = compute_record_hash(seqno, op_type, &prev_hash, &payload);
        }

        if seqno < start_seqno {
            continue;
        }

        let op: Op = rmp_serde::from_slice(&payload).map_err(|e| FieldError::CorruptLog {
            seqno,
            reason: format!("msgpack decode failed: {}", e),
        })?;

        if op_type_byte(&op) != op_type {
            return Err(FieldError::CorruptLog {
                seqno,
                reason: format!(
                    "op_type discriminant mismatch: stored {}, decoded {}",
                    op_type,
                    op_type_byte(&op)
                ),
            });
        }

        f(seqno, op)?;
    }
    Ok(chain_head)
}
