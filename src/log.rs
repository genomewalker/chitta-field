use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use crc32fast::Hasher as CrcHasher;
use crate::error::{FieldError, Result};
use crate::ids::InstanceId;
use crate::ops::Op;

pub const SEGMENT_MAGIC: &[u8; 8] = b"CFLOG001";
pub const MAX_SEGMENT_SIZE: u64 = 256 * 1024 * 1024;

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
use crate::ops::{OP_SESSION_EVENT, OP_TRANSCRIPT_EVENT, OP_TASK_EVENT, OP_USER_MODEL_EVENT, OP_THEME_EVENT, OP_ANALYTICS_EVENT, OP_CLEAR_PROJECT, OP_UPDATE_SYMBOL_DESCRIPTION, OP_UPDATE_MEMORY_CONTENT};

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
    }
}

pub struct OpLog {
    data_dir: PathBuf,
    instance_id: InstanceId,
    current_segment: BufWriter<File>,
    current_segment_path: PathBuf,
    current_segment_size: u64,
    next_seqno: u64,
}

impl OpLog {
    /// Open the write log for a specific instance.
    /// Finds this instance's last segment to continue appending, or creates a new one.
    pub fn open(data_dir: &Path, instance_id: InstanceId, next_seqno: u64) -> Result<Self> {
        let seg_dir = data_dir.join("segments");
        fs::create_dir_all(&seg_dir)?;

        let existing = collect_instance_segments(&seg_dir, instance_id)?;

        if let Some(last_path) = existing.last() {
            let size = last_path.metadata()?.len();
            if size < MAX_SEGMENT_SIZE {
                let f = OpenOptions::new().write(true).append(true).open(last_path)?;
                return Ok(Self {
                    data_dir: data_dir.to_path_buf(),
                    instance_id,
                    current_segment: BufWriter::new(f),
                    current_segment_path: last_path.clone(),
                    current_segment_size: size,
                    next_seqno,
                });
            }
        }

        let path = segment_path(data_dir, instance_id, next_seqno);
        let f = create_segment(&path, next_seqno)?;
        let header_size = (SEGMENT_MAGIC.len() + 8) as u64;
        Ok(Self {
            data_dir: data_dir.to_path_buf(),
            instance_id,
            current_segment: BufWriter::new(f),
            current_segment_path: path,
            current_segment_size: header_size,
            next_seqno,
        })
    }

    /// Append an op, returning its assigned seqno.
    pub fn append(&mut self, op: &Op) -> Result<u64> {
        self.rotate_if_needed()?;

        let seqno = self.next_seqno;
        let op_type = op_type_byte(op);
        let payload = rmp_serde::to_vec(op)
            .map_err(|e| FieldError::Serialization(e.to_string()))?;

        let payload_len = payload.len() as u32;
        let seqno_bytes = seqno.to_be_bytes();
        let op_type_bytes = [op_type];

        let mut hasher = CrcHasher::new();
        hasher.update(&seqno_bytes);
        hasher.update(&op_type_bytes);
        hasher.update(&payload);
        let crc = hasher.finalize();

        self.current_segment.write_all(&payload_len.to_be_bytes())?;
        self.current_segment.write_all(&seqno_bytes)?;
        self.current_segment.write_all(&op_type_bytes)?;
        self.current_segment.write_all(&payload)?;
        self.current_segment.write_all(&crc.to_be_bytes())?;
        self.current_segment.flush()?;

        let entry_size = 4 + 8 + 1 + payload.len() as u64 + 4;
        self.current_segment_size += entry_size;
        self.next_seqno += 1;

        Ok(seqno)
    }

    /// Replay ALL segment files in data_dir/segments/ (from all instances).
    /// Segments are sorted alphabetically — instance_id prefix ensures consistent ordering.
    pub fn replay<F>(&self, _start_seqno: u64, mut f: F) -> Result<()>
    where
        F: FnMut(u64, Op) -> Result<()>,
    {
        let seg_dir = self.data_dir.join("segments");
        let segments = collect_all_segments(&seg_dir)?;
        for seg_path in &segments {
            replay_segment(seg_path, 0, &mut f)?;
        }
        Ok(())
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
        let new_path = segment_path(&self.data_dir, self.instance_id, self.next_seqno);
        let f = create_segment(&new_path, self.next_seqno)?;
        let header_size = (SEGMENT_MAGIC.len() + 8) as u64;
        self.current_segment = BufWriter::new(f);
        self.current_segment_path = new_path;
        self.current_segment_size = header_size;
        Ok(())
    }
}

fn segment_path(data_dir: &Path, instance_id: InstanceId, first_seqno: u64) -> PathBuf {
    data_dir
        .join("segments")
        .join(format!("{:08x}_{:012}.seg", instance_id, first_seqno))
}

fn create_segment(path: &Path, first_seqno: u64) -> Result<File> {
    let mut f = OpenOptions::new().write(true).create(true).truncate(true).open(path)?;
    f.write_all(SEGMENT_MAGIC)?;
    f.write_all(&first_seqno.to_be_bytes())?;
    f.sync_all()?;
    Ok(f)
}

fn is_valid_segment_name(name: &str) -> bool {
    let Some(stem) = name.strip_suffix(".seg") else { return false; };
    let Some((instance_hex, seqno)) = stem.split_once('_') else { return false; };
    instance_hex.len() == 8 && instance_hex.chars().all(|c| c.is_ascii_hexdigit())
        && seqno.len() == 12 && seqno.chars().all(|c| c.is_ascii_digit())
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
            if is_valid_segment_name(name) { Some(p) } else { None }
        })
        .collect();
    paths.sort();
    Ok(paths)
}

/// Only .seg files belonging to this instance (by filename prefix).
fn collect_instance_segments(seg_dir: &Path, instance_id: InstanceId) -> Result<Vec<PathBuf>> {
    let prefix = format!("{:08x}_", instance_id);
    let all = collect_all_segments(seg_dir)?;
    Ok(all.into_iter().filter(|p| {
        p.file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.starts_with(&prefix))
            .unwrap_or(false)
    }).collect())
}

/// Collect all segment files in `data_dir/segments/` not owned by `own_instance_id`.
pub fn collect_foreign_segments(data_dir: &Path, own_instance_id: InstanceId) -> Result<Vec<PathBuf>> {
    let seg_dir = data_dir.join("segments");
    let own_prefix = format!("{:08x}_", own_instance_id);
    let all = collect_all_segments(&seg_dir)?;
    Ok(all.into_iter()
        .filter(|p| p.file_name()
            .and_then(|n| n.to_str())
            .map(|n| !n.starts_with(&own_prefix))
            .unwrap_or(false))
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
    let header_len: u64 = (SEGMENT_MAGIC.len() + 8) as u64;

    let mut cursor = if byte_offset == 0 {
        let mut magic = [0u8; 8];
        if file.read_exact(&mut magic).is_err() { return Ok(0); }
        if &magic != SEGMENT_MAGIC {
            return Err(FieldError::Manifest(format!("bad segment magic in {}", path.display())));
        }
        let mut _buf = [0u8; 8];
        let _ = file.read_exact(&mut _buf);
        header_len
    } else {
        file.seek(std::io::SeekFrom::Start(byte_offset))?;
        byte_offset
    };

    loop {
        let mut len_buf = [0u8; 4];
        match file.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(FieldError::Io(e)),
        }
        let payload_len = u32::from_be_bytes(len_buf) as usize;
        let entry_size = 4u64 + 8 + 1 + payload_len as u64 + 4;

        let mut seqno_buf = [0u8; 8];
        if file.read_exact(&mut seqno_buf).is_err() { break; }
        let seqno = u64::from_be_bytes(seqno_buf);

        let mut op_type_buf = [0u8; 1];
        if file.read_exact(&mut op_type_buf).is_err() { break; }

        let mut payload = vec![0u8; payload_len];
        if file.read_exact(&mut payload).is_err() { break; }

        let mut crc_buf = [0u8; 4];
        if file.read_exact(&mut crc_buf).is_err() { break; }
        let stored_crc = u32::from_be_bytes(crc_buf);

        let mut hasher = CrcHasher::new();
        hasher.update(&seqno_buf);
        hasher.update(&op_type_buf);
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

fn replay_segment<F>(path: &Path, start_seqno: u64, f: &mut F) -> Result<()>
where
    F: FnMut(u64, Op) -> Result<()>,
{
    let mut file = File::open(path)?;

    let mut magic = [0u8; 8];
    if file.read_exact(&mut magic).is_err() { return Ok(()); }
    if &magic != SEGMENT_MAGIC {
        return Err(FieldError::Manifest(format!("bad segment magic in {}", path.display())));
    }

    let mut _first_seqno_buf = [0u8; 8];
    if file.read_exact(&mut _first_seqno_buf).is_err() { return Ok(()); }

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

        let mut payload = vec![0u8; payload_len];
        if file.read_exact(&mut payload).is_err() {
            return Err(FieldError::TruncatedEntry { seqno });
        }

        let mut crc_buf = [0u8; 4];
        if file.read_exact(&mut crc_buf).is_err() {
            return Err(FieldError::TruncatedEntry { seqno });
        }
        let stored_crc = u32::from_be_bytes(crc_buf);

        let mut hasher = CrcHasher::new();
        hasher.update(&seqno_buf);
        hasher.update(&op_type_buf);
        hasher.update(&payload);
        let computed_crc = hasher.finalize();

        if computed_crc != stored_crc {
            return Err(FieldError::CrcMismatch { seqno, expected: stored_crc, actual: computed_crc });
        }

        if seqno < start_seqno { continue; }

        let op: Op = rmp_serde::from_slice(&payload)
            .map_err(|e| FieldError::CorruptLog { seqno, reason: format!("msgpack decode failed: {}", e) })?;

        if op_type_byte(&op) != op_type {
            return Err(FieldError::CorruptLog {
                seqno,
                reason: format!("op_type discriminant mismatch: stored {}, decoded {}", op_type, op_type_byte(&op)),
            });
        }

        f(seqno, op)?;
    }
    Ok(())
}
