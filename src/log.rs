use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use crc32fast::Hasher as CrcHasher;
use crate::error::{FieldError, Result};
use crate::ops::Op;

pub const SEGMENT_MAGIC: &[u8; 8] = b"CFLOG001";
pub const MAX_SEGMENT_SIZE: u64 = 256 * 1024 * 1024;

/// One byte discriminant for each Op variant, stored in the log entry header.
const OP_PUT_PAYLOAD: u8 = 0;
const OP_UPDATE_STATE: u8 = 1;
const OP_DELETE_MEMORY: u8 = 2;
const OP_ADD_ASSOC_EDGE: u8 = 3;
const OP_UPSERT_ARTIFACT: u8 = 4;
const OP_ADD_TRIPLET: u8 = 5;
const OP_INVALIDATE_TRIPLET: u8 = 6;

fn op_type_byte(op: &Op) -> u8 {
    match op {
        Op::PutPayload(_) => OP_PUT_PAYLOAD,
        Op::UpdateState(_) => OP_UPDATE_STATE,
        Op::DeleteMemory(_) => OP_DELETE_MEMORY,
        Op::AddAssocEdge(_) => OP_ADD_ASSOC_EDGE,
        Op::UpsertArtifact(_) => OP_UPSERT_ARTIFACT,
        Op::AddTriplet(_) => OP_ADD_TRIPLET,
        Op::InvalidateTriplet(_) => OP_INVALIDATE_TRIPLET,
    }
}

pub struct OpLog {
    data_dir: PathBuf,
    current_segment: BufWriter<File>,
    current_segment_path: PathBuf,
    current_segment_size: u64,
    next_seqno: u64,
}

impl OpLog {
    /// Open or create the op log in data_dir/segments/.
    /// next_seqno is the seqno to assign to the first new append.
    pub fn open(data_dir: &Path, next_seqno: u64) -> Result<Self> {
        let seg_dir = data_dir.join("segments");
        fs::create_dir_all(&seg_dir)?;

        // Find the last existing segment to continue writing into it,
        // or create a fresh one starting at next_seqno.
        let existing = collect_segment_paths(&seg_dir)?;

        if let Some(last_path) = existing.last() {
            let size = last_path.metadata()?.len();
            if size < MAX_SEGMENT_SIZE {
                // Reopen the last segment for appending.
                let f = OpenOptions::new()
                    .write(true)
                    .append(true)
                    .open(last_path)?;
                return Ok(Self {
                    data_dir: data_dir.to_path_buf(),
                    current_segment: BufWriter::new(f),
                    current_segment_path: last_path.clone(),
                    current_segment_size: size,
                    next_seqno,
                });
            }
        }

        // No usable segment — create a new one.
        let path = Self::segment_path(data_dir, next_seqno);
        let f = create_segment(&path, next_seqno)?;
        let header_size = (SEGMENT_MAGIC.len() + 8) as u64;
        Ok(Self {
            data_dir: data_dir.to_path_buf(),
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

        // Compute CRC over: seqno_bytes || op_type_byte || payload
        let mut hasher = CrcHasher::new();
        hasher.update(&seqno_bytes);
        hasher.update(&op_type_bytes);
        hasher.update(&payload);
        let crc = hasher.finalize();

        // Write entry: [u32 payload_len][u64 seqno][u8 op_type][payload][u32 crc]
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

    /// Replay all ops from start_seqno (inclusive), calling f for each (seqno, Op).
    /// Processes segments in sorted order. Returns on first corruption or EOF.
    pub fn replay<F>(&self, start_seqno: u64, mut f: F) -> Result<()>
    where
        F: FnMut(u64, Op) -> Result<()>,
    {
        let seg_dir = self.data_dir.join("segments");
        let segments = collect_segment_paths(&seg_dir)?;

        for seg_path in &segments {
            replay_segment(seg_path, start_seqno, &mut f)?;
        }

        Ok(())
    }

    /// Rotate to a new segment if the current one exceeds MAX_SEGMENT_SIZE.
    fn rotate_if_needed(&mut self) -> Result<()> {
        if self.current_segment_size < MAX_SEGMENT_SIZE {
            return Ok(());
        }

        // Flush and close the current segment.
        self.current_segment.flush()?;

        let new_path = Self::segment_path(&self.data_dir, self.next_seqno);
        let f = create_segment(&new_path, self.next_seqno)?;
        let header_size = (SEGMENT_MAGIC.len() + 8) as u64;

        self.current_segment = BufWriter::new(f);
        self.current_segment_path = new_path;
        self.current_segment_size = header_size;

        Ok(())
    }

    fn segment_path(data_dir: &Path, first_seqno: u64) -> PathBuf {
        data_dir
            .join("segments")
            .join(format!("{:012}.seg", first_seqno))
    }
}

/// Create a new segment file and write the magic header.
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

/// Return all .seg file paths in a directory, sorted lexicographically (which matches seqno order).
fn collect_segment_paths(seg_dir: &Path) -> Result<Vec<PathBuf>> {
    if !seg_dir.exists() {
        return Ok(Vec::new());
    }
    let mut paths: Vec<PathBuf> = fs::read_dir(seg_dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("seg") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    paths.sort();
    Ok(paths)
}

/// Read and replay one segment file, calling f for each entry with seqno >= start_seqno.
fn replay_segment<F>(path: &Path, start_seqno: u64, f: &mut F) -> Result<()>
where
    F: FnMut(u64, Op) -> Result<()>,
{
    let mut file = File::open(path)?;

    // Read and validate the magic header.
    let mut magic = [0u8; 8];
    if file.read_exact(&mut magic).is_err() {
        // Empty or truncated header — treat as empty segment.
        return Ok(());
    }
    if &magic != SEGMENT_MAGIC {
        return Err(FieldError::Manifest(format!(
            "bad segment magic in {}",
            path.display()
        )));
    }

    // Read first_seqno from header (not used for filtering but consumed).
    let mut _first_seqno_buf = [0u8; 8];
    if file.read_exact(&mut _first_seqno_buf).is_err() {
        return Ok(());
    }

    loop {
        // Read payload_len (4 bytes).
        let mut len_buf = [0u8; 4];
        match file.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(FieldError::Io(e)),
        }
        let payload_len = u32::from_be_bytes(len_buf) as usize;

        // Read seqno (8 bytes).
        let mut seqno_buf = [0u8; 8];
        match file.read_exact(&mut seqno_buf) {
            Ok(()) => {}
            Err(_) => {
                // We already read payload_len so the file is truncated mid-entry.
                // seqno is unknown at this point — use 0 as sentinel.
                return Err(FieldError::TruncatedEntry { seqno: 0 });
            }
        }
        let seqno = u64::from_be_bytes(seqno_buf);

        // Read op_type (1 byte).
        let mut op_type_buf = [0u8; 1];
        if file.read_exact(&mut op_type_buf).is_err() {
            return Err(FieldError::TruncatedEntry { seqno });
        }
        let op_type = op_type_buf[0];

        // Read payload.
        let mut payload = vec![0u8; payload_len];
        if file.read_exact(&mut payload).is_err() {
            return Err(FieldError::TruncatedEntry { seqno });
        }

        // Read CRC (4 bytes).
        let mut crc_buf = [0u8; 4];
        if file.read_exact(&mut crc_buf).is_err() {
            return Err(FieldError::TruncatedEntry { seqno });
        }
        let stored_crc = u32::from_be_bytes(crc_buf);

        // Verify CRC.
        let mut hasher = CrcHasher::new();
        hasher.update(&seqno_buf);
        hasher.update(&op_type_buf);
        hasher.update(&payload);
        let computed_crc = hasher.finalize();

        if computed_crc != stored_crc {
            return Err(FieldError::CrcMismatch {
                seqno,
                expected: stored_crc,
                actual: computed_crc,
            });
        }

        // Skip entries before start_seqno.
        if seqno < start_seqno {
            continue;
        }

        // Deserialize the Op.
        let op: Op = rmp_serde::from_slice(&payload)
            .map_err(|e| FieldError::CorruptLog {
                seqno,
                reason: format!("msgpack decode failed: {}", e),
            })?;

        // Validate that the stored op_type matches the deserialized variant.
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

    Ok(())
}
