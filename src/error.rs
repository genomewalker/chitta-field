use thiserror::Error;

#[derive(Debug, Error)]
pub enum FieldError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Corrupt op log at seqno {seqno}: {reason}")]
    CorruptLog { seqno: u64, reason: String },

    #[error("CRC mismatch at seqno {seqno}: expected {expected:#010x}, got {actual:#010x}")]
    CrcMismatch {
        seqno: u64,
        expected: u32,
        actual: u32,
    },

    #[error("Truncated entry at seqno {seqno}")]
    TruncatedEntry { seqno: u64 },

    #[error("Memory {0} not found")]
    NotFound(u64),

    #[error("Memory {0} is deleted")]
    Deleted(u64),

    #[error("Manifest error: {0}")]
    Manifest(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    InvalidEmbedDim { expected: usize, actual: usize },

    #[error("Store is locked by another process")]
    Locked,

    #[error("Segment full")]
    SegmentFull,
}

pub type Result<T> = std::result::Result<T, FieldError>;
