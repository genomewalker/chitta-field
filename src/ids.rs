use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};

pub type InstanceId = u32;

/// Generate a temporally-ordered instance ID.
/// High 20 bits = seconds since 2020-01-01 (sufficient until ~2053).
/// Low 12 bits = random suffix for uniqueness within the same second.
/// Alphabetical segment sort preserves temporal order across instances.
pub fn new_instance_id() -> InstanceId {
    use std::time::{SystemTime, UNIX_EPOCH};
    const EPOCH_2020: u64 = 1577836800; // 2020-01-01 00:00:00 UTC
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs().saturating_sub(EPOCH_2020))
        .unwrap_or(0);
    let ts_bits = (secs & 0xFFFFF) as u32; // 20-bit seconds

    let mut rand_bits = 0u32;
    use std::fs::File;
    use std::io::Read;
    let mut buf = [0u8; 2];
    if let Ok(mut f) = File::open("/dev/urandom") {
        let _ = f.read_exact(&mut buf);
        rand_bits = u16::from_le_bytes(buf) as u32 & 0xFFF; // 12-bit random
    }
    (ts_bits << 12) | rand_bits
}

pub type MemoryId = u64;
pub type SeqNo = u64;
pub type ChunkHash = [u8; 32];
pub type ArtifactId = u64;

/// Compute the canonical content-addressed hash for a memory payload.
/// Hashes kind + realm + content + raw embedding bytes in canonical order.
pub fn compute_chunk_hash(kind: &str, realm: &str, content: &[u8], embedding: &[f32]) -> ChunkHash {
    let mut hasher = Sha256::new();
    hasher.update(kind.as_bytes());
    hasher.update(realm.as_bytes());
    hasher.update(content);
    for f in embedding {
        hasher.update(f.to_le_bytes());
    }
    hasher.finalize().into()
}

pub struct MemoryIdAllocator {
    next: AtomicU64,
}

impl MemoryIdAllocator {
    pub fn new(start: u64) -> Self {
        Self {
            next: AtomicU64::new(start),
        }
    }

    /// Create an allocator partitioned for a specific instance.
    /// MemoryId = (instance_id as u64) << 32 | local_counter
    pub fn with_instance(instance_id: InstanceId) -> Self {
        Self {
            next: AtomicU64::new((instance_id as u64) << 32 | 1),
        }
    }

    pub fn next_id(&self) -> MemoryId {
        self.next.fetch_add(1, Ordering::Relaxed)
    }

    pub fn current(&self) -> u64 {
        self.next.load(Ordering::Relaxed)
    }

    pub fn reset_to(&self, val: u64) {
        self.next.store(val, Ordering::Relaxed);
    }
}

pub struct ArtifactIdAllocator {
    next: AtomicU64,
}

impl ArtifactIdAllocator {
    pub fn new(start: u64) -> Self {
        Self {
            next: AtomicU64::new(start),
        }
    }

    pub fn with_instance(instance_id: InstanceId) -> Self {
        Self {
            next: AtomicU64::new((instance_id as u64) << 32 | 1),
        }
    }

    pub fn next_id(&self) -> ArtifactId {
        self.next.fetch_add(1, Ordering::Relaxed)
    }

    pub fn current(&self) -> u64 {
        self.next.load(Ordering::Relaxed)
    }

    pub fn reset_to(&self, val: u64) {
        self.next.store(val, Ordering::Relaxed);
    }
}

pub struct TripletIdAllocator {
    next: AtomicU64,
}

impl TripletIdAllocator {
    pub fn new(start: u64) -> Self {
        Self {
            next: AtomicU64::new(start),
        }
    }

    pub fn next_id(&self) -> u64 {
        self.next.fetch_add(1, Ordering::Relaxed)
    }

    pub fn current(&self) -> u64 {
        self.next.load(Ordering::Relaxed)
    }
}
