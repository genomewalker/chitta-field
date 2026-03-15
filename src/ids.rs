use sha2::{Sha256, Digest};
use std::sync::atomic::{AtomicU64, Ordering};

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
        Self { next: AtomicU64::new(start) }
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
        Self { next: AtomicU64::new(start) }
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
        Self { next: AtomicU64::new(start) }
    }

    pub fn next_id(&self) -> u64 {
        self.next.fetch_add(1, Ordering::Relaxed)
    }

    pub fn current(&self) -> u64 {
        self.next.load(Ordering::Relaxed)
    }
}
