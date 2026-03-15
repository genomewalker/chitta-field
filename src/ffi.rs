//! C FFI for chitta-field.
//! Uses typed POD structs for hot-path calls. No JSON in recall path.
//! All functions return 0 on success, negative on error.
//! Errors readable via cf_last_error().

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;
use crate::field::ChittaField;
use crate::ops::EdgeType;
use crate::recall::RecallHit;

/// Opaque handle. C code holds *mut CfHandle.
pub struct CfHandle {
    field: ChittaField,
    last_error: Option<CString>,
}

impl CfHandle {
    fn ok(&mut self) -> c_int {
        self.last_error = None;
        0
    }
    fn err(&mut self, e: impl std::fmt::Display) -> c_int {
        self.last_error = CString::new(e.to_string()).ok();
        -1
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn edge_type_from_u8(v: u8) -> EdgeType {
    match v {
        0 => EdgeType::DerivedFrom,
        1 => EdgeType::SameSession,
        2 => EdgeType::SameArtifact,
        3 => EdgeType::CoRetrieved,
        4 => EdgeType::Supports,
        _ => EdgeType::Contradicts,
    }
}

fn write_hits(hits: Vec<RecallHit>, buf: *mut CfRecallHit, cap: usize, written: *mut usize) {
    let n = hits.len().min(cap);
    for (i, h) in hits.iter().take(n).enumerate() {
        unsafe {
            *buf.add(i) = CfRecallHit {
                memory_id: h.memory_id,
                score: h.score,
                semantic_score: h.semantic_score,
                ts_ms: h.ts_ms,
                strength: h.strength,
                confidence: h.confidence,
            };
        }
    }
    unsafe { *written = n; }
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn cf_open(
    data_dir: *const c_char,
    _lock_dir: *const c_char,
) -> *mut CfHandle {
    // lock_dir is ignored — the Upanishads model needs no locks.
    let data_dir = unsafe {
        match CStr::from_ptr(data_dir).to_str() {
            Ok(s) => PathBuf::from(s),
            Err(_) => return std::ptr::null_mut(),
        }
    };
    match ChittaField::open(data_dir) {
        Ok(field) => Box::into_raw(Box::new(CfHandle { field, last_error: None })),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn cf_close(h: *mut CfHandle) {
    if !h.is_null() {
        unsafe { drop(Box::from_raw(h)) };
    }
}

#[no_mangle]
pub extern "C" fn cf_last_error(h: *const CfHandle) -> *const c_char {
    if h.is_null() {
        return std::ptr::null();
    }
    unsafe {
        (*h).last_error
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(std::ptr::null())
    }
}

// ── Write operations ──────────────────────────────────────────────────────────

/// Store a new memory. Returns MemoryId via out_memory_id (must be non-null).
/// embedding_ptr/embedding_len: pointer to f32 array, len must be 768.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn cf_put_memory(
    h: *mut CfHandle,
    kind: *const c_char,
    realm: *const c_char,
    content_ptr: *const u8,
    content_len: usize,
    embedding_ptr: *const f32,
    embedding_len: usize,
    confidence: f32,
    decay_rate: f32,
    authored_at_ms: i64,
    out_memory_id: *mut u64,
) -> c_int {
    if h.is_null() || out_memory_id.is_null() {
        return -1;
    }
    let handle = unsafe { &mut *h };

    let kind_str = unsafe {
        match CStr::from_ptr(kind).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let realm_str = unsafe {
        match CStr::from_ptr(realm).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let content = unsafe { std::slice::from_raw_parts(content_ptr, content_len) };
    let embedding = unsafe { std::slice::from_raw_parts(embedding_ptr, embedding_len) };

    match handle.field.put_memory(
        kind_str,
        realm_str,
        content,
        embedding,
        confidence,
        decay_rate,
        authored_at_ms,
        vec![],
        None,
        None,
    ) {
        Ok((memory_id, _)) => {
            unsafe { *out_memory_id = memory_id; }
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

/// Update mutable state of a memory.
/// Pass NaN for deltas you don't want to apply (use f32::NAN as sentinel).
#[no_mangle]
pub extern "C" fn cf_update_state(
    h: *mut CfHandle,
    memory_id: u64,
    strength_delta: f32,
    confidence_delta: f32,
    decay_rate: f32,
    touch: u8,
    pin: i8,
) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &mut *h };

    let strength_delta = if strength_delta.is_nan() { None } else { Some(strength_delta) };
    let confidence_delta = if confidence_delta.is_nan() { None } else { Some(confidence_delta) };
    let decay_rate = if decay_rate.is_nan() { None } else { Some(decay_rate) };
    let pin_opt = match pin {
        -1 => None,
        0 => Some(false),
        _ => Some(true),
    };

    match handle.field.update_state(memory_id, strength_delta, confidence_delta, decay_rate, touch != 0, pin_opt) {
        Ok(()) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

/// Soft-delete a memory.
#[no_mangle]
pub extern "C" fn cf_forget(h: *mut CfHandle, memory_id: u64) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &mut *h };
    match handle.field.forget(memory_id) {
        Ok(()) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

/// Add an association edge between two memories.
/// edge_type: 0=DerivedFrom, 1=SameSession, 2=SameArtifact, 3=CoRetrieved, 4=Supports, 5=Contradicts
#[no_mangle]
pub extern "C" fn cf_add_assoc_edge(
    h: *mut CfHandle,
    src: u64,
    dst: u64,
    edge_type: u8,
    weight: f32,
) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &mut *h };
    let et = edge_type_from_u8(edge_type);
    match handle.field.add_assoc_edge(src, dst, et, weight) {
        Ok(()) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

/// Register a file artifact, returns its ArtifactId via out_artifact_id.
#[no_mangle]
pub extern "C" fn cf_upsert_artifact(
    h: *mut CfHandle,
    normalized_path: *const c_char,
    out_artifact_id: *mut u64,
) -> c_int {
    if h.is_null() || out_artifact_id.is_null() {
        return -1;
    }
    let handle = unsafe { &mut *h };
    let path_str = unsafe {
        match CStr::from_ptr(normalized_path).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    match handle.field.upsert_artifact(path_str, None) {
        Ok(artifact_id) => {
            unsafe { *out_artifact_id = artifact_id; }
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

// ── Read operations ───────────────────────────────────────────────────────────

/// A single recall result. Layout must match the C header exactly.
#[repr(C)]
#[derive(Clone)]
pub struct CfRecallHit {
    pub memory_id: u64,
    pub score: f32,
    pub semantic_score: f32,
    pub ts_ms: i64,
    pub strength: f32,
    pub confidence: f32,
}

/// Output buffer for recall results. Caller allocates hits_buf with capacity hits_cap.
/// On return, *hits_written contains number of results written.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn cf_recall_semantic(
    h: *mut CfHandle,
    query_embedding: *const f32,
    embedding_len: usize,
    realm: *const c_char,
    k: usize,
    hits_buf: *mut CfRecallHit,
    hits_cap: usize,
    hits_written: *mut usize,
) -> c_int {
    if h.is_null() || hits_buf.is_null() || hits_written.is_null() {
        return -1;
    }
    let handle = unsafe { &mut *h };
    let embedding = unsafe { std::slice::from_raw_parts(query_embedding, embedding_len) };
    let realm_str = if realm.is_null() {
        None
    } else {
        unsafe {
            match CStr::from_ptr(realm).to_str() {
                Ok(s) => Some(s),
                Err(e) => return handle.err(e),
            }
        }
    };

    match handle.field.recall_semantic(embedding, k, realm_str) {
        Ok(hits) => {
            write_hits(hits, hits_buf, hits_cap, hits_written);
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

#[no_mangle]
pub extern "C" fn cf_recall_temporal(
    h: *mut CfHandle,
    start_ms: i64,
    end_ms: i64,
    realm: *const c_char,
    limit: usize,
    hits_buf: *mut CfRecallHit,
    hits_cap: usize,
    hits_written: *mut usize,
) -> c_int {
    if h.is_null() || hits_buf.is_null() || hits_written.is_null() {
        return -1;
    }
    let handle = unsafe { &mut *h };
    let realm_str = if realm.is_null() {
        None
    } else {
        unsafe {
            match CStr::from_ptr(realm).to_str() {
                Ok(s) => Some(s),
                Err(e) => return handle.err(e),
            }
        }
    };

    match handle.field.recall_temporal(start_ms, end_ms, realm_str, limit) {
        Ok(hits) => {
            write_hits(hits, hits_buf, hits_cap, hits_written);
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

#[no_mangle]
pub extern "C" fn cf_recall_artifact(
    h: *mut CfHandle,
    normalized_path: *const c_char,
    limit: usize,
    hits_buf: *mut CfRecallHit,
    hits_cap: usize,
    hits_written: *mut usize,
) -> c_int {
    if h.is_null() || normalized_path.is_null() || hits_buf.is_null() || hits_written.is_null() {
        return -1;
    }
    let handle = unsafe { &mut *h };
    let path_str = unsafe {
        match CStr::from_ptr(normalized_path).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };

    match handle.field.recall_artifact(path_str, limit) {
        Ok(hits) => {
            write_hits(hits, hits_buf, hits_cap, hits_written);
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

#[no_mangle]
pub extern "C" fn cf_expand_associations(
    h: *mut CfHandle,
    seed_ids: *const u64,
    seed_count: usize,
    max_hops: usize,
    limit: usize,
    hits_buf: *mut CfRecallHit,
    hits_cap: usize,
    hits_written: *mut usize,
) -> c_int {
    if h.is_null() || seed_ids.is_null() || hits_buf.is_null() || hits_written.is_null() {
        return -1;
    }
    let handle = unsafe { &mut *h };
    let seeds = unsafe { std::slice::from_raw_parts(seed_ids, seed_count) };

    match handle.field.expand_associations(seeds, max_hops, limit) {
        Ok(hits) => {
            write_hits(hits, hits_buf, hits_cap, hits_written);
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

/// Get payload content for a memory. Writes UTF-8 content into buf (null-terminated).
/// Returns 0 on success, -1 if not found/deleted, -2 if buf too small.
#[no_mangle]
pub extern "C" fn cf_get_content(
    h: *mut CfHandle,
    memory_id: u64,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &mut *h };

    match handle.field.get_memory(memory_id) {
        Ok(payload) => {
            let content = &payload.content;
            if content.len() > buf_cap {
                return -2;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(content.as_ptr(), buf, content.len());
                *written = content.len();
            }
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

/// Get kind string for a memory into buf.
#[no_mangle]
pub extern "C" fn cf_get_kind(
    h: *mut CfHandle,
    memory_id: u64,
    buf: *mut u8,
    buf_cap: usize,
) -> c_int {
    if h.is_null() || buf.is_null() {
        return -1;
    }
    let handle = unsafe { &mut *h };

    match handle.field.get_memory(memory_id) {
        Ok(payload) => {
            let bytes = payload.kind.as_bytes();
            if bytes.len() >= buf_cap {
                return -2;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len());
                *buf.add(bytes.len()) = 0;
            }
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

/// Get realm string for a memory into buf.
#[no_mangle]
pub extern "C" fn cf_get_realm(
    h: *mut CfHandle,
    memory_id: u64,
    buf: *mut u8,
    buf_cap: usize,
) -> c_int {
    if h.is_null() || buf.is_null() {
        return -1;
    }
    let handle = unsafe { &mut *h };

    match handle.field.get_memory(memory_id) {
        Ok(payload) => {
            let bytes = payload.realm.as_bytes();
            if bytes.len() >= buf_cap {
                return -2;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len());
                *buf.add(bytes.len()) = 0;
            }
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

#[no_mangle]
pub extern "C" fn cf_recall_keyword(
    h: *mut CfHandle,
    query: *const c_char,
    k: usize,
    hits_buf: *mut CfRecallHit,
    hits_cap: usize,
    hits_written: *mut usize,
) -> c_int {
    if h.is_null() || query.is_null() || hits_buf.is_null() || hits_written.is_null() {
        return -1;
    }
    let handle = unsafe { &mut *h };
    let query_str = unsafe {
        match CStr::from_ptr(query).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };

    match handle.field.recall_keyword(query_str, k) {
        Ok(hits) => {
            write_hits(hits, hits_buf, hits_cap, hits_written);
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

// ── Triplet operations ────────────────────────────────────────────────────────

/// Add a triplet fact. Returns triplet_id via out_triplet_id.
/// source_memory_id: pass 0 for no source memory.
#[no_mangle]
pub extern "C" fn cf_add_triplet(
    h: *mut CfHandle,
    subject: *const c_char,
    predicate: *const c_char,
    object: *const c_char,
    weight: f32,
    source_memory_id: u64,
    out_triplet_id: *mut u64,
) -> c_int {
    if h.is_null() || subject.is_null() || predicate.is_null() || object.is_null() || out_triplet_id.is_null() {
        return -1;
    }
    let handle = unsafe { &mut *h };

    let subject_str = unsafe {
        match CStr::from_ptr(subject).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let predicate_str = unsafe {
        match CStr::from_ptr(predicate).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let object_str = unsafe {
        match CStr::from_ptr(object).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };

    let src_mem = if source_memory_id == 0 { None } else { Some(source_memory_id) };

    match handle.field.add_triplet(
        subject_str.to_string(),
        predicate_str.to_string(),
        object_str.to_string(),
        weight,
        src_mem,
        None,
    ) {
        Ok(triplet_id) => {
            unsafe { *out_triplet_id = triplet_id; }
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

/// Invalidate a triplet.
#[no_mangle]
pub extern "C" fn cf_invalidate_triplet(h: *mut CfHandle, triplet_id: u64) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &mut *h };
    match handle.field.invalidate_triplet(triplet_id) {
        Ok(()) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

fn write_triplets_json(
    entries: Vec<crate::organ::triplet::TripletEntry>,
    buf: *mut c_char,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    use serde_json::json;

    let json_val: Vec<_> = entries.iter().map(|e| json!({
        "id": e.id,
        "subject": e.subject,
        "predicate": e.predicate,
        "object": e.object,
        "weight": e.weight,
    })).collect();

    let s = match serde_json::to_string(&json_val) {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let bytes = s.as_bytes();
    // +1 for null terminator
    if bytes.len() + 1 > buf_cap {
        return -2;
    }

    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, bytes.len());
        *(buf as *mut u8).add(bytes.len()) = 0;
        *written = bytes.len();
    }

    0
}

/// Query by subject. Writes results as null-terminated JSON into buf.
/// JSON format: [{"id":1,"subject":"...","predicate":"...","object":"...","weight":0.9}]
/// Returns 0 on success, -1 on error, -2 if buf too small.
#[no_mangle]
pub extern "C" fn cf_query_subject(
    h: *mut CfHandle,
    subject: *const c_char,
    buf: *mut c_char,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || subject.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &mut *h };
    let subject_str = unsafe {
        match CStr::from_ptr(subject).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };

    match handle.field.query_subject(subject_str) {
        Ok(entries) => write_triplets_json(entries, buf, buf_cap, written),
        Err(e) => handle.err(e),
    }
}

/// Query by object. Writes results as null-terminated JSON into buf.
#[no_mangle]
pub extern "C" fn cf_query_object(
    h: *mut CfHandle,
    object: *const c_char,
    buf: *mut c_char,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || object.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &mut *h };
    let object_str = unsafe {
        match CStr::from_ptr(object).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };

    match handle.field.query_object(object_str) {
        Ok(entries) => write_triplets_json(entries, buf, buf_cap, written),
        Err(e) => handle.err(e),
    }
}

/// Query by entity (subject OR object). Writes results as null-terminated JSON into buf.
#[no_mangle]
pub extern "C" fn cf_query_entity(
    h: *mut CfHandle,
    entity: *const c_char,
    buf: *mut c_char,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || entity.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &mut *h };
    let entity_str = unsafe {
        match CStr::from_ptr(entity).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };

    match handle.field.query_entity(entity_str) {
        Ok(entries) => write_triplets_json(entries, buf, buf_cap, written),
        Err(e) => handle.err(e),
    }
}

// ── Learner operations ────────────────────────────────────────────────────────

/// Apply feedback reward to a pending recall episode (route learning).
/// episode_id: returned by cf_select_route. reward: 0.0 = bad, 1.0 = perfect.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn cf_feedback(h: *mut CfHandle, episode_id: u64, reward: f32) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &mut *h };
    match handle.field.feedback(episode_id, reward) {
        Ok(()) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

/// Get recommended context window size for a session type.
/// session_type: null-terminated string (e.g., "code", "general").
/// Returns the recommended window size, or 10 on error.
#[no_mangle]
pub extern "C" fn cf_recommended_window(h: *mut CfHandle, session_type: *const c_char) -> usize {
    if h.is_null() || session_type.is_null() {
        return 10;
    }
    let handle = unsafe { &*h };
    let session_str = unsafe {
        match CStr::from_ptr(session_type).to_str() {
            Ok(s) => s,
            Err(_) => return 10,
        }
    };
    handle.field.recommended_window(session_str)
}

// ── Maintenance ───────────────────────────────────────────────────────────────

/// Flush write buffer to OS.
#[no_mangle]
pub extern "C" fn cf_flush(h: *mut CfHandle) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &mut *h };
    match handle.field.flush() {
        Ok(()) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

/// Get stats.
#[no_mangle]
pub extern "C" fn cf_memory_count(h: *const CfHandle) -> usize {
    if h.is_null() {
        return 0;
    }
    unsafe { (*h).field.memory_count() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use tempfile::TempDir;

    unsafe fn open_tmp() -> (*mut CfHandle, TempDir) {
        let tmp = TempDir::new().unwrap();
        let data = CString::new(tmp.path().join("data").to_str().unwrap()).unwrap();
        let lock = CString::new(tmp.path().join("lock").to_str().unwrap()).unwrap();
        let h = cf_open(data.as_ptr(), lock.as_ptr());
        assert!(!h.is_null());
        (h, tmp)
    }

    #[test]
    fn test_ffi_put_recall() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let kind = CString::new("wisdom").unwrap();
            let realm = CString::new("test").unwrap();
            let content = b"ffi test memory";
            let embedding = vec![0.1f32; 768];
            let mut id: u64 = 0;

            let r = cf_put_memory(h,
                kind.as_ptr(), realm.as_ptr(),
                content.as_ptr(), content.len(),
                embedding.as_ptr(), embedding.len(),
                0.9, 0.001, 0,
                &mut id,
            );
            assert_eq!(r, 0);
            assert!(id > 0);

            // recall it back
            let mut hits = vec![CfRecallHit { memory_id: 0, score: 0.0, semantic_score: 0.0, ts_ms: 0, strength: 0.0, confidence: 0.0 }; 10];
            let mut written: usize = 0;
            let r = cf_recall_semantic(h,
                embedding.as_ptr(), embedding.len(),
                realm.as_ptr(), 5,
                hits.as_mut_ptr(), hits.len(), &mut written,
            );
            assert_eq!(r, 0);
            assert_eq!(written, 1);
            assert_eq!(hits[0].memory_id, id);

            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_forget() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let kind = CString::new("episode").unwrap();
            let realm = CString::new("test").unwrap();
            let content = b"to forget";
            let embedding = vec![0.5f32; 768];
            let mut id: u64 = 0;
            cf_put_memory(h, kind.as_ptr(), realm.as_ptr(),
                content.as_ptr(), content.len(),
                embedding.as_ptr(), embedding.len(),
                1.0, 0.001, 0, &mut id);

            let r = cf_forget(h, id);
            assert_eq!(r, 0);

            // should not appear in recall
            let mut hits = vec![CfRecallHit { memory_id: 0, score: 0.0, semantic_score: 0.0, ts_ms: 0, strength: 0.0, confidence: 0.0 }; 10];
            let mut written: usize = 0;
            cf_recall_semantic(h, embedding.as_ptr(), embedding.len(),
                std::ptr::null(), 10,
                hits.as_mut_ptr(), hits.len(), &mut written);
            assert_eq!(written, 0);

            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_get_content() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let kind = CString::new("correction").unwrap();
            let realm = CString::new("r").unwrap();
            let content = b"hello from ffi";
            let embedding = vec![0.3f32; 768];
            let mut id: u64 = 0;
            cf_put_memory(h, kind.as_ptr(), realm.as_ptr(),
                content.as_ptr(), content.len(),
                embedding.as_ptr(), embedding.len(),
                1.0, 0.001, 0, &mut id);

            let mut buf = vec![0u8; 256];
            let mut written: usize = 0;
            let r = cf_get_content(h, id, buf.as_mut_ptr(), buf.len(), &mut written);
            assert_eq!(r, 0);
            assert_eq!(&buf[..written], content);

            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_update_state() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let kind = CString::new("wisdom").unwrap();
            let realm = CString::new("test").unwrap();
            let content = b"state test";
            let embedding = vec![0.2f32; 768];
            let mut id: u64 = 0;
            cf_put_memory(h, kind.as_ptr(), realm.as_ptr(),
                content.as_ptr(), content.len(),
                embedding.as_ptr(), embedding.len(),
                1.0, 0.001, 0, &mut id);

            // Apply strength delta
            let r = cf_update_state(h, id, 0.1, f32::NAN, f32::NAN, 1, -1);
            assert_eq!(r, 0);

            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_assoc_edge() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let kind = CString::new("wisdom").unwrap();
            let realm = CString::new("test").unwrap();
            let emb = vec![0.1f32; 768];
            let mut id1: u64 = 0;
            let mut id2: u64 = 0;
            cf_put_memory(h, kind.as_ptr(), realm.as_ptr(),
                b"a".as_ptr(), 1, emb.as_ptr(), emb.len(),
                1.0, 0.001, 0, &mut id1);
            cf_put_memory(h, kind.as_ptr(), realm.as_ptr(),
                b"b".as_ptr(), 1, emb.as_ptr(), emb.len(),
                1.0, 0.001, 0, &mut id2);

            let r = cf_add_assoc_edge(h, id1, id2, 0, 0.8); // DerivedFrom
            assert_eq!(r, 0);

            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_upsert_artifact() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let path = CString::new("src/main.cpp").unwrap();
            let mut art_id: u64 = 0;
            let r = cf_upsert_artifact(h, path.as_ptr(), &mut art_id);
            assert_eq!(r, 0);
            assert!(art_id > 0);

            // Idempotent: second call returns same id
            let mut art_id2: u64 = 0;
            let r2 = cf_upsert_artifact(h, path.as_ptr(), &mut art_id2);
            assert_eq!(r2, 0);
            assert_eq!(art_id, art_id2);

            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_recall_temporal() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let kind = CString::new("episode").unwrap();
            let realm = CString::new("test").unwrap();
            let emb = vec![0.4f32; 768];
            let mut id: u64 = 0;
            cf_put_memory(h, kind.as_ptr(), realm.as_ptr(),
                b"temporal".as_ptr(), 8, emb.as_ptr(), emb.len(),
                1.0, 0.001, 1000, &mut id);

            let mut hits = vec![CfRecallHit { memory_id: 0, score: 0.0, semantic_score: 0.0, ts_ms: 0, strength: 0.0, confidence: 0.0 }; 10];
            let mut written: usize = 0;
            let r = cf_recall_temporal(h, 0, 10000, std::ptr::null(), 10,
                hits.as_mut_ptr(), hits.len(), &mut written);
            assert_eq!(r, 0);
            assert_eq!(written, 1);
            assert_eq!(hits[0].memory_id, id);

            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_flush() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let r = cf_flush(h);
            assert_eq!(r, 0);
            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_memory_count() {
        unsafe {
            let (h, _tmp) = open_tmp();
            assert_eq!(cf_memory_count(h as *const CfHandle), 0);

            let kind = CString::new("wisdom").unwrap();
            let realm = CString::new("test").unwrap();
            let emb = vec![0.1f32; 768];
            let mut id: u64 = 0;
            cf_put_memory(h, kind.as_ptr(), realm.as_ptr(),
                b"x".as_ptr(), 1, emb.as_ptr(), emb.len(),
                1.0, 0.001, 0, &mut id);
            assert_eq!(cf_memory_count(h as *const CfHandle), 1);

            cf_forget(h, id);
            assert_eq!(cf_memory_count(h as *const CfHandle), 0);

            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_get_kind_realm() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let kind = CString::new("correction").unwrap();
            let realm = CString::new("myproject").unwrap();
            let emb = vec![0.7f32; 768];
            let mut id: u64 = 0;
            cf_put_memory(h, kind.as_ptr(), realm.as_ptr(),
                b"content".as_ptr(), 7, emb.as_ptr(), emb.len(),
                1.0, 0.001, 0, &mut id);

            let mut buf = [0u8; 64];
            let r = cf_get_kind(h, id, buf.as_mut_ptr(), buf.len());
            assert_eq!(r, 0);
            let kind_result = std::ffi::CStr::from_ptr(buf.as_ptr() as *const c_char).to_str().unwrap();
            assert_eq!(kind_result, "correction");

            let mut buf2 = [0u8; 64];
            let r2 = cf_get_realm(h, id, buf2.as_mut_ptr(), buf2.len());
            assert_eq!(r2, 0);
            let realm_result = std::ffi::CStr::from_ptr(buf2.as_ptr() as *const c_char).to_str().unwrap();
            assert_eq!(realm_result, "myproject");

            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_expand_associations() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let kind = CString::new("wisdom").unwrap();
            let realm = CString::new("test").unwrap();
            let emb = vec![0.1f32; 768];
            let mut id1: u64 = 0;
            let mut id2: u64 = 0;
            cf_put_memory(h, kind.as_ptr(), realm.as_ptr(),
                b"seed".as_ptr(), 4, emb.as_ptr(), emb.len(),
                1.0, 0.001, 0, &mut id1);
            cf_put_memory(h, kind.as_ptr(), realm.as_ptr(),
                b"linked".as_ptr(), 6, emb.as_ptr(), emb.len(),
                1.0, 0.001, 0, &mut id2);
            cf_add_assoc_edge(h, id1, id2, 0, 1.0); // DerivedFrom

            let seeds = [id1];
            let mut hits = vec![CfRecallHit { memory_id: 0, score: 0.0, semantic_score: 0.0, ts_ms: 0, strength: 0.0, confidence: 0.0 }; 10];
            let mut written: usize = 0;
            let r = cf_expand_associations(h,
                seeds.as_ptr(), seeds.len(),
                2, 10,
                hits.as_mut_ptr(), hits.len(), &mut written);
            assert_eq!(r, 0);
            assert_eq!(written, 1);
            assert_eq!(hits[0].memory_id, id2);

            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_last_error_on_failure() {
        unsafe {
            let (h, _tmp) = open_tmp();
            // Try to forget a non-existent memory
            let r = cf_forget(h, 99999);
            assert_eq!(r, -1);
            let err_ptr = cf_last_error(h as *const CfHandle);
            assert!(!err_ptr.is_null());
            let err_msg = std::ffi::CStr::from_ptr(err_ptr).to_str().unwrap();
            assert!(!err_msg.is_empty());
            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_null_handle_returns_error() {
        unsafe {
            let r = cf_forget(std::ptr::null_mut(), 1);
            assert_eq!(r, -1);
            assert!(cf_last_error(std::ptr::null()).is_null());
            assert_eq!(cf_memory_count(std::ptr::null()), 0);
        }
    }

    #[test]
    fn test_ffi_recall_artifact() {
        unsafe {
            let (h, _tmp) = open_tmp();
            // No memories associated yet — should return 0 hits, not an error
            let path = CString::new("src/main.cpp").unwrap();
            let mut hits = vec![CfRecallHit { memory_id: 0, score: 0.0, semantic_score: 0.0, ts_ms: 0, strength: 0.0, confidence: 0.0 }; 10];
            let mut written: usize = 0;
            let r = cf_recall_artifact(h, path.as_ptr(), 10,
                hits.as_mut_ptr(), hits.len(), &mut written);
            assert_eq!(r, 0);
            assert_eq!(written, 0);
            cf_close(h);
        }
    }
}
