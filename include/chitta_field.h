#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CfHandle CfHandle;

typedef struct {
    uint64_t memory_id;
    float    score;
    float    semantic_score;
    int64_t  ts_ms;
    float    strength;
    float    confidence;
} CfRecallHit;

/* Lifecycle */
CfHandle* cf_open(const char* data_dir, const char* lock_dir);
void      cf_close(CfHandle* h);
const char* cf_last_error(const CfHandle* h);

/* Write */
int cf_put_memory(CfHandle* h,
    const char* kind, const char* realm,
    const uint8_t* content_ptr, size_t content_len,
    const float* embedding_ptr, size_t embedding_len,
    float confidence, float decay_rate, int64_t authored_at_ms,
    uint64_t* out_memory_id);

int cf_update_state(CfHandle* h, uint64_t memory_id,
    float strength_delta, float confidence_delta, float decay_rate,
    uint8_t touch, int8_t pin);

int cf_forget(CfHandle* h, uint64_t memory_id);

int cf_add_assoc_edge(CfHandle* h,
    uint64_t src, uint64_t dst, uint8_t edge_type, float weight);

int cf_upsert_artifact(CfHandle* h,
    const char* normalized_path, uint64_t* out_artifact_id);

/* Read */
int cf_recall_semantic(CfHandle* h,
    const float* query_embedding, size_t embedding_len,
    const char* realm, size_t k,
    CfRecallHit* hits_buf, size_t hits_cap, size_t* hits_written);

int cf_recall_temporal(CfHandle* h,
    int64_t start_ms, int64_t end_ms, const char* realm, size_t limit,
    CfRecallHit* hits_buf, size_t hits_cap, size_t* hits_written);

int cf_recall_artifact(CfHandle* h,
    const char* normalized_path, size_t limit,
    CfRecallHit* hits_buf, size_t hits_cap, size_t* hits_written);

int cf_expand_associations(CfHandle* h,
    const uint64_t* seed_ids, size_t seed_count,
    size_t max_hops, size_t limit,
    CfRecallHit* hits_buf, size_t hits_cap, size_t* hits_written);

int cf_recall_keyword(CfHandle* h,
    const char* query, size_t k,
    CfRecallHit* hits_buf, size_t hits_cap, size_t* hits_written);

int cf_get_content(CfHandle* h, uint64_t memory_id,
    uint8_t* buf, size_t buf_cap, size_t* written);

int cf_get_kind(CfHandle* h, uint64_t memory_id,
    uint8_t* buf, size_t buf_cap);

int cf_get_realm(CfHandle* h, uint64_t memory_id,
    uint8_t* buf, size_t buf_cap);

/* Triplets */
int cf_add_triplet(CfHandle* h,
    const char* subject, const char* predicate, const char* object,
    float weight,
    uint64_t source_memory_id,
    uint64_t* out_triplet_id);

int cf_invalidate_triplet(CfHandle* h, uint64_t triplet_id);

int cf_query_subject(CfHandle* h, const char* subject,
    char* buf, size_t buf_cap, size_t* written);

int cf_query_object(CfHandle* h, const char* object,
    char* buf, size_t buf_cap, size_t* written);

int cf_query_entity(CfHandle* h, const char* entity,
    char* buf, size_t buf_cap, size_t* written);

/* Learner */
int    cf_feedback(CfHandle* h, uint64_t episode_id, float reward);
size_t cf_recommended_window(CfHandle* h, const char* session_type);

/* Maintenance */
int    cf_flush(CfHandle* h);
size_t cf_memory_count(const CfHandle* h);

#ifdef __cplusplus
}
#endif
