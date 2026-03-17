#pragma once
#include "chitta_field.h"
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

/// C++ RAII wrapper around CfHandle providing typed methods for chitta-field.
class FieldStore {
public:
    explicit FieldStore(CfHandle* handle) : handle_(handle) {}

    /// Emit a domain event into the chitta-field log.
    /// domain: "session", "transcript", "task", "theme", "analytics"
    /// Returns the assigned event_id, or 0 on error.
    uint64_t emit_event(const std::string& domain, const std::string& kind,
        const std::string& entity_id, const std::string& payload_json,
        const std::string& realm = "", uint64_t fencing_token = 0) {
        uint64_t event_id = 0;
        cf_emit_event(handle_, domain.c_str(), kind.c_str(), entity_id.c_str(),
            reinterpret_cast<const uint8_t*>(payload_json.data()), payload_json.size(),
            realm.c_str(), fencing_token, &event_id);
        return event_id;
    }

    /// Get the payload of the most recent domain event matching domain+kind+entity_id.
    /// Supports domain="user_model" and domain="transcript".
    /// Returns the JSON payload string if found, or nullopt if not found.
    std::optional<std::string> get_latest_event(
        const std::string& domain,
        const std::string& kind,
        const std::string& entity_id) {
        std::vector<uint8_t> buf(65536);
        size_t written = 0;
        int rc = cf_get_latest_event(handle_, domain.c_str(), kind.c_str(), entity_id.c_str(),
                                     buf.data(), buf.size(), &written);
        if (rc == 0 && written > 0) {
            return std::string(reinterpret_cast<char*>(buf.data()), written);
        }
        return std::nullopt;
    }

    /// Iterate log ops starting from from_seqno.
    /// Callback receives the op serialized as JSON and its seqno.
    void iterate_log(uint64_t from_seqno,
        std::function<void(const std::string& op_json, uint64_t seqno)> cb) {
        struct Ctx { std::function<void(const std::string&, uint64_t)>* fn; };
        Ctx ctx{&cb};
        cf_iterate_log(handle_, from_seqno,
            [](const uint8_t* data, size_t len, uint64_t seqno, void* raw_ctx) {
                auto& fn = *static_cast<Ctx*>(raw_ctx)->fn;
                fn(std::string(reinterpret_cast<const char*>(data), len), seqno);
            }, &ctx);
    }

    /// Register a new session. kind is the session type (e.g. "claude", "sadhana").
    int session_register(const std::string& session_id, const std::string& kind,
        const std::string& realm, int64_t now_ms) {
        return cf_session_register(handle_, session_id.c_str(), kind.c_str(),
            realm.c_str(), now_ms);
    }

    /// Heartbeat a session (update last-active timestamp).
    int session_heartbeat(const std::string& session_id, int64_t now_ms) {
        return cf_session_heartbeat(handle_, session_id.c_str(), now_ms);
    }

    /// Mark a session as closed.
    int session_deregister(const std::string& session_id, int64_t now_ms) {
        return cf_session_deregister(handle_, session_id.c_str(), now_ms);
    }

    /// Register a new transcript for a session.
    int transcript_register(const std::string& transcript_id, const std::string& session_id) {
        return cf_transcript_register(handle_, transcript_id.c_str(), session_id.c_str());
    }

    /// Update transcript completion progress (0.0–100.0).
    int transcript_update_progress(const std::string& transcript_id, float progress_pct) {
        return cf_transcript_update_progress(handle_, transcript_id.c_str(), progress_pct);
    }

    /// Add a turn to a transcript. Returns the assigned turn_id, or 0 on error.
    uint64_t transcript_add_turn(const std::string& transcript_id, const std::string& role,
        const std::string& content, int64_t ts_ms) {
        uint64_t turn_id = 0;
        cf_transcript_add_turn(handle_,
            transcript_id.c_str(), role.c_str(),
            reinterpret_cast<const uint8_t*>(content.data()), content.size(),
            ts_ms, &turn_id);
        return turn_id;
    }

    /// Create a task, sadhana, or dream entry.
    /// Returns 0 on success, negative on error.
    int task_create(const std::string& task_id, const std::string& kind,
                    const std::string& payload_json, int64_t now_ms, uint64_t fencing_token = 0) {
        return cf_task_create(handle_,
            task_id.c_str(), kind.c_str(),
            reinterpret_cast<const uint8_t*>(payload_json.data()), payload_json.size(),
            now_ms, fencing_token);
    }

    /// Transition a task's status.
    /// new_status: "start" | "pause" | "resume" | "complete" | "fail"
    bool task_transition(const std::string& task_id, const std::string& new_status,
                         int64_t now_ms, uint64_t fencing_token = 0) {
        return cf_task_transition(handle_,
            task_id.c_str(), new_status.c_str(),
            now_ms, fencing_token) == 0;
    }

    /// List tasks as a JSON array string.
    /// kind_filter: empty string means all kinds.
    std::string task_list(const std::string& kind_filter = "", bool active_only = false) {
        std::string buf(65536, '\0');
        size_t written = 0;
        const char* filter = kind_filter.empty() ? nullptr : kind_filter.c_str();
        int rc = cf_task_list(handle_,
            filter, static_cast<uint8_t>(active_only ? 1 : 0),
            reinterpret_cast<uint8_t*>(buf.data()), buf.size(), &written);
        if (rc != 0) return "[]";
        buf.resize(written);
        return buf;
    }

    /// Upsert a user model entity (profile, goal, habit, anticipation, calibration).
    int user_model_upsert(const std::string& entity_id, const std::string& entity_type,
                          const std::string& payload_json, int64_t now_ms) {
        return cf_user_model_upsert(handle_,
            entity_id.c_str(), entity_type.c_str(),
            reinterpret_cast<const uint8_t*>(payload_json.data()), payload_json.size(),
            now_ms);
    }

    /// Record an observation of a user model entity.
    int user_model_observe(const std::string& entity_id, int64_t now_ms) {
        return cf_user_model_observe(handle_, entity_id.c_str(), now_ms);
    }

    /// List user model entries as a JSON array string.
    /// entity_type_filter: empty string means all types.
    std::string user_model_list(const std::string& entity_type_filter = "") {
        std::string buf(65536, '\0');
        size_t written = 0;
        const char* filter = entity_type_filter.empty() ? nullptr : entity_type_filter.c_str();
        int rc = cf_user_model_list(handle_,
            filter,
            reinterpret_cast<uint8_t*>(buf.data()), buf.size(), &written);
        if (rc != 0) return "[]";
        buf.resize(written);
        return buf;
    }

    /// Create a new named theme.
    int theme_create(uint64_t theme_id, const std::string& name) {
        return cf_theme_create(handle_, theme_id, name.c_str());
    }

    /// Update the centroid for a theme (JSON array of floats).
    int theme_update_centroid(uint64_t theme_id, const std::string& centroid_json) {
        return cf_theme_update_centroid(handle_, theme_id,
            reinterpret_cast<const uint8_t*>(centroid_json.data()), centroid_json.size());
    }

    /// Assign a memory to a theme.
    int theme_assign_member(uint64_t theme_id, uint64_t memory_id) {
        return cf_theme_assign_member(handle_, theme_id, memory_id);
    }

    /// Remove a memory from a theme.
    int theme_remove_member(uint64_t theme_id, uint64_t memory_id) {
        return cf_theme_remove_member(handle_, theme_id, memory_id);
    }

    /// List all themes as a JSON array string.
    /// Each element: {theme_id, name, member_count, centroid_json}
    std::string theme_list() {
        std::string buf(65536, '\0');
        size_t written = 0;
        int rc = cf_theme_list(handle_,
            reinterpret_cast<uint8_t*>(buf.data()), buf.size(), &written);
        if (rc != 0) return "[]";
        buf.resize(written);
        return buf;
    }

    nlohmann::json theme_get(uint64_t theme_id) {
        std::vector<uint8_t> buf(32768);
        size_t written = 0;
        int rc = cf_theme_get(handle_, theme_id, buf.data(), buf.size(), &written);
        if (rc == 0 && written > 0)
            return nlohmann::json::parse(buf.begin(), buf.begin() + written);
        return nullptr;
    }

    nlohmann::json theme_stats(const std::string& realm = "") {
        std::vector<uint8_t> buf(4096);
        size_t written = 0;
        cf_theme_stats(handle_, realm.c_str(), buf.data(), buf.size(), &written);
        if (written > 0)
            return nlohmann::json::parse(buf.begin(), buf.begin() + written);
        return nullptr;
    }

    std::vector<std::pair<uint64_t,float>> theme_recall_by_embedding(
        const std::vector<float>& embedding, size_t k, const std::string& realm = "") {
        std::vector<uint8_t> buf(16384);
        size_t written = 0;
        cf_theme_recall(handle_, embedding.data(), embedding.size(), k,
                        realm.c_str(), buf.data(), buf.size(), &written);
        std::vector<std::pair<uint64_t,float>> results;
        if (written > 0) {
            auto j = nlohmann::json::parse(buf.begin(), buf.begin() + written);
            for (auto& item : j) {
                results.emplace_back(item["theme_id"].get<uint64_t>(),
                                     item["score"].get<float>());
            }
        }
        return results;
    }

    nlohmann::json theme_maintain() {
        std::vector<uint8_t> buf(4096);
        size_t written = 0;
        cf_theme_maintain(handle_, buf.data(), buf.size(), &written);
        if (written > 0)
            return nlohmann::json::parse(buf.begin(), buf.begin() + written);
        return nullptr;
    }

    std::pair<size_t,size_t> theme_assign_orphans(size_t batch_size, const std::string& realm = "") {
        std::vector<uint8_t> buf(256);
        size_t written = 0;
        cf_theme_assign_orphans(handle_, batch_size, realm.c_str(), buf.data(), buf.size(), &written);
        if (written > 0) {
            auto j = nlohmann::json::parse(buf.begin(), buf.begin() + written);
            return {j["assigned"].get<size_t>(), j["remaining"].get<size_t>()};
        }
        return {0, 0};
    }

    /// Append an analytics event. Returns 0 on success, negative on error.
    int analytics_append(const std::string& kind, const std::string& entity_id,
                         const std::string& payload_json, int64_t ts_ms) {
        return cf_analytics_append(handle_,
            kind.c_str(), entity_id.c_str(),
            reinterpret_cast<const uint8_t*>(payload_json.data()), payload_json.size(),
            ts_ms);
    }

    /// Read the most recent `limit` analytics entries as a JSON array string.
    std::string analytics_recent(size_t limit) {
        std::string buf(65536, '\0');
        size_t written = 0;
        int rc = cf_analytics_recent(handle_,
            limit,
            reinterpret_cast<uint8_t*>(buf.data()), buf.size(), &written);
        if (rc != 0) return "[]";
        buf.resize(written);
        return buf;
    }

private:
    CfHandle* handle_;
};
