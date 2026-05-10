pub mod contradiction;
pub mod error;
pub mod ffi;
pub mod field;
pub mod hnsw;
pub mod ids;
pub mod learner;
pub mod log;
pub mod manifest;
pub mod ops;
pub mod organ;
pub mod payload;
pub mod recall;
pub mod repl_executor;
pub mod repl_sessions;
pub mod scoring;
pub mod snapshot;
pub mod state;
pub mod store;

pub use error::FieldError;
pub use field::{AssocEdge, ChittaField};
pub use hnsw::{SemanticHit, SemanticIndex};
pub use ids::{ArtifactId, ChunkHash, MemoryId, SeqNo};
pub use ops::{
    AddAssocEdgeOp, AddTripletOp, AnalyticsEventOp, DeleteMemoryOp, InvalidateTripletOp,
    MsgEventOp, Op, PutPayloadOp, SessionEventOp, StateDeltaOp, TaskEventOp, ThemeEventOp,
    TranscriptEventOp, UpsertArtifactOp, UserModelEventOp, OP_ANALYTICS_EVENT, OP_MSG_EVENT,
    OP_SESSION_EVENT, OP_TASK_EVENT, OP_THEME_EVENT, OP_TRANSCRIPT_EVENT, OP_USER_MODEL_EVENT,
    AssertConstraintOp, RetractConstraintOp, CreateBranchOp, ResolveBranchOp,
    AddTriggerOp, UpdateTriggerOp, FireTriggerOp,
    OP_ASSERT_CONSTRAINT, OP_RETRACT_CONSTRAINT, OP_CREATE_BRANCH, OP_RESOLVE_BRANCH,
    OP_ADD_TRIGGER, OP_UPDATE_TRIGGER, OP_FIRE_TRIGGER,
    RecordSurpriseOp, RegisterDebtOp, UpdateDebtOp, UpdateSourceWeightOp, RecordFeedbackOp,
    OP_RECORD_SURPRISE, OP_REGISTER_DEBT, OP_UPDATE_DEBT, OP_UPDATE_SOURCE_WEIGHT, OP_RECORD_FEEDBACK,
    UpdateSurpriseCreditOp, UpsertWisdomCandidateOp, UpdateWisdomLifecycleOp,
    UpdateScorerModelOp, AttachDebtEvidenceOp,
    OP_UPDATE_SURPRISE_CREDIT, OP_UPSERT_WISDOM_CANDIDATE, OP_UPDATE_WISDOM_LIFECYCLE,
    OP_UPDATE_SCORER_MODEL, OP_ATTACH_DEBT_EVIDENCE,
};
pub use organ::cortex::{CorticalIndex, SparseCode, SparseEncoder};
pub use organ::pq::{ProductQuantizer, PQ_BYTES};
pub use organ::prototype::{ProtoId, PrototypeEntry, PrototypeIndex};
pub use organ::constraint::{ConstraintStore, Constraint, Provenance, Branch, BranchStatus};
pub use organ::predictor::AccessPredictor;
pub use organ::trigger::{TriggerStore, TriggerAutomaton, TriggerCondition, TriggerAction, TriggerStatus};
pub use organ::surprise::{SurpriseStore, SurpriseEvent, BlindSpot};
pub use organ::epistemic_debt::{EpistemicDebtStore, EpistemicDebt, DebtStatus};
pub use organ::integration::{IntegrationKernel, SourceWeight, IntegrationTrace};
pub use organ::surprise_learning::{SurpriseLearningStore, SurpriseLearningState, SurpriseCreditResult};
pub use organ::wisdom_promotion::{WisdomPromotionStore, WisdomCandidate, WisdomLifecycle};
pub use scoring::learned::{LearnedScoringModel, LearnedScoringStats};
pub use organ::triplet::{TripletEntry, TripletStore};
pub use payload::{ArtifactRef, ArtifactRelation, MemoryPayload};
pub use recall::RecallHit;
pub use state::MemoryState;
pub use log::{ChainHash, ZERO_HASH};
pub use store::FilterLevel;
