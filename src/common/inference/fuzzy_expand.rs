use std::collections::HashMap;
use std::time::Duration;

use api::conversions::json::proto_dict_to_json;
use api::grpc::qdrant as grpc;
use api::rest::schema as rest;
use api::rest::schema::VectorInput;
use collection::operations::point_ops::VectorPersisted;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use segment::data_types::vectors::DEFAULT_VECTOR_NAME;
use segment::index::field_index::full_text_index::fuzzy_index::{
    FuzzyCandidate, FuzzyTokenCandidates,
};
use segment::types::FuzzyParams;
use serde_json::Value;
use sparse::common::sparse_vector::SparseVector;
use storage::content_manager::errors::StorageError;
use storage::content_manager::toc::TableOfContent;
use storage::rbac::Auth;

use super::inference_input::{InferenceDataType, InferenceInput};
use super::local_model::{expand_fuzzy_candidates, is_local_model};
use super::params::InferenceParams;
use super::service::{InferenceService, InferenceType};

/// Protocol-agnostic description of a fuzzy expansion request, extracted from either a
/// REST or gRPC query. Carries everything needed to resolve FST candidates and produce
/// a `SparseVector`, with no protocol-specific types.
struct FuzzyIntent {
    text: String,
    model: String,
    params: FuzzyParams,
    vector_name: String,
    /// Document options (carries BM25 config with stemming settings, etc.).
    options: Option<HashMap<String, Value>>,
}

impl FuzzyIntent {
    /// Extracts a `FuzzyIntent` from a REST query request.
    /// Returns `Ok(None)` if no fuzzy params are set (common fast path).
    /// Returns `Err` if fuzzy is set but the query type is incompatible.
    fn from_rest(request: &rest::QueryRequestInternal) -> Result<Option<Self>, StorageError> {
        let params = match request.params.as_ref().and_then(|p| p.fuzzy.as_ref()) {
            Some(p) => p.clone(),
            None => return Ok(None),
        };

        let (text, model, options) = match &request.query {
            Some(rest::QueryInterface::Nearest(VectorInput::Document(doc))) => {
                let opts = doc.options.clone().map(rest::DocumentOptions::into_options);
                (doc.text.clone(), doc.model.clone(), opts)
            }
            Some(rest::QueryInterface::Nearest(_)) => {
                return Err(StorageError::bad_input(
                    "Fuzzy search params can only be used with sparse Document queries, \
                     not with raw vectors or point IDs",
                ));
            }
            _ => {
                return Err(StorageError::bad_input(
                    "Fuzzy search params can only be used with Document queries",
                ));
            }
        };

        Ok(Some(FuzzyIntent {
            text,
            model,
            params,
            vector_name: request
                .using
                .clone()
                .unwrap_or_else(|| DEFAULT_VECTOR_NAME.to_owned()),
            options,
        }))
    }

    /// Extracts a `FuzzyIntent` from a gRPC query request.
    /// Returns `Ok(None)` if no fuzzy params are set (common fast path).
    /// Returns `Err` if fuzzy is set but the query type is incompatible.
    fn from_grpc(request: &grpc::QueryPoints) -> Result<Option<Self>, StorageError> {
        let fuzzy_grpc = match request.params.as_ref().and_then(|p| p.fuzzy.as_ref()) {
            Some(p) => p.clone(),
            None => return Ok(None),
        };

        let (text, model, options) = match request.query.as_ref().and_then(|q| q.variant.as_ref()) {
            Some(grpc::query::Variant::Nearest(vi)) => match vi.variant.as_ref() {
                Some(grpc::vector_input::Variant::Document(doc)) => {
                    let opts = if doc.options.is_empty() {
                        None
                    } else {
                        Some(proto_dict_to_json(doc.options.clone()).map_err(|e| {
                            StorageError::bad_input(format!("Invalid fuzzy document options: {e}"))
                        })?)
                    };
                    (doc.text.clone(), doc.model.clone(), opts)
                }
                _ => {
                    return Err(StorageError::bad_input(
                        "Fuzzy search params can only be used with sparse Document queries, \
                         not with raw vectors or point IDs",
                    ));
                }
            },
            _ => {
                return Err(StorageError::bad_input(
                    "Fuzzy search params can only be used with Document queries",
                ));
            }
        };

        Ok(Some(FuzzyIntent {
            text,
            model,
            params: FuzzyParams {
                max_edits: fuzzy_grpc.max_edits.unwrap_or(1) as u8,
                prefix_length: fuzzy_grpc.prefix_length.unwrap_or(0) as u8,
                max_expansions: fuzzy_grpc.max_expansions.unwrap_or(30) as u8,
            },
            vector_name: request
                .using
                .clone()
                .unwrap_or_else(|| DEFAULT_VECTOR_NAME.to_owned()),
            options,
        }))
    }
}

/// Core resolution: given a validated `FuzzyIntent`, fetches FST candidates from the
/// collection and maps them to a `SparseVector`. Returns `None` if the vector has no
/// fuzzy bind field configured, or if the FST index yields no candidates (normal path).
async fn resolve_fuzzy_intent(
    intent: &FuzzyIntent,
    collection_name: &str,
    toc: &TableOfContent,
    auth: Auth,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
    inference_params: InferenceParams,
) -> Result<Option<SparseVector>, StorageError> {
    let bind_field =
        get_fuzzy_bind_field(toc, collection_name, &intent.vector_name, auth.clone()).await?;
    let Some(bind_field) = bind_field else {
        return Ok(None); // No fuzzy config on this sparse vector, silently skip
    };

    let token_groups = toc
        .get_fuzzy_candidates_grouped(
            collection_name,
            &bind_field,
            &intent.text,
            &intent.params,
            auth,
            timeout,
            hw_measurement_acc,
        )
        .await?;

    let candidates = flatten_fuzzy_token_groups(token_groups);

    if candidates.is_empty() {
        return Ok(None); // No FST candidates — keep original Document query (normal path)
    }

    let sparse = map_candidates_to_sparse(
        &candidates,
        &intent.model,
        inference_params,
        intent.options.clone(),
    )
    .await?;
    Ok(Some(sparse))
}

fn flatten_fuzzy_token_groups(groups: Vec<FuzzyTokenCandidates>) -> Vec<FuzzyCandidate> {
    groups
        .into_iter()
        .flat_map(|group| group.candidates)
        .collect()
}

/// Maps FST candidates to a `SparseVector` using the appropriate strategy:
///
/// - **Local / BM25**: stems each candidate term, then maps via `murmur3(stemmed)` → dimension.
/// - **Remote sparse model** (SPLADE, etc.): batch-infers each candidate term via the
///   inference service; scales each returned sparse vector by the edit-distance weight
///   and takes the per-dimension maximum across all candidates.
async fn map_candidates_to_sparse(
    candidates: &[FuzzyCandidate],
    model: &str,
    inference_params: InferenceParams,
    options: Option<HashMap<String, Value>>,
) -> Result<SparseVector, StorageError> {
    if is_local_model(model) {
        return Ok(expand_fuzzy_candidates(model, candidates, options));
    }

    let service = InferenceService::get_global().ok_or_else(|| {
        StorageError::service_error(
            "Fuzzy expansion for non-BM25 models requires an inference service. \
             Please configure the inference address in the Qdrant config.",
        )
    })?;

    let inputs: Vec<InferenceInput> = candidates
        .iter()
        .map(|c| InferenceInput {
            data: serde_json::Value::String(c.term.clone()),
            data_type: InferenceDataType::Text,
            model: model.to_owned(),
            options: None,
        })
        .collect();

    let response = service
        .infer(inputs, InferenceType::Search, inference_params)
        .await?;

    if response.embeddings.len() != candidates.len() {
        return Err(StorageError::service_error(format!(
            "Inference service returned {} embeddings for {} fuzzy candidates",
            response.embeddings.len(),
            candidates.len(),
        )));
    }

    // For each candidate, scale its sparse vector by the FST edit-distance weight,
    // then take the max per dimension across all candidates.
    // Exact matches (weight=1.0) pass at full strength; 1-edit matches are discounted.
    let mut dim_map: HashMap<u32, f32> = HashMap::new();
    for (candidate, vector) in candidates.iter().zip(response.embeddings) {
        let VectorPersisted::Sparse(sv) = vector else {
            return Err(StorageError::bad_input(format!(
                "Fuzzy expansion requires a sparse model; \
                 model '{model}' returned a non-sparse vector",
            )));
        };
        for (&dim_id, &weight) in sv.indices.iter().zip(sv.values.iter()) {
            let scaled = weight * candidate.weight;
            dim_map
                .entry(dim_id)
                .and_modify(|v| *v = v.max(scaled))
                .or_insert(scaled);
        }
    }

    let mut pairs: Vec<_> = dim_map.into_iter().collect();
    pairs.sort_by_key(|(idx, _)| *idx);
    Ok(SparseVector {
        indices: pairs.iter().map(|(i, _)| *i).collect(),
        values: pairs.iter().map(|(_, v)| *v).collect(),
    })
}

/// REST entry point. Validates fuzzy preconditions, extracts a `FuzzyIntent`,
/// resolves it to a `SparseVector`, and replaces the query in-place.
pub async fn try_expand_fuzzy_query(
    request: &mut rest::QueryRequestInternal,
    collection_name: &str,
    toc: &TableOfContent,
    auth: Auth,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
    inference_params: InferenceParams,
) -> Result<(), StorageError> {
    if request
        .params
        .as_ref()
        .and_then(|p| p.fuzzy.as_ref())
        .is_none()
    {
        return Ok(());
    }
    if request.prefetch.as_ref().is_some_and(|p| !p.is_empty()) {
        return Err(StorageError::bad_input(
            "Fuzzy search params cannot be used together with prefetch queries",
        ));
    }

    let Some(intent) = FuzzyIntent::from_rest(request)? else {
        return Ok(());
    };
    let Some(sparse) = resolve_fuzzy_intent(
        &intent,
        collection_name,
        toc,
        auth,
        timeout,
        hw_measurement_acc,
        inference_params,
    )
    .await?
    else {
        return Ok(());
    };

    request.query = Some(rest::QueryInterface::Nearest(VectorInput::SparseVector(
        sparse,
    )));
    Ok(())
}

/// gRPC entry point. Validates fuzzy preconditions, extracts a `FuzzyIntent`,
/// resolves it to a `SparseVector`, and replaces the query in-place.
pub async fn try_expand_fuzzy_query_grpc(
    request: &mut grpc::QueryPoints,
    toc: &TableOfContent,
    auth: Auth,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
    inference_params: InferenceParams,
) -> Result<(), StorageError> {
    if request
        .params
        .as_ref()
        .and_then(|p| p.fuzzy.as_ref())
        .is_none()
    {
        return Ok(());
    }
    if !request.prefetch.is_empty() {
        return Err(StorageError::bad_input(
            "Fuzzy search params cannot be used together with prefetch queries",
        ));
    }

    let Some(intent) = FuzzyIntent::from_grpc(request)? else {
        return Ok(());
    };
    let Some(sparse) = resolve_fuzzy_intent(
        &intent,
        &request.collection_name,
        toc,
        auth,
        timeout,
        hw_measurement_acc,
        inference_params,
    )
    .await?
    else {
        return Ok(());
    };

    request.query = Some(grpc::Query {
        variant: Some(grpc::query::Variant::Nearest(grpc::VectorInput {
            variant: Some(grpc::vector_input::Variant::Sparse(grpc::SparseVector {
                indices: sparse.indices,
                values: sparse.values,
            })),
        })),
    });
    Ok(())
}

/// Gets the `fuzzy_bind_field` from a collection's sparse vector config.
async fn get_fuzzy_bind_field(
    toc: &TableOfContent,
    collection_name: &str,
    vector_name: &str,
    auth: Auth,
) -> Result<Option<String>, StorageError> {
    use storage::rbac::AccessRequirements;

    let collection_pass = auth.check_collection_access(
        collection_name,
        AccessRequirements::new(),
        "get_fuzzy_bind_field",
    )?;

    let collection = toc.get_collection(&collection_pass).await?;
    Ok(collection.get_fuzzy_bind_field(vector_name).await)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};
    use std::num::NonZeroU32;
    use std::path::Path;
    use std::sync::Arc;

    use api::rest::{Bm25Config, NamedVectorStruct, SearchRequestInternal};
    use collection::collection::{Collection, RequestShardTransfer};
    use collection::config::{CollectionConfigInternal, CollectionParams, WalConfig};
    use collection::operations::CollectionUpdateOperations;
    use collection::operations::point_ops::{
        PointInsertOperationsInternal, PointOperations, PointStructPersisted, VectorPersisted,
        VectorStructPersisted, WriteOrdering,
    };
    use collection::operations::shard_selector_internal::ShardSelectorInternal;
    use collection::operations::types::{FuzzySearchConfig, SparseVectorParams, VectorsConfig};
    use collection::optimizers_builder::OptimizersConfig;
    use collection::shards::CollectionId;
    use collection::shards::channel_service::ChannelService;
    use collection::shards::collection_shard_distribution::CollectionShardDistribution;
    use collection::shards::replica_set::replica_set_state::ReplicaState;
    use collection::shards::replica_set::{AbortShardTransfer, ChangePeerFromState};
    use common::budget::ResourceBudget;
    use common::counter::hardware_accumulator::HwMeasurementAcc;
    use segment::data_types::index::{TextIndexParams, TextIndexType, TokenizerType};
    use segment::data_types::vectors::NamedSparseVector;
    use segment::types::{Payload, PayloadFieldSchema, PayloadSchemaParams, WithPayloadInterface};
    use shard::search::CoreSearchRequestBatch;
    use tempfile::Builder;

    use super::*;
    use crate::common::inference::bm25::Bm25;

    const COLLECTION_NAME: &str = "test";
    const VECTOR_NAME: &str = "text";
    const PAYLOAD_FIELD: &str = "text_content";
    const REST_PORT: u16 = 6333;
    const TEST_OPTIMIZERS_CONFIG: OptimizersConfig = OptimizersConfig {
        deleted_threshold: 0.9,
        vacuum_min_vector_number: 1000,
        default_segment_number: 2,
        max_segment_size: None,
        #[expect(deprecated)]
        memmap_threshold: None,
        indexing_threshold: Some(50_000),
        flush_interval_sec: 30,
        max_optimization_threads: Some(2),
        prevent_unoptimized: None,
    };

    #[tokio::test(flavor = "multi_thread")]
    async fn test_bm25_fuzzy_sparse_search_pipeline_without_manual_setup() {
        let collection_dir = Builder::new().prefix("fuzzy_sparse").tempdir().unwrap();
        let collection = sparse_fuzzy_collection_fixture(collection_dir.path(), 1).await;

        insert_bm25_documents(
            &collection,
            &[
                (0, "the quick brown fox jumps over the lazy dog"),
                (1, "machine learning enables semantic retrieval"),
                (2, "brown bears hibernate during winter months"),
            ],
        )
        .await;

        create_fuzzy_text_index(&collection).await;

        let bm25 = Bm25::new(Bm25Config::default());
        let non_fuzzy = search_sparse(&collection, into_sparse(bm25.search_embed("quuck"))).await;
        assert!(
            non_fuzzy.is_empty(),
            "plain sparse BM25 typo query should not match without fuzzy expansion"
        );

        let params = FuzzyParams {
            max_edits: 1,
            prefix_length: 0,
            max_expansions: 30,
        };
        let candidates = collection
            .get_fuzzy_candidates(
                PAYLOAD_FIELD,
                "quuck",
                &params,
                &ShardSelectorInternal::All,
                None,
                HwMeasurementAcc::disposable(),
            )
            .await
            .unwrap();

        assert!(
            candidates.iter().any(|candidate| candidate.term == "quick"),
            "expected fuzzy candidates to include the corrected term 'quick', got: {candidates:?}"
        );

        let fuzzy_results = search_sparse(
            &collection,
            expand_fuzzy_candidates("qdrant/bm25", &candidates, None),
        )
        .await;

        assert!(
            !fuzzy_results.is_empty(),
            "fuzzy-expanded search should return results"
        );
        assert_eq!(fuzzy_results[0].id, 0.into());
        assert!(fuzzy_results[0].score > 0.0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_exact_match_still_works_with_fuzzy_enabled() {
        let dir = Builder::new().prefix("fuzzy_exact").tempdir().unwrap();
        let collection = sparse_fuzzy_collection_fixture(dir.path(), 1).await;
        insert_bm25_documents(
            &collection,
            &[(0, "the quick brown fox"), (1, "machine learning")],
        )
        .await;
        create_fuzzy_text_index(&collection).await;

        let params = FuzzyParams {
            max_edits: 1,
            prefix_length: 0,
            max_expansions: 30,
        };
        let candidates = collection
            .get_fuzzy_candidates(
                PAYLOAD_FIELD,
                "quick",
                &params,
                &ShardSelectorInternal::All,
                None,
                HwMeasurementAcc::disposable(),
            )
            .await
            .unwrap();

        let exact = candidates
            .iter()
            .find(|candidate| candidate.term == "quick");
        assert!(
            exact.is_some(),
            "exact match term should appear in candidates"
        );
        assert!((exact.unwrap().weight - 1.0).abs() < 1e-6);
        assert_eq!(candidates[0].term, "quick", "exact match should sort first");
        assert_eq!(
            candidates
                .iter()
                .filter(|candidate| candidate.term == "quick")
                .count(),
            1,
            "exact term should not be duplicated in candidate list"
        );

        let results = search_sparse(
            &collection,
            expand_fuzzy_candidates("qdrant/bm25", &candidates, None),
        )
        .await;
        assert!(!results.is_empty());
        assert_eq!(results[0].id, 0.into());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_no_candidates_for_unknown_term() {
        let dir = Builder::new()
            .prefix("fuzzy_nocandidate")
            .tempdir()
            .unwrap();
        let collection = sparse_fuzzy_collection_fixture(dir.path(), 1).await;
        insert_bm25_documents(&collection, &[(0, "hello world")]).await;
        create_fuzzy_text_index(&collection).await;

        let params = FuzzyParams {
            max_edits: 1,
            prefix_length: 0,
            max_expansions: 30,
        };
        let candidates = collection
            .get_fuzzy_candidates(
                PAYLOAD_FIELD,
                "zzzzqqqq",
                &params,
                &ShardSelectorInternal::All,
                None,
                HwMeasurementAcc::disposable(),
            )
            .await
            .unwrap();

        assert!(
            candidates.is_empty(),
            "completely unknown term should yield no fuzzy candidates"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_prefix_length_restricts_expansion() {
        let dir = Builder::new().prefix("fuzzy_prefix").tempdir().unwrap();
        let collection = sparse_fuzzy_collection_fixture(dir.path(), 1).await;
        insert_bm25_documents(&collection, &[(0, "fox fix fax")]).await;
        create_fuzzy_text_index(&collection).await;

        let strict = FuzzyParams {
            max_edits: 1,
            prefix_length: 3,
            max_expansions: 30,
        };
        let candidates_strict = collection
            .get_fuzzy_candidates(
                PAYLOAD_FIELD,
                "fox",
                &strict,
                &ShardSelectorInternal::All,
                None,
                HwMeasurementAcc::disposable(),
            )
            .await
            .unwrap();

        assert!(
            !candidates_strict
                .iter()
                .any(|candidate| candidate.term == "fix" || candidate.term == "fax"),
            "prefix_length=3 should prevent expanding 'fox' to 'fix'/'fax'"
        );

        let loose = FuzzyParams {
            max_edits: 1,
            prefix_length: 0,
            max_expansions: 30,
        };
        let candidates_loose = collection
            .get_fuzzy_candidates(
                PAYLOAD_FIELD,
                "fox",
                &loose,
                &ShardSelectorInternal::All,
                None,
                HwMeasurementAcc::disposable(),
            )
            .await
            .unwrap();

        assert!(
            candidates_loose
                .iter()
                .any(|candidate| candidate.term == "fix" || candidate.term == "fax"),
            "prefix_length=0 should allow expanding 'fox' to 'fix'/'fax'"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_max_expansions_caps_candidate_count() {
        let dir = Builder::new().prefix("fuzzy_maxexp").tempdir().unwrap();
        let collection = sparse_fuzzy_collection_fixture(dir.path(), 1).await;
        insert_bm25_documents(
            &collection,
            &[(0, "aat abt act adt aet aft agt aht ait ajt")],
        )
        .await;
        create_fuzzy_text_index(&collection).await;

        let params = FuzzyParams {
            max_edits: 1,
            prefix_length: 0,
            max_expansions: 3,
        };
        let candidates = collection
            .get_fuzzy_candidates(
                PAYLOAD_FIELD,
                "aat",
                &params,
                &ShardSelectorInternal::All,
                None,
                HwMeasurementAcc::disposable(),
            )
            .await
            .unwrap();

        assert!(
            candidates.len() <= 3,
            "max_expansions=3 should cap candidate count, got {}",
            candidates.len()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_exact_match_ranks_above_fuzzy_match() {
        let dir = Builder::new().prefix("fuzzy_rank").tempdir().unwrap();
        let collection = sparse_fuzzy_collection_fixture(dir.path(), 1).await;
        insert_bm25_documents(
            &collection,
            &[
                (0, "quick brown fox"),
                (1, "quack brown fox"),
                (2, "deep neural networks"),
            ],
        )
        .await;
        create_fuzzy_text_index(&collection).await;

        let params = FuzzyParams {
            max_edits: 1,
            prefix_length: 0,
            max_expansions: 30,
        };
        let candidates = collection
            .get_fuzzy_candidates(
                PAYLOAD_FIELD,
                "quick",
                &params,
                &ShardSelectorInternal::All,
                None,
                HwMeasurementAcc::disposable(),
            )
            .await
            .unwrap();

        assert_eq!(candidates[0].term, "quick");
        let quack = candidates
            .iter()
            .find(|candidate| candidate.term == "quack");
        assert!(
            quack.is_some(),
            "1-edit candidate should be present for ranking test"
        );
        assert!(candidates[0].weight > quack.unwrap().weight);

        let results = search_sparse(
            &collection,
            expand_fuzzy_candidates("qdrant/bm25", &candidates, None),
        )
        .await;

        assert!(!results.is_empty());
        let pos_0 = results
            .iter()
            .position(|result| result.id == 0.into())
            .unwrap();
        let pos_1 = results.iter().position(|result| result.id == 1.into());
        if let Some(pos_1) = pos_1 {
            assert!(
                pos_0 < pos_1,
                "exact match doc should rank above fuzzy match doc"
            );
        }
        assert!(
            results.iter().all(|result| result.id != 2.into()),
            "unrelated doc should not appear in results"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_fuzzy_search_is_case_insensitive() {
        let dir = Builder::new().prefix("fuzzy_case").tempdir().unwrap();
        let collection = sparse_fuzzy_collection_fixture(dir.path(), 1).await;
        insert_bm25_documents(&collection, &[(0, "the quick brown fox")]).await;
        create_fuzzy_text_index(&collection).await;

        let params = FuzzyParams {
            max_edits: 1,
            prefix_length: 0,
            max_expansions: 30,
        };
        let candidates = collection
            .get_fuzzy_candidates(
                PAYLOAD_FIELD,
                "QUUCK",
                &params,
                &ShardSelectorInternal::All,
                None,
                HwMeasurementAcc::disposable(),
            )
            .await
            .unwrap();

        assert!(
            candidates.iter().any(|candidate| candidate.term == "quick"),
            "uppercase fuzzy query should still find lowercase indexed term"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_multi_term_query_expands_all_terms() {
        let dir = Builder::new().prefix("fuzzy_multi").tempdir().unwrap();
        let collection = sparse_fuzzy_collection_fixture(dir.path(), 1).await;
        insert_bm25_documents(
            &collection,
            &[(0, "quick fox"), (1, "quick turtle"), (2, "stone fox")],
        )
        .await;
        create_fuzzy_text_index(&collection).await;

        let params = FuzzyParams {
            max_edits: 1,
            prefix_length: 0,
            max_expansions: 30,
        };

        let c0 = collection
            .get_fuzzy_candidates(
                PAYLOAD_FIELD,
                "quuck",
                &params,
                &ShardSelectorInternal::All,
                None,
                HwMeasurementAcc::disposable(),
            )
            .await
            .unwrap();
        let c1 = collection
            .get_fuzzy_candidates(
                PAYLOAD_FIELD,
                "fex",
                &params,
                &ShardSelectorInternal::All,
                None,
                HwMeasurementAcc::disposable(),
            )
            .await
            .unwrap();

        assert!(c0.iter().any(|candidate| candidate.term == "quick"));
        assert!(c1.iter().any(|candidate| candidate.term == "fox"));

        let mut merged = c0.clone();
        merged.extend(c1);
        let expanded = expand_fuzzy_candidates("qdrant/bm25", &merged, None);

        assert!(
            !expanded.indices.is_empty(),
            "merged fuzzy expansion should produce a non-empty sparse query"
        );

        let results = search_sparse(&collection, expanded).await;
        assert!(!results.is_empty());
        assert_eq!(
            results[0].id,
            0.into(),
            "document matching both expanded terms should rank first"
        );
        let pos_1 = results
            .iter()
            .position(|result| result.id == 1.into())
            .unwrap();
        let pos_2 = results
            .iter()
            .position(|result| result.id == 2.into())
            .unwrap();
        assert!(
            pos_1 > 0,
            "single-term match should rank below the two-term match"
        );
        assert!(
            pos_2 > 0,
            "single-term match should rank below the two-term match"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_multi_term_grouped_candidates_preserve_each_token_budget() {
        let dir = Builder::new()
            .prefix("fuzzy_grouped_budget")
            .tempdir()
            .unwrap();
        let collection = sparse_fuzzy_collection_fixture(dir.path(), 1).await;
        insert_bm25_documents(
            &collection,
            &[(0, "diamond diamonda"), (1, "international internationals")],
        )
        .await;
        create_fuzzy_text_index(&collection).await;

        let params = FuzzyParams {
            max_edits: 1,
            prefix_length: 0,
            max_expansions: 1,
        };

        let grouped = collection
            .get_fuzzy_candidates_grouped(
                PAYLOAD_FIELD,
                "diamondx internationalx",
                &params,
                &ShardSelectorInternal::All,
                None,
                HwMeasurementAcc::disposable(),
            )
            .await
            .unwrap();

        assert_eq!(
            grouped.len(),
            2,
            "expected one candidate group per query token"
        );
        assert_eq!(grouped[0].token, "diamondx");
        assert_eq!(grouped[1].token, "internationalx");
        assert_eq!(grouped[0].candidates.len(), 1);
        assert_eq!(grouped[1].candidates.len(), 1);
        assert_eq!(grouped[0].candidates[0].term, "diamond");
        assert_eq!(grouped[1].candidates[0].term, "international");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_fuzzy_candidates_aggregated_across_shards() {
        let dir = Builder::new().prefix("fuzzy_shards").tempdir().unwrap();
        let collection = sparse_fuzzy_collection_fixture(dir.path(), 2).await;
        insert_bm25_documents(
            &collection,
            &[
                (0, "the quick brown fox"),
                (1, "quick learner"),
                (2, "machine learning"),
            ],
        )
        .await;
        create_fuzzy_text_index(&collection).await;

        let params = FuzzyParams {
            max_edits: 1,
            prefix_length: 0,
            max_expansions: 30,
        };
        let candidates = collection
            .get_fuzzy_candidates(
                PAYLOAD_FIELD,
                "quuck",
                &params,
                &ShardSelectorInternal::All,
                None,
                HwMeasurementAcc::disposable(),
            )
            .await
            .unwrap();

        assert!(
            candidates.iter().any(|candidate| candidate.term == "quick"),
            "cross-shard aggregation should include 'quick' from all shards"
        );
        assert_eq!(
            candidates
                .iter()
                .filter(|candidate| candidate.term == "quick")
                .count(),
            1,
            "cross-shard aggregation should deduplicate identical terms"
        );
        assert_eq!(
            candidates.len(),
            unique_term_count(&candidates),
            "candidate list should not contain duplicate terms after shard aggregation"
        );

        let results = search_sparse(
            &collection,
            expand_fuzzy_candidates("qdrant/bm25", &candidates, None),
        )
        .await;
        assert!(results.iter().any(|result| result.id == 0.into()));
        assert!(results.iter().any(|result| result.id == 1.into()));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_no_fuzzy_config_silently_skips() {
        let dir = Builder::new().prefix("fuzzy_noop").tempdir().unwrap();
        let wal_config = WalConfig {
            wal_capacity_mb: 1,
            wal_segments_ahead: 0,
            wal_retain_closed: 1,
        };
        let collection_params = CollectionParams {
            vectors: VectorsConfig::default(),
            shard_number: NonZeroU32::new(1).unwrap(),
            sparse_vectors: Some(BTreeMap::from([(
                VECTOR_NAME.into(),
                SparseVectorParams {
                    index: None,
                    modifier: None,
                    fuzzy: None,
                },
            )])),
            ..CollectionParams::empty()
        };
        let config = CollectionConfigInternal {
            params: collection_params,
            optimizer_config: TEST_OPTIMIZERS_CONFIG.clone(),
            wal_config,
            hnsw_config: Default::default(),
            quantization_config: Default::default(),
            strict_mode_config: Default::default(),
            uuid: None,
            metadata: None,
        };
        let snap = dir.path().join("snapshots");
        let collection =
            new_local_collection(COLLECTION_NAME.to_string(), dir.path(), &snap, &config)
                .await
                .unwrap();
        for shard_id in collection.get_local_shards().await {
            collection
                .set_shard_replica_state(shard_id, 0, ReplicaState::Active, None)
                .await
                .unwrap();
        }

        insert_bm25_documents(&collection, &[(0, "hello world")]).await;

        let bind = collection.get_fuzzy_bind_field(VECTOR_NAME).await;
        assert!(
            bind.is_none(),
            "no fuzzy config should yield None bind_field"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_candidate_weights_reflect_edit_distance() {
        let dir = Builder::new().prefix("fuzzy_weight").tempdir().unwrap();
        let collection = sparse_fuzzy_collection_fixture(dir.path(), 1).await;
        insert_bm25_documents(&collection, &[(0, "quick quack")]).await;
        create_fuzzy_text_index(&collection).await;

        let params = FuzzyParams {
            max_edits: 1,
            prefix_length: 0,
            max_expansions: 30,
        };
        let candidates = collection
            .get_fuzzy_candidates(
                PAYLOAD_FIELD,
                "quick",
                &params,
                &ShardSelectorInternal::All,
                None,
                HwMeasurementAcc::disposable(),
            )
            .await
            .unwrap();

        let exact = candidates
            .iter()
            .find(|candidate| candidate.term == "quick");
        let fuzzy = candidates
            .iter()
            .find(|candidate| candidate.term == "quack");

        assert!(exact.is_some(), "exact term should be in candidates");
        if let Some(fuzzy) = fuzzy {
            assert!(exact.unwrap().weight > fuzzy.weight);
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_bm25_and_fuzzy_expansion_compose_to_expected_dimensions() {
        let dir = Builder::new()
            .prefix("fuzzy_bm25_compose")
            .tempdir()
            .unwrap();
        let collection = sparse_fuzzy_collection_fixture(dir.path(), 1).await;
        insert_bm25_documents(&collection, &[(0, "quick fox"), (1, "slow turtle")]).await;
        create_fuzzy_text_index(&collection).await;

        let params = FuzzyParams {
            max_edits: 1,
            prefix_length: 0,
            max_expansions: 30,
        };
        let quick_candidates = collection
            .get_fuzzy_candidates(
                PAYLOAD_FIELD,
                "quuck",
                &params,
                &ShardSelectorInternal::All,
                None,
                HwMeasurementAcc::disposable(),
            )
            .await
            .unwrap();
        let fox_candidates = collection
            .get_fuzzy_candidates(
                PAYLOAD_FIELD,
                "fex",
                &params,
                &ShardSelectorInternal::All,
                None,
                HwMeasurementAcc::disposable(),
            )
            .await
            .unwrap();

        let mut merged = quick_candidates;
        merged.extend(fox_candidates);
        let expanded = expand_fuzzy_candidates("qdrant/bm25", &merged, None);

        let quick_dim = Bm25::compute_token_id("quick");
        let fox_dim = Bm25::compute_token_id("fox");
        assert!(expanded.indices.contains(&quick_dim));
        assert!(expanded.indices.contains(&fox_dim));
        assert_eq!(expanded.indices.len(), expanded.values.len());

        let results = search_sparse(&collection, expanded).await;
        assert_eq!(
            results[0].id,
            0.into(),
            "expanded BM25 sparse query should hit the document containing both corrected terms"
        );
        assert!(results.iter().all(|result| result.id != 1.into()));
    }

    async fn sparse_fuzzy_collection_fixture(
        collection_path: &Path,
        shard_number: u32,
    ) -> Collection {
        let wal_config = WalConfig {
            wal_capacity_mb: 1,
            wal_segments_ahead: 0,
            wal_retain_closed: 1,
        };

        let collection_params = CollectionParams {
            vectors: VectorsConfig::default(),
            shard_number: NonZeroU32::new(shard_number).unwrap(),
            sparse_vectors: Some(BTreeMap::from([(
                VECTOR_NAME.into(),
                SparseVectorParams {
                    index: None,
                    modifier: None,
                    fuzzy: Some(FuzzySearchConfig {
                        fuzzy_bind_field: PAYLOAD_FIELD.to_string(),
                    }),
                },
            )])),
            ..CollectionParams::empty()
        };

        let collection_config = CollectionConfigInternal {
            params: collection_params,
            optimizer_config: TEST_OPTIMIZERS_CONFIG.clone(),
            wal_config,
            hnsw_config: Default::default(),
            quantization_config: Default::default(),
            strict_mode_config: Default::default(),
            uuid: None,
            metadata: None,
        };

        let snapshot_path = collection_path.join("snapshots");
        let collection = new_local_collection(
            COLLECTION_NAME.to_string(),
            collection_path,
            &snapshot_path,
            &collection_config,
        )
        .await
        .unwrap();

        for shard_id in collection.get_local_shards().await {
            collection
                .set_shard_replica_state(shard_id, 0, ReplicaState::Active, None)
                .await
                .unwrap();
        }

        collection
    }

    async fn new_local_collection(
        id: CollectionId,
        path: &Path,
        snapshots_path: &Path,
        config: &CollectionConfigInternal,
    ) -> collection::operations::types::CollectionResult<Collection> {
        let on_replica_failure: ChangePeerFromState = Arc::new(|_, _, _| {});
        let request_shard_transfer: RequestShardTransfer = Arc::new(|_| {});
        let abort_shard_transfer: AbortShardTransfer = Arc::new(|_, _| {});

        Collection::new(
            id,
            0,
            path,
            snapshots_path,
            config,
            Default::default(),
            CollectionShardDistribution::all_local(Some(config.params.shard_number.into()), 0),
            None,
            ChannelService::new(REST_PORT, false, None, None),
            on_replica_failure,
            request_shard_transfer,
            abort_shard_transfer,
            None,
            None,
            ResourceBudget::default(),
            None,
        )
        .await
    }

    async fn insert_bm25_documents(collection: &Collection, documents: &[(u64, &str)]) {
        let bm25 = Bm25::new(Bm25Config::default());
        let points = documents
            .iter()
            .map(|(id, text)| PointStructPersisted {
                id: (*id).into(),
                vector: VectorStructPersisted::Named(HashMap::from([(
                    VECTOR_NAME.into(),
                    bm25.doc_embed(text),
                )])),
                payload: Some(document_payload(text)),
            })
            .collect();

        let insert_points = CollectionUpdateOperations::PointOperation(
            PointOperations::UpsertPoints(PointInsertOperationsInternal::PointsList(points)),
        );

        collection
            .update_from_client_simple(
                insert_points,
                true,
                None,
                WriteOrdering::default(),
                HwMeasurementAcc::disposable(),
            )
            .await
            .unwrap();
    }

    async fn create_fuzzy_text_index(collection: &Collection) {
        collection
            .create_payload_index_with_wait(
                PAYLOAD_FIELD.parse().unwrap(),
                PayloadFieldSchema::FieldParams(PayloadSchemaParams::Text(TextIndexParams {
                    r#type: TextIndexType::Text,
                    tokenizer: TokenizerType::Word,
                    lowercase: Some(true),
                    fuzzy_matching: Some(true),
                    ..Default::default()
                })),
                true,
                HwMeasurementAcc::disposable(),
            )
            .await
            .unwrap();
    }

    async fn search_sparse(
        collection: &Collection,
        vector: SparseVector,
    ) -> Vec<segment::types::ScoredPoint> {
        let request = SearchRequestInternal {
            vector: NamedVectorStruct::Sparse(NamedSparseVector {
                name: VECTOR_NAME.into(),
                vector,
            }),
            filter: None,
            params: None,
            limit: 5,
            offset: None,
            with_payload: Some(WithPayloadInterface::Bool(true)),
            with_vector: None,
            score_threshold: Some(0.0001),
        };

        collection
            .core_search_batch(
                CoreSearchRequestBatch {
                    searches: vec![request.into()],
                },
                None,
                ShardSelectorInternal::All,
                None,
                HwMeasurementAcc::disposable(),
            )
            .await
            .unwrap()
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    fn document_payload(text: &str) -> Payload {
        serde_json::from_value(serde_json::json!({ PAYLOAD_FIELD: text })).unwrap()
    }

    fn into_sparse(vector: VectorPersisted) -> SparseVector {
        match vector {
            VectorPersisted::Sparse(sparse) => sparse,
            other => panic!("expected sparse vector, got {other:?}"),
        }
    }

    fn unique_term_count(candidates: &[FuzzyCandidate]) -> usize {
        let mut unique = std::collections::HashSet::new();
        for candidate in candidates {
            unique.insert(candidate.term.as_str());
        }
        unique.len()
    }
}
