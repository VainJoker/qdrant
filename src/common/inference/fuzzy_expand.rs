use std::collections::HashMap;
use std::time::Duration;

use api::grpc::qdrant as grpc;
use api::rest::schema as rest;
use api::rest::schema::VectorInput;
use collection::operations::point_ops::VectorPersisted;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use segment::data_types::vectors::DEFAULT_VECTOR_NAME;
use segment::index::field_index::full_text_index::fuzzy_index::FuzzyCandidate;
use segment::types::FuzzyParams;
use sparse::common::sparse_vector::SparseVector;
use storage::content_manager::errors::StorageError;
use storage::content_manager::toc::TableOfContent;
use storage::rbac::Auth;

use super::bm25::Bm25;
use super::inference_input::{InferenceDataType, InferenceInput};
use super::local_model::is_local_model;
use super::params::InferenceParams;
use super::service::{InferenceService, InferenceType};

/// Maps fuzzy-expanded candidate terms into a `SparseVector` in the model's native
/// dimension space.
///
/// - `LocalBm25`: zero-latency local hash (`murmur3(term)` → dimension), preserves
///   edit-distance-based candidate weights. Used for BM25 and any other locally-handled model.
/// - `RemoteInference`: joins the candidate terms into a space-separated string and
///   calls the configured inference service. The model returns a proper sparse vector
///   in its own vocabulary space. Used for remote sparse models (SPLADE, etc.).
enum CandidateMapper {
    /// BM25 / local-model path: deterministic hash, no network hop.
    LocalBm25,
    /// Remote sparse model: infer on the joined candidate text.
    RemoteInference {
        model: String,
        inference_params: InferenceParams,
    },
}

impl CandidateMapper {
    fn for_model(model: &str, inference_params: InferenceParams) -> Self {
        if is_local_model(model) {
            CandidateMapper::LocalBm25
        } else {
            CandidateMapper::RemoteInference {
                model: model.to_owned(),
                inference_params,
            }
        }
    }

    async fn map_to_sparse(
        &self,
        candidates: &[FuzzyCandidate],
    ) -> Result<SparseVector, StorageError> {
        match self {
            CandidateMapper::LocalBm25 => Ok(bm25_local_expand(candidates)),

            CandidateMapper::RemoteInference {
                model,
                inference_params,
            } => {
                let service = InferenceService::get_global().ok_or_else(|| {
                    StorageError::service_error(
                        "Fuzzy expansion for non-BM25 models requires an \
                         inference service. Please configure the inference \
                         address in the Qdrant config.",
                    )
                })?;

                // Infer each candidate term individually (batched in one request)
                // so we get the model's own dimension IDs for each term.
                // We then scale the returned sparse vector by the FST edit-distance
                // weight, preserving the fuzzy proximity signal.
                let inputs: Vec<InferenceInput> = candidates
                    .iter()
                    .map(|c| InferenceInput {
                        data: serde_json::Value::String(c.term.clone()),
                        data_type: InferenceDataType::Text,
                        model: model.clone(),
                        options: None,
                    })
                    .collect();

                let response = service
                    .infer(inputs, InferenceType::Search, inference_params.clone())
                    .await?;

                if response.embeddings.len() != candidates.len() {
                    return Err(StorageError::service_error(format!(
                        "Inference service returned {} embeddings for {} fuzzy candidates",
                        response.embeddings.len(),
                        candidates.len(),
                    )));
                }

                // Merge: for each candidate, scale its sparse vector by the FST
                // edit-distance weight, then take max per dimension across candidates.
                // Exact matches (weight=1.0) pass at full strength;
                // 1-edit matches (weight≈0.83) are proportionally discounted.
                let mut dim_map: HashMap<u32, f32> = HashMap::new();
                for (candidate, vector) in candidates.iter().zip(response.embeddings) {
                    let VectorPersisted::Sparse(sv) = vector else {
                        return Err(StorageError::bad_input(format!(
                            "Fuzzy expansion requires a sparse model; \
                             model '{model}' returned a non-sparse vector",
                        )));
                    };
                    for (&dim_id, &model_weight) in sv.indices.iter().zip(sv.values.iter()) {
                        let scaled = model_weight * candidate.weight;
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
        }
    }
}

/// Attempts to expand a fuzzy query before it goes through normal inference.
///
/// If the request has `params.fuzzy` set and the query is a Document query, this function:
/// 1. Passes the raw text to the collection's FST index pipeline (tokenized per shard)
/// 2. Aggregated candidates are mapped to a SparseVector via `CandidateMapper`:
///    - Local models (BM25): deterministic hash, no network call
///    - Remote sparse models (SPLADE, etc.): joins candidates as text, re-infers
/// 3. Replaces the query in-place with the expanded SparseVector
///
/// If conditions are not met, the request is left unchanged.
pub async fn try_expand_fuzzy_query(
    request: &mut rest::QueryRequestInternal,
    collection_name: &str,
    toc: &TableOfContent,
    auth: Auth,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
    inference_params: InferenceParams,
) -> Result<(), StorageError> {
    // 1. Extract fuzzy params — if not set, skip entirely
    let fuzzy_params = match request
        .params
        .as_ref()
        .and_then(|p| p.fuzzy.as_ref())
    {
        Some(params) => params.clone(),
        None => return Ok(()), // No fuzzy params, skip
    };

    // Validate: fuzzy is not supported with prefetch
    if request.prefetch.as_ref().is_some_and(|p| !p.is_empty()) {
        return Err(StorageError::bad_input(
            "Fuzzy search params cannot be used together with prefetch queries",
        ));
    }

    // Validate: query must be a Document query when fuzzy is set
    let (doc_text, doc_model) = match &request.query {
        Some(rest::QueryInterface::Nearest(vector_input)) => match vector_input {
            VectorInput::Document(doc) => (doc.text.clone(), doc.model.clone()),
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

    let mapper = CandidateMapper::for_model(&doc_model, inference_params);

    // 2. Get the fuzzy bind field from collection config
    let vector_name = request
        .using
        .clone()
        .unwrap_or_else(|| DEFAULT_VECTOR_NAME.to_owned());

    let bind_field = get_fuzzy_bind_field(toc, collection_name, &vector_name, auth.clone()).await?;
    let Some(bind_field) = bind_field else {
        return Ok(()); // No fuzzy config on sparse vector, silently skip
    };

    // 3. Get fuzzy candidates from the collection's FST index
    let candidates = toc
        .get_fuzzy_candidates(
            collection_name,
            &bind_field,
            &doc_text,
            &fuzzy_params,
            auth,
            timeout,
            hw_measurement_acc,
        )
        .await?;

    if candidates.is_empty() {
        return Ok(()); // No FST candidates — keep original Document query (normal path)
    }

    // 4. Map candidates → SparseVector using the appropriate strategy
    let expanded = mapper.map_to_sparse(&candidates).await?;

    // 5. Replace query with the expanded SparseVector
    request.query = Some(rest::QueryInterface::Nearest(VectorInput::SparseVector(
        expanded,
    )));

    Ok(())
}

/// Get the fuzzy_bind_field from collection's sparse vector config.
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

/// BM25 / local-model expansion: maps each candidate term to a dimension via the
/// same murmur3 hash used during indexing, preserving the edit-distance weight.
fn bm25_local_expand(candidates: &[FuzzyCandidate]) -> SparseVector {
    let mut dim_map: HashMap<u32, f32> = HashMap::new();

    for candidate in candidates {
        if candidate.weight <= 0.0 {
            continue;
        }
        let dim_id = Bm25::compute_token_id(&candidate.term);
        dim_map
            .entry(dim_id)
            .and_modify(|b| *b = b.max(candidate.weight))
            .or_insert(candidate.weight);
    }

    let mut pairs: Vec<_> = dim_map.into_iter().collect();
    pairs.sort_by_key(|(idx, _)| *idx);
    SparseVector {
        indices: pairs.iter().map(|(i, _)| *i).collect(),
        values: pairs.iter().map(|(_, v)| *v).collect(),
    }
}

/// gRPC variant of fuzzy expansion. Works on `grpc::QueryPoints` directly.
pub async fn try_expand_fuzzy_query_grpc(
    request: &mut grpc::QueryPoints,
    toc: &TableOfContent,
    auth: Auth,
    timeout: Option<Duration>,
    hw_measurement_acc: HwMeasurementAcc,
    inference_params: InferenceParams,
) -> Result<(), StorageError> {
    // 1. Extract fuzzy params from search params
    let fuzzy_grpc = match request
        .params
        .as_ref()
        .and_then(|p| p.fuzzy.as_ref())
    {
        Some(params) => params.clone(),
        None => return Ok(()),
    };

    let fuzzy_params = FuzzyParams {
        max_edits: fuzzy_grpc.max_edits.unwrap_or(1) as u8,
        prefix_length: fuzzy_grpc.prefix_length.unwrap_or(0) as u8,
        max_expansions: fuzzy_grpc.max_expansions.unwrap_or(30) as u8,
    };

    // Validate: fuzzy not supported with prefetch
    if !request.prefetch.is_empty() {
        return Err(StorageError::bad_input(
            "Fuzzy search params cannot be used together with prefetch queries",
        ));
    }

    // Validate: query must be a Document query when fuzzy is set
    let (doc_text, doc_model) = match request
        .query
        .as_ref()
        .and_then(|q| q.variant.as_ref())
    {
        Some(grpc::query::Variant::Nearest(vi)) => match vi.variant.as_ref() {
            Some(grpc::vector_input::Variant::Document(doc)) => {
                (doc.text.clone(), doc.model.clone())
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

    let mapper = CandidateMapper::for_model(&doc_model, inference_params);

    // 2. Get fuzzy bind field
    let vector_name = request
        .using
        .clone()
        .unwrap_or_else(|| DEFAULT_VECTOR_NAME.to_owned());

    let bind_field = get_fuzzy_bind_field(
        toc,
        &request.collection_name,
        &vector_name,
        auth.clone(),
    )
    .await?;
    let Some(bind_field) = bind_field else {
        return Ok(());
    };

    // 3. Get fuzzy candidates — shard tokenizes using the field index's tokenizer
    let candidates = toc
        .get_fuzzy_candidates(
            &request.collection_name,
            &bind_field,
            &doc_text,
            &fuzzy_params,
            auth,
            timeout,
            hw_measurement_acc,
        )
        .await?;

    if candidates.is_empty() {
        return Ok(());
    }

    // 4. Map candidates → SparseVector
    let expanded = mapper.map_to_sparse(&candidates).await?;

    let grpc_sparse = grpc::SparseVector {
        indices: expanded.indices,
        values: expanded.values,
    };

    request.query = Some(grpc::Query {
        variant: Some(grpc::query::Variant::Nearest(grpc::VectorInput {
            variant: Some(grpc::vector_input::Variant::Sparse(grpc_sparse)),
        })),
    });

    Ok(())
}
