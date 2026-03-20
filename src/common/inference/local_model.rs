use std::collections::HashMap;

use collection::operations::point_ops::VectorPersisted;
use segment::index::field_index::full_text_index::fuzzy_index::FuzzyCandidate;
use serde_json::Value;
use sparse::common::sparse_vector::SparseVector;
use storage::content_manager::errors::StorageError;

use super::bm25::Bm25;
use super::service::{InferenceInput, InferenceType};
use crate::common::inference::inference_input::InferenceDataType;

enum LocalModelName {
    Bm25,
}

impl LocalModelName {
    fn from_str(model_name: &str) -> Option<Self> {
        match model_name.to_lowercase().as_str() {
            "qdrant/bm25" => Some(LocalModelName::Bm25),
            "bm25" => Some(LocalModelName::Bm25),
            _ => None,
        }
    }
}

/// Run inference with only local models.
///
/// # Panics
/// Panics if one inference input did not target a local model.
pub fn infer_local(
    inference_inputs: Vec<InferenceInput>,
    inference_type: InferenceType,
) -> Result<Vec<VectorPersisted>, StorageError> {
    let mut out = Vec::with_capacity(inference_inputs.len());

    for input in inference_inputs {
        let InferenceInput {
            data,
            data_type,
            model,
            options,
        } = input;

        let Some(model_name) = LocalModelName::from_str(&model) else {
            unreachable!(
                "Non local model has been passed to infer_local(). This can happen if a newly added model wasn't added to infer_local()"
            )
        };

        // Validate it is text
        match data_type {
            InferenceDataType::Text => {}
            InferenceDataType::Image | InferenceDataType::Object => {
                return Err(StorageError::bad_input(format!(
                    "Only text input is supported for {model}."
                )));
            }
        };

        let input_str = data.as_str().ok_or_else(|| {
            StorageError::bad_input(format!("Only text input is supported for {model}."))
        })?;

        let embedding = match model_name {
            LocalModelName::Bm25 => {
                let bm25_config = InferenceInput::parse_bm25_config(options)?;
                let bm25 = Bm25::new(bm25_config);

                match inference_type {
                    InferenceType::Update => bm25.doc_embed(input_str),
                    InferenceType::Search => bm25.search_embed(input_str),
                }
            }
        };

        out.push(embedding);
    }

    Ok(out)
}

/// Expands fuzzy FST candidates into a `SparseVector` using a local model's own mapping
/// strategy. Each local model variant must implement its own expansion here, mirroring the
/// dispatch in `infer_local`.
///
/// # Panics
/// Panics if `model_name` does not resolve to a known local model.
pub fn expand_fuzzy_candidates(
    model_name: &str,
    candidates: &[FuzzyCandidate],
    options: Option<HashMap<String, Value>>,
) -> SparseVector {
    let Some(model) = LocalModelName::from_str(model_name) else {
        unreachable!(
            "Non-local model passed to expand_fuzzy_candidates(). \
             This can happen if a newly added model wasn't added to expand_fuzzy_candidates()"
        )
    };

    match model {
        LocalModelName::Bm25 => bm25_expand(candidates, options),
    }
}

/// BM25 expansion: stems each candidate term using the same tokenization pipeline as
/// doc/search embed, then maps to a sparse dimension via murmur3 hash.
/// Takes the max weight per dimension across all candidates.
fn bm25_expand(
    candidates: &[FuzzyCandidate],
    options: Option<HashMap<String, Value>>,
) -> SparseVector {
    let bm25_config = InferenceInput::parse_bm25_config(options).unwrap_or_default();
    let bm25 = Bm25::new(bm25_config);
    let mut dim_map: HashMap<u32, f32> = HashMap::new();

    for candidate in candidates {
        if candidate.weight <= 0.0 {
            continue;
        }
        // Stem the candidate term through the same pipeline (lowercase, stopwords, stemming)
        // that doc_embed/search_embed use, so hashed dimensions match the index.
        let stemmed_tokens = bm25.tokenize(&candidate.term);
        for token in &stemmed_tokens {
            let dim_id = Bm25::compute_token_id(token);
            dim_map
                .entry(dim_id)
                .and_modify(|b| *b = b.max(candidate.weight))
                .or_insert(candidate.weight);
        }
    }

    let mut pairs: Vec<_> = dim_map.into_iter().collect();
    pairs.sort_by_key(|(idx, _)| *idx);
    SparseVector {
        indices: pairs.iter().map(|(i, _)| *i).collect(),
        values: pairs.iter().map(|(_, v)| *v).collect(),
    }
}

/// Returns `true` if the provided `model_name` targets a local model. Local models
/// are models that are handled by Qdrant and are not forwarded to a remote inference service.
pub fn is_local_model(model_name: &str) -> bool {
    let model_name = LocalModelName::from_str(model_name);
    model_name.is_some()
}
