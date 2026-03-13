use api::rest::models::InferenceUsage;
use api::rest::schema as rest;
use collection::lookup::WithLookup;
use collection::operations::universal_query::collection_query::{
    CollectionPrefetch, CollectionQueryGroupsRequest, CollectionQueryRequest, FeedbackInternal,
    FeedbackStrategy, Mmr, NearestWithMmr, Query, VectorInputInternal, VectorQuery,
};
use collection::operations::universal_query::formula::FormulaInternal;
use collection::operations::universal_query::shard_query::{FusionInternal, SampleInternal};
use ordered_float::OrderedFloat;
use segment::data_types::order_by::OrderBy;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, MultiDenseVectorInternal, VectorInternal};
use segment::index::field_index::full_text_index::tokenizers::Stemmer;
use segment::types::SearchParams;
use segment::vector_storage::query::{
    ContextPair, ContextQuery, DiscoveryQuery, FeedbackItem, RecoQuery,
};
use shard::search::FuzzyBm25Context;
use storage::content_manager::errors::{StorageError, StorageResult};

use crate::common::inference::batch_processing::{
    collect_query_groups_request, collect_query_request,
};
use crate::common::inference::infer_processing::BatchAccumInferred;
use crate::common::inference::params::InferenceParams;
use crate::common::inference::service::{InferenceData, InferenceType};

pub struct CollectionQueryRequestWithUsage {
    pub request: CollectionQueryRequest,
    pub usage: Option<InferenceUsage>,
}

pub struct CollectionQueryGroupsRequestWithUsage {
    pub request: CollectionQueryGroupsRequest,
    pub usage: Option<InferenceUsage>,
}

pub async fn convert_query_groups_request_from_rest(
    request: rest::QueryGroupsRequestInternal,
    inference_params: InferenceParams,
) -> Result<CollectionQueryGroupsRequestWithUsage, StorageError> {
    let batch = collect_query_groups_request(&request);
    let rest::QueryGroupsRequestInternal {
        prefetch,
        query,
        using,
        filter,
        score_threshold,
        params,
        with_vector,
        with_payload,
        lookup_from,
        group_request,
    } = request;

    let (inferred, usage) =
        BatchAccumInferred::from_batch_accum(batch, InferenceType::Search, &inference_params)
            .await?;
    let query = query
        .map(|q| convert_query_with_inferred(q, &inferred))
        .transpose()?;

    let prefetch = prefetch
        .map(|prefetches| {
            prefetches
                .into_iter()
                .map(|p| convert_prefetch_with_inferred(p, &inferred))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();

    let collection_query_groups_request = CollectionQueryGroupsRequest {
        prefetch,
        query,
        using: using.unwrap_or_else(|| DEFAULT_VECTOR_NAME.to_owned()),
        filter,
        score_threshold,
        params,
        with_vector: with_vector.unwrap_or(CollectionQueryRequest::DEFAULT_WITH_VECTOR),
        with_payload: with_payload.unwrap_or(CollectionQueryRequest::DEFAULT_WITH_PAYLOAD),
        lookup_from,
        limit: group_request
            .limit
            .unwrap_or(CollectionQueryRequest::DEFAULT_LIMIT),
        group_by: group_request.group_by,
        group_size: group_request
            .group_size
            .unwrap_or(CollectionQueryRequest::DEFAULT_GROUP_SIZE),
        with_lookup: group_request.with_lookup.map(WithLookup::from),
    };

    Ok(CollectionQueryGroupsRequestWithUsage {
        request: collection_query_groups_request,
        usage,
    })
}

pub async fn convert_query_request_from_rest(
    request: rest::QueryRequestInternal,
    inference_params: &InferenceParams,
) -> Result<CollectionQueryRequestWithUsage, StorageError> {
    let batch = collect_query_request(&request);
    let (inferred, usage) =
        BatchAccumInferred::from_batch_accum(batch, InferenceType::Search, inference_params)
            .await?;

    let rest::QueryRequestInternal {
        prefetch,
        query,
        using,
        filter,
        score_threshold,
        params,
        limit,
        offset,
        with_vector,
        with_payload,
        lookup_from,
    } = request;

    let prefetch = prefetch
        .map(|prefetches| {
            prefetches
                .into_iter()
                .map(|p| convert_prefetch_with_inferred(p, &inferred))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();

    // Extract fuzzy BM25 context before consuming the query
    let fuzzy_context = extract_fuzzy_context(&query, &params);

    // Validate fuzzy parameter constraints (§6.4)
    if let Some(ref search_params) = params
        && search_params.fuzzy.is_some()
    {
        validate_fuzzy_constraints(&query, &prefetch)?;
    }

    let query = query
        .map(|q| convert_query_with_inferred(q, &inferred))
        .transpose()?;

    let collection_query_request = CollectionQueryRequest {
        prefetch,
        query,
        using: using.unwrap_or_else(|| DEFAULT_VECTOR_NAME.to_owned()),
        filter,
        score_threshold,
        limit: limit.unwrap_or(CollectionQueryRequest::DEFAULT_LIMIT),
        offset: offset.unwrap_or(CollectionQueryRequest::DEFAULT_OFFSET),
        params,
        with_vector: with_vector.unwrap_or(CollectionQueryRequest::DEFAULT_WITH_VECTOR),
        with_payload: with_payload.unwrap_or(CollectionQueryRequest::DEFAULT_WITH_PAYLOAD),
        lookup_from,
        fuzzy_context,
    };
    Ok(CollectionQueryRequestWithUsage {
        request: collection_query_request,
        usage,
    })
}

/// Extract fuzzy BM25 context from the query if fuzzy search is enabled
/// and the query is a BM25 document.
fn extract_fuzzy_context(
    query: &Option<rest::QueryInterface>,
    params: &Option<SearchParams>,
) -> Option<FuzzyBm25Context> {
    let fuzzy_params = params.as_ref()?.fuzzy?;

    let doc = match query.as_ref()? {
        rest::QueryInterface::Nearest(rest::VectorInput::Document(doc)) => doc,
        rest::QueryInterface::Query(rest::Query::Nearest(rest::NearestQuery {
            nearest: rest::VectorInput::Document(doc),
            ..
        })) => doc,
        _ => return None,
    };

    if !super::local_model::is_local_model(&doc.model) {
        return None;
    }

    let options = doc.options.clone().map(|o| o.into_options());
    let bm25_config = super::inference_input::InferenceInput::parse_bm25_config(options).ok()?;

    // Build the stemmer from BM25 config BEFORE consuming it.
    // Expanded FST terms must be stemmed with the same stemmer used during indexing
    // so that dim_ids match (e.g., "insurance" → stem → "insur" → hash).
    let stemmer = build_bm25_stemmer(&bm25_config.text_preprocessing_config);

    let bm25 = super::bm25::Bm25::new(bm25_config);
    let tokens: Vec<String> = bm25
        .tokenize(&doc.text)
        .into_iter()
        .map(|t| t.into_owned())
        .collect();

    Some(FuzzyBm25Context {
        tokens,
        fuzzy_params,
        stemmer,
    })
}

/// Build the BM25 stemmer from the text preprocessing config.
///
/// This replicates the stemmer construction logic from `Bm25::new()` so that
/// fuzzy expansion can stem expanded terms before hashing to dim_ids.
fn build_bm25_stemmer(config: &rest::TextPreprocessingConfig) -> Option<Stemmer> {
    let language = config.language.as_deref().unwrap_or("english");
    match &config.stemmer {
        None => Stemmer::try_default_from_language(language),
        Some(stemmer_algorithm) => Some(Stemmer::from_algorithm(stemmer_algorithm)),
    }
}

/// Validate that fuzzy parameters are not combined with incompatible features.
///
/// Returns 400 for:
/// - Raw sparse vector (indices+values) + fuzzy (original tokens are lost)
/// - Prefetch + fuzzy (not supported in current version)
/// - Non-sparse, non-document vector + fuzzy (only applicable to BM25/sparse)
fn validate_fuzzy_constraints(
    query: &Option<rest::QueryInterface>,
    prefetch: &[CollectionPrefetch],
) -> StorageResult<()> {
    if !prefetch.is_empty() {
        return Err(StorageError::bad_request(
            "Fuzzy search cannot be used with prefetch queries",
        ));
    }

    if let Some(q) = query.as_ref() {
        let is_invalid = match q {
            // Raw sparse vector + fuzzy → 400
            rest::QueryInterface::Nearest(rest::VectorInput::SparseVector(_)) => Some(
                "Fuzzy search cannot be used with raw sparse vectors (indices+values). Use a Document query instead.",
            ),
            // Dense/multi-dense + fuzzy → 400
            rest::QueryInterface::Nearest(
                rest::VectorInput::DenseVector(_) | rest::VectorInput::MultiDenseVector(_),
            ) => Some("Fuzzy search is only applicable to BM25/sparse vector queries"),
            rest::QueryInterface::Query(rest::Query::Nearest(rest::NearestQuery {
                nearest: rest::VectorInput::SparseVector(_),
                ..
            })) => Some(
                "Fuzzy search cannot be used with raw sparse vectors (indices+values). Use a Document query instead.",
            ),
            rest::QueryInterface::Query(rest::Query::Nearest(rest::NearestQuery {
                nearest: rest::VectorInput::DenseVector(_) | rest::VectorInput::MultiDenseVector(_),
                ..
            })) => Some("Fuzzy search is only applicable to BM25/sparse vector queries"),
            _ => None,
        };
        if let Some(msg) = is_invalid {
            return Err(StorageError::bad_request(msg));
        }
    }

    Ok(())
}

fn convert_vector_input_with_inferred(
    vector: rest::VectorInput,
    inferred: &BatchAccumInferred,
) -> Result<VectorInputInternal, StorageError> {
    match vector {
        rest::VectorInput::Id(id) => Ok(VectorInputInternal::Id(id)),
        rest::VectorInput::DenseVector(dense) => {
            Ok(VectorInputInternal::Vector(VectorInternal::Dense(dense)))
        }
        rest::VectorInput::SparseVector(sparse) => {
            Ok(VectorInputInternal::Vector(VectorInternal::Sparse(sparse)))
        }
        rest::VectorInput::MultiDenseVector(multi_dense) => Ok(VectorInputInternal::Vector(
            VectorInternal::MultiDense(MultiDenseVectorInternal::new_unchecked(multi_dense)),
        )),
        rest::VectorInput::Document(doc) => {
            let data = InferenceData::Document(doc);
            let vector = inferred.get_vector(&data).ok_or_else(|| {
                StorageError::inference_error("Missing inferred vector for document")
            })?;
            Ok(VectorInputInternal::Vector(VectorInternal::from(
                vector.clone(),
            )))
        }
        rest::VectorInput::Image(img) => {
            let data = InferenceData::Image(img);
            let vector = inferred.get_vector(&data).ok_or_else(|| {
                StorageError::inference_error("Missing inferred vector for image")
            })?;
            Ok(VectorInputInternal::Vector(VectorInternal::from(
                vector.clone(),
            )))
        }
        rest::VectorInput::Object(obj) => {
            let data = InferenceData::Object(obj);
            let vector = inferred.get_vector(&data).ok_or_else(|| {
                StorageError::inference_error("Missing inferred vector for object")
            })?;
            Ok(VectorInputInternal::Vector(VectorInternal::from(
                vector.clone(),
            )))
        }
    }
}

fn convert_query_with_inferred(
    query: rest::QueryInterface,
    inferred: &BatchAccumInferred,
) -> StorageResult<Query> {
    let query = rest::Query::from(query);
    match query {
        rest::Query::Nearest(rest::NearestQuery { nearest, mmr }) => {
            let vector = convert_vector_input_with_inferred(nearest, inferred)?;

            if let Some(mmr) = mmr {
                let mmr = Mmr {
                    diversity: mmr.diversity,
                    candidates_limit: mmr.candidates_limit,
                };
                Ok(Query::Vector(VectorQuery::NearestWithMmr(NearestWithMmr {
                    nearest: vector,
                    mmr,
                })))
            } else {
                Ok(Query::Vector(VectorQuery::Nearest(vector)))
            }
        }
        rest::Query::Recommend(recommend) => {
            let rest::RecommendInput {
                positive,
                negative,
                strategy,
            } = recommend.recommend;
            let positives = positive
                .into_iter()
                .flatten()
                .map(|v| convert_vector_input_with_inferred(v, inferred))
                .collect::<Result<Vec<_>, _>>()?;
            let negatives = negative
                .into_iter()
                .flatten()
                .map(|v| convert_vector_input_with_inferred(v, inferred))
                .collect::<Result<Vec<_>, _>>()?;
            let reco_query = RecoQuery::new(positives, negatives);
            match strategy.unwrap_or_default() {
                rest::RecommendStrategy::AverageVector => Ok(Query::Vector(
                    VectorQuery::RecommendAverageVector(reco_query),
                )),
                rest::RecommendStrategy::BestScore => {
                    Ok(Query::Vector(VectorQuery::RecommendBestScore(reco_query)))
                }
                rest::RecommendStrategy::SumScores => {
                    Ok(Query::Vector(VectorQuery::RecommendSumScores(reco_query)))
                }
            }
        }
        rest::Query::Discover(discover) => {
            let rest::DiscoverInput { target, context } = discover.discover;
            let target = convert_vector_input_with_inferred(target, inferred)?;
            let context = context
                .into_iter()
                .flatten()
                .map(|pair| context_pair_from_rest_with_inferred(pair, inferred))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Query::Vector(VectorQuery::Discover(DiscoveryQuery::new(
                target, context,
            ))))
        }
        rest::Query::Context(context) => {
            let rest::ContextInput(context) = context.context;
            let context = context
                .into_iter()
                .flatten()
                .map(|pair| context_pair_from_rest_with_inferred(pair, inferred))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Query::Vector(VectorQuery::Context(ContextQuery::new(
                context,
            ))))
        }
        rest::Query::OrderBy(order_by) => Ok(Query::OrderBy(OrderBy::from(order_by.order_by))),
        rest::Query::Fusion(fusion) => Ok(Query::Fusion(FusionInternal::from(fusion.fusion))),
        rest::Query::Rrf(rrf) => Ok(Query::Fusion(FusionInternal::from(rrf.rrf))),
        rest::Query::Formula(formula) => Ok(Query::Formula(FormulaInternal::from(formula))),
        rest::Query::Sample(sample) => Ok(Query::Sample(SampleInternal::from(sample.sample))),
        rest::Query::RelevanceFeedback(relevance_feedback) => {
            let rest::RelevanceFeedbackInput {
                target,
                feedback,
                strategy,
            } = relevance_feedback.relevance_feedback;

            let target = convert_vector_input_with_inferred(target, inferred)?;
            let feedback = feedback
                .into_iter()
                .map(|item| {
                    Ok(FeedbackItem {
                        vector: convert_vector_input_with_inferred(item.example, inferred)?,
                        score: item.score.into(),
                    })
                })
                .collect::<StorageResult<Vec<_>>>()?;

            let strategy = FeedbackStrategy::from(strategy);

            Ok(Query::Vector(VectorQuery::Feedback(FeedbackInternal {
                target,
                feedback,
                strategy,
            })))
        }
    }
}

fn convert_prefetch_with_inferred(
    prefetch: rest::Prefetch,
    inferred: &BatchAccumInferred,
) -> Result<CollectionPrefetch, StorageError> {
    let rest::Prefetch {
        prefetch,
        query,
        using,
        filter,
        score_threshold,
        params,
        limit,
        lookup_from,
    } = prefetch;

    let query = query
        .map(|q| convert_query_with_inferred(q, inferred))
        .transpose()?;
    let nested_prefetches = prefetch
        .map(|prefetches| {
            prefetches
                .into_iter()
                .map(|p| convert_prefetch_with_inferred(p, inferred))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();

    Ok(CollectionPrefetch {
        prefetch: nested_prefetches,
        query,
        using: using.unwrap_or_else(|| DEFAULT_VECTOR_NAME.to_owned()),
        filter,
        score_threshold: score_threshold.map(OrderedFloat),
        limit: limit.unwrap_or(CollectionQueryRequest::DEFAULT_LIMIT),
        params,
        lookup_from,
    })
}

fn context_pair_from_rest_with_inferred(
    value: rest::ContextPair,
    inferred: &BatchAccumInferred,
) -> Result<ContextPair<VectorInputInternal>, StorageError> {
    let rest::ContextPair { positive, negative } = value;
    Ok(ContextPair {
        positive: convert_vector_input_with_inferred(positive, inferred)?,
        negative: convert_vector_input_with_inferred(negative, inferred)?,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use api::rest::schema::{Document, Image, InferenceObject, NearestQuery};
    use collection::operations::point_ops::VectorPersisted;
    use serde_json::json;

    use super::*;

    fn create_test_document(text: &str) -> Document {
        Document {
            text: text.to_string(),
            model: "test-model".to_string(),
            options: Default::default(),
        }
    }

    fn create_test_image(url: &str) -> Image {
        Image {
            image: json!({"data": url.to_string()}),
            model: "test-model".to_string(),
            options: Default::default(),
        }
    }

    fn create_test_object(data: &str) -> InferenceObject {
        InferenceObject {
            object: json!({"data": data}),
            model: "test-model".to_string(),
            options: Default::default(),
        }
    }

    fn create_test_inferred_batch() -> BatchAccumInferred {
        let mut objects = HashMap::new();

        let doc = InferenceData::Document(create_test_document("test"));
        let img = InferenceData::Image(create_test_image("test.jpg"));
        let obj = InferenceData::Object(create_test_object("test"));

        let dense_vector = vec![1.0, 2.0, 3.0];
        let vector_persisted = VectorPersisted::Dense(dense_vector);

        objects.insert(doc, vector_persisted.clone());
        objects.insert(img, vector_persisted.clone());
        objects.insert(obj, vector_persisted);

        BatchAccumInferred { objects }
    }

    #[test]
    fn test_convert_vector_input_with_inferred_dense() {
        let inferred = create_test_inferred_batch();
        let vector = rest::VectorInput::DenseVector(vec![1.0, 2.0, 3.0]);

        let result = convert_vector_input_with_inferred(vector, &inferred).unwrap();
        match result {
            VectorInputInternal::Vector(VectorInternal::Dense(values)) => {
                assert_eq!(values, vec![1.0, 2.0, 3.0]);
            }
            _ => panic!("Expected dense vector"),
        }
    }

    #[test]
    fn test_convert_vector_input_with_inferred_document() {
        let inferred = create_test_inferred_batch();
        let doc = create_test_document("test");
        let vector = rest::VectorInput::Document(doc);

        let result = convert_vector_input_with_inferred(vector, &inferred).unwrap();
        match result {
            VectorInputInternal::Vector(VectorInternal::Dense(values)) => {
                assert_eq!(values, vec![1.0, 2.0, 3.0]);
            }
            _ => panic!("Expected dense vector from inference"),
        }
    }

    #[test]
    fn test_convert_vector_input_with_inferred_missing() {
        let inferred = create_test_inferred_batch();
        let doc = create_test_document("missing");
        let vector = rest::VectorInput::Document(doc);

        let result = convert_vector_input_with_inferred(vector, &inferred);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Missing inferred vector"),
        );
    }

    #[test]
    fn test_context_pair_from_rest_with_inferred() {
        let inferred = create_test_inferred_batch();
        let pair = rest::ContextPair {
            positive: rest::VectorInput::DenseVector(vec![1.0, 2.0, 3.0]),
            negative: rest::VectorInput::Document(create_test_document("test")),
        };

        let result = context_pair_from_rest_with_inferred(pair, &inferred).unwrap();
        match (result.positive, result.negative) {
            (
                VectorInputInternal::Vector(VectorInternal::Dense(pos)),
                VectorInputInternal::Vector(VectorInternal::Dense(neg)),
            ) => {
                assert_eq!(pos, vec![1.0, 2.0, 3.0]);
                assert_eq!(neg, vec![1.0, 2.0, 3.0]);
            }
            _ => panic!("Expected dense vectors"),
        }
    }

    #[test]
    fn test_convert_query_with_inferred_nearest() {
        let inferred = create_test_inferred_batch();
        let nearest = NearestQuery {
            nearest: rest::VectorInput::Document(create_test_document("test")),
            mmr: None,
        };
        let query = rest::QueryInterface::Query(rest::Query::Nearest(nearest));

        let result = convert_query_with_inferred(query, &inferred).unwrap();
        match result {
            Query::Vector(VectorQuery::Nearest(vector)) => match vector {
                VectorInputInternal::Vector(VectorInternal::Dense(values)) => {
                    assert_eq!(values, vec![1.0, 2.0, 3.0]);
                }
                _ => panic!("Expected dense vector"),
            },
            _ => panic!("Expected nearest query"),
        }
    }

    // ---------------------------------------------------------------
    // extract_fuzzy_context tests
    // ---------------------------------------------------------------

    fn create_bm25_document(text: &str) -> Document {
        Document {
            text: text.to_string(),
            model: "qdrant/bm25".to_string(),
            options: Default::default(),
        }
    }

    #[test]
    fn test_extract_fuzzy_context_no_params() {
        // No fuzzy params → None
        let query = Some(rest::QueryInterface::Nearest(rest::VectorInput::Document(
            create_bm25_document("hello world"),
        )));
        let params: Option<SearchParams> = None;
        let result = extract_fuzzy_context(&query, &params);
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_fuzzy_context_no_query() {
        // Have fuzzy params but no query → None
        let query: Option<rest::QueryInterface> = None;
        let params = Some(SearchParams {
            fuzzy: Some(segment::types::FuzzyParams {
                max_edit: 1,
                prefix_length: 0,
                max_expansions: 30,
            }),
            ..Default::default()
        });
        let result = extract_fuzzy_context(&query, &params);
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_fuzzy_context_with_bm25_document_direct() {
        // Direct VectorInput::Document with BM25 model → should extract tokens
        let doc = create_bm25_document("hello world fuzzy test");
        let query = Some(rest::QueryInterface::Nearest(rest::VectorInput::Document(
            doc,
        )));
        let fuzzy_params = segment::types::FuzzyParams {
            max_edit: 1,
            prefix_length: 0,
            max_expansions: 30,
        };
        let params = Some(SearchParams {
            fuzzy: Some(fuzzy_params),
            ..Default::default()
        });
        let result = extract_fuzzy_context(&query, &params);
        assert!(result.is_some());
        let ctx = result.unwrap();
        assert!(!ctx.tokens.is_empty());
        assert_eq!(ctx.fuzzy_params, fuzzy_params);
    }

    #[test]
    fn test_extract_fuzzy_context_with_bm25_document_nested() {
        // Nested Query::Nearest(NearestQuery { nearest: Document }) → should extract tokens
        let doc = create_bm25_document("the quick brown fox");
        let query = Some(rest::QueryInterface::Query(rest::Query::Nearest(
            NearestQuery {
                nearest: rest::VectorInput::Document(doc),
                mmr: None,
            },
        )));
        let fuzzy_params = segment::types::FuzzyParams {
            max_edit: 2,
            prefix_length: 1,
            max_expansions: 10,
        };
        let params = Some(SearchParams {
            fuzzy: Some(fuzzy_params),
            ..Default::default()
        });
        let result = extract_fuzzy_context(&query, &params);
        assert!(result.is_some());
        let ctx = result.unwrap();
        assert!(!ctx.tokens.is_empty());
        assert_eq!(ctx.fuzzy_params, fuzzy_params);
    }

    #[test]
    fn test_extract_fuzzy_context_with_dense_vector() {
        // Dense vector + fuzzy → None (not a Document)
        let query = Some(rest::QueryInterface::Nearest(
            rest::VectorInput::DenseVector(vec![1.0, 2.0, 3.0]),
        ));
        let params = Some(SearchParams {
            fuzzy: Some(segment::types::FuzzyParams {
                max_edit: 1,
                prefix_length: 0,
                max_expansions: 30,
            }),
            ..Default::default()
        });
        let result = extract_fuzzy_context(&query, &params);
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_fuzzy_context_non_local_model() {
        // Non-local model (not "qdrant/bm25" or "bm25") → None
        let doc = Document {
            text: "hello world".to_string(),
            model: "openai/text-embedding-3-small".to_string(),
            options: Default::default(),
        };
        let query = Some(rest::QueryInterface::Nearest(rest::VectorInput::Document(
            doc,
        )));
        let params = Some(SearchParams {
            fuzzy: Some(segment::types::FuzzyParams {
                max_edit: 1,
                prefix_length: 0,
                max_expansions: 30,
            }),
            ..Default::default()
        });
        let result = extract_fuzzy_context(&query, &params);
        assert!(result.is_none());
    }

    // ---------------------------------------------------------------
    // validate_fuzzy_constraints tests
    // ---------------------------------------------------------------

    #[test]
    fn test_validate_fuzzy_constraints_prefetch_error() {
        // Fuzzy + prefetch → 400
        let query = Some(rest::QueryInterface::Nearest(rest::VectorInput::Document(
            create_bm25_document("test"),
        )));
        let prefetch = vec![CollectionPrefetch {
            prefetch: vec![],
            query: None,
            using: String::new(),
            filter: None,
            score_threshold: None,
            limit: 10,
            params: None,
            lookup_from: None,
        }];
        let result = validate_fuzzy_constraints(&query, &prefetch);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("prefetch"));
    }

    #[test]
    fn test_validate_fuzzy_constraints_raw_sparse_error() {
        // Raw sparse vector + fuzzy → 400
        // Use JSON deserialization since `sparse` crate is not a direct dependency
        let sparse_input: rest::VectorInput = serde_json::from_value(json!({
            "indices": [1, 2],
            "values": [1.0, 0.5]
        }))
        .unwrap();
        let query = Some(rest::QueryInterface::Nearest(sparse_input));
        let result = validate_fuzzy_constraints(&query, &[]);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("raw sparse vectors"),
        );
    }

    #[test]
    fn test_validate_fuzzy_constraints_dense_error() {
        // Dense vector + fuzzy → 400
        let query = Some(rest::QueryInterface::Nearest(
            rest::VectorInput::DenseVector(vec![1.0, 2.0]),
        ));
        let result = validate_fuzzy_constraints(&query, &[]);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("BM25/sparse vector"),
        );
    }

    #[test]
    fn test_validate_fuzzy_constraints_multi_dense_error() {
        // Multi-dense vector + fuzzy → 400
        let query = Some(rest::QueryInterface::Nearest(
            rest::VectorInput::MultiDenseVector(vec![vec![1.0, 2.0]]),
        ));
        let result = validate_fuzzy_constraints(&query, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_fuzzy_constraints_nested_sparse_error() {
        // Nested Query::Nearest with raw sparse → 400
        let sparse_input: rest::VectorInput = serde_json::from_value(json!({
            "indices": [1],
            "values": [1.0]
        }))
        .unwrap();
        let query = Some(rest::QueryInterface::Query(rest::Query::Nearest(
            NearestQuery {
                nearest: sparse_input,
                mmr: None,
            },
        )));
        let result = validate_fuzzy_constraints(&query, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_fuzzy_constraints_document_ok() {
        // Document query + fuzzy → OK
        let query = Some(rest::QueryInterface::Nearest(rest::VectorInput::Document(
            create_bm25_document("test"),
        )));
        let result = validate_fuzzy_constraints(&query, &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_fuzzy_constraints_no_query_ok() {
        // No query + empty prefetch → OK
        let result = validate_fuzzy_constraints(&None, &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_fuzzy_constraints_id_query_ok() {
        // ID query + fuzzy → OK (not a vector type we restrict)
        let query = Some(rest::QueryInterface::Nearest(rest::VectorInput::Id(
            1.into(),
        )));
        let result = validate_fuzzy_constraints(&query, &[]);
        assert!(result.is_ok());
    }
}
