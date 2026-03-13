use std::sync::Arc;
use std::time::Duration;

use common::counter::hardware_accumulator::HwMeasurementAcc;
use segment::data_types::vectors::VectorInternal;
use segment::json_path::JsonPath;
use segment::types::ScoredPoint;
use shard::common::stopping_guard::StoppingGuard;
use shard::query::fuzzy_resolve::{FuzzyBm25Intent, resolve_fuzzy_intent};
use shard::query::query_enum::QueryEnum;
use shard::search::CoreSearchRequestBatch;
use tokio::runtime::Handle;

use super::LocalShard;
use crate::collection_manager::segments_searcher::SegmentsSearcher;
use crate::config::CollectionConfigInternal;
use crate::operations::types::{CollectionError, CollectionResult};

// Chunk requests for parallelism in certain scenarios
//
// Deeper down, each segment gets its own dedicated search thread. If this shard has just
// one segment, all requests will be executed on a single thread.
//
// To prevent this from being a bottleneck if we have a lot of requests, we can chunk the
// requests into multiple searches to allow more parallelism.
//
// For simplicity, we use a fixed chunk size. Using chunks helps to ensure our 'filter
// reuse optimization' is still properly utilized.
// See: <https://github.com/qdrant/qdrant/pull/813>
// See: <https://github.com/qdrant/qdrant/pull/6326>
const CHUNK_SIZE: usize = 16;

impl LocalShard {
    pub async fn do_search(
        &self,
        core_request: Arc<CoreSearchRequestBatch>,
        search_runtime_handle: &Handle,
        timeout: Duration,
        hw_counter_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        if core_request.searches.is_empty() {
            return Ok(vec![]);
        }

        let skip_batching = if core_request.searches.len() <= CHUNK_SIZE {
            // Don't batch if we have few searches, prevents cloning request
            true
        } else if self.segments.read().len() > self.shared_storage_config.search_thread_count {
            // Don't batch if we have more segments than search threads
            // Not a perfect condition, but it helps to prevent consuming a lot of search threads
            // if the number of segments is large
            // Note: search threads are shared with all other search threads on this Qdrant
            // instance, and other shards also have segments. For simplicity this only considers
            // the global search thread count and local segment count.
            // See: <https://github.com/qdrant/qdrant/pull/6478>
            true
        } else {
            false
        };

        let is_stopped_guard = StoppingGuard::new();

        if skip_batching {
            return self
                .do_search_impl(
                    core_request,
                    search_runtime_handle,
                    timeout,
                    hw_counter_acc,
                    &is_stopped_guard,
                )
                .await;
        }

        // Batch if we have many searches, allows for more parallelism
        let CoreSearchRequestBatch { searches } = core_request.as_ref();

        let chunk_futures = searches
            .chunks(CHUNK_SIZE)
            .map(|chunk| {
                let core_request = CoreSearchRequestBatch {
                    searches: chunk.to_vec(),
                };
                self.do_search_impl(
                    Arc::new(core_request),
                    search_runtime_handle,
                    timeout,
                    hw_counter_acc.clone(),
                    &is_stopped_guard,
                )
            })
            .collect::<Vec<_>>();

        let results = futures::future::try_join_all(chunk_futures)
            .await?
            .into_iter()
            .flatten()
            .collect();

        Ok(results)
    }

    async fn do_search_impl(
        &self,
        core_request: Arc<CoreSearchRequestBatch>,
        search_runtime_handle: &Handle,
        timeout: Duration,
        hw_counter_acc: HwMeasurementAcc,
        is_stopped_guard: &StoppingGuard,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        let start = std::time::Instant::now();

        // Resolve fuzzy BM25 contexts before building query context.
        // This expands fuzzy terms across segments and merges the resulting
        // sparse dimensions into each request's query vector.
        let core_request = {
            let collection_config = self.collection_config.read().await;
            resolve_fuzzy_searches(core_request, &collection_config, &self.segments, timeout)?
        };

        let (query_context, collection_params) = {
            let collection_config = self.collection_config.read().await;
            let query_context_opt = SegmentsSearcher::prepare_query_context(
                self.segments.clone(),
                &core_request,
                &collection_config,
                timeout,
                search_runtime_handle,
                is_stopped_guard,
                hw_counter_acc.clone(),
            )
            .await?;

            let Some(query_context) = query_context_opt else {
                // No segments to search
                return Ok(vec![]);
            };

            (query_context, collection_config.params.clone())
        };

        // update timeout
        let timeout = timeout.saturating_sub(start.elapsed());

        let search_request = SegmentsSearcher::search(
            self.segments.clone(),
            core_request.clone(),
            search_runtime_handle,
            true,
            query_context,
            timeout,
        );

        let res = tokio::time::timeout(timeout, search_request)
            .await
            .map_err(|_| {
                log::debug!("Search timeout reached: {timeout:?}");
                // StoppingGuard takes care of setting is_stopped to true
                CollectionError::timeout(timeout, "Search")
            })??;

        let top_results = res
            .into_iter()
            .zip(core_request.searches.iter())
            .map(|(vector_res, req)| {
                let vector_name = req.query.get_vector_name();
                let distance = collection_params.get_distance(vector_name).unwrap();
                let processed_res = vector_res.into_iter().map(|mut scored_point| {
                    match req.query {
                        QueryEnum::Nearest(_) => {
                            scored_point.score = distance.postprocess_score(scored_point.score);
                        }
                        // Don't post-process if we are dealing with custom scoring
                        QueryEnum::RecommendBestScore(_)
                        | QueryEnum::RecommendSumScores(_)
                        | QueryEnum::Discover(_)
                        | QueryEnum::Context(_)
                        | QueryEnum::FeedbackNaive(_) => {}
                    };
                    scored_point
                });

                if let Some(threshold) = req.score_threshold {
                    processed_res
                        .take_while(|scored_point| {
                            distance.check_threshold(scored_point.score, threshold)
                        })
                        .collect()
                } else {
                    processed_res.collect()
                }
            })
            .collect();
        Ok(top_results)
    }
}
/// Resolve fuzzy BM25 contexts in a search batch.
///
/// For each request with `fuzzy_context`, this function:
/// 1. Looks up the `fuzzy_bind_field` from the collection's sparse vector params
/// 2. Calls `resolve_fuzzy_intent` to expand tokens across segments
/// 3. Merges the expanded sparse dimensions into the request's query vector
///
/// Returns the (potentially modified) batch. If no requests have fuzzy context,
/// the original `Arc` is returned as-is (no clone).
fn resolve_fuzzy_searches(
    core_request: Arc<CoreSearchRequestBatch>,
    collection_config: &CollectionConfigInternal,
    segments: &shard::segment_holder::locked::LockedSegmentHolder,
    timeout: Duration,
) -> CollectionResult<Arc<CoreSearchRequestBatch>> {
    let has_fuzzy = core_request
        .searches
        .iter()
        .any(|req| req.fuzzy_context.is_some());

    if !has_fuzzy {
        return Ok(core_request);
    }

    let mut searches = core_request.searches.clone();

    for req in &mut searches {
        let Some(fuzzy_ctx) = req.fuzzy_context.take() else {
            continue;
        };

        let vector_name = req.query.get_vector_name().to_owned();

        // Look up the fuzzy_bind_field from sparse vector config
        let bind_field_str = collection_config
            .params
            .sparse_vectors
            .as_ref()
            .and_then(|sv| sv.get(&*vector_name))
            .and_then(|params| params.fuzzy_config.as_ref())
            .map(|fc| fc.fuzzy_bind_field.as_str());

        let Some(bind_field_str) = bind_field_str else {
            continue;
        };

        let bind_field: JsonPath = bind_field_str.parse().map_err(|_| {
            CollectionError::bad_request(format!("Invalid fuzzy_bind_field '{bind_field_str}'"))
        })?;

        let intent = FuzzyBm25Intent {
            vector_name: vector_name.clone(),
            bind_field,
            tokens: fuzzy_ctx.tokens,
            params: fuzzy_ctx.fuzzy_params,
            stemmer: fuzzy_ctx.stemmer,
        };

        let result = resolve_fuzzy_intent(&intent, segments, timeout)
            .map_err(|e| CollectionError::service_error(format!("Fuzzy resolve failed: {e}")))?;

        if !result.fuzzy_applied || result.sparse_vector.indices.is_empty() {
            continue;
        }

        // Replace the query sparse vector with the fuzzy-resolved vector.
        // The resolved vector contains:
        // - Expanded dims (stemmed + hashed) with boost weights for tokens with candidates
        // - Original dims with weight 1.0 for tokens without candidates
        replace_query_from_fuzzy(&mut req.query, result.sparse_vector);
    }

    Ok(Arc::new(CoreSearchRequestBatch { searches }))
}

/// Replace the query sparse vector with the fuzzy-resolved vector.
///
/// For `QueryEnum::Nearest` with a `VectorInternal::Sparse` query vector,
/// this REPLACES the entire sparse vector with the resolved one.
/// The resolved vector already contains all necessary dims:
/// - Expanded dims (stemmed + hashed) with boost weights for tokens with FST candidates
/// - Original dims with weight 1.0 for tokens without FST candidates
fn replace_query_from_fuzzy(
    query: &mut QueryEnum,
    fuzzy_sparse: sparse::common::sparse_vector::SparseVector,
) {
    if let QueryEnum::Nearest(named) = query
        && let VectorInternal::Sparse(ref mut existing) = named.query
    {
        // Sort the fuzzy vector by index for consistent sparse vector format
        let mut pairs: Vec<(u32, f32)> = fuzzy_sparse
            .indices
            .into_iter()
            .zip(fuzzy_sparse.values)
            .collect();
        pairs.sort_by_key(|&(idx, _)| idx);
        existing.indices = pairs.iter().map(|&(i, _)| i).collect();
        existing.values = pairs.iter().map(|&(_, v)| v).collect();
    }
}
