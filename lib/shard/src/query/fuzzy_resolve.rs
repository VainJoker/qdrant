//! Collection-level BM25 fuzzy request resolution.
//!
//! This module implements the "Resolve" stage from the BM25 Fuzzy Search RFC:
//! 1. Iterate segments, calling `expand_fuzzy_terms()` per token
//! 2. Global dedup (keep min edit_distance per term)
//! 3. Global truncation (`max_expansions`)
//! 4. Boost computation per candidate
//! 5. Hash to DimId, build SparseVector
//! 6. Rewrite to standard QueryEnum::Nearest
//!
//! This resolution must happen **before** `prepare_query_context()` so that
//! all DimIds are known when `init_query_context()` runs.

use std::borrow::Cow;
use std::collections::HashMap;
use std::time::Duration;

use common::counter::hardware_counter::HardwareCounterCell;
use segment::common::operation_error::OperationResult;
use segment::index::field_index::full_text_index::tokenizers::Stemmer;
use segment::index::sparse_index::bm25_fuzzy_expander::{
    bm25_token_to_dim_id, compute_fuzzy_boost,
};
use segment::json_path::JsonPath;
use segment::types::{FuzzyParams, VectorNameBuf};
use sparse::common::sparse_vector::SparseVector;

use crate::segment_holder::locked::LockedSegmentHolder;

/// Transient intent for a BM25 fuzzy query, constructed from API input.
///
/// This type exists only during the resolve stage. After resolution it is
/// consumed and replaced with a standard `SparseVector` query.
#[derive(Debug, Clone)]
pub struct FuzzyBm25Intent {
    /// The sparse vector field name.
    pub vector_name: VectorNameBuf,
    /// The payload field with a FullTextIndex (fuzzy-enabled) to expand against.
    pub bind_field: JsonPath,
    /// Original query tokens (from `Bm25::tokenize()`).
    pub tokens: Vec<String>,
    /// Fuzzy parameters (max_edit, prefix_length, max_expansions).
    pub params: FuzzyParams,
    /// Optional stemmer from the BM25 model config.
    /// Expanded FST terms must be stemmed before hashing to dim_ids
    /// to match the stemmed tokens used during document indexing.
    pub stemmer: Option<Stemmer>,
}

/// Result of fuzzy resolution.
#[derive(Debug)]
pub struct FuzzyResolveResult {
    /// The resolved sparse vector (dim_id → boost weight).
    pub sparse_vector: SparseVector,
    /// Whether fuzzy expansion was actually applied (at least one segment had FST).
    pub fuzzy_applied: bool,
}

/// Resolve a `FuzzyBm25Intent` by expanding tokens across all segments.
///
/// This function:
/// 1. Iterates all segments, calling `expand_fuzzy_terms()` per token
/// 2. Globally deduplicates candidates (keeping min edit_distance per term)
/// 3. Truncates to `max_expansions` per token
/// 4. Computes boost per candidate
/// 5. Hashes terms to DimIds via `bm25_token_to_dim_id()`
/// 6. Builds the final `SparseVector` (max-boost for DimId conflicts)
///
/// The `bm25_hash_fn` parameter allows injecting the hash function from the
/// inference crate without creating a dependency from shard → inference.
pub fn resolve_fuzzy_intent(
    intent: &FuzzyBm25Intent,
    segments: &LockedSegmentHolder,
    timeout: Duration,
) -> OperationResult<FuzzyResolveResult> {
    log::debug!(
        "[fuzzy] resolve_fuzzy_intent: vector='{}', field='{}', tokens={:?}, params={:?}",
        intent.vector_name,
        intent.bind_field,
        intent.tokens,
        intent.params
    );
    let max_expansions = intent.params.max_expansions as usize;

    let mut any_fst_found = false;

    // Per-token: collect candidates from all segments, dedup, truncate
    let mut final_dim_weights: HashMap<u32, f32> = HashMap::new();

    let hw_counter = HardwareCounterCell::disposable();

    let segments_guard = segments.try_read_for(timeout).ok_or_else(|| {
        segment::common::operation_error::OperationError::timeout(
            timeout,
            "fuzzy resolve: acquire segment lock",
        )
    })?;

    let segment_count = segments_guard
        .non_appendable_then_appendable_segments()
        .count();
    log::debug!("[fuzzy] acquired segment lock, segment_count={segment_count}");

    for token in &intent.tokens {
        let mut token_candidates: HashMap<String, u32> = HashMap::new(); // term → min edit_distance
        let mut segs_with_fst = 0usize;
        let mut segs_no_field = 0usize;

        for locked_segment in segments_guard.non_appendable_then_appendable_segments() {
            let segment = locked_segment.get();
            let segment_guard = segment.read();

            let candidates = segment_guard.expand_fuzzy_terms(
                &intent.bind_field,
                token,
                &intent.params,
                &hw_counter,
            );

            if !candidates.is_empty() {
                any_fst_found = true;
                segs_with_fst += 1;
            } else {
                segs_no_field += 1;
            }

            // Global dedup: keep minimum edit_distance per term
            for candidate in candidates {
                let entry = token_candidates.entry(candidate.term).or_insert(u32::MAX);
                if candidate.edit_distance < *entry {
                    *entry = candidate.edit_distance;
                }
            }
        }

        if token_candidates.is_empty() {
            // No candidates for this token; it won't contribute to the query.
            // The original token dim is preserved in the query by the caller
            // (replace only happens if the result has at least one dim).
            log::debug!(
                "[fuzzy] token='{token}': 0 candidates after dedup \
                 (segs_with_fst={segs_with_fst}, segs_no_candidates={segs_no_field})"
            );
            continue;
        }

        // Sort by (edit_distance ASC, term ASC) for deterministic truncation
        let mut sorted_candidates: Vec<(String, u32)> = token_candidates.into_iter().collect();
        sorted_candidates.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        log::debug!(
            "[fuzzy] token='{token}': {} unique candidates before truncation \
             (segs_with_fst={segs_with_fst}, segs_no_candidates={segs_no_field}), \
             top5={:?}",
            sorted_candidates.len(),
            sorted_candidates.iter().take(5).collect::<Vec<_>>()
        );

        // Global truncation: keep top max_expansions
        sorted_candidates.truncate(max_expansions);

        // Compute boost and hash to DimId.
        // Stem expanded terms through the BM25 stemmer before hashing so that
        // the dim_ids match those produced during document indexing.
        for (term, edit_distance) in sorted_candidates {
            if let Some(boost) = compute_fuzzy_boost(token, &term, edit_distance) {
                let stemmed = match &intent.stemmer {
                    Some(stemmer) => stemmer.stem(Cow::<str>::Borrowed(&term)).into_owned(),
                    None => term,
                };
                let dim_id = bm25_token_to_dim_id(&stemmed);
                // Multi-token DimId conflict: keep max boost (§8.3)
                let entry = final_dim_weights.entry(dim_id).or_insert(0.0);
                if boost > *entry {
                    *entry = boost;
                }
            }
        }
    }

    drop(segments_guard);

    log::debug!(
        "[fuzzy] resolve complete: any_fst_found={any_fst_found}, final_dims={}, dim_ids={:?}",
        final_dim_weights.len(),
        final_dim_weights.keys().take(10).collect::<Vec<_>>(),
    );

    // Build SparseVector from accumulated dim_id → boost
    let (indices, values): (Vec<u32>, Vec<f32>) = final_dim_weights.into_iter().unzip();

    Ok(FuzzyResolveResult {
        sparse_vector: SparseVector { indices, values },
        fuzzy_applied: any_fst_found,
    })
}
