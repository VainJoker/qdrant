mod automaton;
mod immutable_fuzzy_index;
mod mmap_fuzzy_index;
mod mutable_fuzzy_index;
mod scorer;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::PointOffsetType;
pub(super) use immutable_fuzzy_index::ImmutableFuzzyIndex;
pub(super) use mmap_fuzzy_index::MmapFuzzyIndex;
pub(super) use mutable_fuzzy_index::MutableFuzzyIndex;

use crate::index::field_index::CardinalityEstimation;
use crate::index::field_index::full_text_index::inverted_index::{
    Document, InvertedIndex, TokenId, TokenSet,
};
use crate::types::{FieldCondition, FuzzyParams};

#[derive(Debug, Clone, PartialEq)]
pub struct FuzzyCandidate {
    pub term: String,
    // pub token_id: TokenId,
    /// Weight of the fuzzy match between the query term and the matched dictionary term.
    pub weight: f32,
}

impl FuzzyCandidate {
    pub fn new(term: String, len: usize, distance: u32) -> Self {
        let weight = 1.0 - (distance as f32 / (len as f32 + 1.0));
        Self {
            term,
            // token_id,
            weight,
        }
    }
}

// /// Parsed query for fuzzy matching, parallel to [`super::inverted_index::ParsedQuery`].
// #[derive(Debug, Clone)]
// pub enum FuzzyParsedQuery {
//     /// AND semantics: every group must have at least one match in the document.
//     AllTokens(FuzzyDocument),

//     /// OR semantics: any fuzzy-expanded token matching the document is sufficient.
//     AnyTokens(TokenSet),

//     /// Position-ordered phrase match: the document must contain a contiguous window
//     /// where position *i* matches at least one token from `groups[i]`.
//     Phrase(FuzzyDocument),
// }

/// Common interface for fuzzy term expansion.
// pub fn expand_term(
//     term: &str,
//     params: &FuzzyParams,
//     vocab_lookup: &dyn Fn(&str) -> Option<TokenId>,
// ) -> Vec<FuzzyCandidate> {
//     if term.len() < FuzzyParams::MIN_TERM_LENGTH {
//         return vocab_lookup(term)
//             .map(|token_id| {
//                 vec![FuzzyCandidate {
//                     term: term.to_string(),
//                     // token_id,
//                     weight: 1.0,
//                 }]
//             })
//             .unwrap_or_default();
//     }

//     let max_distance = params.max_edits.min(FuzzyParams::MAX_EDITS_DISTANCE);
//     let prefix_len = params.prefix_length as usize;
//     let max_expansions = params
//         .max_expansions
//         .clamp(1, FuzzyParams::MAX_EXPANSIONS_CAP) as usize;

//     let mut candidates = Vec::with_capacity(max_expansions.min(64));
//     let mut seen_ids = std::collections::HashSet::new();

//     for distance in 0..=max_distance {
//         let remaining = max_expansions - candidates.len();
//         if remaining == 0 {
//             break;
//         }
//         let limit = max_expansions;

//         let batch = self
//             .dictionary
//             .search(term, distance, prefix_len, vocab_lookup, limit);

//         for candidate in batch {
//             if candidates.len() >= max_expansions {
//                 break;
//             }
//             if seen_ids.insert(candidate.token_id) {
//                 candidates.push(candidate);
//             }
//         }
//     }

//     candidates.truncate(max_expansions);

//     candidates
// }
// }

/// Trait for fuzzy text matching, parallel to [`InvertedIndex`].
///
/// Each inverted-index implementation provides a `FuzzyIndex` impl that handles
/// fuzzy query filtering and matching using the same internal postings data.
pub trait FuzzyIndex {
    fn search(&mut self, query: &str, params: &FuzzyParams) -> Vec<FuzzyCandidate>;

    // fn fuzzy_filter<'a>(
    //     &'a self,
    //     query: FuzzyParsedQuery,
    //     hw_counter: &'a HardwareCounterCell,
    // ) -> Box<dyn Iterator<Item = PointOffsetType> + 'a>;

    // fn fuzzy_check_match(&self, query: &FuzzyParsedQuery, point_id: PointOffsetType) -> bool;

    // fn fuzzy_estimate_cardinality(
    //     &self,
    //     query: &FuzzyParsedQuery,
    //     condition: &FieldCondition,
    //     hw_counter: &HardwareCounterCell,
    // ) -> CardinalityEstimation {
    //         todo!("Implement fuzzy cardinality estimation using the same heuristics as the exact-match estimation, but applied to the expanded token sets in the query. This may involve estimating the number of matching tokens for each group and combining those estimates according to the query semantics (AND vs OR).")
    //     }
}
