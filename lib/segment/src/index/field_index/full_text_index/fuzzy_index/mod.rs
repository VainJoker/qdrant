mod automaton;
mod immutable_fuzzy_index;
mod mmap_fuzzy_index;
mod mutable_fuzzy_index;

pub(super) use immutable_fuzzy_index::ImmutableFuzzyIndex;
pub(super) use mmap_fuzzy_index::MmapFuzzyIndex;
pub(super) use mutable_fuzzy_index::MutableFuzzyIndex;

use crate::types::FuzzyParams;

#[derive(Debug, Clone, PartialEq)]
pub struct FuzzyCandidate {
    pub term: String,
    /// Weight of the fuzzy match between the query term and the matched dictionary term.
    pub weight: f32,
}

impl FuzzyCandidate {
    pub fn new(term: String, len: usize, distance: u32) -> Self {
        let weight = 1.0 - (distance as f32 / (len as f32 + 1.0));
        Self { term, weight }
    }
}

/// Trait for fuzzy text matching, parallel to [`InvertedIndex`].
///
/// Each inverted-index implementation provides a `FuzzyIndex` impl that handles
/// fuzzy query filtering and matching using the same internal postings data.
pub trait FuzzyIndex {
    fn search(&self, query: &str, params: &FuzzyParams) -> Vec<FuzzyCandidate>;
}
