mod automaton;
mod immutable_fuzzy_index;
mod mmap_fuzzy_index;
mod mutable_fuzzy_index;
mod params;
mod scorer;

pub(super) use immutable_fuzzy_index::ImmutableFuzzyIndex;
pub(super) use mmap_fuzzy_index::MmapFuzzyIndex;
pub(super) use mutable_fuzzy_index::MutableFuzzyIndex;
pub(super) use params::FuzzyParams;
use scorer::ScoredTerm;

pub trait FuzzyIndex {
    fn search(&mut self, query: &str, params: &FuzzyParams) -> Vec<ScoredTerm>;
}
