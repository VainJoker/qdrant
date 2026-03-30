use std::collections::BTreeSet;

use strsim::levenshtein;

use super::FuzzyIndex;
use crate::index::field_index::full_text_index::fuzzy_index::FuzzyCandidate;
use crate::types::FuzzyParams;

pub struct MutableFuzzyIndex {
    terms: BTreeSet<String>,
}

impl MutableFuzzyIndex {
    pub fn new() -> Self {
        Self {
            terms: BTreeSet::new(),
        }
    }

    pub fn build_index(iter: impl Iterator<Item = String>) -> Self {
        let mut index = Self::new();
        for term in iter {
            index.insert(term);
        }
        index
    }

    pub fn add_terms(&mut self, terms: Vec<String>) {
        self.terms.extend(terms);
    }

    fn insert(&mut self, term: String) {
        self.terms.insert(term);
    }

    pub fn get_terms(&self) -> impl Iterator<Item = &str> {
        self.terms.iter().map(|t| t.as_str())
    }
}

impl Default for MutableFuzzyIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl FuzzyIndex for MutableFuzzyIndex {
    fn search(&self, query: &str, params: &FuzzyParams) -> Vec<FuzzyCandidate> {
        let max = params.max_expansions as usize;
        let mut buckets: Vec<Vec<FuzzyCandidate>> =
            (0..=params.max_edits as u32).map(|_| Vec::new()).collect();
        let mut total = 0usize;

        let prefix_len = (params.prefix_length as usize).min(query.len());
        let query_prefix = &query.as_bytes()[..prefix_len];

        'outer: for term in self.terms.iter() {
            if prefix_len > 0 {
                let term_bytes = term.as_bytes();
                if term_bytes.len() < prefix_len || &term_bytes[..prefix_len] != query_prefix {
                    continue;
                }
            }

            let dist = levenshtein(query, term) as u32;
            if dist <= params.max_edits as u32 {
                buckets[dist as usize].push(FuzzyCandidate::new(
                    term.to_string(),
                    query.len(),
                    dist,
                ));
                total += 1;
                if total >= max {
                    break 'outer;
                }
            }
        }

        let mut results: Vec<FuzzyCandidate> = buckets.into_iter().flatten().collect();
        results.truncate(max);
        results
    }
}
