use std::collections::HashSet;

use fst::{IntoStreamer, Set, Streamer};

use super::FuzzyIndex;
use super::params::FuzzyParams;
use super::scorer::ScoredTerm;
use crate::index::field_index::full_text_index::fuzzy_index::automaton::PrefixLevenshtein;
use crate::index::field_index::full_text_index::fuzzy_index::{MmapFuzzyIndex, MutableFuzzyIndex};

pub struct ImmutableFuzzyIndex {
    index: Set<Vec<u8>>,
}

impl ImmutableFuzzyIndex {
    pub fn get_index_as_bytes(&self) -> &[u8] {
        self.index.as_fst().as_bytes()
    }
}

impl FuzzyIndex for ImmutableFuzzyIndex {
    fn search(&mut self, query: &str, params: &FuzzyParams) -> Vec<ScoredTerm> {
        let max = params.max_expansions as usize;
        let mut results: Vec<ScoredTerm> = Vec::with_capacity(max);
        let mut seen: HashSet<String> = HashSet::new();

        'outer: for distance in 0..=params.max_edits as u32 {
            if results.len() >= max {
                break;
            }

            let automaton =
                match PrefixLevenshtein::new(query, params.prefix_length as usize, distance) {
                    Ok(a) => a,
                    Err(_) => break,
                };

            let prefix_bytes =
                &query.as_bytes()[..params.prefix_length.min(query.len() as u8) as usize];
            let mut stream = if prefix_bytes.is_empty() {
                self.index.search(&automaton).into_stream()
            } else {
                self.index.search(&automaton).ge(prefix_bytes).into_stream()
            };

            while let Some(term_bytes) = stream.next() {
                let term = match std::str::from_utf8(term_bytes) {
                    Ok(t) => t.to_string(),
                    Err(_) => continue,
                };
                if seen.insert(term.clone()) {
                    results.push(ScoredTerm::new(term, query.len()));
                    if results.len() >= max {
                        break 'outer;
                    }
                }
            }
        }

        results
    }
}

impl From<MutableFuzzyIndex> for ImmutableFuzzyIndex {
    fn from(value: MutableFuzzyIndex) -> Self {
        // use a unwrap here may not fit, change it in the future
        let index = Set::from_iter(value.get_terms()).unwrap();
        Self { index }
    }
}

impl From<&MmapFuzzyIndex> for ImmutableFuzzyIndex {
    fn from(value: &MmapFuzzyIndex) -> Self {
        let bytes = value.fst_bytes().to_vec();
        let index = Set::new(bytes).unwrap();
        Self { index }
    }
}
