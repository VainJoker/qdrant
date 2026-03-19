use std::collections::HashSet;

use fst::{IntoStreamer, Set, Streamer};

use super::FuzzyIndex;
use crate::index::field_index::full_text_index::fuzzy_index::automaton::PrefixLevenshtein;
use crate::index::field_index::full_text_index::fuzzy_index::{
    FuzzyCandidate, MmapFuzzyIndex, MutableFuzzyIndex,
};
use crate::types::FuzzyParams;

pub struct ImmutableFuzzyIndex {
    index: Set<Vec<u8>>,
}

impl ImmutableFuzzyIndex {
    pub fn get_index_as_bytes(&self) -> &[u8] {
        self.index.as_fst().as_bytes()
    }
}

impl FuzzyIndex for ImmutableFuzzyIndex {
    fn search(&self, query: &str, params: &FuzzyParams) -> Vec<FuzzyCandidate> {
        let max = params.max_expansions as usize;
        let mut results: Vec<FuzzyCandidate> = Vec::with_capacity(max);
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
                    results.push(FuzzyCandidate::new(term, query.len(), distance));
                    if results.len() >= max {
                        break 'outer;
                    }
                }
            }
        }

        results
    }
}

impl TryFrom<MutableFuzzyIndex> for ImmutableFuzzyIndex {
    type Error = crate::common::operation_error::OperationError;

    fn try_from(value: MutableFuzzyIndex) -> Result<Self, Self::Error> {
        let index = Set::from_iter(value.get_terms()).map_err(|e| {
            crate::common::operation_error::OperationError::service_error(format!(
                "Failed to build fuzzy index from MutableFuzzyIndex: {e}"
            ))
        })?;
        Ok(Self { index })
    }
}

impl TryFrom<&MmapFuzzyIndex> for ImmutableFuzzyIndex {
    type Error = crate::common::operation_error::OperationError;

    fn try_from(value: &MmapFuzzyIndex) -> Result<Self, Self::Error> {
        let bytes = value.fst_bytes().to_vec();
        let index = Set::new(bytes).map_err(|e| {
            crate::common::operation_error::OperationError::service_error(format!(
                "Failed to build fuzzy index from MmapFuzzyIndex: {e}"
            ))
        })?;
        Ok(Self { index })
    }
}
