use std::collections::BTreeSet;

use super::FuzzyIndex;
use super::params::FuzzyParams;
use super::scorer::ScoredTerm;
use crate::common::operation_error::OperationResult;

pub struct MutableFuzzyIndex {
    terms: BTreeSet<String>,
}

#[allow(dead_code)]
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
    fn search(&mut self, query: &str, params: &FuzzyParams) -> Vec<ScoredTerm> {
        let max = params.max_expansions as usize;
        let mut buckets: Vec<Vec<ScoredTerm>> =
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

            if let Some(dist) = levenshtein_distance(query, term, params.max_edits as u32) {
                buckets[dist as usize].push(ScoredTerm::new(term.to_string(), query.len()));
                total += 1;
                if total >= max {
                    break 'outer;
                }
            }
        }

        let mut results: Vec<ScoredTerm> = buckets.into_iter().flatten().collect();
        results.truncate(max);
        results
    }
}

fn levenshtein_distance(a: &str, b: &str, max_dist: u32) -> Option<u32> {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let la = a.len();
    let lb = b.len();

    if la.abs_diff(lb) as u32 > max_dist {
        return None;
    }

    let mut prev: Vec<u32> = (0..=(lb as u32)).collect();
    let mut curr: Vec<u32> = vec![0; lb + 1];

    for i in 1..=la {
        curr[0] = i as u32;
        let mut row_min = curr[0];

        for j in 1..=lb {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j - 1] + cost).min(prev[j] + 1).min(curr[j - 1] + 1);
            row_min = row_min.min(curr[j]);
        }

        if row_min > max_dist {
            return None;
        }

        std::mem::swap(&mut prev, &mut curr);
    }

    let dist = prev[lb];
    if dist <= max_dist { Some(dist) } else { None }
}
