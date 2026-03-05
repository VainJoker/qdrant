//! Fuzzy term expansion using FST (Finite State Transducer) and Levenshtein automaton.
//!
//! Given a vocabulary of indexed terms (from the InvertedIndex), this module builds
//! an FST and uses it to efficiently find terms within a given edit distance of a query term.

use std::collections::HashMap;
use std::str;

use fst::automaton::{Automaton, Levenshtein};
use fst::{IntoStreamer, Set, Streamer};

use crate::index::field_index::full_text_index::inverted_index::TokenId;
use crate::types::FuzzyParams;

/// A candidate term found by fuzzy expansion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuzzyCandidate {
    pub term: String,
    pub token_id: TokenId,
}

/// Custom automaton that enforces exact prefix matching before fuzzy matching.
pub struct PrefixFuzzy {
    query: String,
    prefix_len: usize,
    lev: Levenshtein,
}

#[derive(Clone)]
pub enum State {
    Prefix(usize),
    Fuzzy(<Levenshtein as Automaton>::State),
    Dead,
}

impl PrefixFuzzy {
    pub fn new(query: &str, prefix_len: usize, max_edits: u32) -> Option<Self> {
        if prefix_len > query.len() {
            return None;
        }

        let suffix = &query[prefix_len..];
        let lev = Levenshtein::new(suffix, max_edits).ok()?;

        Some(Self {
            query: query.to_string(),
            prefix_len,
            lev,
        })
    }
}

impl Automaton for PrefixFuzzy {
    type State = State;

    fn start(&self) -> Self::State {
        State::Prefix(0)
    }

    fn is_match(&self, state: &Self::State) -> bool {
        match state {
            State::Fuzzy(s) => self.lev.is_match(s),
            _ => false,
        }
    }

    fn can_match(&self, state: &Self::State) -> bool {
        match state {
            State::Prefix(i) => *i <= self.prefix_len,
            State::Fuzzy(s) => self.lev.can_match(s),
            State::Dead => false,
        }
    }

    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        match state {
            State::Prefix(i) => {
                if *i < self.prefix_len {
                    if self.query.as_bytes()[*i] == byte {
                        State::Prefix(i + 1)
                    } else {
                        State::Dead
                    }
                } else {
                    let next = self.lev.start();
                    let s = self.lev.accept(&next, byte);
                    State::Fuzzy(s)
                }
            }
            State::Fuzzy(s) => {
                let next = self.lev.accept(s, byte);
                State::Fuzzy(next)
            }
            State::Dead => State::Dead,
        }
    }
}

/// Builds an FST from a vocabulary and expands query terms using Levenshtein automaton.
pub struct FuzzyExpander {
    fst_set: Set<Vec<u8>>,
}

impl std::fmt::Debug for FuzzyExpander {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FuzzyExpander")
            .field("fst_size", &self.fst_set.as_fst().as_bytes().len())
            .finish()
    }
}

impl FuzzyExpander {
    /// Build an FST from the vocabulary HashMap.
    ///
    /// The FST requires keys to be sorted lexicographically and deduplicated.
    pub fn build(vocab: &HashMap<String, TokenId>) -> Option<Self> {
        if vocab.is_empty() {
            return None;
        }

        let mut terms: Vec<&str> = vocab.keys().map(|s| s.as_str()).collect();
        terms.sort_unstable();
        terms.dedup();

        let fst_set = Set::from_iter(terms).ok()?;
        Some(Self { fst_set })
    }

    /// Serialize the FST to bytes for persistence.
    pub fn to_bytes(&self) -> &[u8] {
        self.fst_set.as_fst().as_bytes()
    }

    /// Deserialize the FST from bytes.
    pub fn from_bytes(data: Vec<u8>) -> Option<Self> {
        let fst_set = Set::new(data).ok()?;
        Some(Self { fst_set })
    }

    /// Expand a single query term into fuzzy candidates.
    ///
    /// Returns a list of candidate terms (with their TokenIds) that are within
    /// the specified edit distance and satisfy the prefix_length constraint.
    ///
    /// Results are collected in order of increasing edit distance (0, 1, 2, ...),
    /// with lexicographic ordering within each distance level, and capped at `max_expansions`.
    ///
    /// `vocab_lookup` maps a matched term string to its `TokenId`.
    pub fn expand_term(
        &self,
        term: &str,
        params: &FuzzyParams,
        vocab_lookup: impl Fn(&str) -> Option<TokenId>,
    ) -> Vec<FuzzyCandidate> {
        // Reject too-short terms
        if term.len() < FuzzyParams::MIN_TERM_LENGTH {
            return vocab_lookup(term)
                .map(|token_id| {
                    vec![FuzzyCandidate {
                        term: term.to_string(),
                        token_id,
                    }]
                })
                .unwrap_or_default();
        }

        let max_distance = params.max_edit.min(FuzzyParams::MAX_EDIT_DISTANCE);
        let prefix_length = params.prefix_length as usize;
        let max_expansions = params
            .max_expansions
            .min(FuzzyParams::MAX_EXPANSIONS_CAP)
            .max(1) as usize;

        // Pre-allocate with reasonable capacity
        let mut all_candidates = Vec::with_capacity(max_expansions.min(64));

        let use_prefix_fuzzy = prefix_length > 0 && prefix_length < term.len();
        let prefix = if prefix_length > 0 && prefix_length <= term.len() {
            &term[..prefix_length]
        } else {
            ""
        };

        // Iterate through edit distances from 0 to max_distance
        for distance in 0..=max_distance {
            let remaining = max_expansions - all_candidates.len();
            if remaining == 0 {
                break;
            };

            let candidates_at_distance = if use_prefix_fuzzy {
                self.search_with_prefix_fuzzy(
                    term,
                    prefix_length,
                    distance,
                    &vocab_lookup,
                    remaining,
                )
            } else {
                self.search_with_levenshtein(term, distance, prefix, &vocab_lookup, remaining)
            };

            all_candidates.extend(candidates_at_distance);
        }

        all_candidates
    }

    /// Search using standard Levenshtein automaton
    fn search_with_levenshtein(
        &self,
        term: &str,
        distance: u32,
        prefix: &str,
        vocab_lookup: &impl Fn(&str) -> Option<TokenId>,
        limit: usize,
    ) -> Vec<FuzzyCandidate> {
        let automaton = match Levenshtein::new(term, distance) {
            Ok(a) => a,
            Err(_) => return Vec::new(),
        };

        let mut stream = self.fst_set.search(&automaton).into_stream();
        let mut candidates = Vec::with_capacity(limit.min(32));

        while let Some(key) = stream.next() {
            let found_term = match str::from_utf8(key) {
                Ok(s) => s,
                Err(_) => continue,
            };

            // Apply prefix filter
            if !prefix.is_empty() && !found_term.starts_with(prefix) {
                continue;
            }

            // Look up token_id
            if let Some(token_id) = vocab_lookup(found_term) {
                candidates.push(FuzzyCandidate {
                    term: found_term.to_string(),
                    token_id,
                });

                // Early exit if we've collected enough
                if candidates.len() >= limit {
                    candidates.sort_unstable_by(|a, b| a.term.cmp(&b.term));
                    candidates.truncate(limit);
                    return candidates;
                }
            }
        }

        // Sort and truncate only if needed
        if candidates.len() > 1 {
            candidates.sort_unstable_by(|a, b| a.term.cmp(&b.term));
        }
        candidates.truncate(limit);
        candidates
    }

    /// Search using PrefixFuzzy automaton
    fn search_with_prefix_fuzzy(
        &self,
        term: &str,
        prefix_length: usize,
        distance: u32,
        vocab_lookup: &impl Fn(&str) -> Option<TokenId>,
        limit: usize,
    ) -> Vec<FuzzyCandidate> {
        let automaton = match PrefixFuzzy::new(term, prefix_length, distance) {
            Some(a) => a,
            None => return Vec::new(),
        };

        let mut stream = self.fst_set.search(&automaton).into_stream();
        let mut candidates = Vec::with_capacity(limit.min(32));

        while let Some(key) = stream.next() {
            let found_term = match str::from_utf8(key) {
                Ok(s) => s,
                Err(_) => continue,
            };

            // Look up token_id (PrefixFuzzy automaton already enforces prefix constraint)
            if let Some(token_id) = vocab_lookup(found_term) {
                candidates.push(FuzzyCandidate {
                    term: found_term.to_string(),
                    token_id,
                });

                // Early exit if we've collected enough
                if candidates.len() >= limit {
                    candidates.sort_unstable_by(|a, b| a.term.cmp(&b.term));
                    candidates.truncate(limit);
                    return candidates;
                }
            }
        }

        // Sort and truncate only if needed
        if candidates.len() > 1 {
            candidates.sort_unstable_by(|a, b| a.term.cmp(&b.term));
        }
        candidates.truncate(limit);
        candidates
    }
}
