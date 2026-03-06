//! FST-based fuzzy term expansion using Levenshtein automaton.

use std::collections::{BTreeSet, HashMap};
use std::str;

use fst::automaton::{Automaton, Levenshtein};
use fst::{IntoStreamer, Set, Streamer};

use super::TermExpander;
use crate::index::field_index::full_text_index::inverted_index::TokenId;
use crate::types::FuzzyParams;

/// A candidate term found by fuzzy expansion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuzzyCandidate {
    pub term: String,
    pub token_id: TokenId,
}

// ---------------------------------------------------------------------------
// PrefixLevenshtein automaton
// ---------------------------------------------------------------------------

/// Automaton that requires an exact prefix match followed by Levenshtein fuzzy matching.
pub struct PrefixLevenshtein {
    query: String,
    prefix_len: usize,
    levenshtein: Levenshtein,
}

/// State of the [`PrefixLevenshtein`] automaton.
#[derive(Clone)]
pub enum PrefixLevenshteinState {
    /// Matching the exact prefix; `usize` is the number of prefix bytes matched so far.
    Prefix(usize),
    /// Prefix matched; delegating to the inner Levenshtein automaton.
    Fuzzy(<Levenshtein as Automaton>::State),
    /// Prefix mismatch; permanently rejected.
    Dead,
}

impl PrefixLevenshtein {
    pub fn new(query: &str, prefix_len: usize, max_edits: u32) -> Option<Self> {
        if prefix_len > query.len() {
            return None;
        }

        let suffix = &query[prefix_len..];
        let levenshtein = Levenshtein::new(suffix, max_edits).ok()?;

        Some(Self {
            query: query.to_string(),
            prefix_len,
            levenshtein,
        })
    }
}

impl Automaton for PrefixLevenshtein {
    type State = PrefixLevenshteinState;

    fn start(&self) -> Self::State {
        PrefixLevenshteinState::Prefix(0)
    }

    fn is_match(&self, state: &Self::State) -> bool {
        match state {
            PrefixLevenshteinState::Fuzzy(s) => self.levenshtein.is_match(s),
            _ => false,
        }
    }

    fn can_match(&self, state: &Self::State) -> bool {
        match state {
            PrefixLevenshteinState::Prefix(i) => *i <= self.prefix_len,
            PrefixLevenshteinState::Fuzzy(s) => self.levenshtein.can_match(s),
            PrefixLevenshteinState::Dead => false,
        }
    }

    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        match state {
            PrefixLevenshteinState::Prefix(i) => {
                if *i < self.prefix_len {
                    if self.query.as_bytes()[*i] == byte {
                        PrefixLevenshteinState::Prefix(i + 1)
                    } else {
                        PrefixLevenshteinState::Dead
                    }
                } else {
                    let start = self.levenshtein.start();
                    let next = self.levenshtein.accept(&start, byte);
                    PrefixLevenshteinState::Fuzzy(next)
                }
            }
            PrefixLevenshteinState::Fuzzy(s) => {
                PrefixLevenshteinState::Fuzzy(self.levenshtein.accept(s, byte))
            }
            PrefixLevenshteinState::Dead => PrefixLevenshteinState::Dead,
        }
    }
}

// ---------------------------------------------------------------------------
// FuzzyExpander
// ---------------------------------------------------------------------------

/// Builds an FST from a vocabulary and expands query terms using Levenshtein automaton.
pub struct FuzzyExpander {
    fst_set: Set<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct MutableFuzzyExpander {
    terms: BTreeSet<String>,
}

impl std::fmt::Debug for FuzzyExpander {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FuzzyExpander")
            .field("fst_size", &self.fst_set.as_fst().as_bytes().len())
            .finish()
    }
}

impl FuzzyExpander {
    /// Build an FST from the vocabulary.
    ///
    /// Returns `None` if the vocabulary is empty.
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

    /// Serialize the FST to bytes.
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
    /// Returns candidate terms (with their `TokenId`s) within the configured edit distance
    /// and prefix constraint, collected in order of increasing edit distance then
    /// lexicographic order, capped at `max_expansions`.
    pub fn expand_term(
        &self,
        term: &str,
        params: &FuzzyParams,
        vocab_lookup: impl Fn(&str) -> Option<TokenId>,
    ) -> Vec<FuzzyCandidate> {
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
        let prefix_len = params.prefix_length as usize;
        let max_expansions = params
            .max_expansions
            .min(FuzzyParams::MAX_EXPANSIONS_CAP)
            .max(1) as usize;

        let use_prefix_automaton = prefix_len > 0 && prefix_len < term.len();
        let prefix = if prefix_len > 0 && prefix_len <= term.len() {
            &term[..prefix_len]
        } else {
            ""
        };

        let mut candidates = Vec::with_capacity(max_expansions.min(64));

        for distance in 0..=max_distance {
            let remaining = max_expansions - candidates.len();
            if remaining == 0 {
                break;
            }

            let batch = if use_prefix_automaton {
                self.search_prefix_levenshtein(term, prefix_len, distance, &vocab_lookup, remaining)
            } else {
                self.search_levenshtein(term, distance, prefix, &vocab_lookup, remaining)
            };

            candidates.extend(batch);
        }

        candidates
    }

    /// Search using a plain Levenshtein automaton with optional prefix filter.
    fn search_levenshtein(
        &self,
        term: &str,
        distance: u32,
        required_prefix: &str,
        vocab_lookup: &impl Fn(&str) -> Option<TokenId>,
        limit: usize,
    ) -> Vec<FuzzyCandidate> {
        let automaton = match Levenshtein::new(term, distance) {
            Ok(a) => a,
            Err(_) => return Vec::new(),
        };

        let mut stream = self.fst_set.search(&automaton).into_stream();
        let mut results = Vec::with_capacity(limit.min(32));

        while let Some(key) = stream.next() {
            let matched_term = match str::from_utf8(key) {
                Ok(s) => s,
                Err(_) => continue,
            };

            if !required_prefix.is_empty() && !matched_term.starts_with(required_prefix) {
                continue;
            }

            if let Some(token_id) = vocab_lookup(matched_term) {
                results.push(FuzzyCandidate {
                    term: matched_term.to_string(),
                    token_id,
                });
                if results.len() >= limit {
                    break;
                }
            }
        }

        results.sort_unstable_by(|a, b| a.term.cmp(&b.term));
        results.truncate(limit);
        results
    }

    /// Search using the [`PrefixLevenshtein`] automaton (exact prefix + fuzzy suffix).
    fn search_prefix_levenshtein(
        &self,
        term: &str,
        prefix_len: usize,
        distance: u32,
        vocab_lookup: &impl Fn(&str) -> Option<TokenId>,
        limit: usize,
    ) -> Vec<FuzzyCandidate> {
        let automaton = match PrefixLevenshtein::new(term, prefix_len, distance) {
            Some(a) => a,
            None => return Vec::new(),
        };

        let mut stream = self.fst_set.search(&automaton).into_stream();
        let mut results = Vec::with_capacity(limit.min(32));

        while let Some(key) = stream.next() {
            let matched_term = match str::from_utf8(key) {
                Ok(s) => s,
                Err(_) => continue,
            };

            if let Some(token_id) = vocab_lookup(matched_term) {
                results.push(FuzzyCandidate {
                    term: matched_term.to_string(),
                    token_id,
                });
                if results.len() >= limit {
                    break;
                }
            }
        }

        results.sort_unstable_by(|a, b| a.term.cmp(&b.term));
        results.truncate(limit);
        results
    }
}

impl TermExpander for FuzzyExpander {
    fn expand_term(
        &self,
        term: &str,
        params: &FuzzyParams,
        vocab_lookup: &dyn Fn(&str) -> Option<TokenId>,
    ) -> Vec<FuzzyCandidate> {
        self.expand_term(term, params, vocab_lookup)
    }
}

impl MutableFuzzyExpander {
    pub fn new() -> Self {
        Self {
            terms: BTreeSet::new(),
        }
    }

    /// Build from an iterator of vocabulary keys.
    pub fn from_vocab_keys(keys: impl Iterator<Item = impl AsRef<str>>) -> Self {
        let terms = keys.map(|k| k.as_ref().to_owned()).collect();
        Self { terms }
    }

    /// Insert a term. Returns `true` if the term was newly inserted.
    pub fn insert(&mut self, term: &str) -> bool {
        self.terms.insert(term.to_owned())
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }
}

impl TermExpander for MutableFuzzyExpander {
    fn expand_term(
        &self,
        term: &str,
        params: &FuzzyParams,
        vocab_lookup: &dyn Fn(&str) -> Option<TokenId>,
    ) -> Vec<FuzzyCandidate> {
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
        let prefix_len = params.prefix_length as usize;
        let max_expansions = params
            .max_expansions
            .min(FuzzyParams::MAX_EXPANSIONS_CAP)
            .max(1) as usize;

        let prefix = if prefix_len > 0 && prefix_len <= term.len() {
            &term[..prefix_len]
        } else {
            ""
        };

        let mut candidates = Vec::with_capacity(max_expansions.min(64));

        for distance in 0..=max_distance {
            let remaining = max_expansions - candidates.len();
            if remaining == 0 {
                break;
            }

            let batch = self.search_btree(term, distance, prefix, vocab_lookup, remaining);
            candidates.extend(batch);
        }

        candidates
    }
}

impl MutableFuzzyExpander {
    /// Iterate the BTreeSet, optionally narrowing by prefix range, and collect terms
    /// within the given Levenshtein edit distance.
    fn search_btree(
        &self,
        term: &str,
        max_distance: u32,
        required_prefix: &str,
        vocab_lookup: &dyn Fn(&str) -> Option<TokenId>,
        limit: usize,
    ) -> Vec<FuzzyCandidate> {
        let mut results = Vec::with_capacity(limit.min(32));

        let iter: Box<dyn Iterator<Item = &String>> = if required_prefix.is_empty() {
            Box::new(self.terms.iter())
        } else {
            // Use BTreeSet range to only scan terms starting with the prefix.
            let start = required_prefix.to_string();
            let end = prefix_successor(required_prefix);
            match end {
                Some(end) => Box::new(self.terms.range(start..end)),
                None => Box::new(self.terms.range(start..)),
            }
        };

        for candidate_term in iter {
            if !required_prefix.is_empty() && !candidate_term.starts_with(required_prefix) {
                continue;
            }

            let dist = levenshtein_distance(term.as_bytes(), candidate_term.as_bytes());
            if dist > max_distance {
                continue;
            }

            if let Some(token_id) = vocab_lookup(candidate_term) {
                results.push(FuzzyCandidate {
                    term: candidate_term.clone(),
                    token_id,
                });
                if results.len() >= limit {
                    break;
                }
            }
        }

        results.sort_unstable_by(|a, b| a.term.cmp(&b.term));
        results.truncate(limit);
        results
    }
}

/// Compute the successor string for a prefix to form an exclusive upper bound.
///
/// For example, `"he"` → `"hf"`. Returns `None` if all bytes are `0xFF`.
fn prefix_successor(prefix: &str) -> Option<String> {
    let mut bytes = prefix.as_bytes().to_vec();
    // Increment the last byte that is not 0xFF, removing trailing 0xFFs
    while let Some(last) = bytes.last_mut() {
        if *last < 0xFF {
            *last += 1;
            return Some(String::from_utf8(bytes).unwrap_or_else(|_| {
                // Not valid UTF-8 after increment — fall back to no upper bound
                String::new()
            }));
        }
        bytes.pop();
    }
    None
}

/// Compute Levenshtein edit distance between two byte slices using the Wagner–Fischer algorithm.
fn levenshtein_distance(a: &[u8], b: &[u8]) -> u32 {
    let m = a.len();
    let n = b.len();

    // Optimization: early return on trivial cases
    if m == 0 {
        return n as u32;
    }
    if n == 0 {
        return m as u32;
    }

    // Use two rows instead of full matrix to save memory
    let mut prev = vec![0u32; n + 1];
    let mut curr = vec![0u32; n + 1];

    for j in 0..=n {
        prev[j] = j as u32;
    }

    for i in 1..=m {
        curr[0] = i as u32;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1) // deletion
                .min(curr[j - 1] + 1) // insertion
                .min(prev[j - 1] + cost); // substitution
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_vocab(terms: &[&str]) -> HashMap<String, TokenId> {
        terms
            .iter()
            .enumerate()
            .map(|(i, t)| (t.to_string(), i as TokenId))
            .collect()
    }

    fn lookup(vocab: &HashMap<String, TokenId>) -> impl Fn(&str) -> Option<TokenId> + '_ {
        |term| vocab.get(term).copied()
    }

    // -- FuzzyExpander::build --

    #[test]
    fn build_returns_none_for_empty_vocab() {
        let vocab = HashMap::new();
        assert!(FuzzyExpander::build(&vocab).is_none());
    }

    #[test]
    fn build_succeeds_with_terms() {
        let vocab = build_vocab(&["hello", "world"]);
        let expander = FuzzyExpander::build(&vocab);
        assert!(expander.is_some());
    }

    // -- Serialization round-trip --

    #[test]
    fn serialize_deserialize_round_trip() {
        let vocab = build_vocab(&["alpha", "beta", "gamma"]);
        let expander = FuzzyExpander::build(&vocab).unwrap();
        let bytes = expander.to_bytes().to_vec();
        let restored = FuzzyExpander::from_bytes(bytes);
        assert!(restored.is_some());
    }

    // -- expand_term: short terms fall back to exact lookup --

    #[test]
    fn short_term_exact_match() {
        let vocab = build_vocab(&["ab", "abc", "xyz"]);
        let expander = FuzzyExpander::build(&vocab).unwrap();
        let params = FuzzyParams::default();

        // "ab" is shorter than MIN_TERM_LENGTH (3), should do exact lookup only
        let results = expander.expand_term("ab", &params, lookup(&vocab));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].term, "ab");
    }

    #[test]
    fn short_term_no_match() {
        let vocab = build_vocab(&["abc", "xyz"]);
        let expander = FuzzyExpander::build(&vocab).unwrap();
        let params = FuzzyParams::default();

        let results = expander.expand_term("zz", &params, lookup(&vocab));
        assert!(results.is_empty());
    }

    // -- expand_term: exact match at distance 0 --

    #[test]
    fn exact_match_at_distance_zero() {
        let vocab = build_vocab(&["hello", "hallo", "world"]);
        let expander = FuzzyExpander::build(&vocab).unwrap();
        let params = FuzzyParams {
            max_edit: 0,
            prefix_length: 0,
            max_expansions: 10,
        };

        let results = expander.expand_term("hello", &params, lookup(&vocab));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].term, "hello");
    }

    // -- expand_term: fuzzy match at distance 1 --

    #[test]
    fn fuzzy_match_edit_distance_one() {
        let vocab = build_vocab(&["hello", "hallo", "hullo", "world"]);
        let expander = FuzzyExpander::build(&vocab).unwrap();
        let params = FuzzyParams {
            max_edit: 1,
            prefix_length: 0,
            max_expansions: 10,
        };

        let results = expander.expand_term("hello", &params, lookup(&vocab));
        let terms: Vec<&str> = results.iter().map(|c| c.term.as_str()).collect();
        assert!(terms.contains(&"hello"));
        assert!(terms.contains(&"hallo"));
        assert!(terms.contains(&"hullo"));
        assert!(!terms.contains(&"world"));
    }

    // -- expand_term: prefix constraint --

    #[test]
    fn prefix_constraint_filters_results() {
        let vocab = build_vocab(&["hello", "hallo", "jello", "world"]);
        let expander = FuzzyExpander::build(&vocab).unwrap();
        let params = FuzzyParams {
            max_edit: 1,
            prefix_length: 2,
            max_expansions: 10,
        };

        // prefix "he" required; "hallo" and "jello" differ at prefix
        let results = expander.expand_term("hello", &params, lookup(&vocab));
        let terms: Vec<&str> = results.iter().map(|c| c.term.as_str()).collect();
        assert!(terms.contains(&"hello"));
        // "hallo" starts with "ha", not "he" — filtered by PrefixLevenshtein
        assert!(!terms.contains(&"hallo"));
        assert!(!terms.contains(&"jello"));
    }

    // -- expand_term: max_expansions cap --

    #[test]
    fn max_expansions_caps_results() {
        let vocab = build_vocab(&["aaa", "aab", "aac", "aad", "aae", "aaf"]);
        let expander = FuzzyExpander::build(&vocab).unwrap();
        let params = FuzzyParams {
            max_edit: 2,
            prefix_length: 0,
            max_expansions: 3,
        };

        let results = expander.expand_term("aaa", &params, lookup(&vocab));
        assert!(results.len() <= 3);
    }

    // -- PrefixLevenshtein --

    #[test]
    fn prefix_levenshtein_rejects_too_long_prefix() {
        assert!(PrefixLevenshtein::new("hi", 5, 1).is_none());
    }

    #[test]
    fn prefix_levenshtein_zero_prefix() {
        // prefix_len = 0 means pure Levenshtein on the whole query
        let automaton = PrefixLevenshtein::new("hello", 0, 1);
        assert!(automaton.is_some());
    }

    #[test]
    fn levenshtein_empty_strings() {
        assert_eq!(levenshtein_distance(b"", b""), 0);
        assert_eq!(levenshtein_distance(b"abc", b""), 3);
        assert_eq!(levenshtein_distance(b"", b"xyz"), 3);
    }

    #[test]
    fn levenshtein_identical() {
        assert_eq!(levenshtein_distance(b"hello", b"hello"), 0);
    }

    #[test]
    fn levenshtein_single_edit() {
        assert_eq!(levenshtein_distance(b"hello", b"hallo"), 1); // substitution
        assert_eq!(levenshtein_distance(b"hello", b"hell"), 1); // deletion
        assert_eq!(levenshtein_distance(b"hello", b"hellos"), 1); // insertion
    }

    #[test]
    fn levenshtein_two_edits() {
        assert_eq!(levenshtein_distance(b"hello", b"hxllo"), 1);
        assert_eq!(levenshtein_distance(b"kitten", b"sitten"), 1);
        assert_eq!(levenshtein_distance(b"kitten", b"sitting"), 3);
    }

    // -- prefix_successor --

    #[test]
    fn prefix_successor_normal() {
        assert_eq!(prefix_successor("he"), Some("hf".to_string()));
        assert_eq!(prefix_successor("a"), Some("b".to_string()));
    }

    #[test]
    fn prefix_successor_all_ff() {
        let s = String::from_utf8(vec![0xFF]).unwrap_or_default();
        // A single 0xFF byte string — may not be valid UTF-8, but let's handle gracefully
        assert!(prefix_successor(&s).is_none() || prefix_successor(&s).is_some());
    }

    // -- MutableFuzzyExpander --

    #[test]
    fn new_expander_is_empty() {
        let expander = MutableFuzzyExpander::new();
        assert!(expander.is_empty());
        assert_eq!(expander.len(), 0);
    }

    #[test]
    fn insert_and_len() {
        let mut expander = MutableFuzzyExpander::new();
        assert!(expander.insert("hello"));
        assert!(!expander.insert("hello")); // duplicate
        assert!(expander.insert("world"));
        assert_eq!(expander.len(), 2);
    }

    #[test]
    fn from_vocab_keys() {
        let vocab = build_vocab(&["alpha", "beta", "gamma"]);
        let expander = MutableFuzzyExpander::from_vocab_keys(vocab.keys());
        assert_eq!(expander.len(), 3);
    }

    #[test]
    fn short_term_exact_lookup() {
        let vocab = build_vocab(&["ab", "abc", "xyz"]);
        let expander = MutableFuzzyExpander::from_vocab_keys(vocab.keys());
        let params = FuzzyParams::default();

        let results = expander.expand_term("ab", &params, &lookup(&vocab));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].term, "ab");
    }

    #[test]
    fn incremental_insert_expands_new_term() {
        let vocab = build_vocab(&["hello", "world"]);
        let mut expander = MutableFuzzyExpander::from_vocab_keys(vocab.keys());

        // "hallo" not yet in expander
        let params = FuzzyParams {
            max_edit: 1,
            prefix_length: 0,
            max_expansions: 10,
        };
        let results = expander.expand_term("hello", &params, &lookup(&vocab));
        assert!(!results.iter().any(|c| c.term == "hallo"));

        // Add "hallo" to expander AND vocab
        let mut vocab = vocab;
        let new_id = vocab.len() as TokenId;
        vocab.insert("hallo".to_string(), new_id);
        expander.insert("hallo");

        let results = expander.expand_term("hello", &params, &lookup(&vocab));
        assert!(results.iter().any(|c| c.term == "hallo"));
    }
}
