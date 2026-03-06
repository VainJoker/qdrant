//! FST-based fuzzy term expansion using Levenshtein automaton.

use std::collections::{BTreeSet, HashMap};
use std::str;

use fst::automaton::{Automaton, Levenshtein};
use fst::{IntoStreamer, Set, Streamer};

use super::TermExpander;
use crate::index::field_index::full_text_index::inverted_index::TokenId;
use crate::types::FuzzyParams;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuzzyCandidate {
    pub term: String,
    pub token_id: TokenId,
}

pub struct PrefixLevenshtein {
    query: String,
    prefix_len: usize,
    levenshtein: Levenshtein,
}

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

pub trait TermDictionary {
    fn search(
        &self,
        term: &str,
        max_distance: u32,
        prefix_len: usize,
        vocab_lookup: &dyn Fn(&str) -> Option<TokenId>,
        limit: usize,
    ) -> Vec<FuzzyCandidate>;
}

pub struct FuzzyExpander<T = Set<Vec<u8>>> {
    dictionary: T,
}

pub type MutableFuzzyExpander = FuzzyExpander<BTreeSet<String>>;

impl Default for FuzzyExpander<Set<Vec<u8>>> {
    fn default() -> Self {
        Self::new()
    }
}

impl TermDictionary for Set<Vec<u8>> {
    fn search(
        &self,
        term: &str,
        max_distance: u32,
        prefix_len: usize,
        vocab_lookup: &dyn Fn(&str) -> Option<TokenId>,
        limit: usize,
    ) -> Vec<FuzzyCandidate> {
        let use_prefix_automaton = prefix_len > 0 && prefix_len < term.len();
        let prefix = if prefix_len > 0 && prefix_len <= term.len() {
            &term[..prefix_len]
        } else {
            ""
        };

        if use_prefix_automaton {
            match PrefixLevenshtein::new(term, prefix_len, max_distance) {
                Some(automaton) => search_fst(self, automaton, "", vocab_lookup, limit),
                None => Vec::new(),
            }
        } else {
            match Levenshtein::new(term, max_distance) {
                Ok(automaton) => search_fst(self, automaton, prefix, vocab_lookup, limit),
                Err(_) => Vec::new(),
            }
        }
    }
}

fn search_fst<A>(
    fst_set: &Set<Vec<u8>>,
    automaton: A,
    required_prefix: &str,
    vocab_lookup: &dyn Fn(&str) -> Option<TokenId>,
    limit: usize,
) -> Vec<FuzzyCandidate>
where
    A: fst::Automaton,
{
    let mut stream = fst_set.search(&automaton).into_stream();
    let mut results = Vec::with_capacity(limit.min(32));

    while let Some(key) = stream.next() {
        let Ok(matched_term) = str::from_utf8(key) else { continue };

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

impl FuzzyExpander<Set<Vec<u8>>> {
    pub fn new() -> Self {
        Self {
            dictionary: Set::from_iter(std::iter::empty::<&str>()).unwrap(),
        }
    }

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
        Some(Self {
            dictionary: fst_set,
        })
    }

    /// Serialize the FST to bytes.
    pub fn to_bytes(&self) -> &[u8] {
        self.dictionary.as_fst().as_bytes()
    }

    /// Deserialize the FST from bytes.
    pub fn from_bytes(data: Vec<u8>) -> Option<Self> {
        let fst_set = Set::new(data).ok()?;
        Some(Self {
            dictionary: fst_set,
        })
    }
}

impl TermDictionary for BTreeSet<String> {
    fn search(
        &self,
        term: &str,
        max_distance: u32,
        prefix_len: usize,
        vocab_lookup: &dyn Fn(&str) -> Option<TokenId>,
        limit: usize,
    ) -> Vec<FuzzyCandidate> {
        let required_prefix = if prefix_len > 0 && prefix_len <= term.len() {
            &term[..prefix_len]
        } else {
            ""
        };

        let mut results = Vec::with_capacity(limit.min(32));

        let iter: Box<dyn Iterator<Item = &String>> = if required_prefix.is_empty() {
            Box::new(self.iter())
        } else {
            let start = required_prefix.to_string();
            let end = prefix_successor(required_prefix);
            match end {
                Some(end) => Box::new(self.range(start..end)),
                None => Box::new(self.range(start..)),
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

impl FuzzyExpander<BTreeSet<String>> {
    pub fn new() -> Self {
        Self {
            dictionary: BTreeSet::new(),
        }
    }

    /// Build from an iterator of vocabulary keys.
    pub fn from_vocab_keys(keys: impl Iterator<Item = impl AsRef<str>>) -> Self {
        let terms = keys.map(|k| k.as_ref().to_owned()).collect();
        Self { dictionary: terms }
    }

    /// Insert a term. Returns `true` if the term was newly inserted.
    pub fn insert(&mut self, term: &str) -> bool {
        self.dictionary.insert(term.to_owned())
    }

    pub fn is_empty(&self) -> bool {
        self.dictionary.is_empty()
    }

    pub fn len(&self) -> usize {
        self.dictionary.len()
    }
}

impl<T: TermDictionary> FuzzyExpander<T> {
    pub fn expand_term(
        &self,
        term: &str,
        params: &FuzzyParams,
        vocab_lookup: impl Fn(&str) -> Option<TokenId>,
    ) -> Vec<FuzzyCandidate> {
        TermExpander::expand_term(self, term, params, &vocab_lookup)
    }
}

impl<T: TermDictionary> TermExpander for FuzzyExpander<T> {
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
        let max_expansions = params.max_expansions.clamp(1, FuzzyParams::MAX_EXPANSIONS_CAP) as usize;

        let mut candidates = Vec::with_capacity(max_expansions.min(64));
        let mut seen_ids = std::collections::HashSet::new();

        for distance in 0..=max_distance {
            let remaining = max_expansions - candidates.len();
            if remaining == 0 {
                break;
            }
            let limit = max_expansions;

            let batch = self
                .dictionary
                .search(term, distance, prefix_len, vocab_lookup, limit);

            for candidate in batch {
                if candidates.len() >= max_expansions {
                    break;
                }
                if seen_ids.insert(candidate.token_id) {
                    candidates.push(candidate);
                }
            }
        }

        candidates.truncate(max_expansions);

        candidates
    }
}

/// Compute the successor string for a prefix to form an exclusive upper bound.
fn prefix_successor(prefix: &str) -> Option<String> {
    let mut bytes = prefix.as_bytes().to_vec();
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
    let mut prev: Vec<u32> = (0..=n as u32).collect();
    let mut curr = vec![0u32; n + 1];

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

    #[test]
    fn prefix_constraint_filters_candidates() {
        let vocab = build_vocab(&["hello", "hallo", "jello"]);
        let expander = FuzzyExpander::build(&vocab).unwrap();

        let params = FuzzyParams {
            max_edit: 1,
            prefix_length: 2,
            max_expansions: 10,
        };

        let results = expander.expand_term("hello", &params, lookup(&vocab));

        let terms: Vec<&str> = results.iter().map(|c| c.term.as_str()).collect();

        assert!(terms.contains(&"hello"));
        assert!(!terms.contains(&"hallo"));
        assert!(!terms.contains(&"jello"));
    }

    #[test]
    fn max_expansions_limits_results() {
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

    #[test]
    fn fuzzy_results_are_lexicographically_sorted() {
        let vocab = build_vocab(&["hello", "haxllo", "hallo", "hullo", "hxllo"]);
        let expander = FuzzyExpander::build(&vocab).unwrap();

        let params = FuzzyParams {
            max_edit: 2,
            prefix_length: 0,
            max_expansions: 10,
        };

        let results = expander.expand_term("hello", &params, lookup(&vocab));

        let terms: Vec<&str> = results.iter().map(|c| c.term.as_str()).collect();

        // With tiered expansion (distance 0 then distance 1):
        // 1. "hello" (dist 0) is found first.
        // 2. "hallo", "hullo", "hxllo" (dist 1) are found next.
        // Within the same distance, results are sorted lexicographically.
        //
        // So expected order is ["hello", "hallo", "hullo", "hxllo", "haxllo"].
        // This is NOT strictly lexicographical overall ("hello" > "hallo"), but
        // it is "distance-then-lexicographical".

        let expected = vec!["hello", "hallo", "hullo", "hxllo", "haxllo"];
        assert_eq!(terms, expected);
    }

    #[test]
    fn mutable_expander_supports_incremental_terms() {
        let vocab = build_vocab(&["hello", "world"]);
        let mut expander = MutableFuzzyExpander::from_vocab_keys(vocab.keys());

        let params = FuzzyParams {
            max_edit: 1,
            prefix_length: 0,
            max_expansions: 10,
        };

        let results = expander.expand_term("hello", &params, &lookup(&vocab));
        assert!(!results.iter().any(|c| c.term == "hallo"));

        let mut vocab = vocab;
        let id = vocab.len() as TokenId;

        vocab.insert("hallo".to_string(), id);
        expander.insert("hallo");

        let results = expander.expand_term("hello", &params, &lookup(&vocab));
        assert!(results.iter().any(|c| c.term == "hallo"));
    }

    #[test]
    fn levenshtein_edits() {
        assert_eq!(levenshtein_distance(b"hello", b"hallo"), 1);
        assert_eq!(levenshtein_distance(b"hello", b"hell"), 1);
        assert_eq!(levenshtein_distance(b"hello", b"hellos"), 1);
        assert_eq!(levenshtein_distance(b"kitten", b"sitting"), 3);
        assert_eq!(levenshtein_distance(b"flaw", b"lawn"), 2);
    }

    #[test]
    fn query_term_not_in_vocab_still_expands() {
        let vocab = build_vocab(&["hello", "hallo", "hell"]);
        let expander = FuzzyExpander::build(&vocab).unwrap();

        let params = FuzzyParams {
            max_edit: 1,
            prefix_length: 0,
            max_expansions: 10,
        };

        let results = expander.expand_term("helo", &params, lookup(&vocab));
        let terms: Vec<&str> = results.iter().map(|c| c.term.as_str()).collect();

        assert!(terms.contains(&"hello"));
        assert!(terms.contains(&"hell"));
    }

    #[test]
    fn mutable_expander_respects_prefix_length() {
        let vocab = build_vocab(&["hello", "hallo", "jello"]);
        let expander = MutableFuzzyExpander::from_vocab_keys(vocab.keys());

        let params = FuzzyParams {
            max_edit: 1,
            prefix_length: 2,
            max_expansions: 10,
        };

        let results = expander.expand_term("hello", &params, &lookup(&vocab));
        let terms: Vec<&str> = results.iter().map(|c| c.term.as_str()).collect();

        // hello prefix = he
        assert!(terms.contains(&"hello"));
        assert!(!terms.contains(&"hallo"));
        assert!(!terms.contains(&"jello"));
    }

    #[test]
    fn mutable_expander_respects_max_expansions() {
        let vocab = build_vocab(&["aaa", "aab", "aac", "aad", "aae", "aaf"]);
        let expander = MutableFuzzyExpander::from_vocab_keys(vocab.keys());

        let params = FuzzyParams {
            max_edit: 2,
            prefix_length: 0,
            max_expansions: 2,
        };

        let results = expander.expand_term("aaa", &params, &lookup(&vocab));

        assert!(results.len() <= 2);
    }

    #[test]
    fn mutable_expander_respects_max_edit() {
        let vocab = build_vocab(&["hello", "hallo", "hell", "hellos", "yellow"]);
        let expander = MutableFuzzyExpander::from_vocab_keys(vocab.keys());

        let params = FuzzyParams {
            max_edit: 1,
            prefix_length: 0,
            max_expansions: 10,
        };

        let results = expander.expand_term("hello", &params, &lookup(&vocab));
        let terms: Vec<&str> = results.iter().map(|c| c.term.as_str()).collect();

        assert!(terms.contains(&"hello"));
        assert!(terms.contains(&"hallo"));
        assert!(terms.contains(&"hell"));
        assert!(terms.contains(&"hellos"));

        // distance = 2
        assert!(!terms.contains(&"yellow"));
    }

    #[test]
    fn mutable_expander_keeps_distance_then_lexicographic_order() {
        let vocab = build_vocab(&["hello", "hallo", "hullo", "hxllo"]);
        let expander = MutableFuzzyExpander::from_vocab_keys(vocab.keys());

        let params = FuzzyParams {
            max_edit: 1,
            prefix_length: 0,
            max_expansions: 10,
        };

        let results = expander.expand_term("hello", &params, &lookup(&vocab));

        let terms: Vec<&str> = results.iter().map(|c| c.term.as_str()).collect();

        // The order is distance-then-lexicographical:
        // dist 0: "hello"
        // dist 1: "hallo", "hullo", "hxllo" (sorted)
        let expected = vec!["hello", "hallo", "hullo", "hxllo"];
        assert_eq!(terms, expected);
    }
}
