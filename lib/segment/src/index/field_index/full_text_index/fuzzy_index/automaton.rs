use std::error::Error;

use fst::Automaton;
use fst::automaton::Levenshtein;
use smallvec::{SmallVec, smallvec};

use super::prefix_chars;
use crate::types::WildcardParams;

pub const WILDCARD_ANY: char = '*';
pub const WILDCARD_ONE: char = '?';

/// Levenshtein automaton that requires an exact byte prefix before fuzzy
/// matching starts.
pub struct PrefixLevenshtein {
    prefix: Vec<u8>,
    lev: Levenshtein,
}

impl PrefixLevenshtein {
    /// Builds an automaton that enforces `prefix_len` exact UTF-8 characters and
    /// then allows up to `distance` edits for the full query.
    pub fn new(query: &str, prefix_len: usize, distance: u32) -> Result<Self, Box<dyn Error>> {
        let prefix = prefix_chars(query, prefix_len).as_bytes().to_vec();
        let lev = Levenshtein::new(query, distance)?;
        Ok(Self { prefix, lev })
    }
}

/// State machine for [`PrefixLevenshtein`].
pub enum PrefixLevState {
    /// Still consuming the exact prefix bytes.
    Prefix(usize),
    /// Exact prefix is complete; delegate to the inner Levenshtein automaton.
    Fuzzy(<Levenshtein as Automaton>::State),
    /// Prefix mismatch made the candidate unreachable.
    Dead,
}

impl Automaton for PrefixLevenshtein {
    type State = PrefixLevState;

    fn start(&self) -> Self::State {
        if self.prefix.is_empty() {
            PrefixLevState::Fuzzy(self.lev.start())
        } else {
            PrefixLevState::Prefix(0)
        }
    }

    /// Returns true if and only if state is a match state.
    ///
    /// Examples with query `"cart"`, `prefix_len = 2`, `distance = 1`:
    /// - after reading `"c"`: state is `Prefix(1)`, so `is_match = false`
    /// - after reading `"ca"`: state moves to `Fuzzy(...)`, but `is_match` still
    ///   depends on the inner Levenshtein state
    /// - after reading `"card"`: state is `Fuzzy(...)` and `is_match = true`
    fn is_match(&self, state: &Self::State) -> bool {
        match state {
            PrefixLevState::Prefix(_) => false,
            PrefixLevState::Fuzzy(s) => self.lev.is_match(s),
            PrefixLevState::Dead => false,
        }
    }

    /// Returns true if and only if `state` can still reach a match in zero or
    /// more future steps.
    ///
    /// Examples with query `"cart"`, `prefix_len = 2`, `distance = 1`:
    /// - after reading `"c"`: state is `Prefix(1)`, so `can_match = true`
    ///   because another `"a"` can continue the exact prefix path
    /// - after reading `"d"`: state is `Dead`, so `can_match = false`
    /// - after reading `"ca"`: state is `Fuzzy(...)`; whether it can still match
    ///   is delegated to the inner Levenshtein automaton
    ///
    /// If this returns false, then no sequence of future inputs should ever
    /// produce a match. Returning true when no match is actually possible is
    /// still correct, but it may force callers to do extra work.
    fn can_match(&self, state: &Self::State) -> bool {
        match state {
            PrefixLevState::Prefix(_) => true,
            PrefixLevState::Fuzzy(s) => self.lev.can_match(s),
            PrefixLevState::Dead => false,
        }
    }

    fn will_always_match(&self, state: &Self::State) -> bool {
        match state {
            PrefixLevState::Prefix(_) => false,
            PrefixLevState::Fuzzy(s) => self.lev.will_always_match(s),
            PrefixLevState::Dead => false,
        }
    }

    /// Consumes one byte and returns the next automaton state.
    ///
    /// Examples with query `"cart"`, `prefix_len = 2`, `distance = 1`:
    /// - `Prefix(0)` + `'c'` -> `Prefix(1)`
    /// - `Prefix(1)` + `'a'` -> `Fuzzy(...)` because the exact prefix `"ca"`
    ///   is now complete
    /// - `Prefix(0)` + `'d'` -> `Dead` because the exact prefix mismatched
    /// - `Fuzzy(s)` + `'r'` -> advances only the inner Levenshtein state
    /// - `Dead` + any byte -> `Dead`
    ///
    /// The transition to `Fuzzy(...)` replays the exact prefix into the inner
    /// Levenshtein automaton so that fuzzy matching continues from the full
    /// query prefix that has already been consumed.
    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        match state {
            PrefixLevState::Prefix(i) => {
                // While we are in the exact-prefix phase, every byte must match
                // the corresponding byte from `self.prefix`.
                if byte == self.prefix[*i] {
                    let next = i + 1;
                    if next == self.prefix.len() {
                        // The exact prefix is fully consumed. Switch to fuzzy
                        // matching, but first replay the accepted prefix bytes
                        // into the inner Levenshtein automaton so its state is
                        // aligned with the input consumed so far.
                        let mut lev_state = self.lev.start();
                        for &b in &self.prefix {
                            lev_state = self.lev.accept(&lev_state, b);
                        }
                        PrefixLevState::Fuzzy(lev_state)
                    } else {
                        // Still consuming the exact prefix.
                        PrefixLevState::Prefix(next)
                    }
                } else {
                    // Any mismatch in the required prefix makes this path
                    // unreachable.
                    PrefixLevState::Dead
                }
            }
            // Once the exact prefix is done, delegate all future bytes to the
            // inner Levenshtein automaton.
            PrefixLevState::Fuzzy(s) => PrefixLevState::Fuzzy(self.lev.accept(s, byte)),
            // Dead is a sink state.
            PrefixLevState::Dead => PrefixLevState::Dead,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum WildcardToken {
    /// A run of literal UTF-8 bytes that must match exactly.
    Literal(Vec<u8>),
    /// `?` – matches exactly one Unicode code point (1‒4 UTF-8 bytes).
    AnyOne,
    /// `*` – matches zero or more arbitrary bytes.
    AnyMany,
}

/// Parse a wildcard pattern string into a sequence of [`WildcardToken`]s.
///
/// * Consecutive `*` are collapsed into a single `AnyMany`.
/// * Adjacent literal characters are merged into one `Literal`.
fn parse_wildcard_pattern(pattern: &str) -> Vec<WildcardToken> {
    debug_assert!(
        !pattern.is_empty(),
        "wildcard patterns are validated by the API layer"
    );
    debug_assert!(
        pattern.len() <= WildcardParams::MAX_PATTERN_LENGTH,
        "wildcard patterns are length-limited by the API layer"
    );

    let mut tokens: Vec<WildcardToken> = Vec::new();
    let mut literal_buf: Vec<u8> = Vec::new();

    for ch in pattern.chars() {
        match ch {
            WILDCARD_ANY => {
                if !literal_buf.is_empty() {
                    tokens.push(WildcardToken::Literal(std::mem::take(&mut literal_buf)));
                }
                // Collapse consecutive `*`
                if !matches!(tokens.last(), Some(WildcardToken::AnyMany)) {
                    tokens.push(WildcardToken::AnyMany);
                }
            }
            WILDCARD_ONE => {
                if !literal_buf.is_empty() {
                    tokens.push(WildcardToken::Literal(std::mem::take(&mut literal_buf)));
                }
                tokens.push(WildcardToken::AnyOne);
            }
            other => {
                let mut buf = [0u8; 4];
                literal_buf.extend_from_slice(other.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    if !literal_buf.is_empty() {
        tokens.push(WildcardToken::Literal(literal_buf));
    }
    tokens
}

/// Extract the longest literal prefix from a wildcard pattern (before the
/// first `*` or `?`).  Returns the prefix as a byte vector suitable for
/// `fst::StreamBuilder::ge()`.
pub fn extract_literal_prefix(pattern: &str) -> Vec<u8> {
    let mut prefix = Vec::new();
    for ch in pattern.chars() {
        if ch == WILDCARD_ANY || ch == WILDCARD_ONE {
            break;
        }
        let mut buf = [0u8; 4];
        prefix.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
    }
    prefix
}

/// A single "position" inside the NFA.
#[derive(Clone, Debug, PartialEq, Eq)]
enum AtomState {
    /// Positioned inside a token, with an optional byte offset into a
    /// `Literal` token.
    AtToken {
        token_index: usize,
        literal_offset: usize,
    },
    /// Mid-way through consuming a multi-byte UTF-8 code point for `AnyOne`.
    AtAnyOne { token_index: usize, remaining: u8 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WildcardStateSet {
    atoms: SmallVec<[AtomState; 8]>,
}

pub struct WildcardAutomaton {
    tokens: Vec<WildcardToken>,
    literal_prefix: Vec<u8>,
    /// Whether the last token is `AnyMany`, precomputed for `will_always_match`.
    trailing_star: bool,
}

/// Push `atom` into `vec` only if it is not already present.
#[inline]
fn push_unique(vec: &mut SmallVec<[AtomState; 8]>, atom: AtomState) {
    if !vec.contains(&atom) {
        vec.push(atom);
    }
}

impl WildcardAutomaton {
    /// Build a new automaton from a wildcard pattern.
    pub fn new(pattern: &str) -> Self {
        let tokens = parse_wildcard_pattern(pattern);
        let literal_prefix = extract_literal_prefix(pattern);
        let trailing_star = matches!(tokens.last(), Some(WildcardToken::AnyMany));
        Self {
            tokens,
            literal_prefix,
            trailing_star,
        }
    }

    /// The literal prefix extracted from the pattern, useful for `.ge()` on
    /// FST streams.
    pub fn literal_prefix(&self) -> &[u8] {
        &self.literal_prefix
    }

    /// Number of tokens in this automaton (exposed for tests).
    #[cfg(test)]
    fn token_count(&self) -> usize {
        self.tokens.len()
    }

    /// Compute the epsilon-closure of the current NFA frontier.
    ///
    /// In this wildcard automaton, the only epsilon transition comes from `*`
    /// (`WildcardToken::AnyMany`): besides the branch that keeps consuming more
    /// input while staying on `*`, there is also a zero-cost branch that skips
    /// over `*` and moves directly to the next token.
    ///
    /// This function repeatedly expands that zero-cost branch for every atom in
    /// `atoms` until no new reachable state can be added.
    ///
    /// Example:
    ///
    /// Suppose the pattern is `ab*cd`, which is tokenized as:
    ///
    /// - `0 => Literal("ab")`
    /// - `1 => AnyMany`
    /// - `2 => Literal("cd")`
    ///
    /// If the current frontier contains:
    ///
    /// - `AtToken { token_index: 1, literal_offset: 0 }`
    ///
    /// it means the automaton is currently positioned on `*`.
    /// At this point there are two legal possibilities:
    ///
    /// - stay on token `1` and let `*` consume one more input byte later;
    /// - skip token `1` immediately and continue from token `2`.
    ///
    /// `epsilon_closure` adds that second possibility by inserting:
    ///
    /// - `AtToken { token_index: 2, literal_offset: 0 }`
    ///
    /// into the frontier.
    ///
    /// The `while i < atoms.len()` loop is important: newly inserted states are
    /// scanned as well, so chained stars such as `a**b` (collapsed during
    /// parsing) or any future epsilon-producing tokens would also be fully
    /// expanded before the function returns.
    ///
    /// Only atoms exactly at the start of a token (`literal_offset == 0`) are
    /// considered. A state inside a literal byte sequence cannot take an
    /// epsilon transition because it must finish matching the remaining bytes of
    /// that literal first.
    fn epsilon_closure(&self, mut atoms: SmallVec<[AtomState; 8]>) -> SmallVec<[AtomState; 8]> {
        let mut i = 0;
        while i < atoms.len() {
            if let AtomState::AtToken {
                token_index,
                literal_offset: 0,
            } = atoms[i]
            {
                if token_index < self.tokens.len() {
                    if let WildcardToken::AnyMany = &self.tokens[token_index] {
                        let next = AtomState::AtToken {
                            token_index: token_index + 1,
                            literal_offset: 0,
                        };
                        if !atoms.contains(&next) {
                            atoms.push(next);
                        }
                    }
                }
            }
            i += 1;
        }
        atoms
    }
}

/// Determine the expected total byte length of a UTF-8 code point from its
/// leading byte.  Returns `None` for invalid leading bytes.
#[inline]
fn utf8_char_len(first_byte: u8) -> Option<u8> {
    match first_byte {
        0x00..=0x7F => Some(1),
        0xC2..=0xDF => Some(2),
        0xE0..=0xEF => Some(3),
        0xF0..=0xF4 => Some(4),
        _ => None,
    }
}

#[inline]
fn is_utf8_continuation(byte: u8) -> bool {
    byte & 0xC0 == 0x80
}

impl Automaton for WildcardAutomaton {
    type State = WildcardStateSet;

    fn start(&self) -> Self::State {
        // `smallvec!` is clearer than `SmallVec::from_elem(..., 1)` whose
        // "repeat N times" semantics are misleading for a single element.
        let init = smallvec![AtomState::AtToken {
            token_index: 0,
            literal_offset: 0,
        }];
        WildcardStateSet {
            atoms: self.epsilon_closure(init),
        }
    }

    fn is_match(&self, state: &Self::State) -> bool {
        let end_index = self.tokens.len();
        state.atoms.iter().any(|a| {
            matches!(a, AtomState::AtToken { token_index, literal_offset: 0 } if *token_index == end_index)
        })
    }

    fn can_match(&self, state: &Self::State) -> bool {
        !state.atoms.is_empty()
    }

    fn will_always_match(&self, state: &Self::State) -> bool {
        // Fast path: if the pattern doesn't end with `*` it can never be in a
        // "will always match" state, regardless of current position.
        if !self.trailing_star {
            return false;
        }
        // The automaton will always match iff we are currently positioned on
        // the very last token and that token is `AnyMany`. We check
        // `token_index + 1 == end_index` directly rather than also verifying
        // that the end sentinel is present in the atom set, because
        // `epsilon_closure` guarantees the sentinel is always pushed alongside
        // any `AnyMany` atom.
        let end_index = self.tokens.len();
        state.atoms.iter().any(|a| {
            matches!(
                a,
                AtomState::AtToken { token_index, literal_offset: 0 }
                    if *token_index + 1 == end_index
                    && matches!(&self.tokens[*token_index], WildcardToken::AnyMany)
            )
        })
    }

    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        // Invariant: state sets must always be deduplicated by callers
        // (epsilon_closure and this function both enforce it).
        debug_assert!(
            (0..state.atoms.len()).all(|i| !state.atoms[i + 1..].contains(&state.atoms[i])),
            "state set should contain no duplicate atoms"
        );

        let mut next: SmallVec<[AtomState; 8]> = SmallVec::new();

        for atom in &state.atoms {
            match atom {
                AtomState::AtToken {
                    token_index,
                    literal_offset,
                } => {
                    let tidx = *token_index;
                    let loff = *literal_offset;
                    if tidx >= self.tokens.len() {
                        // Past the end – no transition.
                        continue;
                    }
                    match &self.tokens[tidx] {
                        WildcardToken::Literal(bytes) => {
                            if loff < bytes.len() && bytes[loff] == byte {
                                let new_off = loff + 1;
                                let new_atom = if new_off == bytes.len() {
                                    AtomState::AtToken {
                                        token_index: tidx + 1,
                                        literal_offset: 0,
                                    }
                                } else {
                                    AtomState::AtToken {
                                        token_index: tidx,
                                        literal_offset: new_off,
                                    }
                                };
                                push_unique(&mut next, new_atom);
                            }
                        }
                        WildcardToken::AnyOne => {
                            // Unify the ASCII (total == 1) and multi-byte
                            // branches: both produce a single new atom.
                            if let Some(total) = utf8_char_len(byte) {
                                let new_atom = if total == 1 {
                                    AtomState::AtToken {
                                        token_index: tidx + 1,
                                        literal_offset: 0,
                                    }
                                } else {
                                    AtomState::AtAnyOne {
                                        token_index: tidx,
                                        remaining: total - 1,
                                    }
                                };
                                push_unique(&mut next, new_atom);
                            }
                            // Invalid leading byte → drop this path.
                        }
                        WildcardToken::AnyMany => {
                            // Consume the byte and stay at the same `*`.
                            push_unique(
                                &mut next,
                                AtomState::AtToken {
                                    token_index: tidx,
                                    literal_offset: 0,
                                },
                            );
                        }
                    }
                }
                AtomState::AtAnyOne {
                    token_index,
                    remaining,
                } => {
                    if is_utf8_continuation(byte) {
                        // Unify the "last continuation byte" and "more bytes
                        // to go" branches: both produce a single new atom.
                        let new_atom = if *remaining == 1 {
                            AtomState::AtToken {
                                token_index: token_index + 1,
                                literal_offset: 0,
                            }
                        } else {
                            AtomState::AtAnyOne {
                                token_index: *token_index,
                                remaining: remaining - 1,
                            }
                        };
                        push_unique(&mut next, new_atom);
                    }
                    // Non-continuation byte → invalid UTF-8, drop path.
                }
            }
        }

        WildcardStateSet {
            atoms: self.epsilon_closure(next),
        }
    }
}

/// Simple helper that runs the automaton byte-by-byte over a string.
/// Useful for the mutable (BTreeSet) backend and for unit tests.
pub fn wildcard_matches(pattern: &str, input: &str) -> bool {
    let automaton = WildcardAutomaton::new(pattern);
    let mut state = automaton.start();
    for &b in input.as_bytes() {
        state = automaton.accept(&state, b);
        if !automaton.can_match(&state) {
            return false;
        }
    }
    automaton.is_match(&state)
}

#[cfg(test)]
mod tests {
    use fst::{Automaton, IntoStreamer, Set, Streamer};

    use super::{PrefixLevState, PrefixLevenshtein, WildcardAutomaton, wildcard_matches};

    fn accept_str(automaton: &PrefixLevenshtein, input: &str) -> PrefixLevState {
        let mut state = automaton.start();
        for byte in input.bytes() {
            state = automaton.accept(&state, byte);
        }
        state
    }

    fn assert_matches(pattern: &str, cases: &[(&str, bool)]) {
        for (input, expected) in cases {
            assert_eq!(
                wildcard_matches(pattern, input),
                *expected,
                "pattern {pattern:?} against input {input:?}"
            );
        }
    }

    fn fst_search(vocab: &[&str], pattern: &str) -> Vec<String> {
        let mut sorted_terms = vocab.to_vec();
        sorted_terms.sort_unstable();
        let index = Set::from_iter(sorted_terms).unwrap();
        let automaton = WildcardAutomaton::new(pattern);
        let prefix = automaton.literal_prefix();

        let mut stream = if prefix.is_empty() {
            index.search(&automaton).into_stream()
        } else {
            index.search(&automaton).ge(prefix).into_stream()
        };

        let mut results = Vec::new();
        while let Some(term_bytes) = stream.next() {
            results.push(std::str::from_utf8(term_bytes).unwrap().to_string());
        }
        results
    }

    #[test]
    fn prefix_state_can_match_but_is_not_match() {
        let automaton = PrefixLevenshtein::new("abc", 2, 1).unwrap();

        let state = accept_str(&automaton, "a");

        assert!(!automaton.is_match(&state));
        assert!(automaton.can_match(&state));
    }

    #[test]
    fn full_prefix_transitions_to_fuzzy_with_true_is_match_and_can_match() {
        let automaton = PrefixLevenshtein::new("ab", 2, 1).unwrap();

        let state = accept_str(&automaton, "ab");

        assert!(matches!(state, PrefixLevState::Fuzzy(_)));
        assert!(automaton.is_match(&state));
        assert!(automaton.can_match(&state));
    }

    #[test]
    fn prefix_mismatch_goes_dead() {
        let automaton = PrefixLevenshtein::new("abc", 2, 1).unwrap();

        let state = accept_str(&automaton, "x");

        assert!(matches!(state, PrefixLevState::Dead));
        assert!(!automaton.is_match(&state));
        assert!(!automaton.can_match(&state));
    }

    #[test]
    fn fuzzy_matching_starts_after_exact_prefix() {
        let automaton = PrefixLevenshtein::new("abc", 2, 1).unwrap();

        let state = accept_str(&automaton, "abd"); // codespell:ignore abd

        assert!(matches!(state, PrefixLevState::Fuzzy(_)));
        assert!(automaton.is_match(&state));
    }

    #[test]
    fn unicode_prefix_length_uses_chars_not_bytes() {
        let automaton = PrefixLevenshtein::new("éclair", 1, 1).unwrap();

        let state = accept_str(&automaton, "éclair");

        assert!(matches!(state, PrefixLevState::Fuzzy(_)));
        assert!(automaton.is_match(&state));
    }

    #[test]
    fn test_will_always_match_trailing_star() {
        let a = WildcardAutomaton::new("foo*");
        let mut state = a.start();
        for &b in b"foo" {
            state = a.accept(&state, b);
        }
        assert!(a.will_always_match(&state));
    }

    #[test]
    fn test_will_always_match_false_without_trailing_star() {
        let a = WildcardAutomaton::new("foo*bar");
        let mut state = a.start();
        for &b in b"foo" {
            state = a.accept(&state, b);
        }
        assert!(!a.will_always_match(&state));
    }

    #[test]
    fn test_question_matches_one_unicode_code_point() {
        assert_matches(
            "a?c",
            &[
                ("abc", true),
                ("aéc", true),
                ("a💡c", true),
                ("a--c", false),
            ],
        );
    }

    #[test]
    fn test_complex_pattern() {
        assert_matches(
            "a*b?c*d",
            &[
                ("abxcd", true),
                ("aXXXbYcZZZd", true),
                ("abcd", false),
                ("abxc", false),
            ],
        );
    }

    #[test]
    fn test_fst_star_prefix_recall() {
        let vocab = ["apple", "afoo", "foo", "xfoo"];
        let results = fst_search(&vocab, "*foo");
        assert!(results.contains(&"afoo".to_string()));
        assert!(results.contains(&"foo".to_string()));
        assert!(results.contains(&"xfoo".to_string()));
        assert!(!results.contains(&"apple".to_string()));
    }

    #[test]
    fn test_worst_case_no_match_star_pattern() {
        let input = "a".repeat(50);
        let result = wildcard_matches("*a*a*a*a*b", &input);
        assert!(!result);
    }

    #[test]
    fn test_dead_state_on_mismatch() {
        let a = WildcardAutomaton::new("abc");
        let mut state = a.start();
        state = a.accept(&state, b'x');
        assert!(!a.can_match(&state));
    }
}
