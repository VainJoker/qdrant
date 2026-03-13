//! BM25 fuzzy expansion helpers used by the collection-level resolve stage.
//!
//! This module provides:
//! - [`FuzzyExpansionParams`]: a small struct carrying pre-validated expansion
//!   parameters derived from [`FuzzyParams`].
//! - [`compute_fuzzy_boost`]: the boost formula
//!   `boost = 1.0 - edit_distance / max(|query|, |candidate|)` (character-level).
//! - [`Bm25FuzzyCandidate`]: a simplified candidate struct for the resolve stage.

use crate::types::FuzzyParams;

/// Compute the dimension ID for a BM25 token using murmur3 hash.
///
/// This is the same hash used by `Bm25::compute_token_id()`, exposed as a
/// standalone public function so that fuzzy expansion can convert expanded
/// dictionary terms into sparse-vector dimension IDs without depending on
/// the inference crate.
pub fn bm25_token_to_dim_id(token: &str) -> u32 {
    (murmur3::murmur3_32_of_slice(token.as_bytes(), 0) as i32).unsigned_abs()
}

/// Pre-validated fuzzy expansion parameters for segment-level calls.
#[derive(Debug, Clone)]
pub struct FuzzyExpansionParams {
    pub max_edit: u32,
    pub prefix_length: usize,
    pub max_expansions: usize,
}

impl From<&FuzzyParams> for FuzzyExpansionParams {
    fn from(params: &FuzzyParams) -> Self {
        Self {
            max_edit: params.max_edit.min(FuzzyParams::MAX_EDIT_DISTANCE),
            prefix_length: params.prefix_length as usize,
            max_expansions: params
                .max_expansions
                .clamp(1, FuzzyParams::MAX_EXPANSIONS_CAP) as usize,
        }
    }
}

/// A simplified candidate returned by segment-level fuzzy expansion.
#[derive(Debug, Clone)]
pub struct Bm25FuzzyCandidate {
    pub term: String,
    pub edit_distance: u32,
}

/// Compute the fuzzy boost weight for a candidate term.
///
/// Formula: `boost = 1.0 - edit_distance / max(|query_chars|, |candidate_chars|)`
///
/// Returns `None` if boost ≤ 0 (candidate too distant).
pub fn compute_fuzzy_boost(
    query_token: &str,
    candidate_term: &str,
    edit_distance: u32,
) -> Option<f32> {
    if edit_distance == 0 {
        return Some(1.0);
    }
    let q_len = query_token.chars().count().max(1);
    let t_len = candidate_term.chars().count().max(1);
    let max_len = q_len.max(t_len) as f32;
    let boost = 1.0 - edit_distance as f32 / max_len;
    if boost > 0.0 { Some(boost) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match_boost() {
        assert_eq!(compute_fuzzy_boost("hello", "hello", 0), Some(1.0));
    }

    #[test]
    fn test_edit_distance_1_boost() {
        let boost = compute_fuzzy_boost("hello", "hallo", 1).unwrap();
        // 1.0 - 1/5 = 0.8
        assert!((boost - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_edit_distance_2_boost() {
        let boost = compute_fuzzy_boost("hello", "hxllo", 2).unwrap();
        // 1.0 - 2/5 = 0.6
        assert!((boost - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_boost_zero_discarded() {
        // edit_distance == max_len → boost = 0.0 → None
        assert!(compute_fuzzy_boost("ab", "cd", 2).is_none());
    }

    #[test]
    fn test_boost_negative_discarded() {
        // edit_distance > max_len → negative → None
        assert!(compute_fuzzy_boost("a", "bcd", 4).is_none());
    }

    #[test]
    fn test_different_length_strings() {
        let boost = compute_fuzzy_boost("qdrant", "qdrantt", 1).unwrap();
        // 1.0 - 1/7 ≈ 0.857
        assert!((boost - (1.0 - 1.0 / 7.0)).abs() < 1e-6);
    }

    #[test]
    fn test_unicode_char_level() {
        // "café" (4 chars) vs "cafe" (4 chars), edit_distance = 1
        let boost = compute_fuzzy_boost("café", "cafe", 1).unwrap();
        assert!((boost - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_expansion_params_from_fuzzy_params() {
        let fp = FuzzyParams {
            max_edit: 5, // should be clamped to MAX_EDIT_DISTANCE (2)
            prefix_length: 2,
            max_expansions: 100, // should be clamped to MAX_EXPANSIONS_CAP (30)
        };
        let ep = FuzzyExpansionParams::from(&fp);
        assert_eq!(ep.max_edit, 2);
        assert_eq!(ep.prefix_length, 2);
        assert_eq!(ep.max_expansions, 30);
    }

    #[test]
    fn test_bm25_token_to_dim_id_deterministic() {
        // Same token always produces the same dim_id
        let id1 = bm25_token_to_dim_id("hello");
        let id2 = bm25_token_to_dim_id("hello");
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_bm25_token_to_dim_id_different_for_different_tokens() {
        let id_hello = bm25_token_to_dim_id("hello");
        let id_world = bm25_token_to_dim_id("world");
        assert_ne!(id_hello, id_world);
    }

    #[test]
    fn test_bm25_token_to_dim_id_matches_murmur3_formula() {
        // The function should produce (murmur3_32_of_slice(token.as_bytes(), 0) as i32).unsigned_abs()
        let token = "test_token";
        let expected = (murmur3::murmur3_32_of_slice(token.as_bytes(), 0) as i32).unsigned_abs();
        assert_eq!(bm25_token_to_dim_id(token), expected);
    }
}
