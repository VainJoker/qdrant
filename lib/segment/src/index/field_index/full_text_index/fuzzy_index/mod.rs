pub mod fuzzy_expander;
mod immutable_fuzzy_index;
mod mmap_fuzzy_index;
mod mutable_fuzzy_index;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::PointOffsetType;
pub use fuzzy_expander::{FuzzyCandidate, FuzzyExpander, MutableFuzzyExpander};

use super::inverted_index::{Document, InvertedIndex, TokenId, TokenSet};
use crate::index::field_index::{CardinalityEstimation, PrimaryCondition};
use crate::types::{FieldCondition, FuzzyParams};

/// Position-ordered groups of fuzzy-expanded token IDs.
///
/// The fuzzy counterpart of [`Document`]: whereas `Document` has exactly
/// one `TokenId` per position, `FuzzyDocument` has a [`TokenSet`] per position
/// (the set of all fuzzy-expanded candidates for that position).
#[derive(Debug, Clone)]
pub struct FuzzyDocument(Vec<TokenSet>);

impl FuzzyDocument {
    pub fn new(groups: Vec<TokenSet>) -> Self {
        Self(groups)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn groups(&self) -> &[TokenSet] {
        &self.0
    }

    pub fn into_inner(self) -> Vec<TokenSet> {
        self.0
    }

    pub fn iter(&self) -> impl Iterator<Item = &TokenSet> {
        self.0.iter()
    }

    /// Checks if a contiguous window of this document's groups matches the exact [`Document`].
    pub fn matches_document(&self, doc: &Document) -> bool {
        let tokens = doc.tokens();
        let groups = self.groups();

        if tokens.len() < groups.len() || tokens.is_empty() || groups.is_empty() {
            return false;
        }

        tokens.windows(groups.len()).any(|window| {
            window
                .iter()
                .zip(groups.iter())
                .all(|(token_id, group)| group.contains(token_id))
        })
    }
}

/// Parsed query for fuzzy matching, parallel to [`super::inverted_index::ParsedQuery`].
#[derive(Debug, Clone)]
pub enum FuzzyParsedQuery {
    /// AND semantics: every group must have at least one match in the document.
    AllTokens(FuzzyDocument),

    /// OR semantics: any fuzzy-expanded token matching the document is sufficient.
    AnyTokens(TokenSet),

    /// Position-ordered phrase match: the document must contain a contiguous window
    /// where position *i* matches at least one token from `groups[i]`.
    Phrase(FuzzyDocument),
}

/// Common interface for fuzzy term expansion.
///
/// [`FuzzyExpander`] (FST-based) is used for immutable/mmap indices;
/// [`MutableFuzzyExpander`] (BTreeSet-based) is used for mutable indices
/// to avoid rebuilding an FST on every vocabulary mutation.
pub trait TermExpander {
    fn expand_term(
        &self,
        term: &str,
        params: &FuzzyParams,
        vocab_lookup: &dyn Fn(&str) -> Option<TokenId>,
    ) -> Vec<FuzzyCandidate>;
}

/// Trait for fuzzy text matching, parallel to [`InvertedIndex`].
///
/// Each inverted-index implementation provides a `FuzzyIndex` impl that handles
/// fuzzy query filtering and matching using the same internal postings data.
pub trait FuzzyIndex: InvertedIndex {
    fn fuzzy_filter<'a>(
        &'a self,
        query: FuzzyParsedQuery,
        hw_counter: &'a HardwareCounterCell,
    ) -> Box<dyn Iterator<Item = PointOffsetType> + 'a>;

    fn fuzzy_check_match(&self, query: &FuzzyParsedQuery, point_id: PointOffsetType) -> bool;

    fn fuzzy_estimate_cardinality(
        &self,
        query: &FuzzyParsedQuery,
        condition: &FieldCondition,
        hw_counter: &HardwareCounterCell,
    ) -> CardinalityEstimation {
        match query {
            FuzzyParsedQuery::AnyTokens(tokens) => {
                self.estimate_has_any_cardinality(tokens, condition, hw_counter)
            }
            FuzzyParsedQuery::AllTokens(fuzzy_doc) | FuzzyParsedQuery::Phrase(fuzzy_doc) => {
                if fuzzy_doc.is_empty() {
                    return CardinalityEstimation::exact(0).with_primary_clause(
                        PrimaryCondition::Condition(Box::new(condition.clone())),
                    );
                }

                let group_estimates: Vec<CardinalityEstimation> = fuzzy_doc
                    .iter()
                    .map(|ts| self.estimate_has_any_cardinality(ts, condition, hw_counter))
                    .collect();

                let min_exp = group_estimates.iter().map(|e| e.exp).min().unwrap_or(0);
                let min_min = group_estimates.iter().map(|e| e.min).min().unwrap_or(0);
                let min_max = group_estimates.iter().map(|e| e.max).min().unwrap_or(0);

                CardinalityEstimation {
                    primary_clauses: vec![PrimaryCondition::Condition(Box::new(condition.clone()))],
                    min: 0,
                    exp: min_exp,
                    max: min_max.min(min_min),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token_set(ids: &[u32]) -> TokenSet {
        ids.iter().copied().collect()
    }

    fn document(ids: &[u32]) -> Document {
        Document::new(ids.to_vec())
    }

    #[test]
    fn test_fuzzy_document_empty() {
        let doc = FuzzyDocument::new(vec![]);
        assert!(doc.is_empty());
        assert_eq!(doc.len(), 0);
        assert!(doc.groups().is_empty());
    }

    #[test]
    fn test_fuzzy_document_matches_document() {
        let fuzzy = FuzzyDocument::new(vec![token_set(&[1, 2]), token_set(&[3, 4])]);

        assert!(fuzzy.matches_document(&document(&[1, 3])));
        assert!(fuzzy.matches_document(&document(&[2, 4])));
        assert!(!fuzzy.matches_document(&document(&[1, 5])));
        assert!(!fuzzy.matches_document(&document(&[1])));
        assert!(!fuzzy.matches_document(&document(&[])));

        let empty_fuzzy = FuzzyDocument::new(vec![]);
        assert!(!empty_fuzzy.matches_document(&document(&[1, 2])));
    }

    #[test]
    fn test_fuzzy_document_sliding_window_match() {
        let fuzzy = FuzzyDocument::new(vec![token_set(&[10]), token_set(&[20])]);

        assert!(fuzzy.matches_document(&document(&[5, 10, 20, 30])));
        assert!(!fuzzy.matches_document(&document(&[5, 10, 15, 20])));
    }

    #[test]
    fn test_fuzzy_document_into_inner() {
        let doc = FuzzyDocument::new(vec![token_set(&[1, 2]), token_set(&[3])]);
        let inner = doc.into_inner();
        assert_eq!(inner.len(), 2);
    }
}
