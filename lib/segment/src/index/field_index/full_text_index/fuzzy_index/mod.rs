pub mod fuzzy_expander;
mod immutable_fuzzy_index;
mod mmap_fuzzy_index;
mod mutable_fuzzy_index;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::PointOffsetType;
pub use fuzzy_expander::{FuzzyCandidate, FuzzyExpander, PrefixFuzzy};

use super::inverted_index::{Document, InvertedIndex, TokenSet};
use crate::index::field_index::{CardinalityEstimation, PrimaryCondition};
use crate::types::FieldCondition;

// ────────────────────────────────────────────────────────────────────────────
// FuzzyDocument
// ────────────────────────────────────────────────────────────────────────────

/// Contains position-ordered groups of fuzzy-expanded token IDs.
///
/// This is the fuzzy counterpart of [`Document`]: whereas `Document` has exactly
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

    pub fn inner(self) -> Vec<TokenSet> {
        self.0
    }

    pub fn iter(&self) -> impl Iterator<Item = &TokenSet> {
        self.0.iter()
    }

    /// Checks if this fuzzy document matches against an exact [`Document`].
    ///
    /// The exact document must contain a contiguous window of `len(groups)` where
    /// `window[i]` matches at least one token in `groups[i]`.
    pub fn matches_document(&self, doc: &Document) -> bool {
        let tokens = doc.tokens();
        let groups = self.groups();

        if tokens.is_empty() || groups.is_empty() {
            return false;
        }

        if tokens.len() < groups.len() {
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

// ────────────────────────────────────────────────────────────────────────────
// FuzzyParsedQuery
// ────────────────────────────────────────────────────────────────────────────

/// Parsed query for fuzzy matching, parallel to [`super::inverted_index::ParsedQuery`].
#[derive(Debug, Clone)]
pub enum FuzzyParsedQuery {
    /// Fuzzy "text" match (AND semantics): for each query token, a set of fuzzy-expanded
    /// token IDs is provided. Every group must have at least one match in the document.
    AllTokens(FuzzyDocument),

    /// Fuzzy "text_any" match (OR semantics): any of these fuzzy-expanded token IDs
    /// matching the document is sufficient.
    AnyTokens(TokenSet),

    /// Fuzzy phrase match: position-ordered groups of fuzzy-expanded token IDs.
    /// The document must contain a contiguous window where position i matches
    /// at least one token from groups[i].
    Phrase(FuzzyDocument),
}

// ────────────────────────────────────────────────────────────────────────────
// FuzzyIndex trait
// ────────────────────────────────────────────────────────────────────────────

/// Trait for fuzzy text matching, parallel to [`InvertedIndex`].
///
/// Each inverted index implementation provides a `FuzzyIndex` impl that handles
/// fuzzy query filtering and matching using the same internal postings data.
pub trait FuzzyIndex: InvertedIndex {
    fn fuzzy_filter<'a>(
        &'a self,
        query: FuzzyParsedQuery,
        hw_counter: &'a HardwareCounterCell,
    ) -> Box<dyn Iterator<Item = PointOffsetType> + 'a>;

    fn fuzzy_check_match(&self, parsed_query: &FuzzyParsedQuery, point_id: PointOffsetType)
    -> bool;

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
                // Estimate as intersection of OR groups: each group is an OR,
                // then we intersect them. Use the smallest group as baseline.
                if fuzzy_doc.is_empty() {
                    return CardinalityEstimation::exact(0).with_primary_clause(
                        PrimaryCondition::Condition(Box::new(condition.clone())),
                    );
                }
                // Estimate each group's cardinality (as OR/any), then intersect
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
