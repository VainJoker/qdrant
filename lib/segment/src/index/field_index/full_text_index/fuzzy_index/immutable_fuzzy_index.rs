use common::counter::hardware_counter::HardwareCounterCell;
use common::types::PointOffsetType;

use super::{FuzzyIndex, FuzzyParsedQuery};
use crate::index::field_index::full_text_index::inverted_index::immutable_inverted_index::ImmutableInvertedIndex;

impl FuzzyIndex for ImmutableInvertedIndex {
    fn fuzzy_filter<'a>(
        &'a self,
        query: FuzzyParsedQuery,
        _hw_counter: &'a HardwareCounterCell,
    ) -> Box<dyn Iterator<Item = PointOffsetType> + 'a> {
        match query {
            FuzzyParsedQuery::AnyTokens(tokens) => Box::new(self.filter_has_any(tokens)),
            FuzzyParsedQuery::AllTokens(fuzzy_doc) => {
                if fuzzy_doc.is_empty() {
                    return Box::new(std::iter::empty());
                }
                let mut sorted = fuzzy_doc.inner();
                sorted.sort_by_key(|ts| ts.len());
                let first = sorted.remove(0);
                let iter = self.filter_has_any(first).filter(move |&point_id| {
                    sorted.iter().all(|ts| self.check_has_any(ts, point_id))
                });
                Box::new(iter)
            }
            FuzzyParsedQuery::Phrase(fuzzy_doc) => {
                if fuzzy_doc.is_empty() {
                    return Box::new(std::iter::empty());
                }
                let fuzzy_doc_clone = fuzzy_doc.clone();
                let mut sorted = fuzzy_doc.inner();
                sorted.sort_by_key(|ts| ts.len());
                let first = sorted.remove(0);
                let remaining = sorted;
                let iter = self
                    .filter_has_any(first)
                    .filter(move |&point_id| {
                        remaining.iter().all(|ts| self.check_has_any(ts, point_id))
                    })
                    .filter(move |&point_id| {
                        self.check_has_fuzzy_phrase(&fuzzy_doc_clone, point_id)
                    });
                Box::new(iter)
            }
        }
    }

    fn fuzzy_check_match(
        &self,
        parsed_query: &FuzzyParsedQuery,
        point_id: PointOffsetType,
    ) -> bool {
        match parsed_query {
            FuzzyParsedQuery::AnyTokens(tokens) => self.check_has_any(tokens, point_id),
            FuzzyParsedQuery::AllTokens(fuzzy_doc) => {
                fuzzy_doc.iter().all(|ts| self.check_has_any(ts, point_id))
            }
            FuzzyParsedQuery::Phrase(fuzzy_doc) => self.check_has_fuzzy_phrase(fuzzy_doc, point_id),
        }
    }
}
