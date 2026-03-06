use common::counter::hardware_counter::HardwareCounterCell;
use common::types::PointOffsetType;

use super::{FuzzyIndex, FuzzyParsedQuery};
use crate::index::field_index::full_text_index::inverted_index::mutable_inverted_index::MutableInvertedIndex;

impl FuzzyIndex for MutableInvertedIndex {
    fn fuzzy_filter<'a>(
        &'a self,
        query: FuzzyParsedQuery,
        _hw_counter: &'a HardwareCounterCell,
    ) -> Box<dyn Iterator<Item = PointOffsetType> + 'a> {
        match query {
            FuzzyParsedQuery::AnyTokens(tokens) => Box::new(self.filter_has_any(tokens)),

            FuzzyParsedQuery::AllTokens(fuzzy_doc) => {
                let mut groups = fuzzy_doc.into_inner();
                if groups.is_empty() {
                    return Box::new(std::iter::empty());
                }
                groups.sort_by_key(|ts| ts.len());
                let smallest = groups.remove(0);
                let remaining = groups;
                let iter = self.filter_has_any(smallest).filter(move |&point_id| {
                    remaining.iter().all(|ts| {
                        self.get_tokens(point_id)
                            .map(|doc_tokens| doc_tokens.has_any(ts))
                            .unwrap_or(false)
                    })
                });
                Box::new(iter)
            }

            FuzzyParsedQuery::Phrase(fuzzy_doc) => {
                if fuzzy_doc.is_empty() {
                    return Box::new(std::iter::empty());
                }
                let fuzzy_doc_for_phrase = fuzzy_doc.clone();
                let mut groups = fuzzy_doc.into_inner();
                groups.sort_by_key(|ts| ts.len());
                let smallest = groups.remove(0);
                let remaining = groups;
                let iter = self
                    .filter_has_any(smallest)
                    .filter(move |&point_id| {
                        remaining.iter().all(|ts| {
                            self.get_tokens(point_id)
                                .map(|doc_tokens| doc_tokens.has_any(ts))
                                .unwrap_or(false)
                        })
                    })
                    .filter(move |&point_id| {
                        self.get_document(point_id)
                            .map(|doc| fuzzy_doc_for_phrase.matches_document(doc))
                            .unwrap_or(false)
                    });
                Box::new(iter)
            }
        }
    }

    fn fuzzy_check_match(&self, query: &FuzzyParsedQuery, point_id: PointOffsetType) -> bool {
        match query {
            FuzzyParsedQuery::AnyTokens(tokens) => self
                .get_tokens(point_id)
                .map(|doc| doc.has_any(tokens))
                .unwrap_or(false),
            FuzzyParsedQuery::AllTokens(fuzzy_doc) => self
                .get_tokens(point_id)
                .map(|doc| fuzzy_doc.iter().all(|ts| doc.has_any(ts)))
                .unwrap_or(false),
            FuzzyParsedQuery::Phrase(fuzzy_doc) => self
                .get_document(point_id)
                .map(|doc| fuzzy_doc.matches_document(doc))
                .unwrap_or(false),
        }
    }
}
