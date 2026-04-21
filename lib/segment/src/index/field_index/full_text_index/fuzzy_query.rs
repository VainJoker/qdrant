use ahash::AHashSet;
use common::counter::hardware_counter::HardwareCounterCell;

use super::fuzzy_index::FuzzyIndex;
use super::inverted_index::{Document, FuzzyDocument, ParsedQuery, TokenSet};
use super::text_index::FullTextIndex;
use crate::types::{Fuzzy, FuzzyParams};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FuzzyQueryKind {
    Text,
    TextAny,
    Phrase,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Fuzziness {
    Exact,
    Fuzzy,
}

impl FullTextIndex {
    pub fn parse_multi_fuzzy_query(
        &self,
        match_fuzzy: &Vec<Fuzzy>,
        hw_counter: &HardwareCounterCell,
    ) -> Option<ParsedQuery> {
        if match_fuzzy.is_empty() {
            return None;
        }

        let mut merged_groups = Vec::new();
        let mut query_kind = None;
        let mut fuzziness = Fuzziness::Exact;

        for fuzzy in match_fuzzy {
            let fuzzy_kind = Self::fuzzy_query_kind(fuzzy);
            // A merged fuzzy query must stay within one query family so it can map to a single ParsedQuery variant.
            if query_kind.is_some_and(|kind| kind != fuzzy_kind) {
                return None;
            }

            query_kind = Some(fuzzy_kind);
            if !Self::is_exact_fuzzy_query(fuzzy) {
                fuzziness = Fuzziness::Fuzzy;
            }
            merged_groups.extend(self.parse_single_fuzzy_query(fuzzy, hw_counter)?);
        }

        Self::build_fuzzy_query(query_kind?, fuzziness, merged_groups)
    }

    fn parse_single_fuzzy_query(
        &self,
        fuzzy: &Fuzzy,
        hw_counter: &HardwareCounterCell,
    ) -> Option<Vec<TokenSet>> {
        match fuzzy {
            Fuzzy::Text { text, params } => {
                if params.as_ref().map_or(0, |p| p.max_edits) == 0 {
                    return self.parse_exact_text_groups(text, hw_counter);
                }

                self.parse_grouped_fuzzy_query(
                    text,
                    params.as_ref().unwrap_or(&FuzzyParams::default()),
                    hw_counter,
                )
            }
            Fuzzy::TextAny { text_any, params } => {
                if params.as_ref().map_or(0, |p| p.max_edits) == 0 {
                    return Some(vec![self.parse_tokenset(text_any, hw_counter)]);
                }

                self.parse_text_any_fuzzy_query(
                    text_any,
                    params.as_ref().unwrap_or(&FuzzyParams::default()),
                    hw_counter,
                )
            }
            Fuzzy::Phrase { phrase, params } => {
                if params.as_ref().map_or(0, |p| p.max_edits) == 0 {
                    return self.parse_exact_phrase_groups(phrase, hw_counter);
                }

                self.parse_grouped_fuzzy_query(
                    phrase,
                    params.as_ref().unwrap_or(&FuzzyParams::default()),
                    hw_counter,
                )
            }
        }
    }

    fn parse_grouped_fuzzy_query(
        &self,
        text: &str,
        params: &FuzzyParams,
        hw_counter: &HardwareCounterCell,
    ) -> Option<Vec<TokenSet>> {
        let mut token_sets = Vec::new();
        let mut has_query_tokens = false;
        let mut has_token_without_candidates = false;
        let mut missing_fuzzy_index = false;

        self.get_tokenizer().tokenize_query(text, |token| {
            has_query_tokens = true;
            match self.expand_fuzzy_token(token.as_ref(), params, hw_counter) {
                Some(token_set) if token_set.is_empty() => {
                    has_token_without_candidates = true;
                }
                Some(token_set) => {
                    token_sets.push(token_set);
                }
                None => {
                    missing_fuzzy_index = true;
                }
            }
        });

        if missing_fuzzy_index || !has_query_tokens || has_token_without_candidates {
            return None;
        }

        Some(token_sets)
    }

    fn fuzzy_query_kind(fuzzy: &Fuzzy) -> FuzzyQueryKind {
        match fuzzy {
            Fuzzy::Text { .. } => FuzzyQueryKind::Text,
            Fuzzy::TextAny { .. } => FuzzyQueryKind::TextAny,
            Fuzzy::Phrase { .. } => FuzzyQueryKind::Phrase,
        }
    }

    fn is_exact_fuzzy_query(fuzzy: &Fuzzy) -> bool {
        match fuzzy {
            Fuzzy::Text { params, .. }
            | Fuzzy::TextAny { params, .. }
            | Fuzzy::Phrase { params, .. } => params.as_ref().map_or(0, |p| p.max_edits) == 0,
        }
    }

    fn expand_fuzzy_token(
        &self,
        token: &str,
        params: &FuzzyParams,
        hw_counter: &HardwareCounterCell,
    ) -> Option<TokenSet> {
        let fuzzy_index: &dyn FuzzyIndex = match self {
            Self::Mutable(index) => index.get_fuzzy_index()?,
            Self::Immutable(index) => index.get_fuzzy_index()?,
            Self::Mmap(index) => index.get_fuzzy_index()?,
        };

        let min_len = self
            .get_tokenizer()
            .tokens_processor()
            .min_token_len
            .unwrap_or(3);

        // Very short tokens are too noisy for Levenshtein expansion, so keep exact lookup semantics.
        if token.chars().count() <= min_len {
            return Some(self.get_token(token, hw_counter).into_iter().collect());
        }

        Some(
            fuzzy_index
                .search_levenshtein(token, params)
                .into_iter()
                .filter_map(|candidate| self.get_token(&candidate.term, hw_counter))
                .collect(),
        )
    }

    fn parse_text_any_fuzzy_query(
        &self,
        text_any: &str,
        params: &FuzzyParams,
        hw_counter: &HardwareCounterCell,
    ) -> Option<Vec<TokenSet>> {
        let mut all_token_ids = AHashSet::new();
        let mut missing_fuzzy_index = false;

        self.get_tokenizer().tokenize_query(text_any, |token| {
            match self.expand_fuzzy_token(token.as_ref(), params, hw_counter) {
                Some(token_set) => all_token_ids.extend(token_set.tokens().iter().copied()),
                None => missing_fuzzy_index = true,
            }
        });

        if missing_fuzzy_index || all_token_ids.is_empty() {
            return None;
        }

        Some(vec![TokenSet::from(all_token_ids)])
    }

    fn parse_exact_text_groups(
        &self,
        text: &str,
        hw_counter: &HardwareCounterCell,
    ) -> Option<Vec<TokenSet>> {
        let mut tokens = AHashSet::new();
        self.get_tokenizer().tokenize_query(text, |token| {
            tokens.insert(self.get_token(token.as_ref(), hw_counter));
        });

        let tokens = tokens.into_iter().collect::<Option<TokenSet>>()?;
        Some(
            tokens
                .tokens()
                .iter()
                .copied()
                .map(|token_id| std::iter::once(token_id).collect())
                .collect(),
        )
    }

    fn parse_exact_phrase_groups(
        &self,
        phrase: &str,
        hw_counter: &HardwareCounterCell,
    ) -> Option<Vec<TokenSet>> {
        let document = self.parse_document(phrase, hw_counter)?;
        Some(
            document
                .tokens()
                .iter()
                .copied()
                .map(|token_id| std::iter::once(token_id).collect())
                .collect(),
        )
    }

    fn build_fuzzy_query(
        query_kind: FuzzyQueryKind,
        fuzziness: Fuzziness,
        merged_groups: Vec<TokenSet>,
    ) -> Option<ParsedQuery> {
        match query_kind {
            FuzzyQueryKind::Text => {
                if fuzziness == Fuzziness::Exact {
                    let tokens = merged_groups
                        .into_iter()
                        .flat_map(|group| group.inner())
                        .collect::<TokenSet>();
                    Some(ParsedQuery::AllTokens(tokens))
                } else {
                    Some(ParsedQuery::FuzzyAllTokens(FuzzyDocument::new(
                        merged_groups,
                    )))
                }
            }
            FuzzyQueryKind::TextAny => {
                let tokens = merged_groups
                    .into_iter()
                    .flat_map(|group| group.inner())
                    .collect::<TokenSet>();

                if tokens.is_empty() {
                    Some(ParsedQuery::AnyTokens(TokenSet::default()))
                // `TextAny` only needs fuzzy iterator semantics when there is exactly one expanded group.
                } else if fuzziness == Fuzziness::Fuzzy && tokens.len() == 1 {
                    Some(ParsedQuery::FuzzyAnyTokens(tokens))
                } else {
                    Some(ParsedQuery::AnyTokens(tokens))
                }
            }
            FuzzyQueryKind::Phrase => {
                if fuzziness == Fuzziness::Exact {
                    // Exact phrases must resolve each position to one token to preserve ordering semantics.
                    let document = merged_groups
                        .into_iter()
                        .map(|group| {
                            let tokens = group.inner();
                            let [token_id] = tokens.as_slice() else {
                                return None;
                            };
                            Some(*token_id)
                        })
                        .collect::<Option<Vec<_>>>()?;

                    Some(ParsedQuery::Phrase(Document::new(document)))
                } else {
                    Some(ParsedQuery::FuzzyPhrase(FuzzyDocument::new(merged_groups)))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use tempfile::Builder;

    use super::super::inverted_index::TokenId;
    use super::*;

    fn ts(ids: &[TokenId]) -> TokenSet {
        ids.iter().copied().collect()
    }

    fn fp(max_edits: u8) -> Option<FuzzyParams> {
        Some(FuzzyParams {
            max_edits,
            prefix_length: 0,
            max_expansions: 30,
        })
    }

    fn new_test_index(phrase_matching: bool) -> FullTextIndex {
        use crate::data_types::index::{TextIndexParams, TextIndexType, TokenizerType};

        let dir = Builder::new().prefix("fq_test").tempdir().unwrap();
        let config = TextIndexParams {
            r#type: TextIndexType::Text,
            tokenizer: TokenizerType::Word,
            phrase_matching: Some(phrase_matching),
            fuzzy_matching: Some(true),
            lowercase: Some(true),
            ..Default::default()
        };

        FullTextIndex::new_gridstore(dir.keep(), config, true)
            .unwrap()
            .unwrap()
    }

    #[test]
    fn test_build_fuzzy_query_variants() {
        let text_exact = FullTextIndex::build_fuzzy_query(
            FuzzyQueryKind::Text,
            Fuzziness::Exact,
            vec![ts(&[1]), ts(&[2])],
        );
        match text_exact {
            Some(ParsedQuery::AllTokens(tokens)) => {
                assert_eq!(tokens.len(), 2);
                assert!(tokens.contains(&1));
                assert!(tokens.contains(&2));
            }
            other => panic!("expected AllTokens, got {other:?}"),
        }

        let cases = [
            (
                FuzzyQueryKind::Text,
                Fuzziness::Fuzzy,
                vec![ts(&[1, 10]), ts(&[2, 20])],
                Some("fuzzy_all_tokens"),
            ),
            (
                FuzzyQueryKind::TextAny,
                Fuzziness::Fuzzy,
                vec![ts(&[42])],
                Some("fuzzy_any_tokens"),
            ),
            (
                FuzzyQueryKind::Phrase,
                Fuzziness::Exact,
                vec![ts(&[5, 6]), ts(&[10])],
                None,
            ),
            (
                FuzzyQueryKind::Phrase,
                Fuzziness::Fuzzy,
                vec![ts(&[5, 6]), ts(&[10, 11])],
                Some("fuzzy_phrase"),
            ),
        ];

        for (query_kind, fuzziness, groups, expected) in cases {
            let result = FullTextIndex::build_fuzzy_query(query_kind, fuzziness, groups);
            match expected {
                Some("fuzzy_all_tokens") => {
                    assert!(matches!(result, Some(ParsedQuery::FuzzyAllTokens(_))));
                }
                Some("fuzzy_any_tokens") => {
                    assert!(matches!(result, Some(ParsedQuery::FuzzyAnyTokens(_))));
                }
                Some("fuzzy_phrase") => {
                    assert!(matches!(result, Some(ParsedQuery::FuzzyPhrase(_))));
                }
                None => assert!(result.is_none()),
                Some(other) => panic!("unexpected expected marker: {other}"),
            }
        }
    }

    #[test]
    fn test_parse_fuzzy_query_empty_returns_none() {
        use common::counter::hardware_counter::HardwareCounterCell;

        let index = new_test_index(false);
        let hw = HardwareCounterCell::new();

        let result = index.parse_fuzzy_query(&vec![], &hw);
        assert!(result.is_none());
    }

    #[test]
    fn test_mixed_kinds_returns_none() {
        use common::counter::hardware_counter::HardwareCounterCell;

        use crate::index::field_index::ValueIndexer;

        let mut index = new_test_index(false);
        let hw = HardwareCounterCell::new();
        index.add_many(0, vec!["hello world".into()], &hw).unwrap();

        let mixed = vec![
            Fuzzy::Text {
                text: "hello".into(),
                params: fp(0),
            },
            Fuzzy::Phrase {
                phrase: "world".into(),
                params: fp(0),
            },
        ];
        assert!(index.parse_fuzzy_query(&mixed, &hw).is_none());
    }

    #[test]
    fn test_exact_text_query_produces_all_tokens() {
        use common::counter::hardware_counter::HardwareCounterCell;

        use crate::index::field_index::ValueIndexer;

        let mut index = new_test_index(false);
        let hw = HardwareCounterCell::new();
        index.add_many(0, vec!["hello world".into()], &hw).unwrap();

        let parsed = index.parse_fuzzy_query(
            &vec![Fuzzy::Text {
                text: "hello world".into(),
                params: fp(0),
            }],
            &hw,
        );
        assert!(matches!(parsed, Some(ParsedQuery::AllTokens(_))));
    }

    #[test]
    fn test_exact_phrase_query_produces_phrase() {
        use common::counter::hardware_counter::HardwareCounterCell;

        use crate::index::field_index::ValueIndexer;

        let mut index = new_test_index(true);
        let hw = HardwareCounterCell::new();
        index
            .add_many(0, vec!["the quick brown fox".into()], &hw)
            .unwrap();

        let parsed = index.parse_fuzzy_query(
            &vec![Fuzzy::Phrase {
                phrase: "quick brown".into(),
                params: fp(0),
            }],
            &hw,
        );
        match parsed {
            Some(ParsedQuery::Phrase(doc)) => {
                assert_eq!(doc.len(), 2, "phrase should have 2 tokens");
            }
            other => panic!("expected Phrase, got {other:?}"),
        }
    }

    #[test]
    fn test_exact_text_any_query_produces_any_tokens() {
        use common::counter::hardware_counter::HardwareCounterCell;

        use crate::index::field_index::ValueIndexer;

        let mut index = new_test_index(false);
        let hw = HardwareCounterCell::new();
        index
            .add_many(0, vec!["alpha beta gamma".into()], &hw)
            .unwrap();

        let parsed = index.parse_fuzzy_query(
            &vec![Fuzzy::TextAny {
                text_any: "alpha delta".into(),
                params: fp(0),
            }],
            &hw,
        );
        match parsed {
            Some(ParsedQuery::AnyTokens(t)) => {
                assert!(t.contains(&index.get_token("alpha", &hw).unwrap()));
            }
            other => panic!("expected AnyTokens, got {other:?}"),
        }
    }
}
