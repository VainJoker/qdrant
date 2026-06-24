use std::borrow::Cow;

use ahash::AHashMap;
use common::counter::hardware_counter::HardwareCounterCell;
use common::iterator_ext::IteratorExt;
use common::types::PointOffsetType;
use common::universal_io::UserData;

use super::fuzzy_index::FuzzyIndex;
use super::inverted_index::{Document, FuzzyDocument, ParsedQuery, TokenId, TokenSet};
use super::tokenizers::{Tokenizer, TokenizerTextKind};
use crate::common::operation_error::OperationResult;
use crate::index::field_index::{CardinalityEstimation, PayloadBlockCondition, ValueIndexer};
use crate::index::payload_config::StorageType;
use crate::telemetry::PayloadIndexTelemetry;
use crate::types::{
    FieldCondition, Fuzzy, FuzzyParams, Match, PayloadKeyType, Wildcard, WildcardParams,
};

/// Shared read surface for the writable [`FullTextIndex`] enum and the
/// read-only `ReadOnlyFullTextIndex<S>` skeleton. Lets the
/// [`PayloadFieldIndexRead`][crate::index::field_index::PayloadFieldIndexRead]
/// bodies live in [`read_ops`][super::read_ops] as free functions instead of
/// being duplicated.
///
/// Object safety is **not** required — `for_each_token_id` is generic over
/// `U: UserData` and `f: impl FnMut(..)`, so callers parameterize with
/// `T: FullTextIndexRead` rather than `&dyn FullTextIndexRead`.
///
/// [`FullTextIndex`]: super::FullTextIndex
pub trait FullTextIndexRead {
    fn tokenizer(&self) -> &Tokenizer;
    fn telemetry_index_type(&self) -> &'static str;

    /// Telemetry shared between [`FullTextIndex`] and `ReadOnlyFullTextIndex<S>`.
    /// Full-text indexes track a single per-point count, so `points_values_count`
    /// and `points_count` are both reported as [`Self::points_count`].
    ///
    /// [`FullTextIndex`]: super::FullTextIndex
    fn get_telemetry_data(&self) -> PayloadIndexTelemetry {
        PayloadIndexTelemetry {
            field_name: None,
            index_type: self.telemetry_index_type(),
            points_values_count: self.points_count(),
            points_count: self.points_count(),
            histogram_bucket_size: None,
        }
    }

    fn points_count(&self) -> usize;
    fn values_count(&self, point_id: PointOffsetType) -> usize;
    fn values_is_empty(&self, point_id: PointOffsetType) -> bool;

    fn for_each_token_id<'a, U: UserData>(
        &self,
        iter: impl Iterator<Item = (U, &'a str)>,
        hw_counter: &HardwareCounterCell,
        f: impl FnMut(U, Option<TokenId>),
    ) -> OperationResult<()>;

    fn filter_query<'a>(
        &'a self,
        query: ParsedQuery,
        hw_counter: &'a HardwareCounterCell,
    ) -> OperationResult<Box<dyn Iterator<Item = PointOffsetType> + 'a>>;

    fn estimate_query_cardinality(
        &self,
        query: &ParsedQuery,
        condition: &FieldCondition,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<CardinalityEstimation>;

    fn check_match(&self, query: &ParsedQuery, point_id: PointOffsetType) -> OperationResult<bool>;

    fn fuzzy_index(&self) -> Option<&dyn FuzzyIndex>;

    /// Walk the inverted-index vocab and emit one [`PayloadBlockCondition`] per
    /// token with at least `threshold` postings. Used to seed payload-block
    /// scans for full-text indexes.
    fn for_each_payload_block_inner(
        &self,
        threshold: usize,
        key: PayloadKeyType,
        f: &mut dyn FnMut(PayloadBlockCondition) -> OperationResult<()>,
    ) -> OperationResult<()>;

    fn get_storage_type(&self) -> StorageType;

    fn ram_usage_bytes(&self) -> usize;

    fn is_on_disk(&self) -> bool;

    /// Parse as [`TokenizerTextKind::Document`] and return [`ParsedQuery::Phrase`].
    /// Returns [`None`] if there are any unseen tokens.
    fn parse_phrase_query(
        &self,
        phrase: &str,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<Option<ParsedQuery>> {
        let document = self.parse_document(phrase, hw_counter)?;
        Ok(document.map(ParsedQuery::Phrase))
    }

    /// Parse as [`TokenizerTextKind::Query`] and return [`ParsedQuery::AllTokens`].
    /// Returns [`None`] if there are any unseen tokens.
    fn parse_text_query(
        &self,
        text: &str,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<Option<ParsedQuery>> {
        let tokenset: Option<TokenSet> = self
            .resolve_tokens(TokenizerTextKind::Query, text, hw_counter)?
            .into_values()
            .collect::<Option<TokenSet>>();
        Ok(tokenset.map(ParsedQuery::AllTokens))
    }

    /// Parse as [`TokenizerTextKind::Query`] and return [`ParsedQuery::AnyTokens`].
    /// Unseen tokens are ignored. Never returns [`None`].
    fn parse_text_any_query(
        &self,
        text: &str,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<Option<ParsedQuery>> {
        let tokenset = self.parse_tokenset(TokenizerTextKind::Query, text, hw_counter)?;
        Ok(Some(ParsedQuery::AnyTokens(tokenset)))
    }

    /// Expand a fuzzy full-text query into index token-id groups.
    ///
    /// Each token produced from `text` becomes one [`TokenSet`] containing all
    /// indexed terms that can satisfy that query token. Later query execution
    /// treats these groups according to the caller's semantics:
    /// - `require_all = true`: every query token must resolve to a non-empty
    ///   group; used by `Fuzzy::Text` and `Fuzzy::Phrase`.
    /// - `require_all = false`: unresolved query tokens are ignored as long as
    ///   at least one group resolves; used by `Fuzzy::TextAny`.
    ///
    /// Very short tokens are resolved exactly instead of through the fuzzy FST.
    /// This keeps approximate matching from becoming too broad/noisy for short
    /// terms, where one edit can change most of the token.
    ///
    /// Returns [`None`] if token-id resolution fails, if `require_all` is set
    /// and any query token has no match, or if no token groups remain.
    fn fuzzy_token_sets(
        &self,
        fuzzy_index: &dyn FuzzyIndex,
        hw_counter: &HardwareCounterCell,
        kind: TokenizerTextKind,
        text: &str,
        params: &FuzzyParams,
        require_all: bool,
    ) -> Option<Vec<TokenSet>> {
        let min_len = self
            .tokenizer()
            .tokens_processor()
            .min_token_len
            .unwrap_or(3);
        let mut result: Result<Vec<TokenSet>, ()> = Ok(Vec::new());
        self.tokenizer().tokenize(kind, text, |token| {
            let Ok(sets) = result.as_mut() else {
                return;
            };
            let mut token_ids = Vec::new();
            let ok = if token.chars().count() <= min_len {
                self.for_each_token_id(
                    std::iter::once(((), token.as_ref())),
                    hw_counter,
                    |(), id| {
                        if let Some(id) = id {
                            token_ids.push(id);
                        }
                    },
                )
            } else {
                let candidates = fuzzy_index.search_levenshtein(token.as_ref(), params);
                self.for_each_token_id(
                    candidates.iter().map(|c| ((), c.term.as_str())),
                    hw_counter,
                    |(), id| {
                        if let Some(id) = id {
                            token_ids.push(id);
                        }
                    },
                )
            };
            match ok {
                Err(_) => result = Err(()),
                Ok(_) => {
                    let ts: TokenSet = token_ids.into_iter().collect();
                    if !ts.is_empty() {
                        sets.push(ts);
                    } else if require_all {
                        result = Err(());
                    }
                }
            }
        });
        result.ok().filter(|sets| !sets.is_empty())
    }

    /// Convert a user-facing fuzzy match clause into the internal parsed query.
    ///
    /// Fuzzy parsing is available only when the fuzzy index exists. For
    /// `max_edits = 0`, this intentionally reuses the exact full-text parsers
    /// while still requiring the fuzzy index to be configured for `Match::Fuzzy`.
    fn parse_fuzzy_query(
        &self,
        fuzzy: &Fuzzy,
        hw_counter: &HardwareCounterCell,
    ) -> Option<ParsedQuery> {
        let fuzzy_index = self.fuzzy_index()?;
        let default_params = FuzzyParams::default();

        match fuzzy {
            Fuzzy::Text { text, params } => {
                let params = params.as_ref().unwrap_or(&default_params);
                if params.max_edits == 0 {
                    return self.parse_text_query(text, hw_counter).ok().flatten();
                }
                let mut groups = self.fuzzy_token_sets(
                    fuzzy_index,
                    hw_counter,
                    TokenizerTextKind::Query,
                    text,
                    params,
                    true,
                )?;
                groups.sort_unstable_by_key(TokenSet::len);
                Some(ParsedQuery::FuzzyAllTokens(FuzzyDocument::new(groups)))
            }
            Fuzzy::TextAny { text_any, params } => {
                let params = params.as_ref().unwrap_or(&default_params);
                if params.max_edits == 0 {
                    return self
                        .parse_text_any_query(text_any, hw_counter)
                        .ok()
                        .flatten();
                }
                let tokens = self
                    .fuzzy_token_sets(
                        fuzzy_index,
                        hw_counter,
                        TokenizerTextKind::Query,
                        text_any,
                        params,
                        false,
                    )?
                    .into_iter()
                    .flat_map(TokenSet::inner)
                    .collect();
                Some(ParsedQuery::FuzzyAnyTokens(tokens))
            }
            Fuzzy::Phrase { phrase, params } => {
                let params = params.as_ref().unwrap_or(&default_params);
                if params.max_edits == 0 {
                    return self.parse_phrase_query(phrase, hw_counter).ok().flatten();
                }
                let groups = self.fuzzy_token_sets(
                    fuzzy_index,
                    hw_counter,
                    TokenizerTextKind::Document,
                    phrase,
                    params,
                    true,
                )?;
                Some(ParsedQuery::FuzzyPhrase(FuzzyDocument::new(groups)))
            }
        }
    }

    fn parse_wildcard_query(
        &self,
        wildcard: &Wildcard,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<Option<ParsedQuery>> {
        let Some(fuzzy_index) = self.fuzzy_index() else {
            return Ok(None);
        };

        let pattern = wildcard.pattern();
        if !WildcardParams::validate_pattern(pattern) {
            return Ok(None);
        }

        let tokenizer = self.tokenizer().tokens_processor();
        let pattern = match tokenizer.lowercase {
            true => Cow::Owned(pattern.to_lowercase()),
            false => Cow::Borrowed(pattern),
        };
        let params = wildcard.params().validate();
        let matched_terms = fuzzy_index.search_wildcard(pattern.as_ref(), &params);
        if matched_terms.is_empty() {
            return Ok(None);
        }

        let mut token_ids = Vec::with_capacity(matched_terms.len());
        self.for_each_token_id(
            matched_terms.iter().map(|term| ((), term.as_str())),
            hw_counter,
            |(), id| {
                if let Some(id) = id {
                    token_ids.push(id);
                }
            },
        )?;

        let tokenset: TokenSet = token_ids.into_iter().collect();
        if tokenset.is_empty() {
            return Ok(None);
        }

        Ok(Some(ParsedQuery::AnyTokens(tokenset)))
    }

    /// Parse as provided [`TokenizerTextKind`] and return [`TokenSet`].
    /// Unseen tokens are ignored.
    fn parse_tokenset(
        &self,
        kind: TokenizerTextKind,
        text: &str,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<TokenSet> {
        let token_ids = self.resolve_tokens(kind, text, hw_counter)?.into_values();
        Ok(token_ids.flatten().collect())
    }

    /// Tokenize the `text` and return a map of token -> token_id.
    /// Missing tokens will have [`None`] as token_id.
    fn resolve_tokens<'a>(
        &self,
        kind: TokenizerTextKind,
        text: &'a str,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<AHashMap<Cow<'a, str>, Option<TokenId>>> {
        let mut token_map = AHashMap::new();
        self.tokenizer().tokenize(kind, text, |token| {
            token_map.insert(token, None);
        });
        let iter = token_map
            .iter_mut()
            .map(|(token, cell)| (cell, token.as_ref()));
        self.for_each_token_id(iter, hw_counter, |cell, token_id| *cell = token_id)?;
        Ok(token_map)
    }

    /// Parse as [`TokenizerTextKind::Document`] and return a [`Document`].
    /// Returns [`None`] if there are any unseen tokens.
    fn parse_document(
        &self,
        text: &str,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<Option<Document>> {
        let mut document_tokens = Vec::new();
        let token_map = self.resolve_tokens(TokenizerTextKind::Document, text, hw_counter)?;
        if token_map.values().any(|token_id| token_id.is_none()) {
            return Ok(None);
        }

        self.tokenizer()
            .tokenize(TokenizerTextKind::Document, text, |token| {
                let token_id = token_map
                    .get(&token)
                    .expect("token should be in map")
                    .expect("token_id should be set for all tokens");
                document_tokens.push(token_id);
            });

        Ok(Some(Document::new(document_tokens)))
    }

    /// Checks a full-text match directly against the payload value using the
    /// full-text index tokenizer.
    fn check_payload_match(
        &self,
        payload_value: &serde_json::Value,
        r#match: &Match,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<bool> {
        let query_opt = match r#match {
            Match::Text(match_text) => self.parse_text_query(&match_text.text, hw_counter)?,
            Match::Phrase(match_phrase) => {
                self.parse_phrase_query(&match_phrase.phrase, hw_counter)?
            }
            Match::TextAny(match_text_any) => {
                self.parse_text_any_query(&match_text_any.text_any, hw_counter)?
            }
            Match::Fuzzy(match_fuzzy) => self.parse_fuzzy_query(&match_fuzzy.fuzzy, hw_counter),
            Match::Wildcard(match_wildcard) => {
                self.parse_wildcard_query(&match_wildcard.wildcard, hw_counter)?
            }
            Match::Value(_) | Match::Any(_) | Match::Except(_) => return Ok(false),
        };

        let Some(query) = query_opt else {
            return Ok(false);
        };

        <super::FullTextIndex as ValueIndexer>::get_values(payload_value)
            .iter()
            .try_any(|value| match &query {
                ParsedQuery::AllTokens(query) => {
                    let tokenset =
                        self.parse_tokenset(TokenizerTextKind::Document, value, hw_counter)?;
                    Ok(tokenset.has_subset(query))
                }
                ParsedQuery::Phrase(query) => {
                    let document = self.parse_document(value, hw_counter)?;
                    Ok(document.is_some_and(|doc| doc.has_phrase(query)))
                }
                ParsedQuery::AnyTokens(query) => {
                    let tokenset =
                        self.parse_tokenset(TokenizerTextKind::Document, value, hw_counter)?;
                    Ok(tokenset.has_any(query))
                }
                ParsedQuery::FuzzyAnyTokens(query) => {
                    let tokenset =
                        self.parse_tokenset(TokenizerTextKind::Document, value, hw_counter)?;
                    Ok(tokenset.has_any(query))
                }
                ParsedQuery::FuzzyAllTokens(fuzzy_doc) => {
                    let tokenset =
                        self.parse_tokenset(TokenizerTextKind::Document, value, hw_counter)?;
                    Ok(!fuzzy_doc.is_empty()
                        && fuzzy_doc.iter().all(|group| tokenset.has_any(group)))
                }
                ParsedQuery::FuzzyPhrase(fuzzy_doc) => {
                    let document = self.parse_document(value, hw_counter)?;
                    Ok(document.is_some_and(|doc| fuzzy_doc.matches_document(&doc)))
                }
            })
    }
}
