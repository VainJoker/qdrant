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
use crate::types::{FieldCondition, Fuzzy, FuzzyParams, Match, MatchFuzzy, PayloadKeyType};

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

    fn parse_fuzzy_query(
        &self,
        match_fuzzy: &MatchFuzzy,
        hw_counter: &HardwareCounterCell,
    ) -> Option<ParsedQuery> {
        // Fuzzy match is enabled only when the dedicated fuzzy index exists.
        // Do not fall back to exact/full-text matching for `Match::Fuzzy` without it.
        let fuzzy_index = self.fuzzy_index()?;

        let default_params = FuzzyParams::default();

        match &match_fuzzy.fuzzy {
            Fuzzy::Text { text, params } => {
                let params = params.as_ref().unwrap_or(&default_params);
                if params.max_edits == 0 {
                    return self.parse_text_query(text, hw_counter).ok().flatten();
                }

                let mut groups = self.parse_fuzzy_token_sets(
                    TokenizerTextKind::Query,
                    text,
                    params,
                    hw_counter,
                    true,
                    fuzzy_index,
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
                    .parse_fuzzy_token_sets(
                        TokenizerTextKind::Query,
                        text_any,
                        params,
                        hw_counter,
                        false,
                        fuzzy_index,
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

                let groups = self.parse_fuzzy_token_sets(
                    TokenizerTextKind::Document,
                    phrase,
                    params,
                    hw_counter,
                    true,
                    fuzzy_index,
                )?;
                Some(ParsedQuery::FuzzyPhrase(FuzzyDocument::new(groups)))
            }
        }
    }

    fn parse_fuzzy_token_sets(
        &self,
        kind: TokenizerTextKind,
        text: &str,
        params: &FuzzyParams,
        hw_counter: &HardwareCounterCell,
        require_each_token: bool,
        fuzzy_index: &dyn FuzzyIndex,
    ) -> Option<Vec<TokenSet>> {
        let mut token_sets = Vec::new();
        let mut failed = false;

        self.tokenizer().tokenize(kind, text, |token| {
            match self.expand_fuzzy_token(token.as_ref(), params, hw_counter, fuzzy_index) {
                Some(token_set) if !token_set.is_empty() => token_sets.push(token_set),
                Some(_) if !require_each_token => {}
                _ => failed = true,
            }
        });

        if failed || token_sets.is_empty() {
            return None;
        }

        Some(token_sets)
    }

    fn expand_fuzzy_token(
        &self,
        token: &str,
        params: &FuzzyParams,
        hw_counter: &HardwareCounterCell,
        fuzzy_index: &dyn FuzzyIndex,
    ) -> Option<TokenSet> {
        let min_len = self
            .tokenizer()
            .tokens_processor()
            .min_token_len
            .unwrap_or(3);
        if token.chars().count() <= min_len {
            return self.resolve_token_set([token], hw_counter);
        }

        let candidates = fuzzy_index.search_levenshtein(token, params);
        self.resolve_token_set(
            candidates.iter().map(|candidate| candidate.term.as_str()),
            hw_counter,
        )
    }

    fn resolve_token_set<'a>(
        &self,
        tokens: impl IntoIterator<Item = &'a str>,
        hw_counter: &HardwareCounterCell,
    ) -> Option<TokenSet> {
        let mut token_ids = Vec::new();
        self.for_each_token_id(
            tokens.into_iter().map(|token| ((), token)),
            hw_counter,
            |(), id| {
                if let Some(id) = id {
                    token_ids.push(id);
                }
            },
        )
        .ok()?;

        Some(token_ids.into_iter().collect())
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
            Match::Fuzzy(match_fuzzy) => self.parse_fuzzy_query(match_fuzzy, hw_counter),
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
