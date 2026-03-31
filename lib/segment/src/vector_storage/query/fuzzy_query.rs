use std::hash::Hash;

use common::types::ScoreType;
use itertools::Itertools;
use ordered_float::OrderedFloat;
use serde::Serialize;

use super::{Query, TransformInto};
use crate::common::operation_error::OperationResult;
use crate::types::FuzzyIntent;

/// Akin to external representation of the query. Unoptimized for scoring.
///
/// Call `into_query` to get the type implementing `Query` trait.
#[derive(Clone, Debug, Serialize, Hash, PartialEq)]
pub struct FuzzyQuery<T> {
    /// The original query vector.
    pub traget: T,

    /// The intent of the fuzzy search, which can be used to adjust the scoring function.
    pub fuzzy: FuzzyIntent,
}

impl FuzzyQuery<()> {
    pub fn new<T>(traget: T, fuzzy: FuzzyIntent) -> FuzzyQuery<T> {
        FuzzyQuery { traget, fuzzy }
    }
}

impl<T: Clone> FuzzyQuery<T> {
    pub fn into_query(self) -> FuzzyQuery<T> {
        self
    }
}

impl<T> FuzzyQuery<T> {
    pub fn flat_iter(&self) -> impl Iterator<Item = &T> {
        std::iter::once(&self.traget)
    }
}

impl<T, U> TransformInto<FuzzyQuery<U>, T, U> for FuzzyQuery<T> {
    fn transform<F>(self, mut f: F) -> OperationResult<FuzzyQuery<U>>
    where
        F: FnMut(T) -> OperationResult<U>,
    {
        let Self { traget, fuzzy } = self;
        Ok(FuzzyQuery::new(f(traget)?, fuzzy))
    }
}

impl<T> Query<T> for FuzzyQuery<T> {
    fn score_by(&self, similarity: impl Fn(&T) -> ScoreType) -> ScoreType {
        let Self {
            traget: vector,
            fuzzy,
        } = self;

        // TODO: implement different scoring functions based on the intent

        todo!(
            "Fuzzy search scoring is not implemented yet. This is a placeholder implementation that just returns the similarity score. The intent is ignored for now, but it can be used to adjust the scoring function in the future."
        )
    }
}
