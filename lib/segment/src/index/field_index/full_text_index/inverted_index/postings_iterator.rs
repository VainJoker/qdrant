use common::types::PointOffsetType;
use itertools::{Either, Itertools};
use posting_list::{PostingIterator, PostingListView, PostingValue};

use super::posting_list::PostingList;
use crate::index::field_index::full_text_index::inverted_index::positions::{
    PartialDocument, Positions, TokenPosition,
};
use crate::index::field_index::full_text_index::inverted_index::{
    Document, FuzzyDocument, TokenId,
};

pub fn intersect_postings_iterator<'a>(
    mut postings: Vec<&'a PostingList>,
) -> impl Iterator<Item = PointOffsetType> + 'a {
    let smallest_posting_idx = postings
        .iter()
        .enumerate()
        .min_by_key(|(_idx, posting)| posting.len())
        .map(|(idx, _posting)| idx)
        .unwrap();
    let smallest_posting = postings.remove(smallest_posting_idx);

    smallest_posting
        .iter()
        .filter(move |doc_id| postings.iter().all(|posting| posting.contains(*doc_id)))
}

pub fn merge_postings_iterator<'a>(
    postings: Vec<&'a PostingList>,
) -> impl Iterator<Item = PointOffsetType> + 'a {
    postings
        .into_iter()
        .map(PostingList::iter)
        .kmerge_by(|a, b| a < b)
        .dedup()
}

pub fn intersect_compressed_postings_iterator<'a, V: PostingValue + 'a>(
    mut postings: Vec<PostingListView<'a, V>>,
    is_active: impl Fn(PointOffsetType) -> bool + 'a,
) -> impl Iterator<Item = PointOffsetType> + 'a {
    let smallest_posting_idx = postings
        .iter()
        .enumerate()
        .min_by_key(|(_idx, posting)| posting.len())
        .map(|(idx, _posting)| idx)
        .unwrap();
    let smallest_posting = postings.remove(smallest_posting_idx);
    let smallest_posting_iterator = smallest_posting.into_iter();

    let mut posting_iterators = postings
        .into_iter()
        .map(PostingListView::into_iter)
        .collect::<Vec<_>>();

    smallest_posting_iterator
        .map(|elem| elem.id)
        .filter(move |id| {
            is_active(*id)
                && posting_iterators.iter_mut().all(|posting_iterator| {
                    // Custom "contains" check, which leverages the fact that smallest posting is sorted,
                    // so the next id that must be in all postings is strictly greater than the previous one.
                    //
                    // This means that the other iterators can remember the last id they returned to avoid extra work
                    posting_iterator
                        // potential optimization: Make posting iterator of just ids, without values (a.k.a. positions).
                        //                         We are discarding them here, thus unnecessarily reading them from the tails of the posting lists.
                        .advance_until_greater_or_equal(*id)
                        .is_some_and(|elem| elem.id == *id)
                })
        })
}

pub fn merge_compressed_postings_iterator<'a, V: PostingValue + 'a>(
    postings: Vec<PostingListView<'a, V>>,
    is_active: impl Fn(PointOffsetType) -> bool + 'a,
) -> impl Iterator<Item = PointOffsetType> + 'a {
    postings
        .into_iter()
        // potential optimization: Make posting iterator of just ids, without values (a.k.a. positions).
        //                         We are discarding them here, thus unnecessarily reading them from the tails of the posting lists.
        .map(|view| view.into_iter().map(|elem| elem.id))
        .kmerge_by(|a, b| a < b)
        .dedup()
        .filter(move |id| is_active(*id))
}

/// Returns an iterator over the points that match the given phrase query.
pub fn intersect_compressed_postings_phrase_iterator<'a>(
    phrase: Document,
    mut postings: Vec<(TokenId, PostingListView<'a, Positions>)>,
    is_active: impl Fn(PointOffsetType) -> bool + 'a,
) -> impl Iterator<Item = PointOffsetType> + 'a {
    if phrase.is_empty() {
        // Empty request -> no matches
        return Either::Left(std::iter::empty());
    }

    let smallest_posting_idx = postings
        .iter()
        .enumerate()
        .min_by_key(|(_idx, (_token_id, posting))| posting.len())
        .map(|(idx, _posting)| idx)
        .unwrap();
    let (smallest_posting_token, smallest_posting) = postings.remove(smallest_posting_idx);
    let smallest_posting_iterator = smallest_posting.into_iter();

    let mut posting_iterators = postings
        .into_iter()
        .map(|(token_id, posting)| (token_id, posting.into_iter()))
        .collect::<Vec<_>>();

    let has_phrase_iter = smallest_posting_iterator
        .filter(move |elem| {
            if !is_active(elem.id) {
                return false;
            }

            let initial_tokens_positions = elem.value.to_token_positions(smallest_posting_token);

            phrase_in_all_postings(
                elem.id,
                &phrase,
                initial_tokens_positions,
                &mut posting_iterators,
            )
        })
        .map(|elem| elem.id);

    Either::Right(has_phrase_iter)
}

/// Reconstructs a partial document from the posting lists (which contain positions)
///
/// Returns true if the document contains the entire phrase, in the same order.
///
/// # Arguments
///
/// - `initial_tokens_positions` - must be prepopulated if iterating over a posting not included in the `posting_iterators`.
fn phrase_in_all_postings<'a>(
    id: PointOffsetType,
    phrase: &Document,
    initial_tokens_positions: Vec<TokenPosition>,
    posting_iterators: &mut Vec<(TokenId, PostingIterator<'a, Positions>)>,
) -> bool {
    let mut tokens_positions = initial_tokens_positions;
    for (token_id, posting_iterator) in posting_iterators.iter_mut() {
        // Custom "contains" check, which leverages the fact that smallest posting is sorted,
        // so the next id that must be in all postings is strictly greater than the previous one.
        //
        // This means that the other iterators can remember the last id they returned to avoid extra work
        let Some(other) = posting_iterator.advance_until_greater_or_equal(id) else {
            return false;
        };

        if id != other.id {
            return false;
        }

        debug_assert!(!other.value.is_empty());
        tokens_positions.extend(other.value.to_token_positions(*token_id))
    }
    PartialDocument::new(tokens_positions).has_phrase(phrase)
}

pub fn check_compressed_postings_phrase(
    phrase: &Document,
    point_id: PointOffsetType,
    token_to_posting: Vec<(TokenId, PostingListView<'_, Positions>)>,
) -> bool {
    let mut posting_iterators = token_to_posting
        .into_iter()
        .map(|(token_id, posting)| (token_id, posting.into_iter()))
        .collect::<Vec<_>>();

    phrase_in_all_postings(point_id, phrase, Vec::new(), &mut posting_iterators)
}

/// Checks if `point_id`'s document satisfies the fuzzy phrase.
pub fn check_compressed_postings_fuzzy_phrase<'a>(
    phrase: &FuzzyDocument,
    point_id: PointOffsetType,
    group_postings: Vec<Vec<(TokenId, PostingListView<'a, Positions>)>>,
) -> bool {
    let mut group_iters: Vec<Vec<(TokenId, PostingIterator<'a, Positions>)>> = group_postings
        .into_iter()
        .map(|group| {
            group
                .into_iter()
                .map(|(token_id, posting)| (token_id, posting.into_iter()))
                .collect()
        })
        .collect();

    let mut tokens_positions: Vec<TokenPosition> = Vec::new();

    for group_iter in &mut group_iters {
        let before = tokens_positions.len();

        for (token_id, iter) in group_iter {
            let Some(elem) = iter.advance_until_greater_or_equal(point_id) else {
                continue;
            };
            if elem.id == point_id {
                tokens_positions.extend(elem.value.to_token_positions(*token_id));
            }
        }

        if tokens_positions.len() == before {
            return false;
        }
    }

    tokens_positions.sort_unstable();
    tokens_positions.dedup();

    PartialDocument::new(tokens_positions).has_fuzzy_phrase(phrase)
}

struct GroupPositionPostingViews<'a> {
    total_len: usize,
    views: Vec<(TokenId, PostingListView<'a, Positions>)>,
}

impl<'a> GroupPositionPostingViews<'a> {
    fn new(views: Vec<(TokenId, PostingListView<'a, Positions>)>) -> Self {
        let total_len = views.iter().map(|(_, view)| view.len()).sum();
        Self { total_len, views }
    }
}

/// Returns an iterator over points whose documents satisfy the fuzzy phrase query.
///
/// Uses stateful peekable iterators (via [`PostingIterator::advance_until_greater_or_equal`])
/// for every group rather than creating fresh random-access visitors per candidate.
/// This eliminates redundant allocations and amortizes the O(log n) seek cost across
/// the monotonically increasing candidate stream.
pub fn intersect_compressed_postings_fuzzy_phrase_iterator<'a>(
    phrase: FuzzyDocument,
    group_postings: Vec<Vec<(TokenId, PostingListView<'a, Positions>)>>,
    is_active: impl Fn(PointOffsetType) -> bool + 'a,
) -> impl Iterator<Item = PointOffsetType> + 'a {
    let mut group_views: Vec<_> = group_postings
        .into_iter()
        .map(GroupPositionPostingViews::new)
        .collect();
    if phrase.is_empty()
        || group_views.is_empty()
        || group_views.iter().any(|group| group.views.is_empty())
    {
        return Either::Left(std::iter::empty());
    }

    let smallest_candidate_group_idx = group_views
        .iter()
        .enumerate()
        .min_by_key(|(_, group)| group.total_len)
        .map(|(idx, _)| idx)
        .unwrap();

    // Clone is cheap here: PostingListView holds only slice references (pointer+length).
    let smallest_candidate_views: Vec<PostingListView<'a, Positions>> = group_views
        [smallest_candidate_group_idx]
        .views
        .iter()
        .map(|(_, pl)| pl.clone())
        .collect();

    group_views.sort_unstable_by_key(|group| group.total_len);

    // Consume group_postings into one set of stateful iterators covering ALL groups.
    // advance_until_greater_or_equal has seek+peek semantics: it positions the cursor
    // at the first element >= target without consuming it, so the iterator retains its
    // position for the next candidate in the monotonically increasing stream.
    let mut group_iters: Vec<Vec<(TokenId, PostingIterator<'a, Positions>)>> = group_views
        .into_iter()
        .map(|group| {
            group
                .views
                .into_iter()
                .map(|(tid, pl)| (tid, pl.into_iter()))
                .collect()
        })
        .collect();

    Either::Right(
        merge_compressed_postings_iterator(smallest_candidate_views, is_active).filter(
            move |&point_id| {
                let mut tokens_positions: Vec<TokenPosition> = Vec::new();
                for group_iter in group_iters.iter_mut() {
                    let before = tokens_positions.len();
                    for (token_id, iter) in group_iter.iter_mut() {
                        // Seek to point_id without consuming the element (peek semantics).
                        // The iterator stays at this position for the next candidate.
                        let Some(elem) = iter.advance_until_greater_or_equal(point_id) else {
                            continue;
                        };
                        if elem.id == point_id {
                            tokens_positions.extend(elem.value.to_token_positions(*token_id));
                        }
                    }
                    // Every group must contribute at least one matching token position.
                    if tokens_positions.len() == before {
                        return false;
                    }
                }
                tokens_positions.sort_unstable();
                tokens_positions.dedup();
                PartialDocument::new(tokens_positions).has_fuzzy_phrase(&phrase)
            },
        ),
    )
}

struct GroupPostingViews<'a, V: PostingValue> {
    total_len: usize,
    views: Vec<PostingListView<'a, V>>,
}

impl<'a, V: PostingValue> GroupPostingViews<'a, V> {
    fn new(views: Vec<PostingListView<'a, V>>) -> Self {
        let total_len = views.iter().map(|view| view.len()).sum();
        Self { total_len, views }
    }
}

struct MaterializedPostingUnion {
    ids: Vec<PointOffsetType>,
    offset: usize,
}

impl MaterializedPostingUnion {
    fn new<'a, V: PostingValue + 'a>(views: Vec<PostingListView<'a, V>>) -> Self {
        let ids = views
            .into_iter()
            .map(|view| view.into_iter().map(|elem| elem.id))
            .kmerge_by(|a, b| a < b)
            .dedup()
            .collect();
        Self { ids, offset: 0 }
    }

    fn contains(&mut self, point_id: PointOffsetType) -> bool {
        while self
            .ids
            .get(self.offset)
            .is_some_and(|&current_id| current_id < point_id)
        {
            self.offset += 1;
        }

        self.ids
            .get(self.offset)
            .is_some_and(|&current_id| current_id == point_id)
    }
}

/// Returns an iterator that yields every active point in which **every group** of the
/// fuzzy document has at least one matching token.
///
/// Strategy: per-group union → cross-group intersect.
/// 1. Bail out early if any group has no postings.
/// 2. Pick the group with the smallest total posting size as the candidate driver.
/// 3. Materialize the union of every other group once.
/// 4. For every candidate, verify it appears in every other group's union.
pub fn intersect_compressed_postings_fuzzy_all_iterator<'a, V: PostingValue + 'a>(
    group_postings: Vec<Vec<PostingListView<'a, V>>>,
    is_active: impl Fn(PointOffsetType) -> bool + 'a,
) -> impl Iterator<Item = PointOffsetType> + 'a {
    let mut group_views: Vec<_> = group_postings
        .into_iter()
        .map(GroupPostingViews::new)
        .collect();
    if group_views.is_empty() || group_views.iter().any(|group| group.views.is_empty()) {
        return Either::Left(std::iter::empty());
    }

    let smallest_idx = group_views
        .iter()
        .enumerate()
        .min_by_key(|(_, group)| group.total_len)
        .map(|(idx, _)| idx)
        .unwrap();
    let candidate_views = group_views.swap_remove(smallest_idx).views;

    group_views.sort_unstable_by_key(|group| group.total_len);

    let mut other_group_unions: Vec<_> = group_views
        .into_iter()
        .map(|group| MaterializedPostingUnion::new(group.views))
        .collect();

    Either::Right(
        merge_compressed_postings_iterator(candidate_views, is_active).filter(move |&point_id| {
            other_group_unions
                .iter_mut()
                .all(|group| group.contains(point_id))
        }),
    )
}

#[cfg(test)]
mod tests {

    use posting_list::IdsPostingList;

    use super::*;

    #[test]
    fn test_postings_iterator() {
        let mut p1 = PostingList::default();
        p1.insert(1);
        p1.insert(2);
        p1.insert(3);
        p1.insert(4);
        p1.insert(5);
        let mut p2 = PostingList::default();
        p2.insert(2);
        p2.insert(4);
        p2.insert(5);
        p2.insert(5);
        let mut p3 = PostingList::default();
        p3.insert(1);
        p3.insert(2);
        p3.insert(5);
        p3.insert(6);
        p3.insert(7);

        let postings = vec![&p1, &p2, &p3];
        let merged = intersect_postings_iterator(postings);

        let res = merged.collect::<Vec<_>>();

        assert_eq!(res, vec![2, 5]);

        let p1_compressed: IdsPostingList = p1.iter().map(|id| (id, ())).collect();
        let p2_compressed: IdsPostingList = p2.iter().map(|id| (id, ())).collect();
        let p3_compressed: IdsPostingList = p3.iter().map(|id| (id, ())).collect();
        let compressed_posting_reades = vec![
            p1_compressed.view(),
            p2_compressed.view(),
            p3_compressed.view(),
        ];
        let merged = intersect_compressed_postings_iterator(compressed_posting_reades, |_| true);

        let res = merged.collect::<Vec<_>>();

        assert_eq!(res, vec![2, 5]);
    }
}
