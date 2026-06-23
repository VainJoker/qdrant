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
///
/// `group_postings` is grouped the same way as [`FuzzyDocument`]: each outer group
/// represents one fuzzy term position, and each inner posting list represents one
/// concrete token that can satisfy that position.
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

        // A fuzzy phrase requires every group to have at least one matching token
        // in the point. If this group contributed nothing, the point cannot match.
        if tokens_positions.len() == before {
            return false;
        }
    }

    // Different fuzzy alternatives may map to the same token/position pair.
    // Normalize before reconstructing the partial document.
    tokens_positions.sort_unstable();
    tokens_positions.dedup();

    PartialDocument::new(tokens_positions).has_fuzzy_phrase(phrase)
}

/// One fuzzy-phrase group after upper layers expand a query term into token alternatives.
///
/// Built by immutable/on-disk `filter_has_phrase_fuzzy` before calling
/// `intersect_compressed_postings_fuzzy_phrase_iterator`.  Phrase matching needs
/// both token ids and positions, so each posting view is kept with its `TokenId`.
/// `total_len` is used as a cheap cost estimate to choose the smallest group as
/// the candidate driver.
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

/// One fuzzy-all group after upper layers expand a query term into token alternatives.
///
/// Used by `intersect_compressed_postings_fuzzy_all_iterator`, which is called from
/// immutable/on-disk `filter_has_all_fuzzy`.  Here only document membership matters,
/// so token ids and positions are not needed; the group is just a set of posting views.
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

/// Materialized OR of one fuzzy-all group.
///
/// After the smallest group is chosen as the sorted candidate stream, every other
/// group is stored as a sorted, deduplicated union.  `contains` can then check
/// candidates with a monotonic cursor instead of repeatedly seeking from the start.
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
        // The candidate stream is sorted, so ids lower than the current candidate
        // will never be needed again. Advance the cursor instead of binary-searching
        // from the beginning for every candidate.
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

    // Candidate generation is driven by the group with the smallest union size.
    // This keeps the number of expensive phrase-position checks as low as possible.
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
                // Convert all matching alternatives into a compact synthetic document,
                // then run the fuzzy phrase matcher over positions.
                tokens_positions.sort_unstable();
                tokens_positions.dedup();
                PartialDocument::new(tokens_positions).has_fuzzy_phrase(&phrase)
            },
        ),
    )
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
    // Generate candidates from the smallest per-group union, then intersect those
    // candidates with the materialized unions of all remaining groups.
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

    use posting_list::{IdsPostingList, PostingList as CompressedPostingList};

    use super::*;
    use crate::index::field_index::full_text_index::inverted_index::TokenSet;

    fn token_set(tokens: &[TokenId]) -> TokenSet {
        tokens.iter().copied().collect()
    }

    fn fuzzy_doc(groups: &[&[TokenId]]) -> FuzzyDocument {
        FuzzyDocument::new(groups.iter().map(|tokens| token_set(tokens)).collect())
    }

    fn ids_posting(ids: &[PointOffsetType]) -> IdsPostingList {
        ids.iter().copied().map(|id| (id, ())).collect()
    }

    fn positions_posting(
        entries: Vec<(PointOffsetType, Vec<u32>)>,
    ) -> CompressedPostingList<Positions> {
        entries
            .into_iter()
            .map(|(id, positions)| {
                let mut value = Positions::default();
                for position in positions {
                    value.push(position);
                }
                (id, value)
            })
            .collect()
    }

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

    #[test]
    fn test_fuzzy_all_iterator_intersects_group_unions() {
        let token_0 = ids_posting(&[1, 2, 10]);
        let token_1 = ids_posting(&[3, 4]);
        let token_2 = ids_posting(&[2, 3, 4, 5]);
        let token_3 = ids_posting(&[10]);

        let group_postings = vec![
            vec![token_0.view(), token_1.view()],
            vec![token_2.view(), token_3.view()],
        ];

        let result = intersect_compressed_postings_fuzzy_all_iterator(group_postings, |id| id != 3)
            .collect::<Vec<_>>();

        assert_eq!(result, vec![2, 4, 10]);
    }

    #[test]
    fn test_fuzzy_iterators_return_empty_for_missing_group() {
        let ids = ids_posting(&[1, 2]);
        let positions = positions_posting(vec![(1, vec![0]), (2, vec![0])]);
        let phrase = fuzzy_doc(&[&[10], &[20]]);

        let all_result = intersect_compressed_postings_fuzzy_all_iterator(
            vec![vec![ids.view()], Vec::new()],
            |_| true,
        )
        .collect::<Vec<_>>();
        assert!(all_result.is_empty());

        let phrase_result = intersect_compressed_postings_fuzzy_phrase_iterator(
            phrase,
            vec![vec![(10, positions.view())], Vec::new()],
            |_| true,
        )
        .collect::<Vec<_>>();
        assert!(phrase_result.is_empty());
    }

    #[test]
    fn test_fuzzy_phrase_iterator_checks_positions_and_alternatives() {
        let token_10 = positions_posting(vec![(1, vec![0]), (4, vec![0]), (5, vec![0])]);
        let token_11 = positions_posting(vec![(2, vec![0]), (3, vec![1])]);
        let token_20 = positions_posting(vec![(1, vec![1]), (4, vec![1])]);
        let token_21 = positions_posting(vec![(2, vec![2]), (3, vec![2]), (5, vec![1])]);
        let token_30 =
            positions_posting(vec![(1, vec![2]), (2, vec![3]), (3, vec![3]), (5, vec![2])]);

        let phrase = fuzzy_doc(&[&[10, 11], &[20, 21], &[30]]);
        let group_postings = || {
            vec![
                vec![(10, token_10.view()), (11, token_11.view())],
                vec![(20, token_20.view()), (21, token_21.view())],
                vec![(30, token_30.view())],
            ]
        };

        let result = intersect_compressed_postings_fuzzy_phrase_iterator(
            phrase.clone(),
            group_postings(),
            |id| id != 5,
        )
        .collect::<Vec<_>>();

        assert_eq!(result, vec![1, 3]);
        assert!(check_compressed_postings_fuzzy_phrase(
            &phrase,
            1,
            group_postings()
        ));
        assert!(check_compressed_postings_fuzzy_phrase(
            &phrase,
            3,
            group_postings()
        ));
        assert!(!check_compressed_postings_fuzzy_phrase(
            &phrase,
            2,
            group_postings()
        ));
        assert!(!check_compressed_postings_fuzzy_phrase(
            &phrase,
            4,
            group_postings()
        ));
    }

    #[test]
    fn test_fuzzy_phrase_check_deduplicates_overlapping_groups() {
        let token_42 = positions_posting(vec![(7, vec![0])]);
        let token_43 = positions_posting(vec![(7, vec![1])]);
        let phrase = fuzzy_doc(&[&[42], &[42, 43]]);

        let group_postings = vec![
            vec![(42, token_42.view())],
            vec![(42, token_42.view()), (43, token_43.view())],
        ];

        assert!(check_compressed_postings_fuzzy_phrase(
            &phrase,
            7,
            group_postings
        ));
    }
}
