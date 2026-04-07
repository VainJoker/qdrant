use std::iter::FusedIterator;

use common::types::PointOffsetType;

use crate::PostingElement;
use crate::value_handler::PostingValue;
use crate::visitor::PostingVisitor;

pub struct PostingIterator<'a, V: PostingValue> {
    visitor: PostingVisitor<'a, V>,
    current_elem: Option<PostingElement<V>>,
    offset: usize,
}

impl<'a, V: PostingValue> PostingIterator<'a, V> {
    pub fn new(visitor: PostingVisitor<'a, V>) -> Self {
        Self {
            visitor,
            current_elem: None,
            offset: 0,
        }
    }

    /// Advances the iterator until the current element id is greater than or equal to the given id.
    ///
    /// Returns `Some(PostingElement)` on the first element that is greater than or equal to the given id. It can be possible that this id is
    /// the head of the iterator, so it does not need to be advanced.
    ///
    /// `None` means the iterator is exhausted.
    pub fn advance_until_greater_or_equal(
        &mut self,
        target_id: PointOffsetType,
    ) -> Option<PostingElement<V>> {
        if let Some(current) = &self.current_elem
            && current.id >= target_id
        {
            return Some(current.clone());
        }

        if self.offset >= self.visitor.len() {
            return None;
        }

        let Some(offset) = self
            .visitor
            .search_greater_or_equal(target_id, Some(self.offset))
        else {
            self.current_elem = None;
            self.offset = self.visitor.len();
            return None;
        };

        debug_assert!(offset >= self.offset);
        let greater_or_equal = self.visitor.get_by_offset(offset);

        self.current_elem = greater_or_equal.clone();
        self.offset = offset;

        greater_or_equal
    }
}

impl<V: PostingValue> Iterator for PostingIterator<'_, V> {
    type Item = PostingElement<V>;

    fn next(&mut self) -> Option<Self::Item> {
        let next_opt = self.visitor.get_by_offset(self.offset).inspect(|_| {
            self.offset += 1;
        });

        self.current_elem = next_opt.clone();

        next_opt
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining_len = self.len();
        (remaining_len, Some(remaining_len))
    }

    fn count(self) -> usize {
        self.size_hint().0
    }
}

impl<V: PostingValue> ExactSizeIterator for PostingIterator<'_, V> {
    fn len(&self) -> usize {
        self.visitor.list.len().saturating_sub(self.offset)
    }
}

impl<V: PostingValue> FusedIterator for PostingIterator<'_, V> {}

/// An ID-only seek iterator over a [`PostingListView`].
///
/// Like [`PostingIterator`] but never reads the value (e.g. positions), only the point IDs.
/// Use this when you need membership / seek checks without caring about the stored value, since
/// skipping value deserialization—especially variable-length position data—can significantly
/// reduce memory bandwidth.
///
/// Has the same **peek semantics** as [`PostingIterator::advance_until_greater_or_equal`]:
/// the cursor is positioned at the first id ≥ target and stays there until a strictly larger
/// target is given.
pub struct PostingIdIterator<'a, V: PostingValue> {
    visitor: PostingVisitor<'a, V>,
    /// Cached id at the current cursor position (peek semantics).
    current_id: Option<PointOffsetType>,
    offset: usize,
}

impl<'a, V: PostingValue> PostingIdIterator<'a, V> {
    pub fn new(visitor: PostingVisitor<'a, V>) -> Self {
        Self {
            visitor,
            current_id: None,
            offset: 0,
        }
    }

    /// Advances the cursor to the first id ≥ `target_id` and returns it, or `None` if
    /// the iterator is exhausted.
    ///
    /// Only the ID bytes are read; the value (e.g. positions stored in `var_size_data`) is
    /// never touched.
    pub fn advance_id_until_greater_or_equal(
        &mut self,
        target_id: PointOffsetType,
    ) -> Option<PointOffsetType> {
        // Fast path: cursor is already at or past the target.
        if let Some(current) = self.current_id
            && current >= target_id
        {
            return Some(current);
        }

        if self.offset >= self.visitor.len() {
            return None;
        }

        let offset = self
            .visitor
            .search_greater_or_equal(target_id, Some(self.offset))?;

        // get_id_by_offset reuses the chunk already decompressed by search_greater_or_equal.
        let id = self.visitor.get_id_by_offset(offset)?;
        self.current_id = Some(id);
        self.offset = offset;
        Some(id)
    }
}
