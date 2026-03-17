use std::collections::HashSet;
use std::path::PathBuf;

use common::fs::clear_disk_cache;
use fst::{IntoStreamer, Streamer};

use super::super::FuzzyIndex;
use crate::common::operation_error::OperationResult;
use crate::index::field_index::full_text_index::fuzzy_index::automaton::PrefixLevenshtein;
use crate::index::field_index::full_text_index::fuzzy_index::mmap_fuzzy_index::MmapFst;
use crate::index::field_index::full_text_index::fuzzy_index::{
    FuzzyCandidate, ImmutableFuzzyIndex,
};
use crate::types::FuzzyParams;

const FUZZY_INDEX_FILE: &str = "fst.dat";

pub struct MmapFuzzyIndex {
    path: PathBuf,
    index: MmapFst,
}

impl MmapFuzzyIndex {
    pub fn create(path: PathBuf, fuzzy_index: &ImmutableFuzzyIndex) -> OperationResult<()> {
        let fuzzy_index_path = path.join(FUZZY_INDEX_FILE);
        MmapFst::create(fuzzy_index_path, fuzzy_index)?;
        Ok(())
    }

    pub fn open(
        path: PathBuf,
        populate: bool,
        enable_fuzzy: bool,
    ) -> OperationResult<Option<Self>> {
        if !enable_fuzzy {
            return Ok(None);
        }

        let fuzzy_index_path = path.join(FUZZY_INDEX_FILE);
        if !fuzzy_index_path.is_file() {
            return Ok(None);
        }

        let index = MmapFst::open(fuzzy_index_path, populate)?;
        Ok(Some(Self { path, index }))
    }

    pub fn files(&self) -> Vec<PathBuf> {
        vec![self.path.join(FUZZY_INDEX_FILE)]
    }

    pub fn immutable_files(&self) -> Vec<PathBuf> {
        vec![self.path.join(FUZZY_INDEX_FILE)]
    }

    pub fn fst_bytes(&self) -> &[u8] {
        self.index.fst_bytes()
    }

    pub fn populate(&self) -> OperationResult<()> {
        self.index.populate();
        Ok(())
    }

    /// Drop disk cache.
    pub fn clear_cache(&self) -> OperationResult<()> {
        let files = self.files();
        for file in files {
            clear_disk_cache(&file)?;
        }

        Ok(())
    }
}

impl FuzzyIndex for MmapFuzzyIndex {
    fn search(&self, query: &str, params: &FuzzyParams) -> Vec<FuzzyCandidate> {
        let max = params.max_expansions as usize;
        let mut results: Vec<FuzzyCandidate> = Vec::with_capacity(max);
        let mut seen: HashSet<String> = HashSet::new();

        'outer: for distance in 0..=params.max_edits as u32 {
            if results.len() >= max {
                break;
            }

            let automaton =
                match PrefixLevenshtein::new(query, params.prefix_length as usize, distance) {
                    Ok(a) => a,
                    Err(_) => break,
                };

            let prefix_bytes =
                &query.as_bytes()[..params.prefix_length.min(query.len() as u8) as usize];
            let mut stream = if prefix_bytes.is_empty() {
                self.index.get_fst().search(&automaton).into_stream()
            } else {
                self.index
                    .get_fst()
                    .search(&automaton)
                    .ge(prefix_bytes)
                    .into_stream()
            };

            while let Some((term_bytes, _)) = stream.next() {
                let term = match std::str::from_utf8(term_bytes) {
                    Ok(t) => t.to_string(),
                    Err(_) => continue,
                };
                if seen.insert(term.clone()) {
                    results.push(FuzzyCandidate::new(term, query.len(), distance));
                    if results.len() >= max {
                        break 'outer;
                    }
                }
            }
        }

        results
    }
}
