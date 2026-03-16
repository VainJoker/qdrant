#![allow(dead_code)]

pub struct FuzzyParams {
    pub max_edits: u8,
    pub prefix_length: u8,
    pub max_expansions: u16,
}

impl FuzzyParams {
    pub fn new(max_edits: u8, prefix_length: u8, max_expansions: u16) -> Self {
        Self {
            max_edits,
            prefix_length,
            max_expansions,
        }
    }
}
