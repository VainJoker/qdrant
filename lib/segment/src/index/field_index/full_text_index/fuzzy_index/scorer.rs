#![allow(dead_code)]

#[derive(Debug, Clone)]
pub struct ScoredTerm {
    pub term: String,
    pub score: f32,
}

impl ScoredTerm {
    pub fn new(term: String, query_len: usize) -> Self {
        let term_len = term.len();
        let denom = query_len.max(term_len) as f32;
        let score = if denom == 0.0 {
            1.0
        } else {
            1.0 - (1.0 / denom)
        };
        Self { term, score }
    }
}
