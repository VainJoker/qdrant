#![allow(dead_code)]
use std::error::Error;

use fst::Automaton;
use fst::automaton::Levenshtein;

pub struct PrefixLevenshtein {
    pub(crate) prefix: Vec<u8>,
    pub(crate) lev: Levenshtein,
}

impl PrefixLevenshtein {
    pub fn new(query: &str, prefix_len: usize, distance: u32) -> Result<Self, Box<dyn Error>> {
        let clamped = prefix_len.min(query.len());
        let prefix = query.as_bytes()[..clamped].to_vec();
        let lev = Levenshtein::new(query, distance)?;
        Ok(Self { prefix, lev })
    }
}

pub enum PrefixLevState {
    Prefix(usize),
    Fuzzy(<Levenshtein as Automaton>::State),
    Dead,
}

impl Automaton for PrefixLevenshtein {
    type State = PrefixLevState;

    fn start(&self) -> Self::State {
        if self.prefix.is_empty() {
            PrefixLevState::Fuzzy(self.lev.start())
        } else {
            PrefixLevState::Prefix(0)
        }
    }

    fn is_match(&self, state: &Self::State) -> bool {
        match state {
            PrefixLevState::Prefix(_) => false,
            PrefixLevState::Fuzzy(s) => self.lev.is_match(s),
            PrefixLevState::Dead => false,
        }
    }

    fn can_match(&self, state: &Self::State) -> bool {
        match state {
            PrefixLevState::Prefix(_) => true,
            PrefixLevState::Fuzzy(s) => self.lev.can_match(s),
            PrefixLevState::Dead => false,
        }
    }

    fn will_always_match(&self, state: &Self::State) -> bool {
        match state {
            PrefixLevState::Prefix(_) => false,
            PrefixLevState::Fuzzy(s) => self.lev.will_always_match(s),
            PrefixLevState::Dead => false,
        }
    }

    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        match state {
            PrefixLevState::Prefix(i) => {
                if byte == self.prefix[*i] {
                    let next = i + 1;
                    if next == self.prefix.len() {
                        let mut lev_state = self.lev.start();
                        for &b in &self.prefix {
                            lev_state = self.lev.accept(&lev_state, b);
                        }
                        PrefixLevState::Fuzzy(lev_state)
                    } else {
                        PrefixLevState::Prefix(next)
                    }
                } else {
                    PrefixLevState::Dead
                }
            }
            PrefixLevState::Fuzzy(s) => PrefixLevState::Fuzzy(self.lev.accept(s, byte)),
            PrefixLevState::Dead => PrefixLevState::Dead,
        }
    }
}
