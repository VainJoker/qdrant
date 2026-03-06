mod test_congruence;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::PointOffsetType;
use tempfile::Builder;

use crate::data_types::index::{TextIndexParams, TextIndexType, TokenizerType};
use crate::index::field_index::full_text_index::text_index::FullTextIndex;
use crate::index::field_index::{FieldIndexBuilderTrait as _, ValueIndexer};
use crate::types::{Fuzzy, FuzzyParams, MatchFuzzy};

fn movie_titles() -> Vec<String> {
    vec![
        "2430 A.D.".to_string(),
        "The Acquisitive Chuckle".to_string(),
        "Author! Author!".to_string(),
        "The Bicentennial Man".to_string(),
        "Big Game".to_string(),
        "The Billiard Ball".to_string(),
        "Birth of a Notion".to_string(),
        "Black Friar of the Flame".to_string(),
        "Blank!".to_string(),
        "Blind Alley".to_string(),
        "Breeds There a Man...?".to_string(),
        "Button, Button".to_string(),
        "Buy Jupiter".to_string(),
        "C-Chute".to_string(),
        "Cal".to_string(),
        "The Callistan Menace".to_string(),
        "Catch That Rabbit".to_string(),
        "Christmas on Ganymede".to_string(),
        "Darwinian Pool Room".to_string(),
        "Day of the Hunters".to_string(),
        "Death Sentence".to_string(),
        "Does a Bee Care?".to_string(),
        "Dreaming Is a Private Thing".to_string(),
        "The Dust of Death".to_string(),
        "The Dying Night".to_string(),
        "Each an Explorer".to_string(),
        "Escape!".to_string(),
        "Everest".to_string(),
        "Evidence".to_string(),
        "The Evitable Conflict".to_string(),
        "Exile to Hell".to_string(),
        "Eyes Do More Than See".to_string(),
        "The Feeling of Power".to_string(),
        "Feminine Intuition".to_string(),
        "First Law".to_string(),
        "Flies".to_string(),
        "For the Birds".to_string(),
        "Founding Father".to_string(),
        "The Fun They Had".to_string(),
        "Galley Slave".to_string(),
        "The Gentle Vultures".to_string(),
        "Getting Even".to_string(),
        "Gimmicks Three".to_string(),
        "Gold".to_string(),
        "Good Taste".to_string(),
        "The Greatest Asset".to_string(),
        "Green Patches".to_string(),
        "Half-Breed".to_string(),
        "Half-Breeds on Venus".to_string(),
        "Hallucination".to_string(),
        "The Hazing".to_string(),
        "Hell-Fire".to_string(),
        "Heredity".to_string(),
        "History".to_string(),
        "Homo Sol".to_string(),
        "Hostess".to_string(),
        "I Just Make Them Up, See!".to_string(),
        "I'm in Marsport Without Hilda".to_string(),
        "The Imaginary".to_string(),
        "The Immortal Bard".to_string(),
        "In a Good Cause—".to_string(),
        "Insert Knob A in Hole B".to_string(),
        "The Instability".to_string(),
        "It's Such a Beautiful Day".to_string(),
        "The Key".to_string(),
        "Kid Stuff".to_string(),
        "The Last Answer".to_string(),
        "The Last Question".to_string(),
        "The Last Trump".to_string(),
        "Left to Right".to_string(),
        "Legal Rites".to_string(),
        "Lenny".to_string(),
        "Lest We Remember".to_string(),
        "Let's Not".to_string(),
        "Liar!".to_string(),
        "Light Verse".to_string(),
        "Little Lost Robot".to_string(),
        "The Little Man on the Subway".to_string(),
        "Living Space".to_string(),
        "A Loint of Paw".to_string(),
        "The Magnificent Possession".to_string(),
        "Marching In".to_string(),
        "Marooned off Vesta".to_string(),
        "The Message".to_string(),
        "Mirror Image".to_string(),
        "Mother Earth".to_string(),
        "My Son, the Physicist".to_string(),
        "No Connection".to_string(),
        "No Refuge Could Save".to_string(),
        "Nobody Here But—".to_string(),
        "Not Final!".to_string(),
        "Obituary".to_string(),
        "Old-fashioned".to_string(),
        "Pâté de Foie Gras".to_string(),
        "The Pause".to_string(),
        "Ph as in Phony".to_string(),
        "The Portable Star".to_string(),
        "The Proper Study".to_string(),
        "Rain, Rain, Go Away".to_string(),
        "Reason".to_string(),
        "The Red Queen's Race".to_string(),
        "Rejection Slips".to_string(),
        "Ring Around the Sun".to_string(),
        "Risk".to_string(),
        "Robot AL-76 Goes Astray".to_string(),
        "Robot Dreams".to_string(),
        "Runaround".to_string(),
        "Sally".to_string(),
        "Satisfaction Guaranteed".to_string(),
        "The Secret Sense".to_string(),
        "Shah Guido G.".to_string(),
        "Silly Asses".to_string(),
        "The Singing Bell".to_string(),
        "Sixty Million Trillion Combinations".to_string(),
        "Spell My Name with an S".to_string(),
        "Star Light".to_string(),
        "A Statue for Father".to_string(),
        "Strikebreaker".to_string(),
        "Super-Neutron".to_string(),
        "Take a Match".to_string(),
        "The Talking Stone".to_string(),
        ". . . That Thou Art Mindful of Him".to_string(),
        "Thiotimoline".to_string(),
        "Time Pussy".to_string(),
        "Trends".to_string(),
        "Truth to Tell".to_string(),
        "The Ugly Little Boy".to_string(),
        "The Ultimate Crime".to_string(),
        "Unto the Fourth Generation".to_string(),
        "The Up-to-Date Sorcerer".to_string(),
        "Waterclap".to_string(),
        "The Watery Place".to_string(),
        "The Weapon".to_string(),
        "The Weapon Too Dreadful to Use".to_string(),
        "What If—".to_string(),
        "What Is This Thing Called Love?".to_string(),
        "What's in a Name?".to_string(),
        "The Winnowing".to_string(),
    ]
}

#[test]
fn test_prefix_search() {
    let temp_dir = Builder::new().prefix("test_dir").tempdir().unwrap();
    let config = TextIndexParams {
        r#type: TextIndexType::Text,
        tokenizer: TokenizerType::Prefix,
        min_token_len: None,
        max_token_len: None,
        lowercase: None,
        phrase_matching: None,
        fuzzy_matching: None,
        stopwords: None,
        on_disk: None,
        stemmer: None,
        ascii_folding: None,
        enable_hnsw: None,
    };

    let mut index =
        FullTextIndex::new_gridstore(temp_dir.path().to_path_buf(), config.clone(), true)
            .unwrap()
            .unwrap();

    let hw_counter = HardwareCounterCell::new();

    let texts = movie_titles();

    for (i, text) in texts.iter().enumerate() {
        index
            .add_many(i as PointOffsetType, vec![text.clone()], &hw_counter)
            .unwrap();
    }

    let res: Vec<_> = index.query("ROBO", &hw_counter).collect();

    let query = index.parse_text_query("ROBO", &hw_counter).unwrap();

    for idx in res.iter().copied() {
        assert!(index.check_match(&query, idx));
    }

    assert_eq!(res.len(), 3);

    let res: Vec<_> = index.query("q231", &hw_counter).collect();
    assert!(res.is_empty());

    assert!(index.parse_text_query("q231", &hw_counter).is_none());
}

#[test]
fn test_phrase_matching() {
    let hw_counter = HardwareCounterCell::default();

    // Create a text index with phrase matching enabled
    let temp_dir = Builder::new().prefix("test_dir").tempdir().unwrap();
    let config = TextIndexParams {
        r#type: TextIndexType::Text,
        tokenizer: TokenizerType::default(),
        min_token_len: None,
        max_token_len: None,
        lowercase: Some(true),
        on_disk: None,
        phrase_matching: Some(true), // Enable phrase matching
        fuzzy_matching: None,
        stopwords: None,
        stemmer: None,
        ascii_folding: None,
        enable_hnsw: None,
    };

    let mut mutable_index =
        FullTextIndex::builder_gridstore(temp_dir.path().to_path_buf(), config.clone())
            .make_empty()
            .unwrap();

    let mut mmap_builder =
        FullTextIndex::builder_mmap(temp_dir.path().to_path_buf(), config.clone(), true);
    mmap_builder.init().unwrap();

    // Add some test documents with phrases
    let documents = vec![
        (0, "the quick brown fox jumps over the lazy dog".to_string()),
        (1, "brown fox quick the jumps over lazy dog".to_string()),
        (2, "quick brown fox runs fast".to_string()),
        (3, "the lazy dog sleeps peacefully".to_string()),
        (4, "the brown brown fox".to_string()),
    ];

    for (point_id, text) in documents {
        mutable_index
            .add_many(point_id, vec![text.clone()], &hw_counter)
            .unwrap();
        mmap_builder
            .add_many(point_id, vec![text], &hw_counter)
            .unwrap();
    }

    let mmap_index = mmap_builder.finalize().unwrap();

    let check_matching = |index: FullTextIndex| {
        // Test regular text matching (should match documents containing all tokens regardless of order)
        let text_query = index
            .parse_text_query("quick brown fox", &hw_counter)
            .unwrap();
        assert!(index.check_match(&text_query, 0));
        assert!(index.check_match(&text_query, 1));
        assert!(index.check_match(&text_query, 2));

        let text_results: Vec<_> = index.filter_query(text_query, &hw_counter).collect();

        // Should match documents 0, 1, and 2 (all contain "quick", "brown", "fox")
        assert_eq!(text_results.len(), 3);
        assert!(text_results.contains(&0));
        assert!(text_results.contains(&1));
        assert!(text_results.contains(&2));

        // Test phrase matching (should only match documents with exact phrase in order)
        let phrase_query = index
            .parse_phrase_query("quick brown fox", &hw_counter)
            .unwrap();
        assert!(index.check_match(&phrase_query, 0));
        assert!(index.check_match(&phrase_query, 2));

        let phrase_results: Vec<_> = index.filter_query(phrase_query, &hw_counter).collect();

        // Should only match documents 0 and 2 (contain "quick brown fox" in that exact order)
        assert_eq!(phrase_results.len(), 2);
        assert!(phrase_results.contains(&0));
        assert!(phrase_results.contains(&2));
        assert!(!phrase_results.contains(&1)); // Document 1 has the words but not in the right order

        // Test phrase that doesn't exist
        let missing_query = index
            .parse_phrase_query("fox brown quick", &hw_counter)
            .unwrap();
        let missing_results: Vec<_> = index.filter_query(missing_query, &hw_counter).collect();

        // Should match no documents (no document contains this exact phrase)
        assert_eq!(missing_results.len(), 0);

        // Test valid phrase up to a token that doesn't exist
        let query_with_unknown_token = index.parse_phrase_query("quick brown bird", &hw_counter);
        // the phrase query is not valid because it contains an unknown token
        assert!(query_with_unknown_token.is_none());

        // Test repeated words
        let phrase_query = index
            .parse_phrase_query("brown brown fox", &hw_counter)
            .unwrap();
        assert!(index.check_match(&phrase_query, 4));

        // Should only match document 4
        let filter_results: Vec<_> = index.filter_query(phrase_query, &hw_counter).collect();
        assert_eq!(filter_results.len(), 1);
        assert!(filter_results.contains(&4));
    };

    check_matching(mutable_index);
    check_matching(mmap_index);
}

#[test]
fn test_ascii_folding_in_full_text_index_word() {
    let hw_counter = HardwareCounterCell::default();

    let temp_dir = Builder::new().prefix("test_dir").tempdir().unwrap();
    let config_enabled = TextIndexParams {
        r#type: TextIndexType::Text,
        tokenizer: TokenizerType::Word,
        min_token_len: None,
        max_token_len: None,
        lowercase: None,
        on_disk: None,
        phrase_matching: None,
        fuzzy_matching: None,
        stopwords: None,
        stemmer: None,
        ascii_folding: Some(true),
        enable_hnsw: None,
    };
    let config_disabled = TextIndexParams {
        ascii_folding: Some(false),
        ..config_enabled.clone()
    };

    // Index with folding enabled
    let mut index_enabled =
        FullTextIndex::new_gridstore(temp_dir.path().to_path_buf(), config_enabled.clone(), true)
            .unwrap()
            .unwrap();

    // Index with folding disabled (separate storage path)
    let temp_dir2 = Builder::new().prefix("test_dir").tempdir().unwrap();
    let mut index_disabled = FullTextIndex::new_gridstore(
        temp_dir2.path().to_path_buf(),
        config_disabled.clone(),
        true,
    )
    .unwrap()
    .unwrap();

    // Documents containing accents
    let docs = vec![
        (0, "ação no coração".to_string()),
        (1, "café com leite".to_string()),
    ];

    for (id, text) in &docs {
        index_enabled
            .add_many(*id as PointOffsetType, vec![text.clone()], &hw_counter)
            .unwrap();
        index_disabled
            .add_many(*id as PointOffsetType, vec![text.clone()], &hw_counter)
            .unwrap();
    }

    // ASCII-only queries should match only when folding is enabled
    let query_enabled = index_enabled.parse_text_query("acao", &hw_counter).unwrap();
    assert!(index_enabled.check_match(&query_enabled, 0));

    let results_enabled: Vec<_> = index_enabled
        .filter_query(query_enabled, &hw_counter)
        .collect();
    assert!(results_enabled.contains(&0));

    let query_disabled_opt = index_disabled.parse_text_query("acao", &hw_counter);
    // Query might still parse, but should not match anything
    if let Some(query_disabled) = query_disabled_opt {
        let results_disabled: Vec<_> = index_disabled
            .filter_query(query_disabled, &hw_counter)
            .collect();
        assert!(!results_disabled.contains(&0));
    }

    // Non-folded query must work in both
    let query_acento = index_enabled.parse_text_query("ação", &hw_counter).unwrap();
    assert!(index_enabled.check_match(&query_acento, 0));
    let results_acento: Vec<_> = index_enabled
        .filter_query(query_acento, &hw_counter)
        .collect();
    assert!(results_acento.contains(&0));

    let query_acento2 = index_disabled
        .parse_text_query("ação", &hw_counter)
        .unwrap();
    let results_acento2: Vec<_> = index_disabled
        .filter_query(query_acento2, &hw_counter)
        .collect();
    assert!(results_acento2.contains(&0));
}

#[test]
fn test_fuzzy_parse_requires_candidates_for_each_token() {
    let hw_counter = HardwareCounterCell::default();

    let temp_dir = Builder::new().prefix("test_dir").tempdir().unwrap();
    let config = TextIndexParams {
        r#type: TextIndexType::Text,
        tokenizer: TokenizerType::default(),
        min_token_len: None,
        max_token_len: None,
        lowercase: Some(true),
        on_disk: None,
        phrase_matching: Some(true),
        fuzzy_matching: Some(true),
        stopwords: None,
        stemmer: None,
        ascii_folding: None,
        enable_hnsw: None,
    };

    let mut index = FullTextIndex::new_gridstore(temp_dir.path().to_path_buf(), config, true)
        .unwrap()
        .unwrap();

    index
        .add_many(0, vec!["quick brown fox".to_string()], &hw_counter)
        .unwrap();
    index
        .add_many(1, vec!["quick red dog".to_string()], &hw_counter)
        .unwrap();

    let params = FuzzyParams {
        max_edit: 1,
        prefix_length: 0,
        max_expansions: 30,
    };

    let fuzzy_text = MatchFuzzy {
        fuzzy: Fuzzy::Text {
            text: "quick zzzzzzz".to_string(),
            params: Some(params.clone()),
        },
    };
    assert!(index.parse_fuzzy_query(&fuzzy_text, &hw_counter).is_none());

    let fuzzy_phrase = MatchFuzzy {
        fuzzy: Fuzzy::Phrase {
            phrase: "quick zzzzzzz".to_string(),
            params: Some(params.clone()),
        },
    };
    assert!(
        index
            .parse_fuzzy_query(&fuzzy_phrase, &hw_counter)
            .is_none()
    );

    let fuzzy_text_any = MatchFuzzy {
        fuzzy: Fuzzy::TextAny {
            text_any: "quick zzzzzzz".to_string(),
            params: Some(params),
        },
    };
    assert!(
        index
            .parse_fuzzy_query(&fuzzy_text_any, &hw_counter)
            .is_some()
    );
}

#[test]
fn test_fuzzy_phrase_preserves_order() {
    let hw_counter = HardwareCounterCell::default();

    let temp_dir = Builder::new().prefix("test_dir").tempdir().unwrap();
    let config = TextIndexParams {
        r#type: TextIndexType::Text,
        tokenizer: TokenizerType::default(),
        min_token_len: None,
        max_token_len: None,
        lowercase: Some(true),
        on_disk: None,
        phrase_matching: Some(true),
        fuzzy_matching: Some(true),
        stopwords: None,
        stemmer: None,
        ascii_folding: None,
        enable_hnsw: None,
    };

    let mut index = FullTextIndex::new_gridstore(temp_dir.path().to_path_buf(), config, true)
        .unwrap()
        .unwrap();

    index
        .add_many(
            0,
            vec!["the quick brown fox jumps".to_string()],
            &hw_counter,
        )
        .unwrap();
    index
        .add_many(
            1,
            vec!["the brown quick fox jumps".to_string()],
            &hw_counter,
        )
        .unwrap();
    index
        .add_many(2, vec!["the quick red fox jumps".to_string()], &hw_counter)
        .unwrap();

    let fuzzy_phrase = MatchFuzzy {
        fuzzy: Fuzzy::Phrase {
            phrase: "quik brown fox".to_string(),
            params: Some(FuzzyParams {
                max_edit: 1,
                prefix_length: 0,
                max_expansions: 30,
            }),
        },
    };

    let query = index
        .parse_fuzzy_query(&fuzzy_phrase, &hw_counter)
        .expect("fuzzy phrase query should parse");

    assert!(index.fuzzy_check_match(&query, 0));
    assert!(!index.fuzzy_check_match(&query, 1));
    assert!(!index.fuzzy_check_match(&query, 2));

    let results: Vec<_> = index.fuzzy_filter_query(query, &hw_counter).collect();
    assert_eq!(results, vec![0]);
}

/// Integration test covering 3 index types (mutable / immutable / mmap) × 3 query modes
/// (text, text_any, phrase) with both exact and fuzzy variants.
#[test]
fn test_all_index_types_all_query_modes() {
    let hw = HardwareCounterCell::default();

    let config = TextIndexParams {
        r#type: TextIndexType::Text,
        tokenizer: TokenizerType::default(),
        min_token_len: None,
        max_token_len: None,
        lowercase: Some(true),
        on_disk: None,
        phrase_matching: Some(true),
        fuzzy_matching: Some(true),
        stopwords: None,
        stemmer: None,
        ascii_folding: None,
        enable_hnsw: None,
    };

    // Shared corpus ──────────────────────────────────────────────
    let docs: Vec<(PointOffsetType, &str)> = vec![
        (0, "the quick brown fox jumps over the lazy dog"),
        (1, "brown fox quick the jumps over lazy dog"),
        (2, "quick brown fox runs fast"),
        (3, "the lazy dog sleeps peacefully"),
        (4, "the brown brown fox"),
        (5, "a completely unrelated sentence about space"),
        // Added for fuzzy multi-expansion:
        (6, "put it in the box"),
        (7, "read a book"),
        (8, "look back"),
        // Added for max_expansions:
        (10, "fuzzytermA"),
        (11, "fuzzytermB"),
        (12, "fuzzytermC"),
        (13, "fuzzytermD"),
        (14, "fuzzytermE"),
    ];

    // Helper: build all 3 index variants with the same data ─────
    let build_indexes = || {
        let dir_mut = Builder::new().prefix("idx_mut").tempdir().unwrap();
        let mut mutable =
            FullTextIndex::new_gridstore(dir_mut.path().to_path_buf(), config.clone(), true)
                .unwrap()
                .unwrap();
        for &(id, text) in &docs {
            mutable.add_many(id, vec![text.to_string()], &hw).unwrap();
        }

        let dir_imm = Builder::new().prefix("idx_imm").tempdir().unwrap();
        let mut imm_builder =
            FullTextIndex::builder_mmap(dir_imm.path().to_path_buf(), config.clone(), false);
        imm_builder.init().unwrap();
        for &(id, text) in &docs {
            imm_builder
                .add_many(id, vec![text.to_string()], &hw)
                .unwrap();
        }
        let immutable = imm_builder.finalize().unwrap();

        let dir_mmap = Builder::new().prefix("idx_mmap").tempdir().unwrap();
        let mut mmap_builder =
            FullTextIndex::builder_mmap(dir_mmap.path().to_path_buf(), config.clone(), true);
        mmap_builder.init().unwrap();
        for &(id, text) in &docs {
            mmap_builder
                .add_many(id, vec![text.to_string()], &hw)
                .unwrap();
        }
        let mmap = mmap_builder.finalize().unwrap();

        vec![(mutable, dir_mut), (immutable, dir_imm), (mmap, dir_mmap)]
    };

    let indexes = build_indexes();

    for (idx, (index, _dir)) in indexes.iter().enumerate() {
        let label = match idx {
            0 => "mutable",
            1 => "immutable",
            _ => "mmap",
        };

        {
            let q = index
                .parse_text_query("quick brown fox", &hw)
                .unwrap_or_else(|| panic!("[{label}] text query should parse"));

            // docs 0, 1, 2 contain all three tokens
            assert!(index.check_match(&q, 0), "[{label}] text match doc 0");
            assert!(index.check_match(&q, 1), "[{label}] text match doc 1");
            assert!(index.check_match(&q, 2), "[{label}] text match doc 2");
            assert!(!index.check_match(&q, 3), "[{label}] text !match doc 3");
            assert!(!index.check_match(&q, 5), "[{label}] text !match doc 5");

            let hits: Vec<_> = index.filter_query(q, &hw).collect();
            assert_eq!(hits.len(), 3, "[{label}] text filter count");
        }

        assert!(
            index.parse_text_query("xyznonexist", &hw).is_none(),
            "[{label}] text unknown token returns None"
        );

        {
            let q = index
                .parse_text_any_query("fox sleeps xyznonexist", &hw)
                .unwrap_or_else(|| panic!("[{label}] text_any query should parse"));

            assert!(index.check_match(&q, 0), "[{label}] any match doc 0");
            assert!(index.check_match(&q, 3), "[{label}] any match doc 3");
            assert!(!index.check_match(&q, 5), "[{label}] any !match doc 5");

            let hits: Vec<_> = index.filter_query(q, &hw).collect();
            assert_eq!(hits.len(), 5, "[{label}] text_any filter count");
        }

        {
            let q = index
                .parse_phrase_query("quick brown fox", &hw)
                .unwrap_or_else(|| panic!("[{label}] phrase query should parse"));

            assert!(index.check_match(&q, 0), "[{label}] phrase match doc 0");
            assert!(index.check_match(&q, 2), "[{label}] phrase match doc 2");
            assert!(
                !index.check_match(&q, 1),
                "[{label}] phrase !match doc 1 (wrong order)"
            );

            let hits: Vec<_> = index.filter_query(q, &hw).collect();
            assert_eq!(hits.len(), 2, "[{label}] phrase filter count");
        }

        {
            let mf = MatchFuzzy {
                fuzzy: Fuzzy::Text {
                    text: "quik brwn fox".to_string(), // typos in quick & brown
                    params: Some(FuzzyParams {
                        max_edit: 1,
                        prefix_length: 0,
                        max_expansions: 30,
                    }),
                },
            };
            let q = index
                .parse_fuzzy_query(&mf, &hw)
                .unwrap_or_else(|| panic!("[{label}] fuzzy text should parse"));

            assert!(index.fuzzy_check_match(&q, 0), "[{label}] fuz-text doc 0");
            assert!(index.fuzzy_check_match(&q, 1), "[{label}] fuz-text doc 1");
            assert!(index.fuzzy_check_match(&q, 2), "[{label}] fuz-text doc 2");
            assert!(!index.fuzzy_check_match(&q, 5), "[{label}] fuz-text !doc 5");

            let hits: Vec<_> = index.fuzzy_filter_query(q, &hw).collect();
            assert_eq!(hits.len(), 3, "[{label}] fuzzy text filter count");
        }

        {
            let mf = MatchFuzzy {
                fuzzy: Fuzzy::Text {
                    text: "zzzzz yyyyy".to_string(),
                    params: Some(FuzzyParams {
                        max_edit: 1,
                        prefix_length: 0,
                        max_expansions: 30,
                    }),
                },
            };
            assert!(
                index.parse_fuzzy_query(&mf, &hw).is_none(),
                "[{label}] fuzzy text no candidates → None"
            );
        }

        {
            let mf = MatchFuzzy {
                fuzzy: Fuzzy::TextAny {
                    text_any: "quik zzzzzzz".to_string(), // "quik"→quick, "zzzzzzz"→nothing
                    params: Some(FuzzyParams {
                        max_edit: 1,
                        prefix_length: 0,
                        max_expansions: 30,
                    }),
                },
            };
            let q = index
                .parse_fuzzy_query(&mf, &hw)
                .unwrap_or_else(|| panic!("[{label}] fuzzy text_any should parse"));

            assert!(index.fuzzy_check_match(&q, 0), "[{label}] fuz-any doc 0");
            assert!(!index.fuzzy_check_match(&q, 3), "[{label}] fuz-any !doc 3");

            let hits: Vec<_> = index.fuzzy_filter_query(q, &hw).collect();
            assert_eq!(hits.len(), 3, "[{label}] fuzzy text_any filter count");
        }

        {
            let mf = MatchFuzzy {
                fuzzy: Fuzzy::Phrase {
                    phrase: "quik brwn fox".to_string(),
                    params: Some(FuzzyParams {
                        max_edit: 1,
                        prefix_length: 0,
                        max_expansions: 30,
                    }),
                },
            };
            let q = index
                .parse_fuzzy_query(&mf, &hw)
                .unwrap_or_else(|| panic!("[{label}] fuzzy phrase should parse"));

            assert!(index.fuzzy_check_match(&q, 0), "[{label}] fuz-phrase doc 0");
            assert!(index.fuzzy_check_match(&q, 2), "[{label}] fuz-phrase doc 2");
            assert!(
                !index.fuzzy_check_match(&q, 1),
                "[{label}] fuz-phrase !doc 1 (wrong order)"
            );
            assert!(
                !index.fuzzy_check_match(&q, 5),
                "[{label}] fuz-phrase !doc 5"
            );

            let hits: Vec<_> = index.fuzzy_filter_query(q, &hw).collect();
            assert_eq!(hits.len(), 2, "[{label}] fuzzy phrase filter count");
        }

        {
            let mf = MatchFuzzy {
                fuzzy: Fuzzy::Text {
                    text: "bok".to_string(),
                    params: Some(FuzzyParams {
                        max_edit: 1,
                        prefix_length: 0,
                        max_expansions: 30,
                    }),
                },
            };
            let q = index
                .parse_fuzzy_query(&mf, &hw)
                .unwrap_or_else(|| panic!("[{label}] fuzzy bok should parse"));

            let hits: Vec<_> = index.fuzzy_filter_query(q, &hw).collect();
            assert!(hits.contains(&6), "[{label}] bok matches box");
            assert!(hits.contains(&7), "[{label}] bok matches book");
        }

        {
            let mf_loose = MatchFuzzy {
                fuzzy: Fuzzy::Text {
                    text: "brwn".to_string(),
                    params: Some(FuzzyParams {
                        max_edit: 1,
                        prefix_length: 2,
                        max_expansions: 30,
                    }),
                },
            };
            let q_loose = index.parse_fuzzy_query(&mf_loose, &hw).unwrap();
            assert!(
                index.fuzzy_check_match(&q_loose, 0),
                "[{label}] prefix=2 matches brown"
            );

            let mf_strict = MatchFuzzy {
                fuzzy: Fuzzy::Text {
                    text: "brwn".to_string(),
                    params: Some(FuzzyParams {
                        max_edit: 1,
                        prefix_length: 3,
                        max_expansions: 30,
                    }),
                },
            };
            if let Some(q_strict) = index.parse_fuzzy_query(&mf_strict, &hw) {
                assert!(
                    !index.fuzzy_check_match(&q_strict, 0),
                    "[{label}] prefix=3 !matches brown"
                );
            }
        }

        {
            let mf = MatchFuzzy {
                fuzzy: Fuzzy::Text {
                    text: "fuzzyterm".to_string(),
                    params: Some(FuzzyParams {
                        max_edit: 1,
                        prefix_length: 0,
                        max_expansions: 2,
                    }),
                },
            };
            let q = index.parse_fuzzy_query(&mf, &hw).unwrap();

            let hits: Vec<_> = index.fuzzy_filter_query(q, &hw).collect();
            assert_eq!(
                hits.len(),
                2,
                "[{label}] max_expansions=2 should limit results"
            );
        }
    }
}
