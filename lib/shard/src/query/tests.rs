use ahash::AHashSet;
use ordered_float::OrderedFloat;
use segment::common::operation_error::OperationError;
use segment::common::reciprocal_rank_fusion::DEFAULT_RRF_K;
use segment::data_types::vectors::{MultiDenseVectorInternal, NamedQuery, VectorInternal};
use segment::json_path::JsonPath;
use segment::types::*;
use sparse::common::sparse_vector::SparseVector;

use super::planned_query::*;
use super::*;

#[test]
fn test_try_from_double_rescore() {
    let dummy_vector = vec![1.0, 2.0, 3.0];
    let filter_inner_inner = Filter::new_must_not(Condition::IsNull(
        JsonPath::try_from("apples").unwrap().into(),
    ));
    let filter_inner = Filter::new_must(Condition::Field(FieldCondition::new_match(
        "has_oranges".try_into().unwrap(),
        true.into(),
    )));
    let filter_outer = Filter::new_must(Condition::HasId(
        AHashSet::from([1.into(), 2.into()]).into(),
    ));

    let request = ShardQueryRequest {
        prefetches: vec![ShardPrefetch {
            prefetches: vec![ShardPrefetch {
                prefetches: Default::default(),
                query: Some(ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery::new(
                    VectorInternal::Dense(dummy_vector.clone()),
                    "byte",
                )))),
                limit: 1000,
                params: None,
                filter: Some(filter_inner_inner.clone()),
                score_threshold: None,
            }],
            query: Some(ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery::new(
                VectorInternal::Dense(dummy_vector.clone()),
                "full",
            )))),
            limit: 100,
            params: None,
            filter: Some(filter_inner.clone()),
            score_threshold: None,
        }],
        query: Some(ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery::new(
            VectorInternal::MultiDense(MultiDenseVectorInternal::new_unchecked(vec![
                dummy_vector.clone(),
            ])),
            "multi",
        )))),
        filter: Some(filter_outer.clone()),
        score_threshold: None,
        limit: 10,
        offset: 0,
        params: Some(SearchParams {
            exact: true,
            ..Default::default()
        }),
        with_vector: WithVector::Bool(true),
        with_payload: WithPayloadInterface::Bool(true),
        fuzzy_context: None,
    };

    let planned_query = PlannedQuery::try_from(vec![request]).unwrap();

    assert_eq!(
        planned_query.searches,
        vec![CoreSearchRequest {
            query: QueryEnum::Nearest(NamedQuery::new(
                VectorInternal::Dense(dummy_vector.clone()),
                "byte",
            )),
            filter: Some(
                filter_outer
                    .merge_owned(filter_inner)
                    .merge_owned(filter_inner_inner)
            ),
            params: None,
            limit: 1000,
            offset: 0,
            with_payload: Some(WithPayloadInterface::Bool(false)),
            with_vector: Some(WithVector::Bool(false)),
            score_threshold: None,
            fuzzy_context: None,
        }]
    );

    assert_eq!(
        planned_query.root_plans,
        vec![RootPlan {
            with_vector: WithVector::Bool(true),
            with_payload: WithPayloadInterface::Bool(true),
            merge_plan: MergePlan {
                sources: vec![Source::Prefetch(Box::from(MergePlan {
                    sources: vec![Source::SearchesIdx(0)],
                    rescore_stages: Some(RescoreStages::shard_level(RescoreParams {
                        rescore: ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery::new(
                            VectorInternal::Dense(dummy_vector.clone()),
                            "full",
                        ))),
                        limit: 100,
                        score_threshold: None,
                        params: None,
                    }))
                }))],
                rescore_stages: Some(RescoreStages::shard_level(RescoreParams {
                    rescore: ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery::new(
                        VectorInternal::MultiDense(MultiDenseVectorInternal::new_unchecked(vec![
                            dummy_vector
                        ])),
                        "multi"
                    ))),
                    limit: 10,
                    score_threshold: None,
                    params: Some(SearchParams {
                        exact: true,
                        ..Default::default()
                    })
                }))
            }
        }]
    );
}

#[test]
fn test_try_from_no_prefetch() {
    let dummy_vector = vec![1.0, 2.0, 3.0];
    let request = ShardQueryRequest {
        prefetches: vec![], // No prefetch
        query: Some(ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery::new(
            VectorInternal::Dense(dummy_vector.clone()),
            "full",
        )))),
        filter: Some(Filter::default()),
        score_threshold: Some(OrderedFloat(0.5)),
        limit: 10,
        offset: 12,
        params: Some(SearchParams::default()),
        with_vector: WithVector::Bool(true),
        with_payload: WithPayloadInterface::Bool(true),
        fuzzy_context: None,
    };

    let planned_query = PlannedQuery::try_from(vec![request]).unwrap();

    assert_eq!(
        planned_query.searches,
        vec![CoreSearchRequest {
            query: QueryEnum::Nearest(
                NamedQuery::new(VectorInternal::Dense(dummy_vector), "full",)
            ),
            filter: Some(Filter::default()),
            params: Some(SearchParams::default()),
            limit: 22,
            offset: 0,
            with_vector: Some(WithVector::Bool(false)),
            with_payload: Some(WithPayloadInterface::Bool(false)),
            score_threshold: Some(0.5),
            fuzzy_context: None,
        }]
    );

    assert_eq!(
        planned_query.root_plans,
        vec![RootPlan {
            with_payload: WithPayloadInterface::Bool(true),
            with_vector: WithVector::Bool(true),
            merge_plan: MergePlan {
                sources: vec![Source::SearchesIdx(0)],
                rescore_stages: None,
            },
        }]
    );
}

#[test]
fn test_try_from_hybrid_query() {
    let dummy_vector = vec![1.0, 2.0, 3.0];
    let dummy_sparse = SparseVector::new(vec![100, 123, 2000], vec![0.2, 0.3, 0.4]).unwrap();

    let filter_inner1 = Filter::new_must(Condition::Field(FieldCondition::new_match(
        "city".try_into().unwrap(),
        "Berlin".to_string().into(),
    )));
    let filter_inner2 = Filter::new_must(Condition::Field(FieldCondition::new_match(
        "city".try_into().unwrap(),
        "Munich".to_string().into(),
    )));
    let filter_outer = Filter::new_must(Condition::Field(FieldCondition::new_match(
        "country".try_into().unwrap(),
        "Germany".to_string().into(),
    )));

    let request = ShardQueryRequest {
        prefetches: vec![
            ShardPrefetch {
                prefetches: Vec::new(),
                query: Some(ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery::new(
                    VectorInternal::Dense(dummy_vector.clone()),
                    "dense",
                )))),
                limit: 100,
                params: None,
                filter: Some(filter_inner1.clone()),
                score_threshold: None,
            },
            ShardPrefetch {
                prefetches: Vec::new(),
                query: Some(ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery::new(
                    VectorInternal::Sparse(dummy_sparse.clone()),
                    "sparse",
                )))),
                limit: 100,
                params: None,
                filter: Some(filter_inner2.clone()),
                score_threshold: None,
            },
        ],
        query: Some(ScoringQuery::Fusion(FusionInternal::Rrf {
            k: DEFAULT_RRF_K,
            weights: None,
        })),
        filter: Some(filter_outer.clone()),
        score_threshold: None,
        limit: 50,
        offset: 0,
        params: None,
        with_payload: WithPayloadInterface::Bool(false),
        with_vector: WithVector::Bool(true),
        fuzzy_context: None,
    };

    let planned_query = PlannedQuery::try_from(vec![request]).unwrap();

    assert_eq!(
        planned_query.searches,
        vec![
            CoreSearchRequest {
                query: QueryEnum::Nearest(NamedQuery::new(
                    VectorInternal::Dense(dummy_vector),
                    "dense",
                )),
                filter: Some(filter_outer.merge(&filter_inner1)),
                params: None,
                limit: 100,
                offset: 0,
                with_payload: Some(WithPayloadInterface::Bool(false)),
                with_vector: Some(WithVector::Bool(false)),
                score_threshold: None,
                fuzzy_context: None,
            },
            CoreSearchRequest {
                query: QueryEnum::Nearest(NamedQuery::new(
                    VectorInternal::Sparse(dummy_sparse),
                    "sparse",
                )),
                filter: Some(filter_outer.merge(&filter_inner2)),
                params: None,
                limit: 100,
                offset: 0,
                with_payload: Some(WithPayloadInterface::Bool(false)),
                with_vector: Some(WithVector::Bool(false)),
                score_threshold: None,
                fuzzy_context: None,
            }
        ]
    );

    assert_eq!(
        planned_query.root_plans,
        vec![RootPlan {
            with_payload: WithPayloadInterface::Bool(false),
            with_vector: WithVector::Bool(true),
            merge_plan: MergePlan {
                sources: vec![Source::SearchesIdx(0), Source::SearchesIdx(1)],
                rescore_stages: Some(RescoreStages::collection_level(RescoreParams {
                    rescore: ScoringQuery::Fusion(FusionInternal::Rrf {
                        k: DEFAULT_RRF_K,
                        weights: None
                    }),
                    limit: 50,
                    score_threshold: None,
                    params: None,
                }))
            }
        }]
    );
}

#[test]
fn test_try_from_rrf_without_source() {
    let request = ShardQueryRequest {
        prefetches: vec![],
        query: Some(ScoringQuery::Fusion(FusionInternal::Rrf {
            k: DEFAULT_RRF_K,
            weights: None,
        })),
        filter: Some(Filter::default()),
        score_threshold: None,
        limit: 50,
        offset: 0,
        params: None,
        with_vector: WithVector::Bool(true),
        with_payload: WithPayloadInterface::Bool(false),
        fuzzy_context: None,
    };

    let planned_query = PlannedQuery::try_from(vec![request]);

    assert!(planned_query.is_err())
}

#[test]
fn test_base_params_mapping_in_try_from() {
    let dummy_vector = vec![1.0, 2.0, 3.0];
    let dummy_params = Some(SearchParams {
        indexed_only: true,
        ..Default::default()
    });
    let dummy_filter = Some(Filter::new_must(Condition::Field(
        FieldCondition::new_match(
            "my_key".try_into().unwrap(),
            Match::new_value(segment::types::ValueVariants::String("hello".to_string())),
        ),
    )));

    let top_level_params = Some(SearchParams {
        exact: true,
        ..Default::default()
    });

    let request = ShardQueryRequest {
        prefetches: vec![ShardPrefetch {
            prefetches: Vec::new(),
            query: Some(ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery::new(
                VectorInternal::Dense(dummy_vector.clone()),
                "dense",
            )))),
            limit: 37,
            params: dummy_params,
            filter: dummy_filter.clone(),
            score_threshold: Some(OrderedFloat(0.1)),
        }],
        query: Some(ScoringQuery::Fusion(FusionInternal::Rrf {
            k: DEFAULT_RRF_K,
            weights: None,
        })),
        filter: Some(Filter::default()),
        score_threshold: Some(OrderedFloat(0.666)),
        limit: 50,
        offset: 49,

        // these params will be ignored because we have a prefetch
        params: top_level_params,
        with_payload: WithPayloadInterface::Bool(true),
        with_vector: WithVector::Bool(false),
        fuzzy_context: None,
    };

    let planned_query = PlannedQuery::try_from(vec![request]).unwrap();

    assert_eq!(
        planned_query.root_plans,
        vec![RootPlan {
            with_payload: WithPayloadInterface::Bool(true),
            with_vector: WithVector::Bool(false),
            merge_plan: MergePlan {
                sources: vec![Source::SearchesIdx(0)],
                rescore_stages: Some(RescoreStages::collection_level(RescoreParams {
                    rescore: ScoringQuery::Fusion(FusionInternal::Rrf {
                        k: DEFAULT_RRF_K,
                        weights: None
                    }),
                    limit: 99,
                    score_threshold: Some(OrderedFloat(0.666)),
                    params: top_level_params,
                }))
            }
        }]
    );

    assert_eq!(
        planned_query.searches,
        vec![CoreSearchRequest {
            query: QueryEnum::Nearest(NamedQuery::new(
                VectorInternal::Dense(dummy_vector),
                "dense",
            ),),
            filter: dummy_filter,
            params: dummy_params,
            limit: 37,
            offset: 0,
            with_payload: Some(WithPayloadInterface::Bool(false)),
            with_vector: Some(WithVector::Bool(false)),
            score_threshold: Some(0.1),
            fuzzy_context: None,
        }]
    )
}

pub fn make_prefetches_at_depth(depth: usize) -> ShardPrefetch {
    // recursive helper for accumulation
    pub fn make_prefetches_at_depth_acc(depth: usize, acc: ShardPrefetch) -> ShardPrefetch {
        if depth == 0 {
            acc
        } else {
            make_prefetches_at_depth_acc(
                depth - 1,
                ShardPrefetch {
                    prefetches: vec![acc],
                    query: Some(ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery::new(
                        VectorInternal::Dense(vec![1.0, 2.0, 3.0]),
                        "dense",
                    )))),
                    limit: 10,
                    params: None,
                    filter: None,
                    score_threshold: None,
                },
            )
        }
    }
    // lowest prefetch
    let prefetch = ShardPrefetch {
        prefetches: Vec::new(),
        query: Some(ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery::new(
            VectorInternal::Dense(vec![1.0, 2.0, 3.0]),
            "dense",
        )))),
        limit: 100,
        params: None,
        filter: None,
        score_threshold: None,
    };
    make_prefetches_at_depth_acc(depth - 1, prefetch)
}

#[test]
fn test_detect_max_depth() {
    // depth 0
    let mut request = ShardQueryRequest {
        prefetches: vec![],
        query: Some(ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery::new(
            VectorInternal::Dense(vec![1.0, 2.0, 3.0]),
            "dense",
        )))),
        filter: None,
        score_threshold: None,
        limit: 10,
        offset: 0,
        params: None,
        with_vector: WithVector::Bool(true),
        with_payload: WithPayloadInterface::Bool(false),
        fuzzy_context: None,
    };
    assert_eq!(request.prefetches_depth(), 0);

    // depth 3
    request.prefetches = vec![ShardPrefetch {
        prefetches: vec![ShardPrefetch {
            prefetches: vec![ShardPrefetch {
                prefetches: vec![],
                query: Some(ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery::new(
                    VectorInternal::Dense(vec![1.0, 2.0, 3.0]),
                    "dense",
                )))),
                limit: 10,
                params: None,
                filter: None,
                score_threshold: None,
            }],
            query: Some(ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery::new(
                VectorInternal::Dense(vec![1.0, 2.0, 3.0]),
                "dense",
            )))),
            limit: 10,
            params: None,
            filter: None,
            score_threshold: None,
        }],
        query: Some(ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery::new(
            VectorInternal::Dense(vec![1.0, 2.0, 3.0]),
            "dense",
        )))),
        limit: 10,
        params: None,
        filter: None,
        score_threshold: None,
    }];
    assert_eq!(request.prefetches_depth(), 3);

    // use with helper for less boilerplate
    request.prefetches = vec![make_prefetches_at_depth(3)];
    assert_eq!(request.prefetches_depth(), 3);
    let _planned_query = PlannedQuery::try_from(vec![request.clone()]).unwrap();

    request.prefetches = vec![make_prefetches_at_depth(64)];
    assert_eq!(request.prefetches_depth(), 64);
    let _planned_query = PlannedQuery::try_from(vec![request.clone()]).unwrap();

    request.prefetches = vec![make_prefetches_at_depth(65)];
    assert_eq!(request.prefetches_depth(), 65);

    // assert error
    assert!(matches!(
        PlannedQuery::try_from(vec![request]),
        Err(OperationError::ValidationError { description }) if description == "prefetches depth 65 exceeds max depth 64",
    ));
}

fn dummy_core_prefetch(limit: usize) -> ShardPrefetch {
    ShardPrefetch {
        prefetches: vec![],
        query: Some(nearest_query()),
        filter: None,
        params: None,
        limit,
        score_threshold: None,
    }
}

fn dummy_scroll_prefetch(limit: usize) -> ShardPrefetch {
    ShardPrefetch {
        prefetches: vec![],
        query: None,
        limit,
        params: None,
        filter: None,
        score_threshold: None,
    }
}

fn nearest_query() -> ScoringQuery {
    ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery::default_dense(vec![
        0.1, 0.2, 0.3, 0.4,
    ])))
}

#[test]
fn test_from_batch_of_requests() {
    let requests = vec![
        // A no-prefetch core_search query
        ShardQueryRequest {
            prefetches: vec![],
            query: Some(nearest_query()),
            filter: None,
            score_threshold: None,
            limit: 10,
            offset: 0,
            params: None,
            with_payload: WithPayloadInterface::Bool(false),
            with_vector: WithVector::Bool(false),
            fuzzy_context: None,
        },
        // A no-prefetch scroll query
        ShardQueryRequest {
            prefetches: vec![],
            query: None,
            filter: None,
            score_threshold: None,
            limit: 20,
            offset: 0,
            params: None,
            with_payload: WithPayloadInterface::Bool(false),
            with_vector: WithVector::Bool(false),
            fuzzy_context: None,
        },
        // A double fusion query
        ShardQueryRequest {
            prefetches: vec![
                ShardPrefetch {
                    prefetches: vec![dummy_core_prefetch(30), dummy_core_prefetch(40)],
                    query: Some(ScoringQuery::Fusion(FusionInternal::Rrf {
                        k: DEFAULT_RRF_K,
                        weights: None,
                    })),
                    filter: None,
                    params: None,
                    score_threshold: None,
                    limit: 10,
                },
                dummy_scroll_prefetch(50),
            ],
            query: Some(ScoringQuery::Fusion(FusionInternal::Rrf {
                k: DEFAULT_RRF_K,
                weights: None,
            })),
            filter: None,
            score_threshold: None,
            limit: 10,
            offset: 0,
            params: None,
            with_payload: WithPayloadInterface::Bool(true),
            with_vector: WithVector::Bool(true),
            fuzzy_context: None,
        },
    ];

    let planned_query = PlannedQuery::try_from(requests).unwrap();
    assert_eq!(planned_query.searches.len(), 3);
    assert_eq!(planned_query.scrolls.len(), 2);
    assert_eq!(planned_query.root_plans.len(), 3);

    assert_eq!(
        planned_query.root_plans,
        vec![
            RootPlan {
                with_vector: WithVector::Bool(false),
                with_payload: WithPayloadInterface::Bool(false),
                merge_plan: MergePlan {
                    sources: vec![Source::SearchesIdx(0)],
                    rescore_stages: None,
                },
            },
            RootPlan {
                with_vector: WithVector::Bool(false),
                with_payload: WithPayloadInterface::Bool(false),
                merge_plan: MergePlan {
                    sources: vec![Source::ScrollsIdx(0)],
                    rescore_stages: None,
                },
            },
            RootPlan {
                with_vector: WithVector::Bool(true),
                with_payload: WithPayloadInterface::Bool(true),
                merge_plan: MergePlan {
                    sources: vec![
                        Source::Prefetch(Box::from(MergePlan {
                            sources: vec![Source::SearchesIdx(1), Source::SearchesIdx(2),],
                            rescore_stages: Some(RescoreStages::shard_level(RescoreParams {
                                rescore: ScoringQuery::Fusion(FusionInternal::Rrf {
                                    k: DEFAULT_RRF_K,
                                    weights: None
                                }),
                                limit: 10,
                                score_threshold: None,
                                params: None,
                            })),
                        })),
                        Source::ScrollsIdx(1),
                    ],
                    rescore_stages: Some(RescoreStages::collection_level(RescoreParams {
                        rescore: ScoringQuery::Fusion(FusionInternal::Rrf {
                            k: DEFAULT_RRF_K,
                            weights: None
                        }),
                        limit: 10,
                        score_threshold: None,
                        params: None,
                    })),
                },
            },
        ]
    );

    assert_eq!(planned_query.searches[0].limit, 10);
    assert_eq!(planned_query.searches[1].limit, 30);
    assert_eq!(planned_query.searches[2].limit, 40);

    assert_eq!(planned_query.scrolls[0].limit, 20);
    assert_eq!(planned_query.scrolls[1].limit, 50);
}

// ===================================================================
// BM25 Fuzzy Resolve Tests
// ===================================================================
//
// These tests exercise `resolve_fuzzy_intent()` — the core algorithm that
// expands query tokens across all segments and builds a SparseVector.
//
// Test setup:
//   1. Build a real Segment with text payloads & a fuzzy-enabled FullTextIndex
//   2. Wrap it in SegmentHolder → LockedSegmentHolder
//   3. Construct FuzzyBm25Intent with known tokens
//   4. Call resolve_fuzzy_intent() and assert on the returned SparseVector

mod fuzzy_resolve_tests {
    use std::time::Duration;

    use common::counter::hardware_counter::HardwareCounterCell;
    use segment::data_types::index::{TextIndexParams, TextIndexType};
    use segment::data_types::vectors::only_default_vector;
    use segment::entry::{NonAppendableSegmentEntry, SegmentEntry};
    use segment::index::sparse_index::bm25_fuzzy_expander::{
        bm25_token_to_dim_id, compute_fuzzy_boost,
    };
    use segment::json_path::JsonPath;
    use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
    use segment::types::{Distance, FuzzyParams, PayloadFieldSchema, PayloadSchemaParams};
    use tempfile::Builder;

    use crate::query::fuzzy_resolve::{FuzzyBm25Intent, resolve_fuzzy_intent};
    use crate::segment_holder::SegmentHolder;
    use crate::segment_holder::locked::LockedSegmentHolder;

    /// Build a SegmentHolder with one segment containing text payloads
    /// and a fuzzy-enabled FullTextIndex.
    fn build_test_holder(dir: &std::path::Path, texts: &[&str]) -> LockedSegmentHolder {
        let hw = HardwareCounterCell::disposable();

        let mut segment = build_simple_segment(dir, 4, Distance::Dot).unwrap();

        // Upsert points with text payloads
        for (i, text) in texts.iter().enumerate() {
            let point_id = (i as u64 + 1).into();
            segment
                .upsert_point(
                    (i + 1) as u64,
                    point_id,
                    only_default_vector(&[1.0, 0.0, 0.0, 0.0]),
                    &hw,
                )
                .unwrap();
            let payload: segment::types::Payload =
                serde_json::from_str(&format!(r#"{{"text_field": "{}"}}"#, text)).unwrap();
            segment
                .set_payload((i + 1) as u64, point_id, &payload, &None, &hw)
                .unwrap();
        }

        // Create text index with fuzzy enabled
        segment
            .create_field_index(
                (texts.len() + 1) as u64,
                &JsonPath::new("text_field"),
                Some(&PayloadFieldSchema::FieldParams(PayloadSchemaParams::Text(
                    TextIndexParams {
                        r#type: TextIndexType::Text,
                        fuzzy_matching: Some(true),
                        lowercase: Some(true),
                        ..Default::default()
                    },
                ))),
                &hw,
            )
            .unwrap();

        let mut holder = SegmentHolder::default();
        holder.add_new(segment);
        LockedSegmentHolder::new(holder)
    }

    fn default_fuzzy_params() -> FuzzyParams {
        FuzzyParams {
            max_edit: 1,
            prefix_length: 0,
            max_expansions: 30,
        }
    }

    #[test]
    fn test_resolve_exact_token() {
        let dir = Builder::new()
            .prefix("fuzzy_resolve_exact")
            .tempdir()
            .unwrap();
        let holder = build_test_holder(
            dir.path(),
            &[
                "the quick brown fox",
                "a lazy brown dog",
                "the quick red rabbit",
            ],
        );

        let intent = FuzzyBm25Intent {
            vector_name: "default".to_string(),
            bind_field: JsonPath::new("text_field"),
            tokens: vec!["quick".to_string()],
            params: default_fuzzy_params(),
            stemmer: None,
        };

        let result = resolve_fuzzy_intent(&intent, &holder, Duration::from_secs(5)).unwrap();
        assert!(result.fuzzy_applied);
        assert!(!result.sparse_vector.indices.is_empty());

        // "quick" exact match → dim_id must be present with boost 1.0
        let quick_dim = bm25_token_to_dim_id("quick");
        let pos = result
            .sparse_vector
            .indices
            .iter()
            .position(|&d| d == quick_dim);
        assert!(pos.is_some(), "Expected dim_id for 'quick' in result");
        assert!((result.sparse_vector.values[pos.unwrap()] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_resolve_fuzzy_expansion() {
        let dir = Builder::new()
            .prefix("fuzzy_resolve_expand")
            .tempdir()
            .unwrap();
        let holder = build_test_holder(dir.path(), &["the quick brown fox", "a lazy brown dog"]);

        // "quik" (typo) should expand to "quick" with edit_distance=1
        let intent = FuzzyBm25Intent {
            vector_name: "default".to_string(),
            bind_field: JsonPath::new("text_field"),
            tokens: vec!["quik".to_string()],
            params: default_fuzzy_params(),
            stemmer: None,
        };

        let result = resolve_fuzzy_intent(&intent, &holder, Duration::from_secs(5)).unwrap();
        assert!(result.fuzzy_applied);

        // "quick" should appear as an expanded dim
        let quick_dim = bm25_token_to_dim_id("quick");
        let pos = result
            .sparse_vector
            .indices
            .iter()
            .position(|&d| d == quick_dim);
        assert!(pos.is_some(), "Expected 'quick' to be expanded from 'quik'");

        // Boost should be < 1.0 (since edit_distance > 0)
        let boost = result.sparse_vector.values[pos.unwrap()];
        assert!(boost > 0.0 && boost < 1.0);

        // Verify boost matches the formula
        let expected_boost = compute_fuzzy_boost("quik", "quick", 1).unwrap();
        assert!((boost - expected_boost).abs() < 1e-6);
    }

    #[test]
    fn test_resolve_no_match() {
        let dir = Builder::new()
            .prefix("fuzzy_resolve_nomatch")
            .tempdir()
            .unwrap();
        let holder = build_test_holder(dir.path(), &["hello world"]);

        // "zzzzzzz" has no fuzzy candidates in the vocabulary
        let intent = FuzzyBm25Intent {
            vector_name: "default".to_string(),
            bind_field: JsonPath::new("text_field"),
            tokens: vec!["zzzzzzz".to_string()],
            params: FuzzyParams {
                max_edit: 1,
                prefix_length: 0,
                max_expansions: 30,
            },
            stemmer: None,
        };

        let result = resolve_fuzzy_intent(&intent, &holder, Duration::from_secs(5)).unwrap();
        // FST was found (fuzzy_applied=true), but no candidates
        assert!(result.sparse_vector.indices.is_empty());
    }

    #[test]
    fn test_resolve_multi_token() {
        let dir = Builder::new()
            .prefix("fuzzy_resolve_multi")
            .tempdir()
            .unwrap();
        let holder = build_test_holder(
            dir.path(),
            &["the quick brown fox jumps", "a lazy brown dog sleeps"],
        );

        // Two tokens: "quik" → expands to "quick", "bron" → expands to "brown"
        let intent = FuzzyBm25Intent {
            vector_name: "default".to_string(),
            bind_field: JsonPath::new("text_field"),
            tokens: vec!["quik".to_string(), "bron".to_string()],
            params: default_fuzzy_params(),
            stemmer: None,
        };

        let result = resolve_fuzzy_intent(&intent, &holder, Duration::from_secs(5)).unwrap();
        assert!(result.fuzzy_applied);

        let quick_dim = bm25_token_to_dim_id("quick");
        let brown_dim = bm25_token_to_dim_id("brown");

        // Both should be present
        assert!(result.sparse_vector.indices.contains(&quick_dim));
        assert!(result.sparse_vector.indices.contains(&brown_dim));
    }

    #[test]
    fn test_resolve_empty_segments() {
        // No segments → no FST found, result is empty
        let holder = LockedSegmentHolder::new(SegmentHolder::default());

        let intent = FuzzyBm25Intent {
            vector_name: "default".to_string(),
            bind_field: JsonPath::new("text_field"),
            tokens: vec!["hello".to_string()],
            params: default_fuzzy_params(),
            stemmer: None,
        };

        let result = resolve_fuzzy_intent(&intent, &holder, Duration::from_secs(5)).unwrap();
        assert!(!result.fuzzy_applied);
        assert!(result.sparse_vector.indices.is_empty());
    }

    #[test]
    fn test_resolve_wrong_bind_field() {
        let dir = Builder::new()
            .prefix("fuzzy_resolve_wrongfield")
            .tempdir()
            .unwrap();
        let holder = build_test_holder(dir.path(), &["hello world"]);

        // Use wrong bind_field → no FullTextIndex found
        let intent = FuzzyBm25Intent {
            vector_name: "default".to_string(),
            bind_field: JsonPath::new("nonexistent_field"),
            tokens: vec!["hello".to_string()],
            params: default_fuzzy_params(),
            stemmer: None,
        };

        let result = resolve_fuzzy_intent(&intent, &holder, Duration::from_secs(5)).unwrap();
        // No candidates
        assert!(result.sparse_vector.indices.is_empty());
    }

    #[test]
    fn test_resolve_max_expansions_truncation() {
        let dir = Builder::new()
            .prefix("fuzzy_resolve_truncate")
            .tempdir()
            .unwrap();
        // Add many similar words to the vocabulary
        let holder = build_test_holder(dir.path(), &["cat car cab cap can", "cup cut cur cub cue"]);

        let intent = FuzzyBm25Intent {
            vector_name: "default".to_string(),
            bind_field: JsonPath::new("text_field"),
            tokens: vec!["cat".to_string()],
            params: FuzzyParams {
                max_edit: 1,
                prefix_length: 0,
                max_expansions: 2, // very restrictive
            },
            stemmer: None,
        };

        let result = resolve_fuzzy_intent(&intent, &holder, Duration::from_secs(5)).unwrap();
        assert!(result.fuzzy_applied);
        // Should be limited to at most 2 dims from this one token
        assert!(result.sparse_vector.indices.len() <= 2);
    }

    #[test]
    fn test_resolve_multi_segment_dedup() {
        // Two segments with overlapping vocabularies — candidates should be deduped
        let dir = Builder::new()
            .prefix("fuzzy_resolve_dedup")
            .tempdir()
            .unwrap();
        let hw = HardwareCounterCell::disposable();

        // Build first segment
        let mut seg1 =
            build_simple_segment(dir.path().join("seg1").as_path(), 4, Distance::Dot).unwrap();
        seg1.upsert_point(
            1,
            1u64.into(),
            only_default_vector(&[1.0, 0.0, 0.0, 0.0]),
            &hw,
        )
        .unwrap();
        let payload: segment::types::Payload =
            serde_json::from_str(r#"{"text_field": "the quick brown fox"}"#).unwrap();
        seg1.set_payload(2, 1u64.into(), &payload, &None, &hw)
            .unwrap();
        seg1.create_field_index(
            3,
            &JsonPath::new("text_field"),
            Some(&PayloadFieldSchema::FieldParams(PayloadSchemaParams::Text(
                TextIndexParams {
                    r#type: TextIndexType::Text,
                    fuzzy_matching: Some(true),
                    lowercase: Some(true),
                    ..Default::default()
                },
            ))),
            &hw,
        )
        .unwrap();

        // Build second segment with overlapping vocabulary
        let mut seg2 =
            build_simple_segment(dir.path().join("seg2").as_path(), 4, Distance::Dot).unwrap();
        seg2.upsert_point(
            1,
            2u64.into(),
            only_default_vector(&[0.0, 1.0, 0.0, 0.0]),
            &hw,
        )
        .unwrap();
        let payload2: segment::types::Payload =
            serde_json::from_str(r#"{"text_field": "the quick red rabbit"}"#).unwrap();
        seg2.set_payload(2, 2u64.into(), &payload2, &None, &hw)
            .unwrap();
        seg2.create_field_index(
            3,
            &JsonPath::new("text_field"),
            Some(&PayloadFieldSchema::FieldParams(PayloadSchemaParams::Text(
                TextIndexParams {
                    r#type: TextIndexType::Text,
                    fuzzy_matching: Some(true),
                    lowercase: Some(true),
                    ..Default::default()
                },
            ))),
            &hw,
        )
        .unwrap();

        let mut holder = SegmentHolder::default();
        holder.add_new(seg1);
        holder.add_new(seg2);
        let locked = LockedSegmentHolder::new(holder);

        let intent = FuzzyBm25Intent {
            vector_name: "default".to_string(),
            bind_field: JsonPath::new("text_field"),
            tokens: vec!["quik".to_string()],
            params: default_fuzzy_params(),
            stemmer: None,
        };

        let result = resolve_fuzzy_intent(&intent, &locked, Duration::from_secs(5)).unwrap();
        assert!(result.fuzzy_applied);

        // "quick" should appear exactly once (deduped)
        let quick_dim = bm25_token_to_dim_id("quick");
        let count = result
            .sparse_vector
            .indices
            .iter()
            .filter(|&&d| d == quick_dim)
            .count();
        assert_eq!(
            count, 1,
            "Expected 'quick' dim_id to appear exactly once (deduped)"
        );
    }

    #[test]
    fn test_resolve_boost_values_correctness() {
        let dir = Builder::new()
            .prefix("fuzzy_resolve_boost")
            .tempdir()
            .unwrap();
        let holder = build_test_holder(dir.path(), &["hello hallo hullo"]);

        // "hello" should match "hello" (exact), "hallo" (dist=1), "hullo" (dist=1)
        let intent = FuzzyBm25Intent {
            vector_name: "default".to_string(),
            bind_field: JsonPath::new("text_field"),
            tokens: vec!["hello".to_string()],
            params: FuzzyParams {
                max_edit: 1,
                prefix_length: 0,
                max_expansions: 30,
            },
            stemmer: None,
        };

        let result = resolve_fuzzy_intent(&intent, &holder, Duration::from_secs(5)).unwrap();
        assert!(result.fuzzy_applied);

        let hello_dim = bm25_token_to_dim_id("hello");
        let hallo_dim = bm25_token_to_dim_id("hallo");
        let hullo_dim = bm25_token_to_dim_id("hullo");

        // Check exact match
        if let Some(pos) = result
            .sparse_vector
            .indices
            .iter()
            .position(|&d| d == hello_dim)
        {
            assert!(
                (result.sparse_vector.values[pos] - 1.0).abs() < 1e-6,
                "Exact match should have boost 1.0"
            );
        }

        // Check fuzzy matches have boost = 0.8 (1 - 1/5)
        for dim in [hallo_dim, hullo_dim] {
            if let Some(pos) = result.sparse_vector.indices.iter().position(|&d| d == dim) {
                let expected = compute_fuzzy_boost("hello", "hallo", 1).unwrap(); // 0.8
                assert!(
                    (result.sparse_vector.values[pos] - expected).abs() < 1e-6,
                    "Fuzzy match should have boost {expected}, got {}",
                    result.sparse_vector.values[pos]
                );
            }
        }
    }

    #[test]
    fn test_resolve_prefix_length() {
        let dir = Builder::new()
            .prefix("fuzzy_resolve_prefix")
            .tempdir()
            .unwrap();
        let holder = build_test_holder(dir.path(), &["cat bat hat mat"]);

        // With prefix_length=1, "cat" can only match terms starting with "c"
        let intent = FuzzyBm25Intent {
            vector_name: "default".to_string(),
            bind_field: JsonPath::new("text_field"),
            tokens: vec!["cat".to_string()],
            params: FuzzyParams {
                max_edit: 1,
                prefix_length: 1, // first char must match
                max_expansions: 30,
            },
            stemmer: None,
        };

        let result = resolve_fuzzy_intent(&intent, &holder, Duration::from_secs(5)).unwrap();
        assert!(result.fuzzy_applied);

        // "cat" itself should be present
        let cat_dim = bm25_token_to_dim_id("cat");
        assert!(result.sparse_vector.indices.contains(&cat_dim));

        // "bat", "hat", "mat" should NOT be present (different first char)
        let bat_dim = bm25_token_to_dim_id("bat");
        let hat_dim = bm25_token_to_dim_id("hat");
        let mat_dim = bm25_token_to_dim_id("mat");
        assert!(
            !result.sparse_vector.indices.contains(&bat_dim),
            "prefix_length=1 should exclude 'bat'"
        );
        assert!(
            !result.sparse_vector.indices.contains(&hat_dim),
            "prefix_length=1 should exclude 'hat'"
        );
        assert!(
            !result.sparse_vector.indices.contains(&mat_dim),
            "prefix_length=1 should exclude 'mat'"
        );
    }

    #[test]
    fn test_resolve_max_edit_0() {
        let dir = Builder::new()
            .prefix("fuzzy_resolve_edit0")
            .tempdir()
            .unwrap();
        let holder = build_test_holder(dir.path(), &["quick brown fox"]);

        // max_edit=0 → only exact matches
        let intent = FuzzyBm25Intent {
            vector_name: "default".to_string(),
            bind_field: JsonPath::new("text_field"),
            tokens: vec!["quik".to_string()], // typo
            params: FuzzyParams {
                max_edit: 0,
                prefix_length: 0,
                max_expansions: 30,
            },
            stemmer: None,
        };

        let result = resolve_fuzzy_intent(&intent, &holder, Duration::from_secs(5)).unwrap();
        // With max_edit=0, "quik" should have no matches (it's not in vocab)
        let _quik_dim = bm25_token_to_dim_id("quik");
        let quick_dim = bm25_token_to_dim_id("quick");
        // "quik" is not in the vocab, so it shouldn't match anything
        // "quick" shouldn't match either because edit distance would be 1 > max_edit
        assert!(!result.sparse_vector.indices.contains(&quick_dim));
    }
}
