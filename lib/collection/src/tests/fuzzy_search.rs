//! Integration tests for the BM25 fuzzy search pipeline at collection level.
//!
//! Tests the full path through LocalShard:
//! 1. Create a local shard with sparse vectors + fuzzy_config
//! 2. Insert documents with text payloads and BM25-style sparse embeddings
//! 3. Create a fuzzy-enabled FullTextIndex on the payload field
//! 4. Run queries with `fuzzy_context` through `shard.core_search()`
//! 5. Verify that fuzzy expansion produces correct results

use std::collections::BTreeMap;
use std::sync::Arc;

use common::budget::ResourceBudget;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use common::save_on_disk::SaveOnDisk;
use segment::data_types::index::{TextIndexParams, TextIndexType};
use segment::data_types::vectors::VectorStructInternal;
use segment::index::sparse_index::bm25_fuzzy_expander::bm25_token_to_dim_id;
use segment::types::{
    Distance, FuzzyParams, PayloadFieldSchema, PayloadSchemaParams, WithPayloadInterface,
    WithVector,
};
use shard::query::query_enum::QueryEnum;
use shard::search::{CoreSearchRequest, CoreSearchRequestBatch, FuzzyBm25Context};
use sparse::common::sparse_vector::SparseVector;
use tempfile::Builder;
use tokio::runtime::Handle;
use tokio::sync::RwLock;

use crate::collection::payload_index_schema::PayloadIndexSchema;
use crate::config::{CollectionConfigInternal, CollectionParams, WalConfig};
use crate::operations::point_ops::{
    PointInsertOperationsInternal, PointOperations, PointStructPersisted, VectorPersisted,
};
use crate::operations::types::{FuzzySearchConfig, SparseVectorParams, VectorsConfig};
use crate::operations::vector_params_builder::VectorParamsBuilder;
use crate::operations::{CollectionUpdateOperations, CreateIndex, FieldIndexOperations};
use crate::shards::local_shard::LocalShard;
use crate::shards::shard_trait::ShardOperation;
use crate::tests::fixtures::TEST_OPTIMIZERS_CONFIG;

/// Simple BM25-like tokenize + embed: tokenize by whitespace, lowercase, deduplicate, hash to dim_ids.
fn simple_bm25_embed(text: &str) -> (Vec<String>, SparseVector) {
    let tokens: Vec<String> = text
        .to_lowercase()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();

    // Deduplicate tokens for embedding
    let mut seen = std::collections::HashSet::new();
    let unique_tokens: Vec<&str> = tokens
        .iter()
        .filter(|t| seen.insert(t.as_str()))
        .map(|t| t.as_str())
        .collect();

    let indices: Vec<u32> = unique_tokens
        .iter()
        .map(|t| bm25_token_to_dim_id(t))
        .collect();
    let values: Vec<f32> = vec![1.0; indices.len()];

    (tokens, SparseVector { indices, values })
}

/// Create a collection config with one dense vector (required) + one sparse vector with fuzzy_config.
fn create_fuzzy_collection_config() -> CollectionConfigInternal {
    let wal_config = WalConfig {
        wal_capacity_mb: 1,
        wal_segments_ahead: 0,
        wal_retain_closed: 1,
    };

    // We need at least one vector config. Use a minimal dense vec + sparse vec.
    let collection_params = CollectionParams {
        vectors: VectorsConfig::Single(VectorParamsBuilder::new(4, Distance::Dot).build()),
        sparse_vectors: Some(BTreeMap::from([(
            "text".to_string(),
            SparseVectorParams {
                index: None,
                modifier: None,
                fuzzy_config: Some(FuzzySearchConfig {
                    fuzzy_bind_field: "text_content".to_string(),
                }),
            },
        )])),
        ..CollectionParams::empty()
    };

    let mut optimizer_config = TEST_OPTIMIZERS_CONFIG.clone();
    optimizer_config.default_segment_number = 1;
    optimizer_config.flush_interval_sec = 0;

    CollectionConfigInternal {
        params: collection_params,
        optimizer_config,
        wal_config,
        hnsw_config: Default::default(),
        quantization_config: Default::default(),
        strict_mode_config: Default::default(),
        uuid: None,
        metadata: None,
    }
}

/// Insert documents with BM25-style sparse embeddings and text payloads.
fn create_upsert_ops(documents: &[&str]) -> CollectionUpdateOperations {
    let points: Vec<PointStructPersisted> = documents
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let (_tokens, sparse) = simple_bm25_embed(text);

            // Create vector struct with both dense (dummy) and sparse
            let mut vectors = std::collections::HashMap::<String, VectorPersisted>::new();
            vectors.insert(
                "".to_string(),
                VectorPersisted::Dense(vec![1.0, 0.0, 0.0, 0.0]),
            );
            vectors.insert("text".to_string(), VectorPersisted::Sparse(sparse));

            PointStructPersisted {
                id: (i as u64).into(),
                vector: VectorStructInternal::Named(
                    vectors
                        .into_iter()
                        .map(|(k, v): (String, VectorPersisted)| (k, v.into()))
                        .collect(),
                )
                .into(),
                payload: Some(
                    serde_json::from_str(&format!(r#"{{"text_content": "{}"}}"#, text)).unwrap(),
                ),
            }
        })
        .collect();

    CollectionUpdateOperations::PointOperation(PointOperations::UpsertPoints(
        PointInsertOperationsInternal::PointsList(points),
    ))
}

/// Create a text index with fuzzy matching enabled.
fn create_text_index_op() -> CollectionUpdateOperations {
    CollectionUpdateOperations::FieldIndexOperation(FieldIndexOperations::CreateIndex(
        CreateIndex {
            field_name: "text_content".parse().unwrap(),
            field_schema: Some(PayloadFieldSchema::FieldParams(PayloadSchemaParams::Text(
                TextIndexParams {
                    r#type: TextIndexType::Text,
                    fuzzy_matching: Some(true),
                    lowercase: Some(true),
                    ..Default::default()
                },
            ))),
        },
    ))
}

/// Build a CoreSearchRequest with fuzzy context for BM25 sparse search.
fn build_fuzzy_search_request(query_text: &str, fuzzy_params: FuzzyParams) -> CoreSearchRequest {
    let (tokens, sparse) = simple_bm25_embed(query_text);

    CoreSearchRequest {
        query: QueryEnum::Nearest(segment::data_types::vectors::NamedQuery::new(
            segment::data_types::vectors::VectorInternal::Sparse(sparse),
            "text",
        )),
        filter: None,
        params: None,
        limit: 10,
        offset: 0,
        with_payload: Some(WithPayloadInterface::Bool(true)),
        with_vector: Some(WithVector::Bool(false)),
        score_threshold: None,
        fuzzy_context: Some(FuzzyBm25Context {
            tokens,
            fuzzy_params,
            stemmer: None,
        }),
    }
}

/// Build a CoreSearchRequest WITHOUT fuzzy context (exact BM25 search).
fn build_exact_search_request(query_text: &str) -> CoreSearchRequest {
    let (_tokens, sparse) = simple_bm25_embed(query_text);

    CoreSearchRequest {
        query: QueryEnum::Nearest(segment::data_types::vectors::NamedQuery::new(
            segment::data_types::vectors::VectorInternal::Sparse(sparse),
            "text",
        )),
        filter: None,
        params: None,
        limit: 10,
        offset: 0,
        with_payload: Some(WithPayloadInterface::Bool(true)),
        with_vector: Some(WithVector::Bool(false)),
        score_threshold: None,
        fuzzy_context: None,
    }
}

const TEST_DOCS: &[&str] = &[
    "the quick brown fox jumps over the lazy dog",
    "a fast brown cat leaps across the sleeping puppy",
    "quantum computing will revolutionize cryptography",
    "the lazy dog slept in the warm sunshine",
    "brown bears hibernate during winter months",
];

/// Build a LocalShard with fuzzy config, insert documents, create text index.
async fn setup_shard() -> (tempfile::TempDir, tempfile::TempDir, LocalShard, Handle) {
    let collection_dir = Builder::new()
        .prefix("test_fuzzy_collection")
        .tempdir()
        .unwrap();
    let payload_index_schema_dir = Builder::new()
        .prefix("test_fuzzy_schema")
        .tempdir()
        .unwrap();
    let payload_index_schema_file = payload_index_schema_dir.path().join("payload-schema.json");
    let payload_index_schema: Arc<SaveOnDisk<PayloadIndexSchema>> =
        Arc::new(SaveOnDisk::load_or_init_default(payload_index_schema_file).unwrap());

    let config = create_fuzzy_collection_config();
    let current_runtime = Handle::current();

    let shard = LocalShard::build(
        0,
        "test_fuzzy".to_string(),
        collection_dir.path(),
        Arc::new(RwLock::new(config.clone())),
        Arc::new(Default::default()),
        payload_index_schema,
        current_runtime.clone(),
        current_runtime.clone(),
        ResourceBudget::default(),
        config.optimizer_config.clone(),
    )
    .await
    .unwrap();

    // Insert documents
    let upsert_ops = create_upsert_ops(TEST_DOCS);
    shard
        .update(upsert_ops.into(), true, None, HwMeasurementAcc::new())
        .await
        .unwrap();

    // Create fuzzy-enabled text index
    let index_ops = create_text_index_op();
    shard
        .update(index_ops.into(), true, None, HwMeasurementAcc::new())
        .await
        .unwrap();

    (
        collection_dir,
        payload_index_schema_dir,
        shard,
        current_runtime,
    )
}

// ===================================================================
// Integration test: exact BM25 search through LocalShard
// ===================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_fuzzy_shard_exact_query() {
    let (_dir, _schema_dir, shard, runtime) = setup_shard().await;

    // Exact query for "quick brown fox"
    let req = build_exact_search_request("quick brown fox");
    let batch = CoreSearchRequestBatch {
        searches: vec![req],
    };
    let hw_acc = HwMeasurementAcc::new();
    let results = shard
        .core_search(Arc::new(batch), &runtime, None, hw_acc)
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert!(
        !results[0].is_empty(),
        "Expected results for 'quick brown fox'"
    );
    // Doc 0 should be top result
    assert_eq!(
        results[0][0].id,
        0.into(),
        "Expected doc 0 as top result for exact 'quick brown fox'"
    );

    shard.stop_gracefully().await;
}

// ===================================================================
// Integration test: fuzzy BM25 search with typos
// ===================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_fuzzy_shard_typo_query() {
    let (_dir, _schema_dir, shard, runtime) = setup_shard().await;

    // Fuzzy query with typo: "quik" → should expand to "quick"
    let fuzzy_params = FuzzyParams {
        max_edit: 1,
        prefix_length: 0,
        max_expansions: 30,
    };
    let req = build_fuzzy_search_request("quik", fuzzy_params);
    let batch = CoreSearchRequestBatch {
        searches: vec![req],
    };
    let hw_acc = HwMeasurementAcc::new();
    let results = shard
        .core_search(Arc::new(batch), &runtime, None, hw_acc)
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    // "quik" alone has no matches in BM25 vocabulary, but fuzzy expansion
    // to "quick" should find docs 0 and possibly others
    assert!(
        !results[0].is_empty(),
        "Expected fuzzy expansion to find results for 'quik'"
    );

    shard.stop_gracefully().await;
}

// ===================================================================
// Integration test: fuzzy vs non-fuzzy comparison
// ===================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_fuzzy_shard_fuzzy_vs_exact() {
    let (_dir, _schema_dir, shard, runtime) = setup_shard().await;

    let fuzzy_params = FuzzyParams {
        max_edit: 1,
        prefix_length: 0,
        max_expansions: 30,
    };

    // Non-fuzzy query with typo
    let req_exact = build_exact_search_request("quik brwn");
    // Fuzzy query with same typo
    let req_fuzzy = build_fuzzy_search_request("quik brwn", fuzzy_params);

    let batch_exact = CoreSearchRequestBatch {
        searches: vec![req_exact],
    };
    let batch_fuzzy = CoreSearchRequestBatch {
        searches: vec![req_fuzzy],
    };

    let results_exact = shard
        .core_search(
            Arc::new(batch_exact),
            &runtime,
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap();
    let results_fuzzy = shard
        .core_search(
            Arc::new(batch_fuzzy),
            &runtime,
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap();

    // Fuzzy should find more or equal results compared to exact (with typos)
    assert!(
        results_fuzzy[0].len() >= results_exact[0].len(),
        "Fuzzy should find at least as many results: fuzzy={}, exact={}",
        results_fuzzy[0].len(),
        results_exact[0].len()
    );

    shard.stop_gracefully().await;
}

// ===================================================================
// Integration test: batch fuzzy search
// ===================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_fuzzy_shard_batch_query() {
    let (_dir, _schema_dir, shard, runtime) = setup_shard().await;

    let fuzzy_params = FuzzyParams {
        max_edit: 1,
        prefix_length: 0,
        max_expansions: 30,
    };

    // Batch with multiple fuzzy queries
    let req1 = build_fuzzy_search_request("quik fox", fuzzy_params);
    let req2 = build_fuzzy_search_request("lzy dog", fuzzy_params);
    let req3 = build_exact_search_request("brown"); // one non-fuzzy in the batch

    let batch = CoreSearchRequestBatch {
        searches: vec![req1, req2, req3],
    };
    let hw_acc = HwMeasurementAcc::new();
    let results = shard
        .core_search(Arc::new(batch), &runtime, None, hw_acc)
        .await
        .unwrap();

    assert_eq!(results.len(), 3, "Expected 3 result sets for batch of 3");
    // Each result set should be non-empty
    assert!(
        !results[0].is_empty(),
        "Fuzzy 'quik fox' should have results"
    );
    assert!(
        !results[1].is_empty(),
        "Fuzzy 'lzy dog' should have results"
    );
    assert!(!results[2].is_empty(), "Exact 'brown' should have results");

    shard.stop_gracefully().await;
}

// ===================================================================
// Integration test: fuzzy search without fuzzy_context (passthrough)
// ===================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_fuzzy_shard_no_fuzzy_context_passthrough() {
    let (_dir, _schema_dir, shard, runtime) = setup_shard().await;

    // Regular search without fuzzy_context should work normally
    let req = build_exact_search_request("brown");
    let batch = CoreSearchRequestBatch {
        searches: vec![req],
    };
    let hw_acc = HwMeasurementAcc::new();
    let results = shard
        .core_search(Arc::new(batch), &runtime, None, hw_acc)
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert!(!results[0].is_empty(), "Exact 'brown' should find results");

    shard.stop_gracefully().await;
}
