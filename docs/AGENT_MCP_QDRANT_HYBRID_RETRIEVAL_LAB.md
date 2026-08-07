# Agent + MCP + Qdrant Learning Project: Hybrid Retrieval Lab

## 1. Background

This document proposes a learning-oriented project built around three pieces:

- an agent that can plan and explain experiments,
- an MCP server that exposes retrieval and evaluation tools,
- Qdrant as the execution engine for dense, sparse, and hybrid retrieval.

The project focus is not a generic chat interface. The goal is to build a small but complete experimentation system that can answer a practical question:

> For a given dataset and retrieval workload, which Qdrant retrieval strategy works better, under what settings, and why?

This makes the project suitable for learning because it covers:

- agent orchestration instead of one-shot prompting,
- MCP tool design instead of hardcoded glue code,
- Qdrant retrieval behavior instead of only CRUD operations,
- evaluation and reporting instead of subjective trial and error.

## 2. Problem Statement

Teams using Qdrant for RAG or semantic search often know that multiple retrieval strategies are available, but they usually lack a repeatable way to compare them.

Typical options include:

- dense retrieval,
- sparse retrieval,
- hybrid retrieval,
- reranking after retrieval,
- different values for top-k, filters, and fusion weights.

In practice, people often test these options manually and reach conclusions based on a small number of examples. This causes unstable quality decisions and makes it hard to explain why one setup should be used in production.

## 3. Pain Points To Solve

### Pain Point 1: Strategy choice is unclear

Most users know hybrid retrieval sounds stronger than dense-only or sparse-only, but they cannot tell whether it actually improves their own corpus.

What the project should solve:

- run the same query set across multiple retrieval strategies,
- compare them with the same metrics,
- produce an explicit recommendation instead of a vague impression.

### Pain Point 2: Experimentation is manual and slow

Manual experiments usually involve changing code, rerunning scripts, saving screenshots, and comparing outputs by hand.

What the project should solve:

- let an agent plan experiment batches,
- let MCP tools execute them in a repeatable way,
- persist results so runs can be compared later.

### Pain Point 3: Results are not explainable

Even when one setup looks better on average, it is hard to answer questions like:

- Which queries improved?
- Which queries regressed?
- Did sparse search help with keyword-heavy queries?
- Is reranking worth the latency cost?

What the project should solve:

- include per-query result inspection,
- surface representative failures and wins,
- attach explanations to the final recommendation.

### Pain Point 4: Agent demos are often shallow

Many agent demos only translate natural language into API calls. That looks convenient, but it does not show why an agent is useful for engineering work.

What the project should solve:

- give the agent a structured research workflow,
- make the agent plan, execute, compare, and summarize,
- keep real work inside tools so the behavior stays reliable.

### Pain Point 5: Learning projects often lack an end-to-end loop

Many learning projects stop after the first successful query. That does not teach how to evaluate quality, compare tradeoffs, or design safe tool boundaries.

What the project should solve:

- build a full loop from dataset to evaluation to report,
- make each stage inspectable,
- keep the scope small enough to finish.

## 4. Proposed Project

Project name:

**Hybrid Retrieval Lab**

Project definition:

A local experimentation assistant that uses an agent through MCP to run retrieval benchmarks on Qdrant collections, compare dense, sparse, and hybrid strategies, and generate a report with metrics, latency, and failure analysis.

## 5. Target Users

This project is suitable for:

- developers learning MCP through a real use case,
- engineers evaluating Qdrant retrieval quality,
- RAG builders who need evidence before choosing a retrieval strategy,
- anyone who wants an agent to help with structured experiments rather than only chat.

## 6. Core Value

The project should make these questions answerable with one workflow:

- Is dense retrieval enough for this dataset?
- Does sparse retrieval help on keyword-dominant queries?
- Does hybrid retrieval improve recall enough to justify complexity?
- Does reranking produce meaningful gains relative to latency cost?
- Which configuration should be used as the default baseline?

## 7. Scope

### In Scope

- local or dev-environment execution,
- one or a small number of Qdrant collections,
- offline evaluation on a labeled query set,
- dense, sparse, and hybrid retrieval comparison,
- report generation in markdown or JSON,
- agent-driven planning and summarization,
- MCP tools for retrieval and evaluation.

### Out of Scope For MVP

- online traffic A/B testing,
- production deployment automation,
- automatic training of rankers,
- large-scale dashboarding,
- multi-tenant permissions,
- autonomous destructive operations on Qdrant.

## 8. Requirements And The Pain Points They Address

| Requirement | Description | Pain Point Addressed |
| --- | --- | --- |
| Unified experiment schema | Every run must store strategy, parameters, metrics, latency, and sample results in one structure. | Manual experiments are hard to compare and reproduce. |
| Repeatable retrieval tools | Dense, sparse, and hybrid searches must be callable through stable MCP tools. | Ad hoc scripts make results inconsistent. |
| Offline evaluation support | The system must load a labeled dataset with queries and ground truth. | Strategy quality cannot be judged objectively without labels. |
| Per-query analysis | The report must highlight wins, regressions, and failure cases by query. | Average metrics hide what actually improved or broke. |
| Agent planning layer | The agent must propose and sequence experiments instead of only issuing single search calls. | Agent demos feel shallow without multi-step reasoning. |
| Guardrails | Limit number of experiment combinations, top-k values, and expensive optional steps. | Open-ended agents can generate wasteful or unsafe workloads. |
| Result persistence | Results must be saved so later runs can be compared. | One-off experiments produce no reusable knowledge. |
| Recommendation summary | The final output must explain which setup is preferred and under what tradeoff. | Raw metrics alone do not support engineering decisions. |

## 9. Functional Requirements

### Dataset Management

- Load a local evaluation dataset.
- Support a query list and expected relevant document ids.
- Allow inspecting a sample of source documents or payloads.

### Retrieval Execution

- Run dense retrieval against Qdrant.
- Run sparse retrieval against Qdrant.
- Run hybrid retrieval with configurable fusion parameters.
- Optionally support a rerank stage on top of retrieved candidates.

### Evaluation

- Compute Recall@k.
- Compute MRR@k.
- Compute HitRate@k.
- Track response latency per experiment.
- Summarize aggregate metrics for each run.

### Analysis And Reporting

- Compare runs side by side.
- Identify queries where one strategy strongly outperforms another.
- Identify queries where hybrid retrieval regresses.
- Generate a markdown report with recommendation text.

### Agent Workflow

- Accept a user goal in natural language.
- Inspect collection and dataset metadata.
- Generate an experiment plan.
- Execute experiment cases via MCP tools.
- Summarize findings and tradeoffs.

## 10. Non-Functional Requirements

- The system should run locally without production dependencies.
- Tool outputs should be structured and easy to validate.
- The agent should orchestrate, not directly implement scoring logic.
- The MVP should be understandable by one developer within a short learning cycle.
- The system should fail safely when required fields, datasets, or vector configurations are missing.

## 11. Recommended Architecture

### Layer 1: Dataset Layer

Responsible for loading:

- evaluation queries,
- ground truth labels,
- example documents,
- experiment presets.

### Layer 2: MCP Tool Layer

Responsible for exposing operations such as:

- collection inspection,
- dense search,
- sparse search,
- hybrid search,
- metric calculation,
- report persistence.

### Layer 3: Agent Layer

Responsible for:

- reading context,
- choosing which experiments to run,
- deciding comparison order,
- interpreting result differences,
- writing the final recommendation.

### Layer 4: Report Layer

Responsible for:

- structuring experiment outputs,
- generating markdown or JSON reports,
- preserving reproducibility for future comparison.

Design rule:

**The agent should decide and explain. Tools should execute and calculate.**

This boundary keeps the system reliable and easier to debug.

## 12. Suggested MCP Tools

### Dataset Tools

- `list_datasets`
- `load_eval_queries`
- `load_ground_truth`
- `load_document_samples`

### Qdrant Tools

- `qdrant_collection_info`
- `qdrant_dense_search`
- `qdrant_sparse_search`
- `qdrant_hybrid_search`

### Evaluation Tools

- `compute_recall_at_k`
- `compute_mrr_at_k`
- `compute_hit_rate_at_k`
- `summarize_latency`

### Experiment Tools

- `run_experiment_case`
- `batch_run_experiments`
- `compare_experiment_runs`
- `save_experiment_result`
- `generate_report`

## 13. MVP Definition

The MVP should answer one concrete question:

> On a small labeled dataset, which of dense-only, sparse-only, or hybrid retrieval performs best on Qdrant, and what is the tradeoff in latency?

### Minimum Deliverables

- one evaluation dataset,
- one Qdrant collection prepared for the experiment,
- dense, sparse, and hybrid retrieval execution,
- metric computation for each run,
- a markdown report with recommendation text,
- one agent workflow that can run the experiment end to end.

### Nice-To-Have After MVP

- reranking support,
- experiment history comparison,
- multiple fusion weights,
- HTML report rendering,
- CLI shortcuts for common experiment presets.

## 14. Example User Requests

- Compare dense and hybrid retrieval on this FAQ collection.
- Run a benchmark for short keyword-like queries.
- Show me which queries improved after enabling sparse retrieval.
- Tell me whether reranking is worth the extra latency.
- Recommend a default retrieval strategy for this dataset.

These requests are appropriate for an agent because they involve planning, repeated tool use, and result interpretation.

## 15. Guardrails

To keep the system practical and safe, the first version should include clear limits:

- cap the number of experiment combinations per request,
- cap top-k to a reasonable bound,
- require explicit opt-in for reranking,
- avoid destructive Qdrant operations,
- return validation errors for missing dataset labels or unsupported collection setups.

## 16. Why This Is A Strong Learning Project

This project is stronger than a plain natural-language admin tool because it teaches four important engineering patterns at once:

- how to design MCP tools around stable interfaces,
- how to make an agent useful through orchestration rather than magic,
- how to evaluate retrieval systems with reproducible evidence,
- how to separate planning, execution, and reporting concerns.

It is also advanced enough to feel substantial, while still small enough to build as a personal learning project.

## 17. Recommended First Implementation Order

1. Define the experiment input and output schema.
2. Implement a unified retrieval execution layer.
3. Implement metric calculation and result persistence.
4. Expose the execution and evaluation functions through MCP.
5. Add the agent planning and summarization workflow.
6. Add reporting and representative failure analysis.

This order reduces the risk of building a polished agent around unstable experiment logic.

## 18. Final Summary

The proposed project is a learning-oriented but technically meaningful system built around agent orchestration, MCP tools, and Qdrant retrieval evaluation.

Its core contribution is not chat-based control. Its core contribution is a structured experimental workflow that can:

- compare dense, sparse, and hybrid retrieval fairly,
- explain quality and latency tradeoffs,
- preserve experiment results,
- recommend a strategy based on evidence.

If completed well, this project can serve both as a personal learning vehicle and as a reusable internal tool for retrieval experimentation.