# The Analyst — Working Brief
Last updated: 2025-09-14

## Goal
Reliable **NL→SQL over messy CSVs** with strong guardrails and reproducibility.

## Pipeline
Upload → Validate/Profile → **Semantic Schema** (types, dates, categories, units, synonyms) →
NL→SQL (SELECT‑only, LIMIT 1000, blocked PII) → **Woodpecker** checks → Charts → Export (Charlie‑opt).

## Guardrails
- `sqlglot` validation; read‑only DB role; blocked DDL/DML
- Column deny‑lists & redaction; row/column sanity checks
- Interceptor pre‑clean; Woodpecker post‑verify; full **Audit**

## API (v1 sketch)
- `POST /upload` → returns dataset_id, profile summary
- `GET  /profile/{dataset_id}` → schema/profile
- `POST /query` (dataset_id, question) → {"sql","result_preview","charts","audit_id"}
- `POST /export` (audit_id, format) → governed by Charlie (if enabled)

## Eval Harness
- 5 sample datasets × ~20 canonical queries:
  - totals, group‑bys, time series, top‑N, filters, joins (future)
- Metrics: accuracy (exact/within tolerance), latency (p95), cost
