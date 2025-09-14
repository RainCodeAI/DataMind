# Schema Builder v1 — Design Spec
Last updated: 2025-09-14
[...trimmed in chat view: see full file after download...]

## Purpose
Turn messy CSVs into a **semantic schema** that the NL→SQL engine can safely and reproducibly use.

## Functional Requirements
- Profiling (types, roles, stats, unit hints, PII heuristics)
- User Confirmation (edit types/roles, PII, synonyms; approve & version)
- Persistence (versioned schema JSON + profile snapshot)
- Governance (deny-list, redaction; no queries until approved)

## Semantic Schema — JSON Example
```json
{
  "dataset_id": "ds_2025_001",
  "version": 1,
  "columns": [
    {
      "name": "order_date", "type": "date", "role": "time", "format": "YYYY-MM-DD",
      "pii": false, "deny": false, "synonyms": ["date"]
    }
  ],
  "denylist": ["email","phone","ssn"]
}
```
