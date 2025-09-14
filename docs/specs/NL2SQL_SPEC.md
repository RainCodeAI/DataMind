# Constrained NL→SQL — Design Spec
Last updated: 2025-09-14

## Guardrails
SELECT-only, LIMIT, deny-list/PII blocks, schema-constrained, read-only role.

## Validation Pipeline
Load schema → LLM draft → sqlglot parse → SELECT-only → LIMIT → columns in schema
→ deny-list → execute → Woodpecker checks → charts → audit.

## Pseudocode
```python
def answer(dataset_id, question, limit=1000):
    cleaned = interceptor.clean(question)
    schema = load_schema(dataset_id)
    sql_draft = llm_generate_sql(schema, cleaned, limit)
    node = sqlglot.parse_one(sql_draft, read='postgres')
    assert node.is_select()
    enforce_limit(node, limit)
    ensure_columns_in_schema(node, schema)
    ensure_no_denied(node, schema.denylist)
    df = execute_readonly(node, dataset_id)
    ok, flags = woodpecker.evaluate(schema, node, df.head(100))
    if not ok: return {"error":"Validation failed", "flags": flags}
    charts = suggest_charts(df, schema)
    audit_id = audit.log({"dataset_id":dataset_id,"prompt":question,"sql":node.sql()})
    return {"sql": node.sql(), "preview": df.head(50).to_dict("records"),
            "charts": charts, "audit_id": audit_id, "woodpecker": {"ok": ok, "flags": flags}}
```
