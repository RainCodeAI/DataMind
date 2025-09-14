# Product Roadmap (Q3–Q4 2025)
Last updated: 2025-09-14

## Milestones
**Week 1 (MVP):**
- Schema Builder v1 (profile → confirm)
- Constrained NL→SQL (SELECT‑only, LIMIT, deny‑lists)
- Basic charts; full audit logging

**Week 2 (Alpha pilots):**
- Interceptor + Woodpecker integration
- 10–15 scripted queries per dataset; usability pass
- Start 2–3 pilot users (Stratford)

**Weeks 3–4 (Pilot/Beta):**
- Saved dashboards, CSV/PDF export
- Simple RBAC (org/project/role)
- Expand pilots to 5–8 users

## Success Criteria
- ≥95% accuracy on eval suite; **p95 < 5s** on 1M rows
- 0 write queries executed; 100% PII deny‑list enforced
- Complete, replayable audit trails

## Near-Term Next
- Nexus control plane MVP (AuthZ, Router, Audit, Connectors)
- AnomalyAlert incident model + evidence packs; partner installer pilot
