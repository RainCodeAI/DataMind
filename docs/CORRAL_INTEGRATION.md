# Corral Integration Map
Last updated: 2025-09-14

## Products & Roles
- **Interceptor** (pre‑LLM): normalize/defuse prompts; enforce input policy.
- **Woodpecker** (post‑LLM): evaluate outputs for factual/consistency, banned columns, policy violations.
- **Charlie** (compliance): govern exports (privacy, retention), generate compliance reports.
- **Nomad** (secure routing): policy‑based, cross‑border/multi‑office data movement (future).

## Per-Product Flows
### The Analyst
1. User question → Interceptor cleans
2. NL→SQL against **semantic schema** (SELECT‑only, LIMIT)
3. Woodpecker verifies (schema/agg/rowcount, banned columns)
4. Results + suggested charts; **Charlie on export** if regulated

### AnomalyAlert
- Detector events (MAD z‑score, IQR, Isolation Forest) → **Incident builder** (burst collapse, severity)
- Woodpecker checks: duplicate bursts, zone/time policy, evidence completeness
- Notify (Telegram/email), export **Evidence Pack** (snapshots + hashes) → Audit

### Nexus
- All agent prompts: User → Interceptor → Agent → Woodpecker → Audit
- Central RBAC, Memory/Profiles, Connectors, and org/project workspaces
