# RainCode AI — System Overview
Last updated: 2025-09-14

## Mission
Applied AI that turns messy, real-world workflows into reliable tools for **security, intelligence, and productivity**.

## What We Build
- **Flagships:** The Analyst (SMB data clarity), AnomalyAlert (AI surveillance anomaly detection), Nexus (AI operating system for workflows).
- **Ecosystem:** Modular agents that stand alone or plug into each other; governance by design through Corral Software.

## Operating Principles
1. **Deterministic-first:** SQL/Pandas or rule-based engines where possible; LLMs for translation/UX only.
2. **Governed by default:** Prompt → **Interceptor** → Agent → **Woodpecker** → Audit; **Charlie** on export in regulated contexts.
3. **Privacy by design:** PII deny-lists, read-only analytics roles, explicit user approval before risky actions.
4. **SaaS first, hybrid later:** Cloud baseline with optional local/air‑gapped modes for regulated clients.

## Shared Services (cross-agent)
- **Auth/RBAC**, **Connectors** (Gmail/Calendar/Drive/Slack), **Audit Log**, **Secrets/Keys**, **Jobs/Queues**, **Storage (S3‑compatible)**.

## Current Focus
- Build **Schema Builder v1** and **constrained NL→SQL** for The Analyst.
- Start pilots with 2–3 Stratford businesses.
- Integrate Interceptor/Woodpecker (phase 1), Charlie (opt‑in exports).

## High-Level Architecture
User → **Nexus UI** → **Interceptor** → Target Agent (Analyst/AnomalyAlert/…) → **Woodpecker** → **Audit** → UI export/report.

```text
apps/
  analyst/        # The Analyst (Streamlit UI → FastAPI)
  anomalyalert/   # Event detectors, incident model, notifier
  nexus/          # Control plane (AuthZ, Router, Memory, Audit)
shared/
  corral/         # Interceptor, Woodpecker, Charlie clients
  connectors/     # Gmail, Calendar, Drive, Slack
  infra/          # queues, storage, config
docs/             # system briefs (this folder)
```
