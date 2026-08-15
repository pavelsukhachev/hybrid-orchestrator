# Demo 0 — Claim intake playground

Reference walkthrough of **claim intake → routing → human escalation**.

This is **not** a production claims system. The agent is a mock router that reads `scenarios.json`. There are no API keys, no live model, and no network calls except same-origin JSON (when you serve the folder).

## Open it

From this folder:

```bash
# Option A — any static server
python3 -m http.server 8080
# then open http://127.0.0.1:8080/

# Option B — open index.html directly
# CSS/JS load as relative files. scenarios.json is also embedded in
# index.html so file:// works when fetch is blocked.
```

Files: `index.html` · `demo.css` · `demo.js` · `scenarios.json`

## What you should see

1. Headline: *AI files routine claims. Humans take over when the decision is regulated.*
2. Default scenario `claim_intake_066` already selected.
3. Click **Run**. The caller message types in ~2s. The decision board shows intent `file_liability_claim`, routing `human_handoff`, compliance `litigation_risk`, and why: *AI must not admit fault or settle.*
4. Human queue card: Connect legal · preserve evidence · do not admit liability, with a full context packet so the human never asks the customer to repeat.
5. Toggle **058 · Fender-bender** and Run: green `ai_handle`, queue empty.
6. Optional **021 · Foundation / storm**: `flag_preexisting_condition` → adjuster desk, not a lawsuit handoff.

HF labels (`ai_handle` / `human_handoff`) are the main chips. Library actions (`ESCALATE` / `DASHBOARD_ALERT`) appear in smaller type. Do not conflate them with the agent API (`CONTINUE` / `ESCALATE`).

## Scenarios

Encoded from the [Insurance AI Agent Reliability Benchmark](https://huggingface.co/datasets/pashas/insurance-ai-reliability-benchmark) (Apache-2.0). Cite `sukhachev2026insurance`.

| ID | Line | Routing | Library |
| --- | --- | --- | --- |
| `claim_intake_066` | Commercial GL · hard | `human_handoff` | `ESCALATE` |
| `claim_intake_058` | Personal auto · easy | `ai_handle` | `DASHBOARD_ALERT` |
| `claim_intake_021` | Homeowners · hard | `human_handoff` | `DASHBOARD_ALERT` (adjuster) |

Framework: [pavelsukhachev/hybrid-orchestrator](https://github.com/pavelsukhachev/hybrid-orchestrator).

```bibtex
@dataset{sukhachev2026insurance,
  title={Insurance AI Agent Reliability Benchmark},
  author={Sukhachev, Pavel},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/pashas/insurance-ai-reliability-benchmark}
}
```

## License

Apache-2.0. Scenario text follows the benchmark license.
