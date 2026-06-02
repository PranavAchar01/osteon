# Osteon

**An AI agent inside Blender that autonomously designs orthopedic implants by iterating geometry against biomechanical stress constraints — built so the design loop never breaks when models, providers, or tools fail.**

Built for the **Resilient Agents** online hackathon (TrueFoundry + AWS Bedrock, June 1–7 2026).

---

## 1. What it does

Given a clinical case (a bone mesh, a defect, and a load profile), Osteon runs a closed agentic loop:

```
CaseSpec ─▶ [A] Localize ─▶ PlacementPlan ─▶ [B] Synthesize ─┬▶ ImplantCandidate ─▶ [C] Evaluate ─▶ StressReport
                                                             ▲                                          │
                                                             └──────────── iterate on θ ◀───────────────┘
```

- **[A] Localization** — finds the exact coordinates, anchor points, and resection planes inside the bone.
- **[B] Synthesis + Controller** — generates a parametric implant mesh in Blender and drives the iteration loop.
- **[C] Biomechanical Evaluation** — runs stress analysis (FEA) and decides whether the candidate survives.

The point of the project is **resilience**: every stage degrades gracefully when an LLM, provider, or tool fails, and the loop always returns a schema-valid result.

---

## 2. Why it is resilient (the hackathon thesis)

Every model call goes through the **TrueFoundry AI Gateway** to **AWS Bedrock**; every tool call goes through the **MCP Gateway**; risky steps are wrapped in **Guardrails**; everything emits one OpenTelemetry trace.

| Failure injected | Handled by | Mechanism |
|---|---|---|
| Rate limit | AI Gateway | Rate-limit rules + unhealthy-model cooldown, traffic shifts to fallback model |
| Provider/model outage | AI Gateway | Fallback config (401/403/5xx → fallback Bedrock model) |
| Slow response | AI Gateway + app | Per-target timeout → latency routing; app deadline → next fallback rung |
| Tool failure / timeout | MCP Gateway + app | Circuit breaker + result-size bounds + local rung-3 floor |
| Bad intermediate output | Guardrails | `mcp_tool_post_invoke` / `llm_output` hooks reject or repair before the next stage |
| Cascading errors | Fallback ladder | Each stage's rung-3 floor returns a valid low-confidence output and **never raises** |

---

## 3. Stack

- **AWS Bedrock** — foundation models (Claude, Llama 3, Mistral) behind one managed API.
- **TrueFoundry AI Gateway** — OpenAI-compatible endpoint; routing, fallback, retries, rate limits, observability.
- **TrueFoundry MCP Gateway** — scoped, audited, authenticated access to the project's tools.
- **Guardrails** — validate tool arguments before execution, mask/inspect results before the model sees them.
- **Blender (bpy)** — parametric geometry generation and rendering.
- **CalculiX / surrogate FEA** — biomechanical stress, factor of safety, fatigue, stress shielding.
- **Python 3.11**, `uv`, `pydantic`, `trimesh` / `pymeshlab` / `open3d`, `cma`, OpenTelemetry.

---

## 4. Repository layout

```
osteon/
├── common/                  # JOINTLY OWNED — frozen after Phase 0 (see STANDARDIZATION.md)
│   ├── settings.py          # env loader
│   ├── llm.py               # the ONE gateway client wrapper
│   ├── contracts.py         # CaseSpec, PlacementPlan, ImplantCandidate, StressReport, LoopTrace
│   ├── errors.py            # shared error taxonomy
│   ├── ladder.py            # standardized 3-rung fallback ladder
│   └── trace.py             # OTel span + JSONL emitter
├── gateway/                 # config-as-code, applied with `tfy apply`
│   ├── routing.yaml
│   ├── guardrails.yaml
│   └── mcp-registry.yaml
├── split_a_localization/    # Person A
│   ├── engine.py  mcp_server.py  fixtures/  test_acceptance.py  SETUP.md
├── split_b_synthesis/       # Person B
│   ├── engine.py  mcp_server.py  fixtures/  test_acceptance.py  SETUP.md
├── split_c_evaluation/      # Person C
│   ├── engine.py  mcp_server.py  fixtures/  test_acceptance.py  SETUP.md
├── orchestrator.py          # chains the three stages (integration day)
├── README.md
└── STANDARDIZATION.md       # the compatibility contract — read this first
```

---

## 5. Quickstart (Phase 0 — everyone, once)

```bash
git clone https://github.com/PranavAchar01/osteon.git && cd osteon
uv venv && source .venv/bin/activate
uv pip install -e .

cp .env.example .env        # fill TFY_TOKEN, TFY_GATEWAY_URL=https://gateway.truefoundry.ai
tfy login                   # TrueFoundry CLI

# apply shared gateway config (routing + guardrails + MCP registry)
tfy apply -f gateway/routing.yaml
tfy apply -f gateway/guardrails.yaml
tfy apply -f gateway/mcp-registry.yaml

# smoke test the gateway connection
python -c "from common.llm import call_llm; print(call_llm(stage='smoke', messages=[{'role':'user','content':'ping'}]).choices[0].message.content)"
```

**Prerequisites:** a TrueFoundry account with an AWS Bedrock provider account added, a Virtual Account Token, and Bedrock model access requested for Claude / Llama 3 70B / Mistral Large (aliased in the gateway as `bedrock/claude-sonnet`, `bedrock/llama-70b`, `bedrock/mistral-large`).

---

## 6. The three splits

| Split | Owner | Input → Output | Standalone demo |
|---|---|---|---|
| **A — Localization** | _A_ | `CaseSpec` → `PlacementPlan` | 5 labeled bones: landmark RMS < 3 mm, anchors in cortical bone, resection angle err < 5°; then kill ML + Bedrock key → rung2 + gateway fallback still produce a valid plan |
| **B — Synthesis + Controller** | _B_ | `PlacementPlan` + `StressReport` → `ImplantCandidate` | Frozen plan + mock stress oracle: converges with 100% valid meshes; out-of-bounds θ blocked pre-Blender; LLM killed → CMA-ES still converges |
| **C — Evaluation** | _C_ | `ImplantCandidate` → `StressReport` | Analytic benchmarks within 10%; force solver timeout → surrogate → analytic floor; injected NaN report caught by guardrail |

Each split is independently runnable against **frozen fixtures + mocks** — nobody is blocked on anyone else. See each split's `SETUP.md`.

---

## 7. Demo script (for judging)

Run one full case live, then inject three failures mid-loop:
1. Revoke the Bedrock key → AI Gateway fails Claude → Llama.
2. Force a CalculiX timeout → C falls full-FEA → surrogate → analytic floor.
3. Feed an out-of-bounds θ → guardrail blocks it before Blender runs.

Show the single OpenTelemetry trace where all three recoveries appear and the final valid implant is still produced. That one trace evidences AI Gateway, MCP Gateway, Guardrails, and Resilience simultaneously.

---

## 8. Scope note

This is a hackathon MVP, **not** a clinical/FDA-validated tool. It uses open/synthetic bone data and simplified FEA. The critical path for the competition is the resilience and recovery story, not implant fidelity — if time runs short, A can ship on geometric heuristics only and C on the surrogate+analytic rungs, and the gateway/MCP/guardrail/recovery demo stays fully intact.

---

## 9. Contributing

Read **[STANDARDIZATION.md](./STANDARDIZATION.md) before writing any code.** It defines the contracts, the gateway client, the fallback ladder, the trace format, config-as-code conventions, the MCP tool pattern, the git workflow, and the Definition of Done. Conformance to it is what makes the three independently built parts snap together on integration day.
