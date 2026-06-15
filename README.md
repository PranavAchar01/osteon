# Osteon

A resilient multi-agent system that designs patient-specific orthopedic bone implants and proves they survive mechanically, built so the design loop never breaks when a model, a provider, a tool, or a solver fails.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)
![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-FF9900?style=flat&logo=amazonwebservices&logoColor=white)
![TrueFoundry](https://img.shields.io/badge/TrueFoundry-AI%20Gateway-5b2be6?style=flat)
![MCP](https://img.shields.io/badge/MCP-Gateway-111111?style=flat)
![Blender](https://img.shields.io/badge/Blender-bpy-F5792A?style=flat&logo=blender&logoColor=white)
![FEA](https://img.shields.io/badge/FEA-sfepy%20%2F%20CalculiX-006699?style=flat)
![pydantic](https://img.shields.io/badge/pydantic-v2-E92063?style=flat&logo=pydantic&logoColor=white)
![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-traced-425CC7?style=flat&logo=opentelemetry&logoColor=white)
![Offline path](https://img.shields.io/badge/offline%20path-zero%20credentials-2ea043?style=flat)
![Hackathon](https://img.shields.io/badge/Resilient%20Agents-Hackathon-1f6feb?style=flat)

A structured case file for one patient goes in: a 3D bone scan, the defect, the loads the bone has to survive, and the material properties. A closed agent loop figures out where the implant goes, generates and places a parametric implant inside the bone's own coordinate frame in Blender, runs a finite-element stress analysis to prove it holds under the patient's loads, and iterates the geometry against the biomechanical result until the design passes. Every model call, every tool call, and every solver in that loop can fail mid-run, and the loop still returns a schema-valid implant. That recovery behavior is the point of the project.

Built for the Resilient Agents online hackathon (TrueFoundry plus AWS Bedrock, June 1 to 7, 2026) by Pranav Achar, Sahiel Bose, and Shanay Gaitonde.

---

## Table of contents

1. [What it does](#1-what-it-does)
2. [Why it is resilient](#2-why-it-is-resilient)
3. [The three splits](#3-the-three-splits)
4. [Shared foundation](#4-shared-foundation)
5. [The resilience ladder](#5-the-resilience-ladder)
6. [Gateway: config as code](#6-gateway-config-as-code)
7. [Data contracts](#7-data-contracts)
8. [Repository layout](#8-repository-layout)
9. [Quickstart](#9-quickstart)
10. [Running the pipeline](#10-running-the-pipeline)
11. [The dashboard](#11-the-dashboard)
12. [Failure-injection demo](#12-failure-injection-demo)
13. [Tech stack](#13-tech-stack)
14. [Testing and quality](#14-testing-and-quality)
15. [Scope and disclaimer](#15-scope-and-disclaimer)
16. [Authors](#16-authors)
17. [License](#17-license)

---

## 1. What it does

The input is not a free-text doctor's note. It is a structured, machine-readable `CaseSpec`: a watertight bone mesh in millimeters, the defect (fracture, resection, or void), a load profile in newtons, the implant material, and the manufacturing constraints. The system computes where the implant goes; it is never told.

Given that case file, Osteon runs a closed agentic loop:

```
              CaseSpec  (bone scan + defect + loads + materials)
                  |
                  v
        +---------------------+
        |  A  Localization    |   "Where does the implant go?"
        |  CaseSpec ->        |   anchors on cortical bone + a PCA coordinate frame
        |  PlacementPlan      |
        +----------+----------+
                   | PlacementPlan
                   v
        +---------------------+
        |  B  Synthesis       |   "Generate, parameterize, and place the implant."
        |  PlacementPlan ->   |   a real implant placed into the bone's own frame
        |  ImplantCandidate   |
        +----------+----------+
                   | ImplantCandidate            StressReport
                   v                                  ^
        +---------------------+                       |
        |  C  Evaluation      | ----------------------+
        |  Candidate + Case ->|   "Does it survive?" FEA stress, factor of safety,
        |  StressReport       |   fatigue, displacement, stress shielding + heat-map
        +----------+----------+
                   | StressReport (+ stress heat-map)
                   v
   orchestrator iterates B against C until the implant passes,
   or returns the best safe attempt
```

- A (Localization): loads and normalizes the bone mesh, builds an anatomical coordinate frame from PCA principal axes, finds anchor points on cortical bone (surface sampling with a cortical-thickness ray-cast filter at 1.5 mm and farthest-point spreading), and emits a valid `PlacementPlan`.
- B (Synthesis and Controller): generates a parametric implant mesh, places it into the bone's coordinate frame (not at the origin) so the STL overlays the bone and the screw holes land on the anchors, and tunes the implant's parameters across iterations using the feedback from C.
- C (Biomechanical Evaluation): runs the stress analysis, computes peak von Mises stress, factor of safety, fatigue safety, maximum displacement, and the stress-shielding index, decides whether the candidate survives, and colors a stress heat-map over the implant geometry.

---

## 2. Why it is resilient

Resilience is the defining property of the system. Every stage degrades gracefully instead of crashing. If an ML model, an LLM, a solver, or a cloud service fails mid-run, that stage falls back to a simpler method and the design loop keeps going. It cannot hard-fail.

Every model call goes through the TrueFoundry AI Gateway to AWS Bedrock. Every cross-stage tool call goes through the MCP Gateway. Risky steps are wrapped in guardrails. Everything emits one OpenTelemetry trace that flows from A to B to C.

| Failure injected | Handled by | Mechanism |
|---|---|---|
| Rate limit | AI Gateway | Rate-limit rules and unhealthy-model cooldown; traffic shifts to the fallback model |
| Provider or model outage | AI Gateway | Fallback config (401, 403, or 5xx routes to a fallback Bedrock model) |
| Slow response | AI Gateway and app | Per-target timeout drives latency routing; an app deadline advances to the next fallback rung |
| Tool failure or timeout | MCP Gateway and app | Circuit breaker, result-size bounds, and a local floor rung |
| Bad intermediate output | Guardrails | Pre-invoke and post-invoke hooks reject or repair output before the next stage sees it |
| Cascading errors | Fallback ladder | Each stage's floor rung returns a valid low-confidence output and never raises |

The result: kill the ML model and pull the cloud key in the middle of a run, and a single trace shows the rung-1 to rung-2 fallback plus the gateway swapping models, while the final implant is still produced and still valid.

---

## 3. The three splits

The system is built as three independent splits that snap together by contract. Each one is independently runnable against frozen fixtures and mocks, so no split is ever blocked on another. Each split owns one contract transformation, one MCP server, one pre-invoke guardrail, one post-invoke guardrail, and a three-rung fallback ladder.

### Split A: Localization and Anchoring (`split_a_localization/`)

Owns `CaseSpec -> PlacementPlan`, stage tag `localize`, MCP server `localization-mcp`.

Builds the anatomical coordinate frame, segments the defect, detects anchor points on cortical bone, computes resection planes, and emits the fit-target submesh the implant must conform to. An LLM turns the free-text `defect.description` into a structured target and self-checks the proposed coordinates for anatomical plausibility.

- Fallback ladder: rung 1 ML landmark regressor (PointNet), rung 2 geometric heuristic (PCA plus curvature), floor conservative default frame.
- Acceptance bar: landmark RMS error under 3 mm, every anchor in cortical bone (thickness at least 1.5 mm at the point), resection-plane angular error under 5 degrees, valid `PlacementPlan` for all five labeled bones.
- Viewer: `blender_render.py` renders the bone, anchors, defect, and frame axes headless to PNG, saves an interactive `.blend`, and can open the live 3D scene in Blender.
- Fixture bone: `scripts/make_femur.py` generates an anatomically proportioned femur (roughly 440 mm, SDF plus marching cubes, hollow medullary canal) used as the demo bone.

### Split B: Parametric Synthesis and Placement (`split_b_synthesis/`)

Owns `PlacementPlan` (plus `StressReport`) `-> ImplantCandidate`, stage tag `synthesize`, MCP server `blender-mcp`.

Produces the implant and places it into the bone's coordinate frame so the STL overlays the bone and the screw holes land on the anchors. The controller tunes the implant's parameters across iterations from C's stress feedback.

- Fallback ladder: rung 1 LLM parameter proposer (via the gateway), rung 2 CMA-ES numeric optimizer (no LLM, immune to model outages), floor last-known-good geometry plus a guaranteed-watertight plate.
- Geometry tools run offline without `bpy`: trimesh plate generation with pymeshlab and CGAL boolean screw holes.
- Guardrails: `theta-bounds-check` (pre-invoke) rejects out-of-range parameters before any meshing or Blender call; `mesh-validity-check` (post-invoke) blocks non-watertight meshes.

### Split C: Biomechanical Evaluation and Stress Oracle (`split_c_evaluation/`)

Owns `ImplantCandidate + CaseSpec -> StressReport`, stage tag `evaluate`, MCP server `fea-mcp`.

Decides whether a candidate survives: peak von Mises stress, factor of safety (yield over peak), a fatigue check against the endurance limit, maximum displacement, and the stress-shielding index (a Wolff's-law concern, since a too-stiff implant offloads the bone). The output also drives the picture: a stress heat-map colored over the implant geometry.

- Fallback ladder: rung 1 sfepy 3D linear-elastic FEA (ASTM F382-style bending), rung 2 1D Euler-Bernoulli beam FE in pure numpy, floor closed-form analytic bound that never raises.
- Guardrails: `mesh-watertight-gate` (pre-invoke) blocks the solver on garbage meshes; `report-nan-gate` (post-invoke) rejects NaN, infinite, or negative factor-of-safety reports before they cascade back into B.
- Visualization: `heatmap_render.py` turns the stress field into a colored render.

Each split ships five fixture files of its own output under `split_x/fixtures/`, so the downstream split can develop without the upstream split existing. See each split's `SETUP.md` for the full work assignment.

---

## 4. Shared foundation

`common/` is the frozen integration layer. Every split behaves identically here. After Phase 0 it is change-controlled: any edit requires all three owners to approve.

| Module | Purpose |
|---|---|
| `contracts.py` | The frozen pydantic v2 data contracts. Units fixed: mm, N, MPa, degrees. |
| `ladder.py` | `with_fallback(rungs, floor)`, the standardized three-rung ladder used by all splits. The floor never raises, so the loop cannot crash. |
| `llm.py` | The single LLM entry point, `call_llm(stage=...)`. All routing and fallback live in the gateway, never in code. |
| `trace.py` | One `trace_id` per case, flowing A to B to C. JSONL spans with hashed payloads, so raw patient data is never logged. |
| `errors.py` | The shared error taxonomy. The ladder branches on `.code`, not on class names. |
| `mcp_base.py` | The `osteon_tool` wrapper that normalizes every MCP tool error to `ToolFailError` and enforces a result-size bound. |
| `settings.py` | The single source of environment config (`TFY_TOKEN`, `TFY_GATEWAY_URL`, `OSTEON_TRACE_DIR`, and more), read only here. |

---

## 5. The resilience ladder

Every stage's `run()` is built from the same shape: `run = with_fallback([rung1, rung2], floor)`.

- Rung 1 is the best method (ML, LLM, or full FEA). It may fail on a model outage, a bad output, or solver divergence, which is mapped to a `RetryableError` or `RejectedOutput`.
- Rung 2 is a simpler, dependency-light method (geometry, CMA-ES, or reduced FE).
- The floor is a deterministic closed-form result that cannot raise. Low confidence, but always valid.

| Stage | Rung 1 | Rung 2 | Floor |
|---|---|---|---|
| A | ML landmark regressor | geometric heuristic (PCA plus curvature) | conservative default frame, low confidence |
| B | LLM parameter proposer (Bedrock) | CMA-ES numeric optimizer (no LLM) | last-known-good geometry, stop flag |
| C | full tetrahedral FEA (sfepy / CalculiX) | reduced beam or surrogate FEA | analytic closed-form bound |

Each result object records which rung produced it in its `fallback_rung` field. The orchestrator chains the stage floors structurally, so the A to B to C loop always produces a valid result.

---

## 6. Gateway: config as code

Model routing and guardrails live as YAML in `gateway/`, applied with `tfy apply` and changed only by pull request. None of this lives in application code.

- `routing.yaml`: per-stage model plus fallback (for example `bedrock/claude-sonnet` routes to `bedrock/llama-70b`), with `max_failures` and a `cooldown_seconds` window per model.
- `guardrails.yaml`: the pre-invoke and post-invoke guardrails referenced by each MCP server, scoped by model or by MCP server.
- `mcp-registry.yaml`: registers `localization-mcp`, `blender-mcp`, and `fea-mcp`, each exposing only its own split's tools.

Model aliases set once in the gateway and used everywhere: `bedrock/claude-sonnet` (rung 1), `bedrock/llama-70b` (rung 2), `bedrock/mistral-large` (rung 2b).

---

## 7. Data contracts

The pipeline is wired entirely through frozen pydantic v2 contracts in `common/contracts.py`. Field names, types, and units are fixed. Every contract carries `fallback_rung` (which rung produced it) and `trace_id` (the shared thread).

```
CaseSpec  --A-->  PlacementPlan  --B-->  ImplantCandidate  --C-->  StressReport
(system input)    (anchors+frame)        (placed implant STL)      (survives? + heat-map)
```

| Contract | Produced by | Consumed by |
|---|---|---|
| `CaseSpec` | system or fixture | A |
| `PlacementPlan` | A | B |
| `ImplantCandidate` | B | C |
| `StressReport` | C | B (controller) |

Units are fixed everywhere with no exceptions: millimeters for length, newtons for force, MPa for stress, degrees for angles. The coordinate frame is right-handed with positive Z along the bone long axis, and meshes are watertight STL in millimeters.

---

## 8. Repository layout

```
osteon/
  orchestrator.py          A -> (B against C) loop with the stage-level circuit breaker
  common/                  frozen contracts, ladder, llm, trace, errors, mcp_base, settings
  split_a_localization/    engine, mcp_server, blender_render, model, fixtures, SETUP.md
  split_b_synthesis/       engine, mcp_server, fixtures, SETUP.md
  split_c_evaluation/      engine, fea, guardrails, heatmap_render, mcp_server, fixtures, SETUP.md
  webapp/                  Flask dashboard (app.py, blender renderers, templates, static)
  gateway/                 routing.yaml, guardrails.yaml, mcp-registry.yaml
  scripts/                 make_femur.py, make_implant.py fixture generators
  fixtures/                shared golden fixtures plus implant_library/
  tests/                   smoke_test.py end-to-end orchestrator and contract validation
  STANDARDIZATION.md       the frozen compatibility contract (read this first to contribute)
  OVERVIEW.md              the deeper architecture writeup
  PHASE0.md                the shared-foundation setup everyone runs once
  README.md
  LICENSE
```

---

## 9. Quickstart

Prerequisites for the offline path: Python 3.11 and [uv](https://github.com/astral-sh/uv). The tests and the orchestrator run with no credentials.

```bash
git clone https://github.com/PranavAchar01/osteon.git && cd osteon
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

pytest -q                  # contracts, ladder-to-floor, trace, and orchestrator all green
python orchestrator.py     # prints passed=true, factor of safety, and a trace_id
```

For the full pipeline with the live engines, install the split extras:

```bash
uv pip install -e ".[dev,localization,synthesis,evaluation]"
```

For the live Bedrock-via-gateway path, add credentials and apply the shared gateway config. A TrueFoundry account with an AWS Bedrock provider, a Virtual Account Token, and Bedrock model access for Claude, Llama 3 70B, and Mistral Large are required.

```bash
cp .env.example .env        # fill TFY_TOKEN and TFY_GATEWAY_URL=https://gateway.truefoundry.ai
tfy login

tfy apply -f gateway/routing.yaml
tfy apply -f gateway/guardrails.yaml
tfy apply -f gateway/mcp-registry.yaml

# smoke test the gateway connection
python -c "from common.llm import call_llm; print(call_llm(stage='smoke', messages=[{'role':'user','content':'ping'}]).choices[0].message.content)"
```

See [PHASE0.md](./PHASE0.md) for the full shared-foundation setup that every contributor runs once.

---

## 10. Running the pipeline

```bash
# full pipeline: A localizes the femur, B places the implant, C runs the FEA stress report
python orchestrator.py fixtures/example_case.json

# force a stage to fail and watch the loop still complete via the floor
OSTEON_FORCE_FAIL=evaluate python orchestrator.py

# Blender visualization (set the binary path if Blender is not on PATH)
export OSTEON_BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
#   Split A scenes render through split_a_localization/blender_render.py
#   Split C renders the stress heat-map through split_c_evaluation/heatmap_render.py
```

The MCP servers (`localization-mcp`, `blender-mcp`, `fea-mcp`) each run as `python <split>/mcp_server.py` and are registered with the gateway through `tfy apply -f gateway/mcp-registry.yaml`.

---

## 11. The dashboard

`webapp/` is a Flask dashboard that reads like a clinical workspace. It drives the whole resilient pipeline from a `CaseSpec`, exactly as `orchestrator.design_implant` does, runs the real engines (each with its three-rung ladder), shows which rung fired at every stage, serves B's actual placed STL, and renders the dual implant-and-bone stress heat-map.

```bash
cd osteon && source .venv/bin/activate
python webapp/app.py        # http://127.0.0.1:5001
```

Highlights:

- Preset clinical cases (tibial and femoral fractures, comminuted fractures, osteoporotic bone) across titanium, cobalt-chromium, stainless steel, and PEEK.
- A new-patient form that builds a custom `CaseSpec` and runs the full A to B to C pipeline live.
- Per-patient implant sizing: the plate thickness is sized to the thinnest value that meets the factor-of-safety target for that patient's load (3-point bending), so a heavier load or a weaker material produces a genuinely thicker plate.
- Failure-injection toggles for the live resilience demo, each a simulated and reversible switch that touches no real credentials, meshes, or files, and always ends in a valid result.
- Browser-side three.js rendering of the placed implant and bone, with an option to open the rendered `.blend` in a local Blender install.

True to the resilience theme, the dashboard falls back to the shipped A and B fixtures if a live stage is ever unavailable, so it always produces a valid result.

---

## 12. Failure-injection demo

The judging demo runs one full case live, then injects three failures in the middle of the loop and shows that the final valid implant is still produced.

1. Revoke or fail the Bedrock primary model. The AI Gateway fails Claude over to Llama.
2. Force a solver timeout in C. Evaluation falls from full FEA to the beam surrogate to the analytic floor.
3. Feed an out-of-bounds parameter to B. The pre-invoke guardrail blocks it before Blender runs, and CMA-ES substitutes a valid implant.

The single OpenTelemetry trace shows all three recoveries in one place. That one trace is evidence of the AI Gateway, the MCP Gateway, the guardrails, and the resilience story at the same time.

---

## 13. Tech stack

- AWS Bedrock for foundation models (Claude, Llama 3, Mistral) behind one managed API.
- TrueFoundry AI Gateway for an OpenAI-compatible endpoint with routing, fallback, retries, rate limits, and observability.
- TrueFoundry MCP Gateway for scoped, audited, authenticated access to the project's tools.
- Guardrails to validate tool arguments before execution and inspect results before the model sees them.
- Blender (bpy) for parametric geometry generation and rendering.
- sfepy and CalculiX for the finite-element stress analysis, with a pure-numpy beam surrogate and a closed-form analytic floor.
- Python 3.11, uv, pydantic v2, trimesh, pymeshlab, open3d, CMA-ES, OpenTelemetry.
- Flask and three.js for the dashboard.

---

## 14. Testing and quality

- `pytest -q` runs the contract validation, the ladder-to-floor behavior, the trace format, and the full orchestrator, all with no credentials.
- Each split owns a `test_acceptance.py` that must pass and must pass again with one injected failure (model killed, tool timeout, or bad output), proving recovery.
- Formatting and linting use black (line length 100) and ruff; run `ruff check . && black --check .` before every commit.
- Contracts are change-controlled. Any change to `common/` or `gateway/` requires all three owners to approve.

Read [STANDARDIZATION.md](./STANDARDIZATION.md) before writing any code. It defines the contracts, the gateway client, the fallback ladder, the trace format, the config-as-code conventions, the MCP tool pattern, the git workflow, and the Definition of Done. Conformance to it is what makes the three independently built parts snap together on integration day.

---

## 15. Scope and disclaimer

This is a hackathon project, not a clinical or FDA-validated tool. It uses open and synthetic bone data and simplified finite-element analysis. The critical path for the competition is the resilience and recovery story, not implant fidelity. Some rungs are intentionally demo-grade (for example Split A's ML landmark rung and defect segmentation), but the floors and the contracts are final, so those rungs can be upgraded in place without touching the integration layer. Do not use any output of this system for actual medical decisions or patient care.

---

## 16. Authors

- Pranav Achar
- Sahiel Bose
- Shanay Gaitonde

Built for the Resilient Agents online hackathon (TrueFoundry plus AWS Bedrock, June 2026).

---

## 17. License

Released under the MIT License. See [LICENSE](./LICENSE) for the full text.
