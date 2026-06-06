# Osteon — Project Overview

**Osteon is a resilient AI agent that designs patient-specific orthopedic bone implants.**
Given a digital case file for one patient — a 3D bone scan, the defect, the loads it must
survive, and material properties — Osteon decides *where* the implant goes, *selects and places*
a real implant onto the bone, and *proves* it survives mechanically, then visualizes the whole
thing in Blender.

The system's defining property is **resilience**: every stage degrades gracefully instead of
crashing. If an ML model, an LLM, a solver, or a cloud service fails mid-run, that stage falls
back to a simpler method and the design loop keeps going — it can never hard-fail.

---

## The big idea

```
            CaseSpec  (one patient: bone scan + defect + loads + materials)
                │
   ┌────────────▼────────────┐
   │  Split A — Localization  │   "Where does the implant go?"
   │  CaseSpec → PlacementPlan │   anchors on cortical bone + coordinate frame
   └────────────┬────────────┘
                │ PlacementPlan
   ┌────────────▼─────────────┐
   │  Split B — Synthesis      │   "Pick + place the implant."
   │  PlacementPlan → Candidate│   real implant placed into the bone frame
   └────────────┬─────────────┘
                │ ImplantCandidate                 ▲ StressReport (feedback loop)
   ┌────────────▼─────────────┐                    │
   │  Split C — Evaluation      │ ──────────────────┘
   │  Candidate+Case → Report   │   "Does it survive?" FEA stress, FoS, fatigue, shielding
   └────────────┬─────────────┘
                │ StressReport (+ stress heat-map)
                ▼
     orchestrator iterates B↔C until the implant passes, or returns the best safe attempt
```

The **input is not a doctor's note** — it's a structured `CaseSpec` (machine-readable case
file). The system *computes* where the implant goes; it isn't told.

---

## The three splits

### Split A — Localization & Anchoring  (`split_a_localization/`)
**Owns:** `CaseSpec → PlacementPlan`  ·  stage tag `localize`  ·  MCP server `localization-mcp`

Loads and normalizes the bone mesh, builds an anatomical coordinate frame (PCA principal axes),
finds anchor points on cortical bone (surface sampling + a ≥1.5 mm cortical-thickness ray-cast
filter + farthest-point spreading), and emits a valid `PlacementPlan`. An LLM turns the
free-text `defect.description` into a structured target.

- **Fallback ladder:** rung 1 ML landmark regressor (PointNet) → rung 2 geometric heuristic
  (PCA + curvature) → floor conservative default frame.
- **Viewer:** `blender_render.py` renders the bone + anchors + defect + frame axes headless to
  PNG, saves an interactive `.blend`, and can open the live 3D scene in Blender.
- **Fixture bone:** `scripts/make_femur.py` generates an anatomically-proportioned ~440 mm femur
  (SDF + marching cubes, hollow medullary canal) used as the demo bone.

### Split B — Parametric Synthesis & Placement  (`split_b_synthesis/`)
**Owns:** `PlacementPlan (+ StressReport) → ImplantCandidate`  ·  stage tag `synthesize`  ·  MCP server `blender-mcp`

Produces the implant and **places it into the bone's coordinate frame** (not at the origin), so
the STL overlays the femur and screw holes land on the anchors. The controller tunes the
implant's parameters across iterations using the `StressReport` feedback from Split C.

- **Fallback ladder:** rung 1 LLM θ-proposer (via the gateway) → rung 2 CMA-ES numeric optimizer
  (no LLM, immune to model outages) → floor last-known-good + guaranteed-watertight plate.
- **Geometry tools** (offline, no `bpy`): trimesh plate + pymeshlab/CGAL boolean screw holes.
- **Guardrails:** `theta-bounds-check` (pre) rejects out-of-range parameters before meshing;
  `mesh-validity-check` (post) blocks non-watertight meshes.
- **Library variant** (on branch `b/library-implant`): selects a *real* implant CAD from
  `fixtures/implant_library/` by clinical inference and rigidly registers it to the anchors.

### Split C — Biomechanical Evaluation & Stress Oracle  (`split_c_evaluation/`)
**Owns:** `ImplantCandidate + CaseSpec → StressReport`  ·  stage tag `evaluate`  ·  MCP server `fea-mcp`

Decides whether a candidate survives: peak von Mises stress, factor of safety (yield/peak),
fatigue check vs. the endurance limit, max displacement, and the **stress-shielding index**
(Wolff's-law concern — a too-stiff implant offloads the bone). Output drives the picture, too:
a **stress heat-map** colored over the implant geometry.

- **Fallback ladder:** rung 1 sfepy 3D linear-elastic FEA (ASTM F382-style bending) → rung 2
  1D Euler-Bernoulli beam FE (pure numpy) → floor closed-form analytic bound (never raises).
- **Guardrails:** `mesh-watertight-gate` (pre) blocks the solver on garbage meshes;
  `report-nan-gate` (post) rejects NaN/∞/negative-FoS reports before they cascade into B.
- **Visualization:** `heatmap_render.py` (stress field → colored render); `webapp/` serves a
  dashboard of the results.

---

## Shared foundation (`common/`)

The frozen integration layer — every split behaves identically here.

| Module | Purpose |
|---|---|
| `contracts.py` | The **frozen** Pydantic data contracts. Units fixed: mm, N, MPa, degrees. |
| `ladder.py` | `with_fallback(rungs, floor)` — the standardized 3-rung ladder used by all splits. The floor **never raises**, so the loop can't crash. |
| `llm.py` | The single LLM entry point: `call_llm(stage=…)`. All routing/fallback lives in the gateway, never in code. |
| `trace.py` | One `trace_id` per case, flowing A→B→C; JSONL spans, payloads hashed (never raw patient data). |
| `errors.py` | Shared error taxonomy; the ladder branches on `.code`, not class names. |
| `mcp_base.py` | `osteon_tool` wrapper — normalizes every MCP tool error to `ToolFailError` + a result-size bound. |
| `settings.py` | Single source of env config (`TFY_TOKEN`, `TFY_GATEWAY_URL`, `OSTEON_TRACE_DIR`, …). |

### The data contracts

```
CaseSpec  ──A──▶  PlacementPlan  ──B──▶  ImplantCandidate  ──C──▶  StressReport
(system input)    (anchors+frame)        (placed implant STL)      (survives? + heat-map)
```

Every contract carries `fallback_rung` (which rung produced it) and `trace_id` (the shared
thread). `CaseSpec`/`PlacementPlan`/`ImplantCandidate`/`StressReport` are defined in
`common/contracts.py` and are change-controlled (all three owners must approve a field change).

---

## Resilience ladder (why nothing can crash)

Each stage is `run = with_fallback([rung1, rung2], floor)`:

- **rung 1** — the best method (ML / LLM / full FEA). May fail on a model outage, bad output, or
  solver divergence → mapped to a `RetryableError`/`RejectedOutput`.
- **rung 2** — a simpler, dependency-light method (geometry / CMA-ES / reduced FE).
- **floor** — a deterministic closed-form result that **cannot raise**; low confidence, but valid.

The orchestrator chains the stage floors structurally, so the A→B→C loop always produces a
valid result. The demo proves it: kill the ML model **and** pull the cloud key mid-run, and the
trace shows `rung1→rung2` plus the gateway swapping models — with a still-valid output.

---

## Gateway (`gateway/`)

Model routing and guardrails live as config, not code (TrueFoundry AI Gateway):

- `routing.yaml` — per-stage model + fallback (e.g. `bedrock/claude-sonnet` → `bedrock/llama-70b`).
- `guardrails.yaml` — the pre/post-invoke guardrails referenced by each MCP server.
- `mcp-registry.yaml` — registers `localization-mcp`, `blender-mcp`, `fea-mcp`.

---

## Repository map

```
osteon/
├── orchestrator.py          A→(B↔C) loop with the stage-level circuit breaker
├── common/                  frozen contracts, ladder, llm, trace, errors, settings
├── split_a_localization/    engine, mcp_server, blender_render, model, fixtures
├── split_b_synthesis/       engine, mcp_server, fixtures (+ implant_library on a branch)
├── split_c_evaluation/      engine, fea, guardrails, heatmap_render, mcp_server, fixtures
├── webapp/                  Flask dashboard (app.py, implants_gen.py, templates/)
├── gateway/                 routing.yaml, guardrails.yaml, mcp-registry.yaml
├── scripts/make_femur.py    synthetic femur fixture generator
├── fixtures/                shared golden fixtures + implant_library/
├── tests/smoke_test.py      end-to-end orchestrator + contract validation
├── STANDARDIZATION.md       the frozen contract + integration spec (read this first)
└── README.md / PHASE0.md / OVERVIEW.md
```

---

## Running it

```bash
# setup (Python 3.11)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,localization,synthesis,evaluation]"
# .env needs TFY_TOKEN + TFY_GATEWAY_URL for the live LLM path (offline paths work without it)

# full pipeline: A femur → B placement → C FEA stress report
python orchestrator.py fixtures/example_case.json

# tests (offline, no credentials required)
pytest -q

# Blender visualization (bone + anchors, or bone + implant)
export OSTEON_BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"   # if not on PATH
#   Split A scenes open via split_a_localization/blender_render.py
#   Split C renders the stress heat-map via split_c_evaluation/heatmap_render.py

# dashboard
python webapp/app.py
```

The MCP servers (`localization-mcp`, `blender-mcp`, `fea-mcp`) run as
`python <split>/mcp_server.py` and are registered with the gateway via
`tfy apply -f gateway/mcp-registry.yaml`.

---

## Status

All three splits are integrated and the A→B→C loop runs green end-to-end. The resilience
ladders, contracts, gateway config, Blender visualization, FEA evaluation, stress heat-map, and
dashboard are in place. Some rungs remain demo-grade (e.g. Split A's ML landmark rung and defect
segmentation), but the floors and contracts are final — the system is designed so those upgrade
in place without touching the integration layer.
