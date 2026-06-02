# Split B — Parametric Synthesis & Iteration Controller  (Owner: Person B)

> Read `../STANDARDIZATION.md` first. This file is your complete work assignment + setup guide.

**One-liner:** Turn a placement into a buildable shape, then refine it against stress feedback. (This is the agent loop — the heaviest gateway/guardrail user.)

**You own:** `PlacementPlan` + `StressReport` → next `ImplantCandidate`  (stage tag: `synthesize`)

---

## 1. Engine work

1. Parametric mesh generator in `bpy` driven by a named θ (e.g. plate: length, width, thickness, screw-hole positions, contour-fit offset; stem: stem length, taper, neck angle).
2. Conform the mesh to `fit_target_surface` (project/wrap to bone) and make it pass through every anchor point.
3. **Controller:** consume a `StressReport`, propose the next θ that minimizes peak von Mises subject to FoS ≥ target, `stress_shielding_index` in band (0.6–0.9), and volume ≤ max.
4. Termination: constraints satisfied, or max iterations, or no-improvement plateau.

Output must be a valid `ImplantCandidate` (watertight STL + validity flags).

---

## 2. Model usage (Bedrock via AI Gateway)

- **Rung 1 model:** `bedrock/claude-sonnet`  —  **gateway fallback:** `bedrock/mistral-large`
- **What the LLM does:** propose a structured θ-delta with rationale given the latest `StressReport`.
- Call only through `common.llm.call_llm(stage="synthesize", ...)`.

---

## 3. MCP tools  (server name: `blender-mcp`)

| Tool | Purpose |
|---|---|
| `generate_mesh(theta)` | build the parametric implant mesh (expensive — enforce timeout + result-size bound) |
| `repair_mesh(path)` | fill holes, remove self-intersections (pymeshlab/trimesh) |
| `check_contacts(mesh, anchors)` | verify mesh touches all anchor points |

---

## 4. Guardrails (your demo centerpiece)

| Hook | Name | Behavior |
|---|---|---|
| pre-invoke | `implant/theta-bounds-check` | reject out-of-range θ **before** Blender runs (saves compute, blocks garbage geometry) |
| post-invoke | `implant/mesh-validity-check` | reject non-watertight / self-intersecting meshes before they reach C |

A rejected output raises `RejectedOutput` → the ladder advances.

---

## 5. Fallback ladder (`common/ladder.py`)

| Rung | Method |
|---|---|
| 1 | LLM θ-proposer (Bedrock) |
| 2 | CMA-ES numeric optimizer (`cma`, no LLM — immune to model outages) |
| floor | last-known-good θ, stop with flag, never raises |

Record the rung in `ImplantCandidate.fallback_rung`.

---

## 6. Full setup

```bash
# prereq: Phase 0 done; .env has TFY_TOKEN, TFY_GATEWAY_URL
git clone https://github.com/PranavAchar01/osteon.git && cd osteon
uv venv && source .venv/bin/activate
uv pip install -e ".[synthesis]"           # bpy (or blender --background), trimesh, pymeshlab, cma

tfy apply -f gateway/guardrails.yaml        # theta-bounds-check, mesh-validity-check
tfy apply -f gateway/mcp-registry.yaml      # blender-mcp
blender --background --python split_b_synthesis/mcp_server.py &
python -m split_b_synthesis.engine --plan split_b_synthesis/fixtures/plan_frozen.json --oracle mock
```

---

## 7. Acceptance test (your verifiable output)

Against a **frozen `PlacementPlan`** fixture and a **mock analytic stress oracle** (closed-form beam bending: stress ∝ load / (thickness·width²)):
- every iteration produces a watertight, manifold, non-self-intersecting mesh (100% after repair)
- every mesh contacts all anchor points (< 1 mm)
- the controller drives the mock objective below threshold within N iterations

**Resilience demo:** (a) feed an out-of-bounds θ → pre-invoke guardrail blocks it, no Blender call; (b) kill the LLM → CMA-ES takes over and still converges (shown in trace). Runs with **zero dependency on A's live code or C.**

Ship 5 `ImplantCandidate` fixtures (STL + JSON) under `fixtures/` so C can build without you.
