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

---

## 8. Implementation notes & offline runbook (Day 1–2)

**θ — plate parametrization** (`parameter_vector`, all mm except `n_screws`). `THETA_BOUNDS` in
`engine.py` is the single source of truth (bounds guardrail + CMA-ES search space):

| param | bounds | | param | bounds |
|---|---|---|---|---|
| `length_mm` | 40–200 | | `n_screws` | 2–12 |
| `width_mm` | 8–30 | | `screw_spacing_mm` | 8–40 |
| `thickness_mm` | 2–8 | | `contour_offset_mm` | 0–5 |

`THETA_BOUNDS` is intentionally **static**: the frozen `synthesize` input is `{plan, report,
iteration}`, so `CaseSpec.constraints` (`min_thickness_mm`, `max_volume_mm3`) never reach Split B.

**Ladder** — `with_fallback([_rung1, _rung2], _floor)`:
- **rung 1** — LLM θ-proposer via `call_llm(stage="synthesize", model="bedrock/claude-sonnet")`;
  gateway fallback **`bedrock/mistral-large`** (in `gateway/routing.yaml`). Bad JSON / out-of-bounds
  θ / model outage → `RetryableError` / `RejectedOutput` → ladder advances.
- **rung 2** — CMA-ES (`cma`) over `THETA_BOUNDS`, no LLM. Optimizes against an injected stress
  oracle (tests) or a built-in analytic beam-bending proxy (live/offline). `fallback_rung=2`.
- **floor** — last-known-good θ + `parameter_vector["_stop"]=True`; guaranteed-watertight solid
  plate; never raises.

**Geometry (offline, no bpy):** trimesh builds the plate + screw cylinders; **pymeshlab/CGAL** cuts
the holes (trimesh's own boolean needs `manifold3d` or the Blender binary). `validity.self_intersect`
is an `is_volume` proxy. `contacts_anchor_ids` come from `check_contacts` (<1 mm) — empty on Split A's
current stub anchors (~388 mm off the plate), an A-stub artifact, not a bug.

**Guardrails** (mirror the gateway `synthesize-guards` so they fire offline): `theta_bounds_check`
(pre-invoke, before `generate_mesh`) and `mesh_validity_check` (post-invoke, before returning to C).

> **Gateway TODO (one person, in the TrueFoundry UI):** create the `implant/theta-bounds-check` and
> `implant/mesh-validity-check` guardrail *integrations* so the existing `synthesize-guards` rule in
> `gateway/guardrails.yaml` resolves. No YAML edit required.

**Offline commands** (no creds, no Blender):
```bash
uv pip install -e ".[dev,localization,synthesis]"
pytest -q split_b_synthesis/test_acceptance.py    # convergence + 2 injected failures + trace
python -m split_b_synthesis.engine                # regenerate the 5 fixtures (rung 2 offline)
```

---

## 9. Coordinate-frame placement (integration — the implant goes ONTO the bone)

The implant is modeled INTO the PlacementPlan frame and recomputed per patient (nothing static):
- **Position:** body centered at `defect_region.centroid` if populated, else the **anchor centroid**.
- **Orientation:** long axis along `coordinate_frame.basis` +Z (bone long axis); X = width, Y = thickness.
- **Size:** `seed_theta` spans the anchors in the local frame (length along +Z, width across) and takes
  thickness from `cortical_thickness_mm`, all clamped to `THETA_BOUNDS`. The controller (LLM/CMA) tunes
  **thickness**; plate length/width/screw layout follow the anatomy.
- **Screws:** a hole is drilled at each on-plate anchor (axis along the anchor normal), then the finished
  plate is rigid-transformed **local → world** so the STL overlays the bone mesh.
- **Contacts:** `contacts_anchor_ids` come from the real `check_contacts` (<1 mm) on the world-frame mesh.

**Units / overlay:** anchors and the implant STL are in **mm**; A's `dummy_bone.stl` is in **meters** —
scale the bone ×1000 (A's pipeline already does) to overlay the implant in the same frame.

**On A's current fixtures:** the 4 anchors are scattered ~15–110 mm around the 3-D shaft (not a coplanar
fracture line), so a single flat bounded plate contacts only the anchors that land on it (e.g. test_case_02
→ a0, a3; the others none). Placement itself is correct and per-patient — proven by
`test_placement_is_frame_driven_not_static` (two frames, every screw < 1 mm). Full screws-at-all-anchors
needs clustered fracture-line anchors from A, or a curved surface-conform plate (future work).
