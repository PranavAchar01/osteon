# Split A — Localization & Anchoring  (Owner: Person A)

> Read `../STANDARDIZATION.md` first. This file is your complete work assignment + setup guide.

**One-liner:** Given a defect, find exactly where the implant goes.

**You own:** `CaseSpec` → `PlacementPlan`  (stage tag: `localize`)

---

## 1. Engine work

1. Load + normalize the bone mesh (units to mm, recenter).
2. Build an anatomical coordinate frame: PCA principal axes → align +Z to the long-bone axis; disambiguate proximal/distal by cross-sectional-area gradient.
3. Segment the defect region (resection = cut volume; fracture = gap via connected-component analysis).
4. Detect candidate anchor points: sample the cortical surface, reject points where local cortical thickness (inner/outer wall ray-cast) < 1.5 mm, then farthest-point-sample a well-spread set that brackets the defect.
5. Compute resection planes as the principal cut surfaces.
6. Emit the `fit_target_surface` submesh the implant must conform to.

Output must be a valid `PlacementPlan` (see contracts §3 in STANDARDIZATION.md).

---

## 2. Model usage (Bedrock via AI Gateway)

- **Rung 1 model:** `bedrock/claude-sonnet`  —  **gateway fallback:** `bedrock/llama-70b`
- **What the LLM does:** turn the free-text `defect.description` into a structured target spec, then run a self-consistency check on its own coordinates (“are these anchors anatomically plausible?”).
- Call only through `common.llm.call_llm(stage="localize", ...)`.

---

## 3. MCP tools  (server name: `localization-mcp`)

| Tool | Purpose |
|---|---|
| `load_bone_mesh(path)` | load + normalize STL |
| `measure_cortical_thickness(xyz)` | ray-cast wall distance at a point |
| `render_markers(plan)` | Blender scene PNG: anchors green, defect red, frame axes |

Expose nothing else. No delete/overwrite tools.

---

## 4. Guardrails

| Hook | Name | Behavior |
|---|---|---|
| pre-invoke | `coords-in-bbox-check` | reject anchor coords outside the bone bounding box before render |
| post-invoke | `landmark-sanity` | if confidence < 0.3, reject → forces the rung-3 floor |

---

## 5. Fallback ladder (`common/ladder.py`)

| Rung | Method |
|---|---|
| 1 | ML landmark regressor (small PointNet, PyTorch) |
| 2 | geometric heuristic (PCA + curvature extrema) |
| floor | conservative default frame, `confidence` low, never raises |

Record which rung produced the output in `PlacementPlan.fallback_rung`.

---

## 6. Full setup

```bash
# prereq: Phase 0 done; .env has TFY_TOKEN, TFY_GATEWAY_URL
git clone https://github.com/PranavAchar01/osteon.git && cd osteon
uv venv && source .venv/bin/activate
uv pip install -e ".[localization]"        # open3d, trimesh, scikit-learn, torch (CPU), pydicom

tfy apply -f gateway/mcp-registry.yaml      # registers localization-mcp
python split_a_localization/mcp_server.py &  # serve tools
python -m split_a_localization.engine --case split_a_localization/fixtures/case_01.json
```

---

## 7. Acceptance test (your verifiable output)

Five open bone meshes with hand-annotated ground-truth landmarks + a defined defect. **Pass:**
- landmark RMS error < **3 mm**
- 100% of anchors land in cortical bone (thickness ≥ 1.5 mm at point)
- resection-plane angular error < **5°**
- valid `PlacementPlan` JSON for all 5

**Resilience demo:** mid-run, kill the ML model **and** pull the Bedrock key → trace shows rung1→rung2 and gateway Claude→Llama, with a still-valid `PlacementPlan`. Runs with **zero dependency on B or C.**

Ship 5 `PlacementPlan` fixtures under `fixtures/` so B can build without you.
