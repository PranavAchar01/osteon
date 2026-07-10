# Split C — Stress Heat-Map Visualization  (Owner: Person C)

> Companion to `SETUP.md`. This adds a **visual** deliverable on top of the numeric `StressReport`.
> Read `../STANDARDIZATION.md` (contracts) and Split A's `blender_render.py` (the headless-Blender
> rendering pattern you will reuse) before starting.

**One-liner:** Take the final implant design and show *exactly where the load goes* — a colored
stress map over the geometry, not just a pass/fail number.

---

## 1. What this is

Today Split C emits a `StressReport` (scalars: peak stress, FoS, shielding index, pass/fail).
That tells you *whether* it survives, not *where* it's about to break. This feature adds the
**where**: a heat map of the von Mises stress field painted onto the implant surface.

| | |
|---|---|
| **Input** | The **final implant design** — the watertight mesh produced by Split B (`ImplantCandidate.mesh_path`), optionally fused with the bone surface (`PlacementPlan.fit_target_surface_path`). This is "the final Blender design of the bone implant." Plus the `CaseSpec` loads/materials that define the stress state. |
| **Output** | A **stress heat map**: the implant (and bone) surface colored by local von Mises stress, with a calibrated color legend, rendered to PNG **and** saved as an interactive `.blend`. |

> Terminology: colloquial "pressure" → the engineering field is **von Mises stress** (σ_vM, in MPa) —
> the standard scalar for "how close is this point to yielding." That is what the map colors.

### Where it sits in the pipeline

```
Split A  CaseSpec ─▶ PlacementPlan ─┐
                                    ├─▶ Split B ─▶ ImplantCandidate (final mesh) ─┐
Split A  bone fit_target_surface ───┘                                            │
                                                                                 ▼
                                                       Split C: FE solve ─▶ σ_vM field
                                                                 │
                                                                 ├─▶ StressReport   (existing, scalars)
                                                                 └─▶ stress heat map (NEW: PNG + .blend)
```

The heat map is a **new artifact**; it does not replace the `StressReport`. It is produced from the
same FE solve that already populates the report, so the numbers and the picture are guaranteed consistent.

---

## 2. Inputs (precise)

1. `ImplantCandidate.mesh_path` — watertight STL of the final implant (required).
2. `PlacementPlan.fit_target_surface_path` — bone submesh to show the implant in context (optional; if
   absent, render the implant alone).
3. `CaseSpec.load_profile` — force vectors + application regions → boundary conditions.
4. `CaseSpec.implant_material` / `bone_material` — `E_MPa`, `yield_MPa` → stress + normalization.
5. The **σ_vM field** from the FE solve (per node/element). This is the new quantity you must surface
   out of `run_calculix`, not just the scalar peak.

---

## 3. Pipeline (FE solve → field → color → render)

1. **Solve and keep the field.** Extend `run_calculix` so it returns the full nodal stress field, not
   just the peak. CalculiX writes a `.frd` results file; parse the stress tensor per node and compute
   `σ_vM = sqrt(0.5·((σ1-σ2)² + (σ2-σ3)² + (σ3-σ1)²) + 3·(τxy²+τyz²+τzx²))`. Result: an array of
   `σ_vM` aligned to the mesh nodes.
2. **Map field → surface vertices.** The FE tet mesh ≠ the render STL. Sample the volumetric field
   onto the surface STL's vertices (nearest-node or barycentric interpolation). Output: one `σ_vM`
   value per surface vertex.
3. **Normalize + colorize.** Map `σ_vM` → color via a fixed colormap (below). Normalization range is
   `[0, yield_MPa]` so colors mean the same thing across every case (red = at yield).
4. **Render.** Bake per-vertex colors onto the mesh as a Blender vertex-color attribute and render
   headless — reuse Split A's `blender_render.py` machinery (scene setup, camera/light framing,
   `.blend` save, `OSTEON_BLENDER` resolver). Add a **color legend** (scale bar with MPa labels).
5. **Return artifact paths** alongside the report.

---

## 4. New MCP tool  (server: `fea-mcp`)

Add one tool to `split_c_evaluation/mcp_server.py`. Do **not** add delete/overwrite tools.

| Tool | Signature | Purpose |
|---|---|---|
| `render_stress_heatmap` | `(mesh_path: str, stress_field: list[float], yield_mpa: float, bone_path: str = "") -> dict` | Color the surface by per-vertex σ_vM, render headless, save PNG + `.blend`. Returns `{ "png_path": ..., "blend_path": ..., "peak_mpa": ..., "peak_location": {x,y,z} }`. |

- Wrap with `@osteon_tool(mcp)` (error normalization + size bound) like the others.
- Enforce its own timeout if the mesh is large (the wrapper does **not** add one — see Split B's
  `generate_mesh` for the `concurrent.futures` deadline pattern using `settings.OSTEON_DEADLINE_MS`).
- `stress_field` must be length-aligned to the mesh vertices; raise `ToolFailError` on mismatch.

---

## 5. Color mapping (make it honest and comparable)

- **Colormap:** blue → cyan → green → yellow → red for low → high σ_vM (the conventional FEA map).
  Optionally use `matplotlib`'s `turbo`/`jet` for the LUT.
- **Normalization:** linear, fixed range `[0, yield_MPa]`. **Do not auto-scale per case** — auto-scaling
  makes a safe part and a failing part look identical. Clamp `σ_vM ≥ yield` to full red.
- **Legend:** render a vertical color bar with tick labels in MPa, plus annotations for `peak σ_vM`,
  `yield`, and the resulting factor of safety. Composite it onto the PNG (e.g. `matplotlib` colorbar
  side-by-side, or a gradient strip object in the Blender scene).
- **Peak marker:** drop a small marker (like Split A's anchor spheres) at the peak-stress vertex so the
  hot spot is unambiguous.

---

## 6. Rendering (reuse, don't reinvent)

`split_a_localization/blender_render.py` already solves the hard parts headless — copy the approach:
- `_blender_bin()` resolver (`OSTEON_BLENDER` → PATH → macOS default) — see `split_a_localization/mcp_server.py`.
- STL import (`wm.stl_import` for Blender 4.x/5.x), a created camera + sun light, and **bbox-based
  framing** (the viewport `view3d` ops don't exist in `--background`).
- Output dispatch by extension: `.png` → render, `.blend` → `save_as_mainfile` (interactive), `view` →
  GUI only.
- For color: add a **vertex color attribute** (`mesh.color_attributes.new(domain='POINT', type='FLOAT_COLOR')`),
  write the per-vertex RGBA, and use a material with an **Attribute → Base Color/Emission** node so the
  colors show in a solid render.
- Save both a PNG (the deliverable) and a `.blend` (so the surgeon/engineer can orbit the hot spots),
  mirroring `open_in_blender()`.

Render at least two views (anterior + lateral) so a hot spot can't hide behind the mesh.

---

## 7. Fallback ladder (the map degrades with the solver)

The heat map must follow the same resilience contract as the report (`common/ladder.py`). Record the
source in the output so the picture never lies about its own fidelity:

| Rung | Field source | Map fidelity |
|---|---|---|
| 1 | full tet FEA (CalculiX) σ_vM per node | true distribution |
| 2 | reduced/voxel surrogate field | coarse distribution (label it "surrogate") |
| floor | analytic closed-form (e.g. beam-bending gradient along the load path) | indicative only — low confidence |

- The **floor must still produce a map** (a smooth analytic gradient from the load point), never raise.
- Stamp the rung / `solver_used` onto the legend so a coarse map is never mistaken for FEA truth.

---

## 8. Contract considerations (read before you touch `common/contracts.py`)

`StressReport` is **frozen** — adding a field needs a PR with all three owners approving
(STANDARDIZATION §3). Two clean options:

- **Preferred (no contract change):** return the artifact paths from the `render_stress_heatmap` MCP
  tool and write the files under `traces/` (or `OSTEON_TRACE_DIR`). The orchestrator/demo picks them up
  via the trace; B and C contracts are untouched.
- **If you want it in the contract:** add `heatmap_png_path: str` and `heatmap_blend_path: str` to
  `StressReport` via the 3-owner PR. Keep them optional/defaulted so older fixtures still validate.

Either way, log the artifact paths as a span (`trace.emit(heatmap_png=..., heatmap_blend=...)`) under the
existing `trace_id` so A→B→C stays one trace.

---

## 9. Guardrails

- Reuse `mesh-watertight-gate` (pre): don't render a garbage mesh.
- Add **`heatmap-field-gate` (pre-render):** reject if `stress_field` has NaN/∞, is empty, or its length
  ≠ vertex count — a bad field paints a meaningless map. Map to `RejectedOutput`/`ToolFailError` so the
  ladder advances to the coarser source.

---

## 10. Acceptance criteria (verifiable)

1. On the **cantilever benchmark** (SETUP §7), the hot spot lands at the fixed end (max moment) and the
   peak color matches `PL/Z` within **10%**.
2. On the **notched plate**, the map clearly concentrates color at the notch (visual Kt).
3. Color is **comparable across cases**: a low-load case renders mostly blue/green; an overloaded case
   shows red at/above yield — same color = same MPa (fixed `[0, yield]` normalization).
4. The **peak marker** coincides with `StressReport.peak_location`, and the legend's peak value equals
   `StressReport.peak_von_mises_MPa` (picture and numbers agree — same solve).
5. **Resilience:** force a solver timeout → the map still renders from the surrogate/floor field with the
   rung stamped on the legend; no crash.
6. PNG + `.blend` written for all 5 fixture candidates; opens in Blender showing the colored field.

---

## 11. How to run (target)

```bash
source .venv/bin/activate
export OSTEON_BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"   # if not on PATH

# end-to-end: A -> B (final implant) -> C (report + heat map)
python orchestrator.py fixtures/example_case.json

# heat map directly from a final candidate + its solved field
python -c "
import json
from split_c_evaluation import mcp_server, engine
from common.contracts import CaseSpec, ImplantCandidate
cand = ImplantCandidate(**json.load(open('split_b_synthesis/fixtures/implant_candidate_test_case_01.json')))
case = CaseSpec(**json.load(open('fixtures/example_case.json')))
field = engine.stress_field(cand, case)          # NEW: nodal sigma_vM from the FE solve
out = mcp_server.render_stress_heatmap(cand.mesh_path, field, case.implant_material['yield_MPa'])
print(out)                                        # {png_path, blend_path, peak_mpa, peak_location}
"
```

---

## 12. Implementation notes / starting points

- **Stress field plumbing:** `run_calculix` currently returns nothing usable for color — make it parse the
  `.frd` and return `{"nodes": [...], "von_mises": [...], "peak": ..., "peak_xyz": ...}`. `engine.stress_field()`
  then interpolates that onto the render STL's vertices.
- **Reuse the renderer:** factor the scene-building out of `blender_render.py` (it already has `build_scene`,
  `setup_camera_and_light`, `_blender_bin`) so Split C calls the same headless path with vertex colors instead
  of marker spheres. Avoid duplicating the camera/framing logic.
- **Keep it offline-capable:** like Split B, the heat map should render without gateway creds (the LLM only
  writes the natural-language summary, SETUP §2). The field + render must work air-gapped for the demo.
- **Files you'll touch:** `split_c_evaluation/mcp_server.py` (new tool), `split_c_evaluation/engine.py`
  (`stress_field` + wire the artifact into the report's trace), and a small shared/reused renderer. Confine
  changes to `split_c_evaluation/` except a reviewed renderer refactor.
