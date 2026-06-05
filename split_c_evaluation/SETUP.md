# Split C — Biomechanical Evaluation & Stress Oracle  (Owner: Person C)

> Read `../STANDARDIZATION.md` first. This file is your complete work assignment + setup guide.

**One-liner:** Decide whether a candidate survives.

**You own:** `ImplantCandidate` + `CaseSpec` loads → `StressReport`  (stage tag: `evaluate`)

---

## 1. Engine work

1. Mesh → FE model: tetrahedralize the implant (+ a bone block around the anchors), apply material properties from `CaseSpec`, apply boundary conditions (fixed distal, load vectors from `load_profile`).
2. Solve with CalculiX (`ccx`, generated `.inp`) for static von Mises stress and displacement.
3. Compute factor of safety (yield / peak) and fatigue check (peak vs endurance limit under cyclic load).
4. **Stress-shielding index:** solve the bone twice — intact vs implanted — and take the ratio of bone strain energy (Wolff's-law concern; this is what makes it a real tool, not a toy).

Output must be a valid `StressReport` with `solver_used` recorded.

---

## 2. Model usage (Bedrock via AI Gateway)

- **Rung 1 model:** `bedrock/claude-sonnet`  —  **gateway fallback:** `bedrock/llama-70b`
- **What the LLM does:** triage solver failures (“CalculiX diverged — retry coarser or fall to surrogate?”) and write the natural-language `StressReport` summary for the demo.
- Call only through `common.llm.call_llm(stage="evaluate", ...)`.

---

## 3. MCP tools  (server name: `fea-mcp`)

| Tool | Purpose |
|---|---|
| `meshing_to_fe(mesh)` | tetrahedralize + assign materials + BCs → `.inp` |
| `run_calculix(inp)` | run solver (expensive — enforce timeout + result-size bound) |
| `compute_shielding_index(intact, implanted)` | strain-energy ratio |

---

## 4. Guardrails

| Hook | Name | Behavior |
|---|---|---|
| pre-invoke | `mesh-watertight-gate` | block the solver call on an invalid mesh (don't burn 90s on garbage) |
| post-invoke | `implant/report-nan-gate` | reject StressReports with NaN / ∞ / negative FoS before B consumes them (kills cascading errors) |

---

## 5. Fallback ladder (`common/ladder.py`)

| Rung | Method |
|---|---|
| 1 | full tet FEA (CalculiX) |
| 2 | reduced voxel / surrogate FEA |
| floor | analytic closed-form bound (beam/plate), `solver_used="analytic_fallback"`, low confidence, never raises |

Record the rung in `StressReport.fallback_rung` and the solver in `solver_used`.

---

## 6. Full setup

```bash
# prereq: Phase 0 done; .env has TFY_TOKEN, TFY_GATEWAY_URL
git clone https://github.com/PranavAchar01/osteon.git && cd osteon
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[evaluation]" rtree     # sfepy, trimesh, pycalculix, numpy/scipy (+rtree for trimesh)

# CalculiX (ccx) has no Homebrew formula on macOS, so the FEA runs on sfepy (a
# pure-Python 3D linear-elastic solver) as the rung-1 "full_fea" engine. If a `ccx`
# binary is on PATH (Linux apt / conda-forge), run_calculix prefers it automatically:
#   linux:  sudo apt-get install -y calculix-ccx   &&  ccx -v
#   conda:  conda install -c conda-forge calculix

# gateway configs already declare evaluate-guards + fea-mcp; apply when a gateway exists:
# tfy apply -f gateway/guardrails.yaml        # implant/mesh-watertight-gate, implant/report-nan-gate
# tfy apply -f gateway/mcp-registry.yaml      # fea-mcp

python split_c_evaluation/mcp_server.py &     # serve fea-mcp tools
python -m split_c_evaluation.engine           # evaluate the example candidate -> StressReport
pytest split_c_evaluation/test_acceptance.py -q
```

---

## 7. Acceptance test (your verifiable output)

Validate against **analytic benchmarks** with known answers:
- cantilever beam (max stress = PL/Z)
- notched plate (known stress-concentration factor Kt)
- a simple contact case

**Pass:** FEA result within **10%** of the analytic answer on all three; stress-shielding index computed correctly on an intact-vs-implanted toy pair.

**Resilience demo:** force a solver timeout → trace shows rung1→rung2→floor with `solver_used` recorded; injected NaN report caught by the post-invoke guardrail. Runs with **zero dependency on A or B.**

Ship 5 `StressReport` fixtures under `fixtures/` so B's controller can be tested without you.

---

## 8. Implementation status — DONE ✅

| File | What it does |
|---|---|
| `fea.py` | sfepy 3D linear-elastic solver (`solve_block_fea`, modes axial/cantilever/three_point), 1D Euler-Bernoulli surrogate (`surrogate_beam_fea`), closed-form analytic bounds, notched-plate Kt, `shielding_index`. |
| `engine.py` | `run = with_fallback([_rung1, _rung2], _floor)`. rung1 `full_fea` (sfepy), rung2 `reduced_surrogate`, floor `analytic_fallback`. Derives geometry/load/material from the contracts; best-effort LLM summary via `call_llm(stage="evaluate")`. |
| `guardrails.py` | `mesh_watertight_gate` (pre-invoke), `report_nan_gate` (post-invoke) → raise `RejectedOutput`. |
| `mcp_server.py` | `fea-mcp`: `meshing_to_fe`, `run_calculix` (ccx-if-present else sfepy, SIGALRM timeout), `compute_shielding_index`. |
| `test_acceptance.py` | 11 tests; `fixtures/stress_report_0{1..5}.json` ship 5 reports across all tiers. |

**Design choices (all defensible, documented in code):**
- **sfepy is the rung-1 FEA** — CalculiX has no macOS Homebrew formula; `run_calculix` auto-prefers `ccx` if the binary appears. Contract tier stays `full_fea`.
- **Bending model = three-point (ASTM F382-style)** — the standard bone-plate bench test (M = PL/4). Default reference load 700 N when `load_profile` is empty.
- **Shielding = composite-beam strain-energy ratio** `(EI_bone/(EI_bone+EI_implant))²`.

**Verified benchmarks (real sfepy 3D FEA vs analytic):** axial σ & δ **0–0.4%**; cantilever tip δ **0.3%**, mid-span bending σ **4.3%**; surrogate **<1%**; Kt(d/w→0)=3.0 and Kt(d/w=0.5)≈2.16 within 10%. The three-point peak von Mises sits ~18% below 1-D beam theory — expected St-Venant load-spreading; the 3-D field is the more accurate truth, the analytic bound is the floor.

**Run the resilience demo:**
```bash
python -m split_c_evaluation.engine                       # rung1 full_fea: FoS 2.26, passed
OSTEON_FORCE_FAIL=evaluate python -m split_c_evaluation.engine   # -> rung2 reduced_surrogate
```
