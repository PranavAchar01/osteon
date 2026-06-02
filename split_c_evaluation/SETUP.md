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
uv venv && source .venv/bin/activate
uv pip install -e ".[evaluation]"          # pycalculix or ccx binary, sfepy/numpy, trimesh
sudo apt-get install -y calculix-ccx        # or build ccx; verify `ccx -v`

tfy apply -f gateway/guardrails.yaml        # mesh-watertight-gate, report-nan-gate
tfy apply -f gateway/mcp-registry.yaml      # fea-mcp
python split_c_evaluation/mcp_server.py &
python -m split_c_evaluation.engine --candidate split_c_evaluation/fixtures/candidate_frozen.stl
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
