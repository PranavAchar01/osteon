# STANDARDIZATION.md — Osteon Compatibility Contract

**Read this before writing a single line of code.** This document is the contract that lets three people build three independent parts that snap together with zero rework on integration day. If your code conforms to everything here, it is compatible by construction.

Conformance is the **Definition of Done**. A split is not "finished" until it (1) reads/writes the exact contracts below, (2) calls models only through `common/llm.py`, (3) raises only the shared errors, (4) wraps its logic in the standard fallback ladder, (5) emits the standard trace, and (6) passes its acceptance test with one injected failure.

---

## 0. Golden rules (non-negotiable)

1. **Nobody calls Bedrock or any model directly.** All model calls go through `common/llm.call_llm(...)`. Routing/fallback/retries live in `gateway/routing.yaml`, never in app code.
2. **Nobody calls a tool directly across stages.** Tools are MCP tools registered in `gateway/mcp-registry.yaml` and called through the MCP Gateway, so every call is scoped and audited.
3. **Stages communicate only through the frozen contracts in §3.** No stage reads another stage's internals.
4. **No function that touches a model or tool may raise to the orchestrator.** It must go through the ladder (§6); the rung-3 floor always returns a valid object.
5. **Every model/tool function takes `trace: LoopTrace` and emits a span.** (§7)
6. **Config is code.** Gateway routing, guardrails, and MCP registry are YAML in `gateway/`, applied with `tfy apply`, changed only via PR.
7. **`common/` is frozen after Phase 0.** Changes to it require all three people to agree in the PR.

---

## 1. Tooling (identical for everyone)

| Concern | Standard |
|---|---|
| Language | Python **3.11** (exact) |
| Env / deps | `uv`; deps declared in `pyproject.toml` with extras `[localization]`, `[synthesis]`, `[evaluation]` |
| Format / lint | `black` (line length 100) + `ruff`; run `ruff check . && black --check .` before every commit |
| Types | `pydantic` v2 models for all contracts; type hints everywhere |
| Tests | `pytest`; each split owns `test_acceptance.py` |
| Units | **millimeters** for length, **newtons** for force, **MPa** for stress, **degrees** for angles — everywhere, no exceptions |
| Coordinate frame | right-handed, +Z = bone long axis (proximal positive); meshes are watertight STL in mm |

---

## 2. Environment variables (`.env`, never committed)

```
TFY_TOKEN=<virtual-account-token>
TFY_GATEWAY_URL=https://gateway.truefoundry.ai
OTEL_EXPORTER_OTLP_ENDPOINT=<optional, for Grafana/Datadog>
OSTEON_TRACE_DIR=./traces            # JSONL fallback sink
OSTEON_DEADLINE_MS=20000             # per-stage wall-clock deadline
```

`common/settings.py` is the only place these are read (via `pydantic-settings`). Do not call `os.environ` anywhere else.

---

## 3. Data contracts (the freeze) — `common/contracts.py`

These pydantic models are the integration interface. Field names, types, and units are fixed. Add fields only by PR with all three approving.

```python
from pydantic import BaseModel
from typing import Literal

class Vec3(BaseModel):
    x: float; y: float; z: float            # mm

class CaseSpec(BaseModel):                  # SYSTEM INPUT
    case_id: str
    bone_mesh_path: str                     # watertight STL, mm
    bone_material: dict                      # {E_cortical_MPa, E_trabecular_MPa, density}
    defect: dict                             # {type: "fracture"|"resection"|"void", region, severity, description}
    load_profile: list                       # [{name, force_vector_N: Vec3, application_region, cycles}]
    implant_material: dict                   # {name, E_MPa, yield_MPa, endurance_limit_MPa}
    constraints: dict                        # {min_thickness_mm, max_volume_mm3, process: "additive"|"subtractive"}

class AnchorPoint(BaseModel):
    id: str; xyz: Vec3; normal: Vec3
    cortical_thickness_mm: float

class PlacementPlan(BaseModel):             # A -> B
    case_id: str
    coordinate_frame: dict                   # {origin: Vec3, basis: 3x3 list}
    anchor_points: list[AnchorPoint]
    resection_planes: list                   # [{point: Vec3, normal: Vec3}]
    defect_region: dict                      # {centroid: Vec3, obb, volume_mm3}
    fit_target_surface_path: str             # submesh STL the implant must conform to
    confidence: float                        # 0..1
    fallback_rung: int | Literal["floor"]
    trace_id: str

class ImplantCandidate(BaseModel):          # B -> C
    case_id: str; candidate_id: str; iteration: int
    parameter_vector: dict                   # named theta
    mesh_path: str                           # watertight STL
    contacts_anchor_ids: list[str]
    volume_mm3: float; min_thickness_mm: float
    validity: dict                           # {watertight: bool, manifold: bool, self_intersect: bool}
    fallback_rung: int | Literal["floor"]
    trace_id: str

class StressReport(BaseModel):              # C -> B
    case_id: str; candidate_id: str; iteration: int
    peak_von_mises_MPa: float; peak_location: Vec3
    factor_of_safety: float
    fatigue_safe: bool
    stress_shielding_index: float            # 0 = full shielding, 1 = natural bone
    displacement_max_mm: float
    passed: bool
    solver_used: Literal["full_fea", "reduced_surrogate", "analytic_fallback"]
    confidence: float
    fallback_rung: int | Literal["floor"]
    trace_id: str
```

**Producer/consumer map (who owns what):**

| Contract | Produced by | Consumed by |
|---|---|---|
| `CaseSpec` | system / fixture | A |
| `PlacementPlan` | A | B |
| `ImplantCandidate` | B | C |
| `StressReport` | C | B (controller) |

Every split commits **5 fixture files of its own output** under `split_x/fixtures/` so the downstream split can develop without the upstream split existing.

---

## 4. Error taxonomy — `common/errors.py`

Raise only these. Guardrails and the ladder branch on `.code`.

```python
class OsteonError(Exception):
    code: str = "E_UNKNOWN"

class RetryableError(OsteonError): ...      # transient — ladder advances to next rung
class RateLimitError(RetryableError):  code = "E_RATE_LIMIT"
class ProviderOutageError(RetryableError): code = "E_PROVIDER_OUTAGE"   # 5xx
class TimeoutError_(RetryableError):   code = "E_TIMEOUT"
class ToolFailError(RetryableError):   code = "E_TOOL_FAIL"
class RejectedOutput(OsteonError):     code = "E_BAD_OUTPUT"            # raised by a guardrail
class CascadeError(OsteonError):       code = "E_CASCADE"
```

Map every external failure onto one of these at the boundary. A raw `openai.APIStatusError(5xx)` becomes `ProviderOutageError`; a guardrail rejection becomes `RejectedOutput`; a Blender/CalculiX crash becomes `ToolFailError`.

---

## 5. The one gateway client — `common/llm.py`

```python
from openai import OpenAI
from common.settings import settings
client = OpenAI(api_key=settings.TFY_TOKEN, base_url=settings.TFY_GATEWAY_URL)

def call_llm(*, stage: str, messages, model: str = "bedrock/claude-sonnet", **kw):
    """The ONLY entry point to a model. `stage` is mandatory and tags the trace + gateway routing."""
    return client.chat.completions.create(
        model=model, messages=messages,
        extra_headers={"x-tfy-metadata": f'{{"stage":"{stage}"}}'}, **kw)
```

Rules: always pass `stage`. Never hardcode a provider key. Never add retry/fallback logic here — that is the gateway's job (`routing.yaml`). If you need a different model as your rung-1, change the `model=` default for your stage only and document it in your `SETUP.md`.

**Model aliases (set once in the gateway, used everywhere):** `bedrock/claude-sonnet` (rung 1), `bedrock/llama-70b` (rung 2), `bedrock/mistral-large` (rung 2b).

---

## 6. The standardized fallback ladder — `common/ladder.py`

Every stage's `run()` is built from this. **Identical shape in all three splits.** The floor never raises.

```python
def with_fallback(rungs, floor):
    """rungs: list of callables (best -> worse); floor: deterministic callable that cannot fail."""
    def run(inp, trace):
        for i, rung in enumerate(rungs, start=1):
            try:
                out = rung(inp, trace); trace.emit(rung=i, fallback=i > 1); return out
            except (RetryableError, RejectedOutput) as e:
                trace.emit(rung=i, error=e.code, fallback=True); continue
        out = floor(inp, trace)                       # deterministic local floor
        trace.emit(rung="floor", fallback=True, confidence="low")
        return out                                     # NEVER raises -> loop cannot break
    return run
```

**Required rung structure per stage (so resilience is uniform):**

| Stage | Rung 1 (LLM/tool via gateway) | Rung 2 (degraded) | Rung 3 (floor, deterministic) |
|---|---|---|---|
| A | ML landmark regressor | geometric heuristic (PCA + curvature) | conservative default frame, `confidence` low |
| B | LLM θ-proposer (Bedrock) | CMA-ES numeric optimizer (no LLM) | last-known-good θ, stop flag |
| C | full tet FEA (CalculiX) | reduced voxel/surrogate FEA | analytic closed-form bound |

The result object's `fallback_rung` field must record which rung produced it.

---

## 7. Observability standard — `common/trace.py`

One trace per case, one span per stage attempt. Spans are OpenTelemetry (gateway is OTel-compliant) **and** mirrored to JSONL at `OSTEON_TRACE_DIR` for the demo.

Mandatory span fields:
```
trace_id, span_id, stage, rung, fallback (bool), error (code|null),
model_or_tool, latency_ms, input_hash, output_hash, confidence
```

Rules: never log raw patient/case payloads (hash them); one `trace_id` flows A→B→C and is carried in every contract's `trace_id` field; the orchestrator opens the root span and passes `trace` down.

---

## 8. Config-as-code conventions (`gateway/`)

All three files are applied with `tfy apply -f` and edited only via PR.

- **`routing.yaml`** — `type: gateway-load-balancing-config`. One rule per stage model; rung-1 target `weight: 100`, fallback targets `weight: 0, fallback: true`; `model_configs` set `max_failures` + `cooldown_seconds: 300`.
- **`guardrails.yaml`** — `type: gateway-guardrails-config`. Hooks are exactly: `llm_input_guardrails`, `llm_output_guardrails`, `mcp_tool_pre_invoke_guardrails`, `mcp_tool_post_invoke_guardrails`. Scope each rule by `model` or `mcpServers`.
- **`mcp-registry.yaml`** — register one MCP server per split, aggregator mode, exposing only that split's tools.

> Note: field names above reflect TrueFoundry's documented config shape; confirm exact keys against your deployed gateway version's docs before the first `tfy apply`, then freeze.

### Naming conventions (so configs don't collide)
- MCP servers: `localization-mcp`, `blender-mcp`, `fea-mcp`
- Guardrails: `<domain>/<check>` → `implant/theta-bounds-check`, `implant/mesh-validity-check`, `implant/report-nan-gate`, `pii/pii-detection`
- Routing rules: `<stage>-route` → `proposer-route`, `localizer-route`, `evaluator-route`
- Stage tags (in `call_llm`): `localize`, `synthesize`, `evaluate`

---

## 9. MCP tool pattern (identical scaffolding)

Each split exposes its tools from `split_x/mcp_server.py` using the same skeleton. Tools are pure functions over the contracts; no tool may write outside `OSTEON_TRACE_DIR` or the split's `fixtures/`.

```python
# split_x/mcp_server.py  (FastMCP-style skeleton — keep this shape in all splits)
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("blender-mcp")               # name MUST match mcp-registry.yaml

@mcp.tool()
def generate_mesh(theta: dict) -> dict:
    """Returns {mesh_path, validity}. Raises ToolFailError on failure (never a bare Exception)."""
    ...

if __name__ == "__main__":
    mcp.run()
```

**Tool scoping rules:** expose the minimum tools needed; never expose a delete/overwrite tool; expensive tools (`generate_mesh`, `run_calculix`) must accept a timeout and enforce a result-size bound; risky tools must have a `mcp_tool_pre_invoke` guardrail registered.

### Guardrail responsibilities per split
| Split | pre-invoke (before tool runs) | post-invoke (before model/next stage sees result) |
|---|---|---|
| A | `coords-in-bbox-check` | `landmark-sanity` (confidence floor) |
| B | `implant/theta-bounds-check` | `implant/mesh-validity-check` |
| C | `mesh-watertight-gate` | `implant/report-nan-gate` |

---

## 10. Git workflow

- Default branch `main` is protected; no direct pushes.
- Branch names: `a/<topic>`, `b/<topic>`, `c/<topic>`, `common/<topic>`, `gateway/<topic>`.
- Commits: imperative, scoped — `b: add CMA-ES rung-2 proposer`.
- PRs require: green `ruff` + `black` + `pytest`, and for `common/` or `gateway/` changes, **all three reviewers**.
- Never commit `.env`, `traces/`, large meshes, or solver binaries (`.gitignore` them; meshes go in `fixtures/` only if < 5 MB).

---

## 11. Definition of Done (per split)

A split is mergeable to `main` only when all are true:
1. Reads its input contract and writes its output contract exactly (validated by pydantic).
2. All model calls go through `common/llm.call_llm` with the correct `stage` tag.
3. All cross-stage tool calls go through its registered MCP server.
4. Logic is wrapped in `with_fallback` with the three rungs from §6; floor never raises.
5. Its two guardrails (§9) are defined in `guardrails.yaml` and demonstrably fire.
6. Emits standard spans (§7) with one `trace_id`.
7. `test_acceptance.py` passes **and** passes again with one injected failure (model killed / tool timeout / bad output), proving recovery.
8. Ships 5 output fixtures and a complete `SETUP.md`.

---

## 12. Integration day checklist (shared)

- [ ] `common/` and `gateway/*.yaml` frozen and applied.
- [ ] Each split passes its acceptance test independently.
- [ ] `orchestrator.py` chains A→B↔C using only contracts + `with_fallback`.
- [ ] One end-to-end case produces a valid implant.
- [ ] Live failure demo (revoke Bedrock key, force CalculiX timeout, inject bad θ) shows recovery in a single trace.
- [ ] Equal-work audit: each split = engine + LLM-via-gateway (1+1 fallback) + 2 MCP tools + 2 guardrails + ladder + SETUP + demo.
