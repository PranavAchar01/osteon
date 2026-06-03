"""Split B acceptance test (STANDARDIZATION §11.7) - runs fully offline.

Drives the B<->C loop with a mock analytic stress oracle (the C-simulator), and proves the
two resilience paths: an out-of-bounds theta is blocked by the pre-invoke guardrail before
generate_mesh runs, and a killed LLM falls through to the CMA-ES rung which still converges.
"""

import json
from pathlib import Path

import pytest

from common.contracts import CaseSpec, PlacementPlan, StressReport, Vec3
from common.errors import RejectedOutput
from common.trace import LoopTrace
from split_b_synthesis import engine, mcp_server

ROOT = Path(__file__).resolve().parents[1]
PLAN_FIXTURE = ROOT / "split_a_localization" / "fixtures" / "placement_plan_test_case_01.json"
CASE_FIXTURE = ROOT / "fixtures" / "example_case.json"


# --- Block 2.1: mock analytic stress oracle (test double for Split C) ----------------------
def make_mock_oracle(case: CaseSpec):
    """Closed-form beam bending (SETUP §7): peak von Mises ~ load / (thickness * width^2)."""
    yield_mpa = float(case.implant_material.get("yield_MPa", 830.0))
    endurance = float(case.implant_material.get("endurance_limit_MPa", 510.0))
    load_n = 700.0
    if case.load_profile:
        fv = case.load_profile[0].get("force_vector_N", {})
        mag = (fv.get("x", 0.0) ** 2 + fv.get("y", 0.0) ** 2 + fv.get("z", 0.0) ** 2) ** 0.5
        load_n = mag or 700.0

    def oracle(theta: dict) -> StressReport:
        thickness, width = theta["thickness_mm"], theta["width_mm"]
        peak = 550.0 * load_n / (thickness * width**2)
        fos = yield_mpa / peak
        ssi = max(0.0, min(1.0, 1.0 - thickness / 16.0))
        return StressReport(
            case_id=case.case_id,
            candidate_id="mock",
            iteration=0,
            peak_von_mises_MPa=peak,
            peak_location=Vec3(x=0.0, y=0.0, z=0.0),
            factor_of_safety=fos,
            fatigue_safe=bool(peak < endurance),
            stress_shielding_index=ssi,
            displacement_max_mm=0.3,
            passed=bool(fos >= 1.5 and 0.6 <= ssi <= 0.9),
            solver_used="analytic_fallback",
            confidence=0.7,
            fallback_rung="floor",
            trace_id=case.case_id,
        )

    return oracle


# --- fake gateway response so the LLM rung can run offline ---------------------------------
class _FakeResp:
    def __init__(self, content: str):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]


@pytest.fixture
def plan() -> PlacementPlan:
    return PlacementPlan(**json.load(open(PLAN_FIXTURE)))


@pytest.fixture
def case() -> CaseSpec:
    return CaseSpec(**json.load(open(CASE_FIXTURE)))


def _assert_valid(candidate):
    v = candidate.validity
    assert v["watertight"] and v["manifold"] and not v["self_intersect"], v


# --- Block 2.4: convergence via the controller loop (rung 1 / LLM path) -------------------
def test_convergence_drives_passed_within_n_iters(monkeypatch, plan, case):
    oracle = make_mock_oracle(case)
    monkeypatch.setattr(engine, "STRESS_ORACLE", oracle)

    # Mock the gateway: an improving proposer that steps thickness down out of the shielding zone.
    steps = {"n": 0}
    schedule = [7.0, 6.0, 5.0, 5.0, 5.0, 5.0]

    def fake_call_llm(*, stage, messages, model="bedrock/claude-sonnet", **kw):
        assert stage == "synthesize"
        thickness = schedule[min(steps["n"], len(schedule) - 1)]
        steps["n"] += 1
        return _FakeResp(json.dumps({"thickness_mm": thickness, "width_mm": 14.0}))

    monkeypatch.setattr(engine.llm, "call_llm", fake_call_llm)

    trace = LoopTrace(plan.case_id, trace_id=plan.trace_id, stage=None)
    report, candidate, iters = None, None, 0
    for i in range(8):
        iters = i + 1
        candidate = engine.run(
            {"plan": plan, "report": report, "iteration": i}, trace.child("synthesize")
        )
        assert candidate.fallback_rung == 1  # rung 1 (LLM) succeeded
        _assert_valid(candidate)
        # 100% valid after repair_mesh, too
        repaired = mcp_server.repair_mesh(candidate.mesh_path)["validity"]
        assert repaired["watertight"] and repaired["manifold"] and not repaired["self_intersect"]
        report = oracle(candidate.parameter_vector)
        if report.passed:
            break

    assert report.passed, f"did not converge in {iters} iters: {report}"
    assert iters <= 5


# --- Block 2.4: injected failure 1 - bad output blocked before generate_mesh --------------
def test_bad_output_guardrail_blocks_generate_mesh(monkeypatch, plan, case):
    monkeypatch.setattr(engine, "STRESS_ORACLE", make_mock_oracle(case))

    def fake_call_llm(*, stage, messages, model="bedrock/claude-sonnet", **kw):
        return _FakeResp(json.dumps({"thickness_mm": 99.0}))  # wildly out of bounds

    monkeypatch.setattr(engine.llm, "call_llm", fake_call_llm)

    calls = []
    real_generate = engine.generate_mesh

    def spy(theta):
        calls.append(theta)
        return real_generate(theta)

    monkeypatch.setattr(engine, "generate_mesh", spy)

    trace = LoopTrace(plan.case_id, stage=None)
    inp = {"plan": plan, "report": None, "iteration": 0}

    # rung 1 in isolation: the pre-invoke guardrail rejects the theta, generate_mesh never runs.
    with pytest.raises(RejectedOutput):
        engine._rung1(inp, trace.child("synthesize"))
    assert calls == [], "generate_mesh must NOT be called for out-of-bounds theta"

    # full ladder: RejectedOutput advances rung 1 -> rung 2 (CMA-ES), which produces a valid candidate.
    calls.clear()
    candidate = engine.run(inp, trace.child("synthesize"))
    assert candidate.fallback_rung == 2
    _assert_valid(candidate)
    assert len(calls) >= 1  # rung 2 did call generate_mesh


# --- Block 2.4: injected failure 2 - model killed -> CMA-ES still converges ----------------
def test_model_killed_falls_to_cma(monkeypatch, plan, case):
    oracle = make_mock_oracle(case)
    monkeypatch.setattr(engine, "STRESS_ORACLE", oracle)
    monkeypatch.setenv("OSTEON_FORCE_FAIL", "synthesize")  # kill rung 1

    trace = LoopTrace(plan.case_id, trace_id=plan.trace_id, stage=None)
    report, candidate = None, None
    for i in range(4):
        candidate = engine.run(
            {"plan": plan, "report": report, "iteration": i}, trace.child("synthesize")
        )
        assert candidate.fallback_rung == 2, "LLM killed -> must land on the CMA-ES rung"
        _assert_valid(candidate)
        report = oracle(candidate.parameter_vector)
        if report.passed:
            break

    assert report.passed, f"CMA-ES did not converge: {report}"


# --- Block 2.4: trace assertion - failed attempts carry rung/fallback/error ----------------
def test_trace_records_rung_fallback_and_error(monkeypatch, plan, case):
    monkeypatch.setattr(engine, "STRESS_ORACLE", make_mock_oracle(case))
    monkeypatch.setenv("OSTEON_FORCE_FAIL", "synthesize")  # rung 1 fails -> rung 2 succeeds

    trace = LoopTrace(plan.case_id, stage=None)
    engine.run({"plan": plan, "report": None, "iteration": 0}, trace.child("synthesize"))

    spans = [
        json.loads(line) for line in Path(trace._path).read_text().splitlines() if line.strip()
    ]
    failed = [s for s in spans if s.get("rung") == 1]
    assert failed, "expected a span for the failed rung-1 attempt"
    assert any(s.get("fallback") and s.get("error") == "E_RETRYABLE" for s in failed), spans
    assert any(s.get("rung") == 2 for s in spans), "expected a span for the rung-2 success"
