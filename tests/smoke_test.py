"""Phase 0 smoke test - proves the shared foundation works identically for everyone.

Run from the repo root:  pytest -q
No credentials required for any test here.
"""
import json
from pathlib import Path

from common.contracts import CaseSpec, ImplantCandidate, PlacementPlan, StressReport
from common.errors import RetryableError
from common.ladder import with_fallback
from common.settings import settings
from common.trace import LoopTrace

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "fixtures"


def test_settings_loads():
    assert settings.TFY_GATEWAY_URL.startswith("http")


def test_golden_fixtures_validate():
    CaseSpec(**json.load(open(FIX / "example_case.json")))
    PlacementPlan(**json.load(open(FIX / "example_placement_plan.json")))
    ImplantCandidate(**json.load(open(FIX / "example_implant_candidate.json")))
    StressReport(**json.load(open(FIX / "example_stress_report.json")))


def test_ladder_falls_to_floor():
    def bad(inp, trace):
        raise RetryableError("boom")

    def floor(inp, trace):
        return "floor-result"

    run = with_fallback([bad, bad], floor)
    assert run(None, LoopTrace("t")) == "floor-result"


def test_trace_writes_jsonl():
    t = LoopTrace("smoke")
    t.emit(rung=1, fallback=False)
    assert t._path.exists()


def test_orchestrator_runs_green():
    from orchestrator import design_implant

    case = CaseSpec(**json.load(open(FIX / "example_case.json")))
    result = design_implant(case)
    assert result["report"].passed is True
    assert result["report"].factor_of_safety > 1.0
