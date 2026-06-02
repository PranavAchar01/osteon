"""Phase 0 orchestrator: chains A -> (B <-> C) with the stage-level circuit breaker.

Runs GREEN today on the Phase 0 stubs. Each owner swaps real rungs into their split's
engine.py without touching this file. The circuit breaker is structural: every stage's
ladder floor never raises, so the loop cannot crash.
"""
import json
import sys
from pathlib import Path

from common.contracts import CaseSpec
from common.trace import LoopTrace
from split_a_localization.engine import run as localize
from split_b_synthesis.engine import run as synthesize
from split_c_evaluation.engine import run as evaluate

ROOT = Path(__file__).resolve().parent
MAX_ITERS = 5


def design_implant(case: CaseSpec) -> dict:
    trace = LoopTrace(case_id=case.case_id)
    plan = localize(case, trace.child("localize"))
    report = None
    candidate = None
    for i in range(MAX_ITERS):
        candidate = synthesize(
            {"plan": plan, "report": report, "iteration": i}, trace.child("synthesize")
        )
        report = evaluate({"candidate": candidate, "case": case}, trace.child("evaluate"))
        if report.passed:
            break
    return {"plan": plan, "candidate": candidate, "report": report, "trace_id": trace.trace_id}


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "fixtures" / "example_case.json")
    case = CaseSpec(**json.load(open(path)))
    result = design_implant(case)
    r = result["report"]
    print(
        json.dumps(
            {
                "passed": r.passed,
                "factor_of_safety": r.factor_of_safety,
                "solver_used": r.solver_used,
                "iterations": r.iteration + 1,
                "trace_id": result["trace_id"],
            },
            indent=2,
        )
    )
    print(f"trace written to: traces/{result['trace_id']}.jsonl")
