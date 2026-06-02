"""Split A engine - Localization & Anchoring.

PHASE 0 STUB: rungs return a conservative valid PlacementPlan so the loop runs green.
Person A replaces _rung1 with the real ML + geometric methods (see SETUP.md).
The contract (CaseSpec -> PlacementPlan) and the ladder shape are FINAL.
"""
import argparse
import json
import os
from pathlib import Path

from common.contracts import AnchorPoint, CaseSpec, PlacementPlan, Vec3
from common.errors import RetryableError
from common.ladder import with_fallback

ROOT = Path(__file__).resolve().parent.parent


def _build(case, trace, rung, confidence):
    return PlacementPlan(
        case_id=case.case_id,
        coordinate_frame={"origin": {"x": 0, "y": 0, "z": 0}, "basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        anchor_points=[
            AnchorPoint(id="a0", xyz=Vec3(x=-8, y=0, z=40), normal=Vec3(x=-1, y=0, z=0), cortical_thickness_mm=4.2),
            AnchorPoint(id="a1", xyz=Vec3(x=-8, y=0, z=-40), normal=Vec3(x=-1, y=0, z=0), cortical_thickness_mm=3.9),
        ],
        resection_planes=[],
        defect_region={"centroid": {"x": 0, "y": 0, "z": 0}, "obb": [], "volume_mm3": 320.0},
        fit_target_surface_path="fixtures/fit_target.stl",
        confidence=confidence,
        fallback_rung=rung,
        trace_id=trace.trace_id,
    )


def _rung1(case, trace):
    if os.getenv("OSTEON_FORCE_FAIL") == "localize":
        raise RetryableError("forced failure for demo")
    # TODO(A): ML landmark regressor + cortical-thickness-gated anchor selection
    return _build(case, trace, rung=1, confidence=0.91)


def _floor(case, trace):
    # TODO(A): conservative geometric default (PCA frame)
    return _build(case, trace, rung="floor", confidence=0.2)


run = with_fallback([_rung1], _floor)


if __name__ == "__main__":
    from common.trace import LoopTrace

    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=str(ROOT / "fixtures" / "example_case.json"))
    args = ap.parse_args()
    case = CaseSpec(**json.load(open(args.case)))
    plan = run(case, LoopTrace(case.case_id, stage="localize"))
    print(plan.model_dump_json(indent=2))
