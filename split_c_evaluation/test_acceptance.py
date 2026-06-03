"""Split C acceptance — STANDARDIZATION.md §11 Definition of Done.

Proves: (1) the FEA matches analytic benchmarks within 10%; (2) the ladder degrades
rung1 -> rung2 -> floor and the floor never raises; (3) both guardrails fire.
Runs with ZERO dependency on Split A or B.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from common.contracts import CaseSpec, ImplantCandidate
from common.errors import RejectedOutput
from common.trace import LoopTrace
from split_c_evaluation import engine, fea
from split_c_evaluation.guardrails import mesh_watertight_gate, report_nan_gate

ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures"
E_TI, P = 110000.0, 700.0


def _pct(a, b):
    return 100.0 * abs(a - b) / abs(b)


# ----------------------------- fixtures ----------------------------------- #
@pytest.fixture(scope="module")
def watertight_stl():
    import trimesh

    FIX.mkdir(parents=True, exist_ok=True)
    p = FIX / "block_watertight.stl"
    trimesh.creation.box(extents=(96.0, 14.0, 4.0)).export(p)
    return str(p)


@pytest.fixture(scope="module")
def open_stl():
    import trimesh

    FIX.mkdir(parents=True, exist_ok=True)
    box = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
    box.faces = box.faces[:-4]  # delete faces -> a hole -> not watertight
    p = FIX / "block_open.stl"
    box.export(p)
    return str(p)


@pytest.fixture(scope="module")
def case():
    return CaseSpec(**json.load(open(ROOT / "fixtures" / "example_case.json")))


@pytest.fixture(scope="module")
def candidate():
    return ImplantCandidate(**json.load(open(ROOT / "fixtures" / "example_implant_candidate.json")))


# ------------------- 1. FEA within 10% of analytic ------------------------ #
def test_axial_fea_within_10pct():
    g = fea.BeamGeom(L=100.0, b=10.0, h=10.0)
    fe = fea.solve_block_fea(g, E_TI, P, mode="axial")
    an = fea.analytic_axial(g, E_TI, P)
    assert _pct(fe["peak_von_mises_MPa"], an["peak_von_mises_MPa"]) < 10.0
    assert _pct(fe["displacement_max_mm"], an["displacement_max_mm"]) < 10.0


def test_cantilever_fea_within_10pct():
    g = fea.BeamGeom(L=100.0, b=10.0, h=10.0)
    fe = fea.solve_block_fea(g, E_TI, P, mode="cantilever")
    an = fea.analytic_cantilever(g, E_TI, P)
    # tip deflection
    assert _pct(fe["displacement_max_mm"], an["displacement_max_mm"]) < 10.0
    # mid-span top-fibre bending stress vs beam theory at x = L/2
    c, s = fe["centroids"], fe["stress"]
    sel = (np.abs(c[:, 0] - g.L / 2) < g.L * 0.06) & (c[:, 2] > g.h / 2 - g.h * 0.2)
    sxx = float(np.abs(s[sel, 0]).max())
    assert _pct(sxx, fea.analytic_cantilever_sigma_at(g, P, g.L / 2)) < 10.0


def test_compression_contact_within_10pct():
    # simplest contact case: a block bearing on a flat platen -> uniform normal (bearing)
    # stress P/A. (Bearing stress is the validated quantity for a flat contact.)
    g = fea.BeamGeom(L=30.0, b=20.0, h=20.0)
    fe = fea.solve_block_fea(g, E_TI, P, mode="axial")
    assert _pct(fe["peak_von_mises_MPa"], P / g.area) < 10.0


def test_surrogate_is_exact_beam_theory():
    g = fea.BeamGeom(L=100.0, b=10.0, h=10.0)
    su = fea.surrogate_beam_fea(g, E_TI, P, mode="three_point")
    an = fea.analytic_three_point(g, E_TI, P)
    assert _pct(su["peak_von_mises_MPa"], an["peak_von_mises_MPa"]) < 1.0
    assert _pct(su["displacement_max_mm"], an["displacement_max_mm"]) < 1.0


def test_kt_notched_plate_matches_textbook():
    # d/w -> 0 approaches the Kirsch value Kt = 3.0
    assert _pct(fea.stress_concentration_factor_hole(0.01, 100.0), 3.0) < 5.0
    # d/w = 0.5: Howland finite-width Kt ~ 2.16
    assert _pct(fea.stress_concentration_factor_hole(50.0, 100.0), 2.16) < 10.0


def test_shielding_index_toy_pair():
    assert _pct(fea.shielding_index(1.0, 0.6), 0.6) < 1.0
    assert fea.shielding_index(1.0, 1.0) == pytest.approx(1.0)
    # a stiffer implant shields more -> lower index (monotonic)
    soft = fea.shielding_index(1.0 / 1.0, 1.0 / (1.0 + 0.2) ** 2 * 1.0)
    stiff = fea.shielding_index(1.0 / 1.0, 1.0 / (1.0 + 5.0) ** 2 * 1.0)
    assert stiff < soft


# --------------------------- 2. the ladder -------------------------------- #
def test_rung1_full_fea_passes(candidate, case):
    rep = engine.run(
        {"candidate": candidate, "case": case}, LoopTrace(case.case_id, stage="evaluate")
    )
    assert rep.solver_used == "full_fea" and rep.fallback_rung == 1
    assert rep.passed is True and rep.factor_of_safety > 1.0
    assert 0.0 <= rep.stress_shielding_index <= 1.0


def test_ladder_falls_to_surrogate(candidate, case, monkeypatch):
    monkeypatch.setenv("OSTEON_FORCE_FAIL", "evaluate")  # kill rung 1
    rep = engine.run(
        {"candidate": candidate, "case": case}, LoopTrace(case.case_id, stage="evaluate")
    )
    assert rep.solver_used == "reduced_surrogate" and rep.fallback_rung == 2
    assert rep.passed is True


def test_ladder_falls_to_floor_never_raises(candidate, case, monkeypatch):
    # inject failure into BOTH FE rungs; the deterministic floor must still return a report
    def boom(inp, trace):
        raise engine.RetryableError("injected solver failure")

    monkeypatch.setattr(engine, "_rung1", boom)
    monkeypatch.setattr(engine, "_rung2", boom)
    run = engine.with_fallback([engine._rung1, engine._rung2], engine._floor)
    rep = run({"candidate": candidate, "case": case}, LoopTrace(case.case_id, stage="evaluate"))
    assert rep.solver_used == "analytic_fallback" and rep.fallback_rung == "floor"
    assert math.isfinite(rep.factor_of_safety) and rep.confidence < 0.5


# ------------------------- 3. guardrails fire ----------------------------- #
def test_report_nan_gate_rejects_nan(candidate, case):
    rep = engine._floor({"candidate": candidate, "case": case}, LoopTrace("t", stage="evaluate"))
    bad = rep.model_copy(update={"peak_von_mises_MPa": float("nan")})
    with pytest.raises(RejectedOutput):
        report_nan_gate(bad)
    worse = rep.model_copy(update={"factor_of_safety": -1.0})
    with pytest.raises(RejectedOutput):
        report_nan_gate(worse)


def test_mesh_watertight_gate_blocks_bad_mesh(watertight_stl, open_stl):
    assert mesh_watertight_gate(watertight_stl) is True
    with pytest.raises(RejectedOutput):
        mesh_watertight_gate(open_stl)
    with pytest.raises(RejectedOutput):
        mesh_watertight_gate("does/not/exist.stl")


# --------------------------- 4. MCP tools --------------------------------- #
def test_mcp_meshing_and_solve(watertight_stl):
    from split_c_evaluation import mcp_server

    fe_model = mcp_server.meshing_to_fe(watertight_stl)
    assert fe_model["validity"]["watertight"] is True
    assert os.path.exists(fe_model["inp_path"])
    res = mcp_server.run_calculix(fe_model["inp_path"], timeout_s=120)
    assert res["solver_used"] == "full_fea"
    assert math.isfinite(res["peak_von_mises_MPa"]) and res["peak_von_mises_MPa"] > 0


def test_mcp_shielding_solves_bone_twice(watertight_stl):
    from split_c_evaluation import mcp_server

    out = mcp_server.compute_shielding_index(watertight_stl, watertight_stl)
    assert out["bone_solves"] == 2  # intact + implanted
    assert 0.0 <= out["stress_shielding_index"] <= 1.0
    assert out["strain_energy_implanted"] <= out["strain_energy_intact"]
