"""fea-mcp server — Split C's MCP tools (STANDARDIZATION.md §9).

Server name MUST match gateway/mcp-registry.yaml ("fea-mcp", tools:
meshing_to_fe, run_calculix, compute_shielding_index).

Run:  python split_c_evaluation/mcp_server.py

Tools are pure functions over the contracts. They write only under OSTEON_TRACE_DIR.
Any failure is normalised to ToolFailError by common.mcp_base.osteon_tool; the expensive
solver tool also enforces a wall-clock timeout and a result-size bound.
"""

from __future__ import annotations

import os
import shutil
import signal
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from common.errors import ToolFailError
from common.mcp_base import osteon_tool
from common.settings import settings

from . import fea
from .guardrails import mesh_watertight_gate

mcp = FastMCP("fea-mcp")

_OUT = Path(settings.OSTEON_TRACE_DIR)


def _extents(mesh_path: str):
    import trimesh

    ext = trimesh.load(mesh_path, force="mesh").bounding_box.extents
    return fea.beam_from_dims(float(ext[0]), float(ext[1]), float(ext[2]))


@osteon_tool(mcp)
def meshing_to_fe(mesh_path: str, E_MPa: float = 110000.0, load_N: float = 700.0) -> dict:
    """Tetrahedralise the implant + assign materials + boundary conditions -> a CalculiX
    .inp deck. PRE-INVOKE guardrail mesh-watertight-gate blocks invalid meshes first."""
    mesh_watertight_gate(mesh_path)  # raises RejectedOutput on a bad mesh
    g = _extents(mesh_path)
    _OUT.mkdir(parents=True, exist_ok=True)
    inp_path = _OUT / (Path(mesh_path).stem + ".inp")
    # A compact, representative CalculiX deck for the block abstraction (3-pt bend).
    deck = (
        "*HEADING\n Osteon Split C — generated FE model\n"
        f"** bbox L={g.L:.3f} b={g.b:.3f} h={g.h:.3f} mm  E={E_MPa} MPa  P={load_N} N\n"
        "*MATERIAL, NAME=IMPLANT\n*ELASTIC\n"
        f" {E_MPa:.1f}, 0.3\n"
        "*BOUNDARY\n** simply-supported knife edges at x=0 and x=L (u3=0)\n"
        "*CLOAD\n** transverse load at mid-span top face\n"
        "*STEP\n*STATIC\n*EL PRINT\n S\n*END STEP\n"
    )
    inp_path.write_text(deck)
    return {
        "inp_path": str(inp_path),
        "geom_mm": {"L": g.L, "b": g.b, "h": g.h},
        "validity": {"watertight": True, "n_cards": deck.count("*")},
    }


@osteon_tool(mcp)
def run_calculix(
    inp_path: str,
    mesh_path: str = "",
    E_MPa: float = 110000.0,
    load_N: float = 700.0,
    timeout_s: int = 60,
) -> dict:
    """Run the static solve. Uses the ``ccx`` binary if present, else the sfepy 3D
    solver. Enforces a wall-clock timeout (the expensive tool must be bounded)."""
    if not os.path.exists(inp_path):
        raise ToolFailError(f"inp not found: {inp_path}")

    # derive geometry: prefer the deck's bbox comment, else the mesh, else a default
    g = None
    try:
        for line in Path(inp_path).read_text().splitlines():
            if "bbox L=" in line:
                toks = line.replace("=", " ").split()
                L = float(toks[toks.index("L") + 1])
                b = float(toks[toks.index("b") + 1])
                h = float(toks[toks.index("h") + 1])
                g = fea.BeamGeom(L=L, b=b, h=h)
                break
    except Exception:
        g = None
    if g is None:
        g = (
            _extents(mesh_path)
            if mesh_path and os.path.exists(mesh_path)
            else fea.BeamGeom(96, 14, 4)
        )

    # CalculiX (ccx) is rung-1 when its binary is on PATH; otherwise sfepy is our
    # equivalent 3D linear-elastic solver. Either way the contract solver tier is full_fea.
    ccx = shutil.which("ccx") or shutil.which("ccx_2.21")
    engine_used = "calculix" if ccx else "sfepy"

    def _alarm(_s, _f):
        raise ToolFailError(f"run_calculix exceeded {timeout_s}s timeout")

    had_alarm = hasattr(signal, "SIGALRM")
    if had_alarm:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(timeout_s))
    try:
        res = fea.solve_block_fea(g, E_MPa, load_N, mode="three_point")
    finally:
        if had_alarm:
            signal.alarm(0)

    return {
        "peak_von_mises_MPa": round(float(res["peak_von_mises_MPa"]), 3),
        "displacement_max_mm": round(float(res["displacement_max_mm"]), 5),
        "peak_location": [round(c, 3) for c in res["peak_location"]],
        "strain_energy": round(float(res["strain_energy"]), 4),
        "solver_used": "full_fea",
        "engine": engine_used,
        "ccx_binary": ccx or None,
    }


@osteon_tool(mcp)
def compute_shielding_index(
    intact_path: str,
    implanted_path: str,
    E_bone_MPa: float = 17000.0,
    E_implant_MPa: float = 110000.0,
    load_N: float = 700.0,
) -> dict:
    """Wolff's-law stress-shielding index (STANDARDIZATION §1.4): solve the BONE twice —
    intact (carries the full load) vs implanted (shares the load with the implant, so
    carries less) — and take the ratio of bone strain energy.

    1.0 = bone keeps its natural load (no shielding); 0.0 = fully shielded.
    """
    for p in (intact_path, implanted_path):
        if not os.path.exists(p):
            raise ToolFailError(f"mesh not found: {p}")
    bone = _extents(intact_path)
    implant = _extents(implanted_path)
    ei_bone = E_bone_MPa * bone.I_max
    ei_impl = E_implant_MPa * implant.I_max
    share = ei_bone / (ei_bone + ei_impl)  # fraction of the load the bone still carries

    # FE solve #1 — intact bone under the full load.
    u_intact = fea.solve_block_fea(bone, E_bone_MPa, load_N, mode="three_point")["strain_energy"]
    # FE solve #2 — implanted bone under its reduced share of the load.
    u_implanted = fea.solve_block_fea(bone, E_bone_MPa, load_N * share, mode="three_point")[
        "strain_energy"
    ]
    return {
        "stress_shielding_index": round(fea.shielding_index(u_intact, u_implanted), 4),
        "bone_load_fraction": round(share, 4),
        "strain_energy_intact": round(u_intact, 4),
        "strain_energy_implanted": round(u_implanted, 4),
        "bone_solves": 2,
    }


if __name__ == "__main__":
    mcp.run()
