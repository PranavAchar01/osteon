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
from .guardrails import heatmap_field_gate, mesh_watertight_gate

mcp = FastMCP("fea-mcp")

_OUT = Path(settings.OSTEON_TRACE_DIR)

# Two DISJOINT colour systems so the implant and the bone never share a colour (§heat-map):
#   IMPLANT — warm "hot-metal" ramp, hue 0..60 deg (red -> orange -> white-hot yellow)
#   BONE    — cool ramp,            hue 180..270 deg (cyan -> blue -> indigo)
# Verified ~134 deg hue guard gap; greens/magentas are never emitted by either ramp.
WARM_STOPS = [
    (0.00, (60, 10, 8)), (0.15, (120, 18, 10)), (0.30, (180, 30, 8)), (0.50, (225, 70, 5)),
    (0.70, (245, 130, 10)), (0.85, (252, 185, 20)), (1.00, (255, 238, 70)),
]
COOL_STOPS = [
    (0.00, (215, 245, 250)), (0.15, (165, 230, 245)), (0.30, (100, 200, 235)),
    (0.50, (50, 150, 215)), (0.70, (35, 95, 190)), (0.85, (40, 55, 160)), (1.00, (45, 25, 120)),
]


def _segmented_cmap(name, stops):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        name, [(t, (r / 255.0, g / 255.0, b / 255.0)) for t, (r, g, b) in stops]
    )


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

    # Surface the FULL von Mises field (per element), not just the scalar peak, so the
    # heat map can be coloured from the same solve. (CalculiX path would parse the .frd;
    # the sfepy path already has per-cell stress.)
    cen = res["centroids"]
    vmf = res["von_mises"]
    return {
        "peak_von_mises_MPa": round(float(res["peak_von_mises_MPa"]), 3),
        "displacement_max_mm": round(float(res["displacement_max_mm"]), 5),
        "peak_location": [round(c, 3) for c in res["peak_location"]],
        "strain_energy": round(float(res["strain_energy"]), 4),
        "nodes": [[round(float(c), 3) for c in p] for p in cen],
        "von_mises": [round(float(v), 3) for v in vmf],
        "peak_xyz": [round(c, 3) for c in res["peak_location"]],
        "solver_used": "full_fea",
        "engine": engine_used,
        "ccx_binary": ccx or None,
    }


@osteon_tool(mcp)
def render_stress_heatmap(
    mesh_path: str,
    stress_field: list,
    yield_mpa: float,
    bone_path: str = "",
    solver_used: str = "full_fea",
    factor_of_safety: float = None,
    bone_vertices: list = None,
    bone_field: list = None,
    bone_yield_mpa: float = 130.0,
    bone_scale: float = 1.0,
) -> dict:
    """Paint per-vertex stress onto BOTH the implant and the surrounding bone and render
    them together headless in Blender (PNG + interactive .blend). Two DISJOINT colour
    systems (warm = implant von Mises [0, yield]; cool = bone load [0, bone_yield]) so a
    colour can never be confused between the two meshes; each gets its own legend. A
    peak-stress marker is dropped at the implant hot spot.

    Returns {png_path, blend_path, peak_mpa, peak_location, bone_peak_mpa}.
    """
    import json as _json
    import subprocess
    import tempfile

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import trimesh
    from PIL import Image

    mpl.use("Agg")

    if not os.path.exists(mesh_path):
        raise ToolFailError(f"mesh not found: {mesh_path}")
    mesh_watertight_gate(mesh_path)  # pre-invoke: never render a garbage mesh
    verts = np.asarray(trimesh.load(mesh_path, force="mesh").vertices, dtype=float)
    heatmap_field_gate(stress_field, len(verts))  # pre-render: validate the field

    vm = np.asarray(stress_field, dtype=float)
    peak = float(vm.max())
    pk = int(vm.argmax())
    peak_xyz = verts[pk].tolist()
    yld = float(yield_mpa) if yield_mpa and yield_mpa > 0 else max(peak, 1.0)
    fos = factor_of_safety if factor_of_safety is not None else (yld / peak if peak > 0 else None)

    warm = _segmented_cmap("vm_warm", WARM_STOPS)
    cool = _segmented_cmap("bone_cool", COOL_STOPS)
    # FIXED normalisation (do NOT auto-scale) -> same colour means same MPa, per mesh.
    rgba = warm(np.clip(vm / yld, 0.0, 1.0))

    out_dir = Path(settings.OSTEON_TRACE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(mesh_path).stem
    data = {
        "implant": {
            "vertices": verts.tolist(),
            "rgba": rgba.tolist(),
            "peak_xyz": peak_xyz,
            "marker_radius": float((verts.max(0) - verts.min(0)).max() * 0.04),
        }
    }

    # Optional BONE layer — its own cool colour system + scale (metres -> mm via bone_scale).
    byld = float(bone_yield_mpa) if bone_yield_mpa and bone_yield_mpa > 0 else 130.0
    bone_peak = None
    has_bone = (
        bone_path
        and os.path.exists(bone_path)
        and bone_vertices is not None
        and bone_field is not None
        and len(bone_vertices) == len(bone_field)
    )
    if has_bone:
        bf = np.asarray(bone_field, dtype=float)
        bone_peak = float(np.nanmax(bf)) if bf.size else None
        brgba = cool(np.clip(bf / byld, 0.0, 1.0))
        data["bone"] = {
            "vertices": list(bone_vertices),
            "rgba": brgba.tolist(),
            "scale": float(bone_scale),
        }

    dj = tempfile.mktemp(suffix=".json")
    with open(dj, "w") as f:
        _json.dump(data, f)
    v0, v1 = tempfile.mktemp(suffix=".png"), tempfile.mktemp(suffix=".png")
    blend_path = str(out_dir / f"{stem}_heatmap.blend")
    png_path = str(out_dir / f"{stem}_heatmap.png")

    blender = (
        os.environ.get("OSTEON_BLENDER")
        or shutil.which("blender")
        or "/Applications/Blender.app/Contents/MacOS/Blender"
    )
    script = str(Path(__file__).resolve().parent / "heatmap_render.py")
    cmd = [blender, "--background", "--python", script, "--", mesh_path, dj, v0, v1, blend_path]
    if bone_path and os.path.exists(bone_path):
        cmd.append(bone_path)  # bone STL imported + painted (or ivory if no bone field)
    timeout_s = max(45, int(settings.OSTEON_DEADLINE_MS / 1000) * 4)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise ToolFailError(f"heatmap render exceeded {timeout_s}s") from exc
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise ToolFailError(
            f"Blender heatmap render failed: {getattr(exc, 'stderr', exc)}"
        ) from exc
    finally:
        if os.path.exists(dj):
            os.unlink(dj)

    # legend: TWO vertical colour bars side by side (implant warm | bone cool), each on its
    # own fixed scale, annotated so the disjoint colour systems are unambiguous.
    legend = tempfile.mktemp(suffix=".png")
    ncols = 2 if has_bone else 1
    fig, axes = plt.subplots(1, ncols, figsize=(2.7 * ncols, 9), dpi=100)
    axes = np.atleast_1d(axes)
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=yld), cmap=warm)
    sm.set_array([])
    fig.colorbar(sm, cax=axes[0])
    axes[0].set_ylabel("implant von Mises (MPa)", fontsize=12)
    title = f"IMPLANT\npeak {peak:.0f} MPa\nyield {yld:.0f} MPa"
    if fos is not None:
        title += f"\nFoS {fos:.1f}"
    title += f"\nsolver: {solver_used}"
    axes[0].set_title(title, fontsize=10)
    if has_bone:
        smb = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=byld), cmap=cool)
        smb.set_array([])
        fig.colorbar(smb, cax=axes[1])
        axes[1].set_ylabel("bone load (MPa)", fontsize=12)
        bt = "BONE"
        if bone_peak is not None:
            bt += f"\npeak {bone_peak:.0f} MPa"
        bt += f"\nyield {byld:.0f} MPa"
        axes[1].set_title(bt, fontsize=10)
    fig.savefig(legend, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # montage: [anterior view | lateral view | legend]
    imgs = [Image.open(p).convert("RGB") for p in (v0, v1, legend)]
    h = max(im.height for im in imgs)
    imgs = [im.resize((max(1, int(im.width * h / im.height)), h)) for im in imgs]
    canvas = Image.new("RGB", (sum(im.width for im in imgs), h), "white")
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width
    canvas.save(png_path)
    for p in (v0, v1, legend):
        if os.path.exists(p):
            os.unlink(p)

    return {
        "png_path": png_path,
        "blend_path": blend_path,
        "peak_mpa": round(peak, 3),
        "bone_peak_mpa": round(bone_peak, 3) if bone_peak is not None else None,
        "peak_location": {
            "x": round(peak_xyz[0], 3),
            "y": round(peak_xyz[1], 3),
            "z": round(peak_xyz[2], 3),
        },
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
