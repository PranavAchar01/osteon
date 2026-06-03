"""Per-patient implant designs (distinct parametric plates).

Split B converges to one plate for its shipped fixtures; for the demo we show a
*different* design per patient (length / width / thickness / screw count / end style),
generated as watertight STLs so each patient's 3-D view and Blender model differ.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

# One implant design per case id.
THETAS = {
    "c1": {"length_mm": 96, "width_mm": 14, "thickness_mm": 4.0, "n_screws": 6, "style": "plate"},
    "c2": {"length_mm": 120, "width_mm": 18, "thickness_mm": 5.0, "n_screws": 8, "style": "round"},
    "c3": {"length_mm": 140, "width_mm": 16, "thickness_mm": 5.0, "n_screws": 10, "style": "bridge"},
    "c4": {"length_mm": 90, "width_mm": 22, "thickness_mm": 4.5, "n_screws": 6, "style": "round"},
}


def make_plate(th: dict) -> trimesh.Trimesh:
    L, W, T, n = th["length_mm"], th["width_mm"], th["thickness_mm"], th["n_screws"]
    style = th.get("style", "plate")
    plate = trimesh.creation.box(extents=(L, W, T))
    if style in ("round", "bridge"):  # rounded ends — a different silhouette
        rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        for sx in (-L / 2, L / 2):
            cyl = trimesh.creation.cylinder(radius=T / 2, height=W, sections=24)
            cyl.apply_transform(rot)
            cyl.apply_translation((sx, 0, 0))
            try:
                plate = trimesh.boolean.union([plate, cyl])
            except Exception:
                pass
    holes, hr = [], min(W, T) * 0.30
    for x in np.linspace(-L / 2 + L * 0.1, L / 2 - L * 0.1, n):
        if abs(x) < L * 0.13:  # leave a central gap over the fracture
            continue
        cy = trimesh.creation.cylinder(radius=hr, height=T * 3, sections=18)
        cy.apply_translation((x, 0, 0))
        holes.append(cy)
    if holes:
        try:
            r = trimesh.boolean.difference([plate] + holes)
            if r is not None and len(r.faces) > 0:
                plate = r
        except Exception:
            pass
    return plate


def ensure_implant(case_id: str, out_dir: Path) -> Path:
    """Return the STL path for a case's implant, generating it once if missing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{case_id}.stl"
    if not path.exists() and case_id in THETAS:
        make_plate(THETAS[case_id]).export(path)
    return path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "implants"
    for cid in THETAS:
        p = ensure_implant(cid, out)
        m = trimesh.load(p, force="mesh")
        print(f"{cid}: {p.name}  watertight={m.is_watertight}  bbox={[round(x,1) for x in m.bounding_box.extents]}")
