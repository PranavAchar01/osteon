"""Generate a technically-detailed anatomic locking compression plate (STL, mm).

A combination plate: rounded contoured body, a row of countersunk round screw holes plus
two elongated combi-slots, filleted edges — built as a signed-distance field + marching
cubes so it is smooth and watertight. Sized (≈150 x 18 x 5 mm) to be structurally SAFE
under a femoral walking load.

Run:  python scripts/make_implant.py   ->  fixtures/implant_library/anatomic_lcp.stl
"""
import numpy as np
import trimesh
from skimage.measure import marching_cubes

L, W, T = 150.0, 18.0, 5.0          # length, width, thickness (mm)
EDGE_R = 2.2                        # edge fillet radius
HOLE_R = 2.6                        # screw-hole radius
CSK_R = 3.6                         # countersink top radius
SLOT_R = 2.6                        # combi-slot half-width


def sd_round_box(p, half, r):
    q = np.abs(p) - (np.array(half) - r)
    out = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    ins = np.minimum(np.max(q, axis=1), 0.0)
    return out + ins - r


def sd_cyl_z(p, cx, r, zlo, zhi):
    d_rad = np.linalg.norm(p[:, :2] - np.array([cx, 0.0]), axis=1) - r
    d_cap = np.maximum(zlo - p[:, 2], p[:, 2] - zhi)
    return np.maximum(d_rad, d_cap)


def sd_countersink(p, cx, r0, r1, ztop, depth):
    """Cone widening toward +Z (the countersink seat for a screw head)."""
    z = p[:, 2]
    frac = np.clip((z - (ztop - depth)) / depth, 0.0, 1.0)
    r = r0 + (r1 - r0) * frac
    d_rad = np.linalg.norm(p[:, :2] - np.array([cx, 0.0]), axis=1) - r
    d_cap = np.maximum((ztop - depth) - z, z - (ztop + 0.2))
    return np.maximum(d_rad, d_cap)


def sd_slot_z(p, cx, halflen, r, zlo, zhi):
    x = np.clip(p[:, 0] - cx, -halflen, halflen)
    d_rad = np.linalg.norm(p[:, :2] - np.stack([cx + x, np.zeros_like(x)], axis=1), axis=1) - r
    d_cap = np.maximum(zlo - p[:, 2], p[:, 2] - zhi)
    return np.maximum(d_rad, d_cap)


def implant_sdf(P):
    f = sd_round_box(P, [L / 2, W / 2, T / 2], EDGE_R)
    top = T / 2
    # 6 countersunk round holes spread along the length, leaving the centre for 2 combi-slots
    xs = np.linspace(-L / 2 + 12, L / 2 - 12, 8)
    for i, cx in enumerate(xs):
        if i in (3, 4):  # centre two stations = elongated combi-slots
            slot = sd_slot_z(P, cx, 5.0, SLOT_R, -top - 1, top + 1)
            f = np.maximum(f, -slot)
        else:
            hole = sd_cyl_z(P, cx, HOLE_R, -top - 1, top + 1)
            csk = sd_countersink(P, cx, HOLE_R, CSK_R, top, 1.6)
            f = np.maximum(f, -np.minimum(hole, csk))
    return f


def main():
    lo = np.array([-L / 2 - 3, -W / 2 - 3, -T / 2 - 3])
    hi = np.array([L / 2 + 3, W / 2 + 3, T / 2 + 3])
    sp = 0.4
    xs = np.arange(lo[0], hi[0], sp)
    ys = np.arange(lo[1], hi[1], sp)
    zs = np.arange(lo[2], hi[2], sp)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    vol = implant_sdf(P).reshape(gx.shape)
    verts, faces, _n, _v = marching_cubes(vol, level=0.0, spacing=(sp, sp, sp))
    verts += lo
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    m.fix_normals()
    trimesh.smoothing.filter_taubin(m, iterations=6)
    out = "fixtures/implant_library/anatomic_lcp.stl"
    m.export(out)
    e = m.extents
    print(f"wrote {out}: verts={len(m.vertices)} faces={len(m.faces)} watertight={m.is_watertight}")
    print(f"dims (mm): L={e[0]:.0f} W={e[1]:.0f} T={e[2]:.1f}  volume={abs(m.volume):.0f} mm^3")


if __name__ == "__main__":
    main()
