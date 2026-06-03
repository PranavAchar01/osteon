"""Generate an anatomically-proportioned synthetic femur STL (in METERS).

The engine treats input meshes as metres and scales x1000 -> mm, so a 0.46 m model
becomes a ~460 mm femur. Built as a signed-distance field with smooth-union blending
(head, neck, trochanters, bowed tapering shaft, flared condyles) then meshed with
marching cubes. The diaphysis is hollow (medullary canal) so cortical-thickness
ray-casts return realistic ~6 mm walls instead of a solid bar.

Anatomy / proportions (right femur, +Z proximal, X = medial(-)/lateral(+), Y = anterior(+)/posterior(-)):
  length ~460 mm; head dia ~48 mm; neck-shaft angle ~125 deg; mid-shaft outer dia ~26 mm;
  cortical wall ~6 mm; bicondylar width ~80 mm; gentle anterior bow.

Run:  python scripts/make_femur.py
"""
import numpy as np
import trimesh
from skimage.measure import marching_cubes

# ---- units: metres -----------------------------------------------------------
MM = 1e-3

def smin(a, b, k):
    """Polynomial smooth-min: blends two SDFs over a radius ~k."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)

def sd_sphere(P, c, r):
    return np.linalg.norm(P - c, axis=1) - r

def sd_capsule(P, a, b, r):
    a = np.asarray(a); b = np.asarray(b)
    pa = P - a
    ba = b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0.0, 1.0)
    return np.linalg.norm(pa - np.outer(h, ba), axis=1) - r

def sd_ellipsoid(P, c, radii):
    q = (P - c) / radii
    k0 = np.linalg.norm(q, axis=1)
    k1 = np.linalg.norm(q / radii, axis=1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)

def bowed_centerline(n=14):
    """Shaft axis from distal (-Z) to proximal (+Z) with a gentle anterior (+Y) bow."""
    z = np.linspace(-0.185, 0.150, n)
    bow = 0.012 * (1.0 - (z / 0.185) ** 2)  # max ~12 mm anterior bow mid-shaft
    x = np.zeros_like(z)
    return np.stack([x, bow, z], axis=1)

def femur_sdf(P):
    cl = bowed_centerline()
    k = 0.018  # blend radius

    # Shaft: tube along the bowed centerline, slightly tapered (min Ø mid-shaft).
    shaft = np.full(len(P), 1e9)
    for a, b in zip(cl[:-1], cl[1:]):
        zc = 0.5 * (a[2] + b[2])
        r = 0.013 + 0.010 * (abs(zc) / 0.185) ** 2  # flare toward metaphyses
        shaft = np.minimum(shaft, sd_capsule(P, a, b, r))
    f = shaft

    # Proximal: neck (125 deg) + head + greater & lesser trochanters.
    top = cl[-1]
    head_c = np.array([-0.040, 0.006, 0.196])           # superomedial
    neck = sd_capsule(P, top + [0, 0, 0.005], head_c, 0.012)
    head = sd_sphere(P, head_c, 0.024)                   # ~48 mm dia
    gt = sd_ellipsoid(P, [0.028, 0.0, 0.150], [0.020, 0.018, 0.026])   # greater trochanter
    lt = sd_sphere(P, [-0.014, -0.014, 0.120], 0.010)    # lesser trochanter
    for part in (neck, head, gt, lt):
        f = smin(f, part, k)

    # Distal: medial + lateral condyles + patellar flare.
    med = sd_ellipsoid(P, [-0.024, -0.004, -0.196], [0.022, 0.026, 0.024])
    lat = sd_ellipsoid(P, [0.024, -0.004, -0.196], [0.022, 0.026, 0.024])
    pat = sd_ellipsoid(P, [0.0, 0.020, -0.182], [0.026, 0.014, 0.020])
    for part in (med, lat, pat):
        f = smin(f, part, k)

    # Hollow medullary canal through the diaphysis only (endpoints enclosed).
    canal = sd_capsule(P, [0, 0.006, -0.085], [0, 0.006, 0.060], 0.007)
    f = np.maximum(f, -canal)
    return f

def main():
    lo = np.array([-0.075, -0.050, -0.235])
    hi = np.array([0.075, 0.055, 0.235])
    sp = 0.0015  # 1.5 mm voxels
    xs = np.arange(lo[0], hi[0], sp)
    ys = np.arange(lo[1], hi[1], sp)
    zs = np.arange(lo[2], hi[2], sp)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    vol = femur_sdf(P).reshape(gx.shape)
    verts, faces, normals, _ = marching_cubes(vol, level=0.0, spacing=(sp, sp, sp))
    verts += lo  # index space -> world (metres)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.fix_normals()
    trimesh.smoothing.filter_taubin(mesh, iterations=12)  # organic surface

    out = "fixtures/dummy_bone.stl"
    mesh.export(out)
    e = mesh.extents * 1000
    print(f"wrote {out}: verts={len(mesh.vertices)} faces={len(mesh.faces)} "
          f"watertight={mesh.is_watertight}")
    print(f"size after x1000 (mm): L={e[2]:.0f}  ML={e[0]:.0f}  AP={e[1]:.0f}  "
          f"components={mesh.body_count}")

if __name__ == "__main__":
    main()
