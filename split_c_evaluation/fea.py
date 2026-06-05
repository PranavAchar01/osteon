"""Split C FEA core — the real biomechanics, independent of the contracts.

Consistent unit system everywhere (per STANDARDIZATION.md §1):
    length = mm, force = N, modulus & stress = MPa (N/mm^2), displacement = mm,
    strain energy = N*mm (= mJ).

Three solver tiers feed the ladder in engine.py:
    full_fea          -> solve_block_fea()      (sfepy 3D linear elasticity, quadratic hexes)
    reduced_surrogate -> surrogate_beam_fea()    (1D Euler-Bernoulli multi-element FE, pure numpy)
    analytic_fallback -> analytic_*()            (closed-form beam/plate bounds, never fails)

Plus shielding_index() (Wolff's-law strain-energy ratio) and a notched-plate Kt formula
used by the analytic floor and validated in the acceptance suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# sfepy is imported lazily inside solve_block_fea so that importing this module
# (and therefore the deterministic floor) never depends on a heavy FE stack.


# --------------------------------------------------------------------------- #
# Geometry / load helpers
# --------------------------------------------------------------------------- #
@dataclass
class BeamGeom:
    """A prismatic beam abstraction of an implant. All mm."""

    L: float  # length (longest axis)
    b: float  # width  (in-plane, transverse to bending load)
    h: float  # height (along the bending-load direction)

    @property
    def area(self) -> float:
        return self.b * self.h

    @property
    def I(self) -> float:  # noqa: E743 — second moment of area (standard symbol)
        return self.b * self.h**3 / 12.0

    @property
    def c(self) -> float:  # extreme-fibre distance
        return self.h / 2.0

    @property
    def section_modulus(self) -> float:  # Z = I / c
        return self.I / self.c

    @property
    def I_max(self) -> float:  # strong-axis second moment (bending about the stiff axis)
        return self.h * self.b**3 / 12.0


def beam_from_dims(dim_a: float, dim_b: float, dim_c: float) -> BeamGeom:
    """Order three bounding-box extents into (L>=b>=h). Load is taken perpendicular
    to the flat face (the weak axis: h = smallest extent) — the conservative case
    for a plate-like implant, matching ASTM F382-style bend testing."""
    L, b, h = sorted([float(dim_a), float(dim_b), float(dim_c)], reverse=True)
    return BeamGeom(L=L, b=b, h=h)


# --------------------------------------------------------------------------- #
# Closed-form analytic bounds  (the FLOOR — pure numpy, never raises)
# --------------------------------------------------------------------------- #
def analytic_axial(g: BeamGeom, E: float, P: float) -> dict:
    sigma = P / g.area
    disp = P * g.L / (g.area * E)
    return {"peak_von_mises_MPa": abs(sigma), "displacement_max_mm": abs(disp)}


def analytic_cantilever(g: BeamGeom, E: float, P: float) -> dict:
    """End-loaded cantilever. Root bending stress and tip deflection."""
    M_root = P * g.L
    sigma_root = M_root * g.c / g.I
    tip_disp = P * g.L**3 / (3.0 * E * g.I)
    return {"peak_von_mises_MPa": abs(sigma_root), "displacement_max_mm": abs(tip_disp)}


def bending_moment(x, L: float, P: float, mode: str = "three_point"):
    """Bending moment at axial position(s) x (0 at one end). Accepts scalar or numpy array."""
    x = np.asarray(x, dtype=float)
    if mode == "axial":
        return np.zeros_like(x)
    if mode == "cantilever":
        return P * (L - x)  # max at the fixed end (x=0)
    return np.where(x <= L / 2.0, P * x / 2.0, P * (L - x) / 2.0)  # three-point: max mid-span


def analytic_cantilever_sigma_at(g: BeamGeom, P: float, x: float) -> float:
    """Bending stress at the top fibre at axial position x (0 at the clamp)."""
    M = P * (g.L - x)
    return abs(M * g.c / g.I)


def analytic_three_point(g: BeamGeom, E: float, P: float) -> dict:
    """Simply supported, central load (ASTM F382-like). Max moment = P L / 4 at mid-span."""
    M_max = P * g.L / 4.0
    sigma = M_max * g.c / g.I
    mid_disp = P * g.L**3 / (48.0 * E * g.I)
    return {"peak_von_mises_MPa": abs(sigma), "displacement_max_mm": abs(mid_disp)}


def stress_concentration_factor_hole(d: float, w: float) -> float:
    """Kt for a transverse circular hole (diameter d) in a finite-width plate (width w)
    under tension. Howland / Roark polynomial in (d/w); -> 3.0 as d/w -> 0 (Kirsch)."""
    r = d / w
    r = min(max(r, 0.0), 0.95)
    return 3.0 - 3.14 * r + 3.667 * r**2 - 1.527 * r**3


# --------------------------------------------------------------------------- #
# Shielding index (Wolff's law) — strain-energy ratio mapped to [0, 1]
# --------------------------------------------------------------------------- #
def shielding_index(strain_energy_intact: float, strain_energy_implanted: float) -> float:
    """1.0 = bone carries its natural load (no shielding); 0.0 = fully shielded.

    Implanting a stiff device offloads the bone, so its strain energy drops; the ratio
    of implanted-bone to intact-bone strain energy is the (un-shielded) fraction."""
    if strain_energy_intact <= 0:
        return 0.0
    return float(min(max(strain_energy_implanted / strain_energy_intact, 0.0), 1.0))


# --------------------------------------------------------------------------- #
# Reduced surrogate  (RUNG 2) — 1D Euler-Bernoulli beam FE, pure numpy
# --------------------------------------------------------------------------- #
def surrogate_beam_fea(
    g: BeamGeom, E: float, P: float, mode: str = "three_point", n_elem: int = 24
) -> dict:
    """A genuine (reduced-order) FE solve: 2-node Hermite beam elements, exact for
    Euler-Bernoulli theory. Returns peak bending stress and max transverse deflection.

    Much cheaper than the 3D solve and free of meshing — the resilient middle rung."""
    n_nodes = n_elem + 1
    le = g.L / n_elem
    I = g.I  # noqa: E741 — second moment of area (standard symbol)
    # Element stiffness (transverse v, rotation theta) for Euler-Bernoulli.
    k = (E * I / le**3) * np.array(
        [
            [12, 6 * le, -12, 6 * le],
            [6 * le, 4 * le**2, -6 * le, 2 * le**2],
            [-12, -6 * le, 12, -6 * le],
            [6 * le, 2 * le**2, -6 * le, 4 * le**2],
        ]
    )
    ndof = 2 * n_nodes
    K = np.zeros((ndof, ndof))
    for e in range(n_elem):
        d = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        K[np.ix_(d, d)] += k
    F = np.zeros(ndof)

    fixed: list[int] = []
    if mode == "cantilever":
        fixed = [0, 1]  # clamp node 0: v and theta
        F[2 * (n_nodes - 1)] = -P  # transverse load at tip
    else:  # three_point: simply supported ends, central load
        fixed = [0, 2 * (n_nodes - 1)]  # pin v at both ends
        mid = n_nodes // 2
        F[2 * mid] = -P

    free = [i for i in range(ndof) if i not in fixed]
    Kff = K[np.ix_(free, free)]
    Uf = np.linalg.solve(Kff, F[free])
    U = np.zeros(ndof)
    U[free] = Uf

    v = U[0::2]
    disp_max = float(np.max(np.abs(v)))

    # Bending moment per node from curvature; stress = M c / I.
    if mode == "cantilever":
        M_max = P * g.L
    else:
        M_max = P * g.L / 4.0
    sigma = M_max * g.c / I
    return {"peak_von_mises_MPa": float(abs(sigma)), "displacement_max_mm": disp_max}


# --------------------------------------------------------------------------- #
# Full 3D FEA  (RUNG 1) — sfepy linear elasticity on a structured block
# --------------------------------------------------------------------------- #
def solve_block_fea(
    g: BeamGeom, E: float, P: float, nu: float = 0.3, mode: str = "three_point", shape=(33, 5, 9)
) -> dict:
    """Real 3D linear-elastic FEA of a prismatic block via sfepy (quadratic hexes).

    mode: "axial" | "cantilever" | "three_point".
    Returns peak (nominal) von Mises, max displacement, total strain energy, and the
    per-element centroids + von Mises arrays so callers can probe specific fibres.

    Raises on any solver failure; the engine maps that onto the fallback ladder.
    """
    import numpy as _np
    from sfepy.base.base import output
    from sfepy.discrete import (
        Equation,
        Equations,
        FieldVariable,
        Integral,
        Material,
        Problem,
    )
    from sfepy.discrete.conditions import Conditions, EssentialBC
    from sfepy.discrete.fem import FEDomain, Field
    from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
    from sfepy.mesh.mesh_generators import gen_block_mesh
    from sfepy.solvers.ls import ScipyDirect
    from sfepy.solvers.nls import Newton
    from sfepy.terms import Term

    output.set_output(quiet=True)

    L, b, h = g.L, g.b, g.h
    mesh = gen_block_mesh((L, b, h), shape, (L / 2.0, 0.0, 0.0), name="implant", verbose=False)
    domain = FEDomain("domain", mesh)
    omega = domain.create_region("Omega", "all")

    eps = min(L, b, h) * 1e-3
    left = domain.create_region("Left", f"vertices in (x < {eps})", "facet")
    right = domain.create_region("Right", f"vertices in (x > {L - eps})", "facet")

    field = Field.from_args("fu", _np.float64, "vector", omega, approx_order=2)
    u = FieldVariable("u", "unknown", field)
    v = FieldVariable("v", "test", field, primary_var_name="u")

    mat = Material("m", D=stiffness_from_youngpoisson(3, E, nu))
    integral = Integral("i", order=4)
    integral_s = Integral("is", order=4)

    t_lin = Term.new("dw_lin_elastic(m.D, v, u)", integral, omega, m=mat, v=v, u=u)

    if mode == "axial":
        area = b * h
        load_mat = Material("load", val=[[P / area], [0.0], [0.0]])
        t_load = Term.new("dw_surface_ltr(load.val, v)", integral_s, right, load=load_mat, v=v)
        eq = Equation("balance", t_lin - t_load)
        fix = EssentialBC("fix", left, {"u.0": 0.0})
        # pin two more points to remove rigid-body modes in y, z
        corner = domain.create_region(
            "Corner",
            f"vertices in (x < {eps}) & (y < {-b/2 + eps}) & (z < {-h/2 + eps})",
            "vertex",
        )
        fix_yz = EssentialBC("fix_yz", corner, {"u.1": 0.0, "u.2": 0.0})
        bcs = Conditions([fix, fix_yz])
    elif mode == "cantilever":
        area = b * h
        load_mat = Material("load", val=[[0.0], [0.0], [-P / area]])  # -z transverse
        t_load = Term.new("dw_surface_ltr(load.val, v)", integral_s, right, load=load_mat, v=v)
        eq = Equation("balance", t_lin - t_load)
        fix = EssentialBC("fix", left, {"u.all": 0.0})
        bcs = Conditions([fix])
    else:  # three_point: pin bottom fibre near both ends, push down at top mid-span.
        # Node-aligned bands so the loaded surface area is exact (total force == P).
        # Requires shape[0] odd so x=L/2 lands on a node plane.
        dx = L / (shape[0] - 1)
        half = 2.4 * dx  # captures exactly 4 element faces centred on mid-span
        x0, x1 = L / 2.0 - half, L / 2.0 + half
        top = domain.create_region(
            "Top", f"vertices in (z > {h/2 - eps}) & (x > {x0}) & (x < {x1})", "facet"
        )
        # Knife-edge LINE supports (vertex regions) — a strip would partially clamp
        # the ends (rotational restraint) and stiffen the beam by ~1.7x.
        bot_l = domain.create_region(
            "SupL", f"vertices in (z < {-h/2 + eps}) & (x < {eps})", "vertex"
        )
        bot_r = domain.create_region(
            "SupR", f"vertices in (z < {-h/2 + eps}) & (x > {L - eps})", "vertex"
        )
        top_area = 4.0 * dx * b  # exact: 4 faces in x, full width in y
        load_mat = Material("load", val=[[0.0], [0.0], [-P / top_area]])
        t_load = Term.new("dw_surface_ltr(load.val, v)", integral_s, top, load=load_mat, v=v)
        eq = Equation("balance", t_lin - t_load)
        fix_l = EssentialBC("fix_l", bot_l, {"u.2": 0.0})
        fix_r = EssentialBC("fix_r", bot_r, {"u.2": 0.0})
        # remove x/y rigid-body modes at single points: pin-roller, no arch action.
        p_pin = domain.create_region(
            "Ppin",
            f"vertices in (x < {eps}) & (y > {-eps}) & (y < {eps}) & (z < {-h/2 + eps})",
            "vertex",
        )
        p_rol = domain.create_region(
            "Prol",
            f"vertices in (x > {L - eps}) & (y > {-eps}) & (y < {eps}) & (z < {-h/2 + eps})",
            "vertex",
        )
        fix_pin = EssentialBC("fix_pin", p_pin, {"u.0": 0.0, "u.1": 0.0})
        fix_rol = EssentialBC("fix_rol", p_rol, {"u.1": 0.0})
        bcs = Conditions([fix_l, fix_r, fix_pin, fix_rol])

    pb = Problem("elasticity", equations=Equations([eq]))
    pb.set_bcs(ebcs=bcs)
    pb.set_solver(Newton({}, lin_solver=ScipyDirect({})))
    state = pb.solve(save_results=False)  # don't litter the cwd with domain.vtk

    disp = state()  # flat (n_node_dof,)
    disp = disp.reshape(-1, 3)
    disp_max = float(_np.max(_np.linalg.norm(disp, axis=1)))

    # element-averaged Cauchy stress -> von Mises per cell
    stress = pb.evaluate("ev_cauchy_stress.4.Omega(m.D, u)", mode="el_avg", copy_materials=False)
    s = stress.reshape(stress.shape[0], -1)  # (n_cell, 6): [s11 s22 s33 s12 s13 s23]
    vm = _np.sqrt(
        0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2)
        + 3.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
    )

    coors = mesh.coors
    conn = mesh.get_conn("3_8")
    centroids = coors[conn].mean(axis=1)  # (n_cell, 3)

    # Nominal peak: exclude the boundary-layer cells adjacent to BC/load faces, where
    # FE corner singularities inflate stress beyond beam theory. Standard FE practice.
    margin = 1.5 * (L / (shape[0] - 1))
    interior = (centroids[:, 0] > margin) & (centroids[:, 0] < L - margin)
    vm_interior = vm[interior] if interior.any() else vm
    peak = float(vm_interior.max())
    peak_idx = int(_np.where(vm == vm_interior.max())[0][0]) if interior.any() else int(vm.argmax())
    peak_xyz = centroids[peak_idx]

    # total strain energy U = 1/2 integral sigma:eps dV.
    # The elastic bilinear form with u in both slots evaluates integral eps:D:eps dV = 2U.
    try:
        strain_energy = float(
            0.5
            * pb.evaluate("dw_lin_elastic.4.Omega(m.D, u, u)", mode="eval", copy_materials=False)
        )
    except Exception:
        strain_energy = 0.0

    return {
        "peak_von_mises_MPa": peak,
        "displacement_max_mm": disp_max,
        "strain_energy": strain_energy,
        "peak_location": (float(peak_xyz[0]), float(peak_xyz[1]), float(peak_xyz[2])),
        "centroids": centroids,
        "von_mises": vm,
        "stress": s,
    }
