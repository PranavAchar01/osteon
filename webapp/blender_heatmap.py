"""Blender-side FEA-style jet von Mises heat-map (run via `blender --background`).

Reads {"implant","bone","out_png","out_blend"}. Paints a CONTINUOUS jet contour
(blue->cyan->green->yellow->red) across the FULL surface of the implant (bending field,
hot at the loaded mid-span) and the bone (diaphyseal load), using one shared colormap and
an emission shader so it reads like a real FEA contour plot.
"""
import json
import sys

import bpy
import mathutils
import numpy as np

spec = json.load(open(sys.argv[sys.argv.index("--") + 1:][0]))

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
world = bpy.data.worlds.new("w"); bpy.context.scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.05, 0.05, 0.06, 1)

# --- jet colormap ---------------------------------------------------------------------
JET = np.array([
    [0.00, 0.00, 0.00, 0.52], [0.12, 0.00, 0.20, 1.00], [0.36, 0.00, 0.86, 1.00],
    [0.50, 0.30, 1.00, 0.55], [0.62, 0.85, 1.00, 0.10], [0.78, 1.00, 0.55, 0.00],
    [0.90, 1.00, 0.10, 0.00], [1.00, 0.55, 0.00, 0.00],
])

def jet(t):
    t = np.clip(t, 0, 1)
    r = np.interp(t, JET[:, 0], JET[:, 1])
    g = np.interp(t, JET[:, 0], JET[:, 2])
    b = np.interp(t, JET[:, 0], JET[:, 3])
    return np.stack([r, g, b], axis=1)

def imp(path):
    before = set(bpy.context.scene.objects)
    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=path)
    else:
        bpy.ops.import_mesh.stl(filepath=path)
    return [o for o in bpy.context.scene.objects if o not in before][0]

def world_verts(o):
    mw = o.matrix_world
    return np.array([(mw @ v.co)[:] for v in o.data.vertices])

def pca(P):
    c = P.mean(0)
    _, v = np.linalg.eigh(np.cov((P - c).T))
    return c, v[:, ::-1]  # center, axes major->minor

def paint(o, t):
    """t in [0,1] per vertex -> jet vertex-color attribute + emission material."""
    cols = jet(t)
    me = o.data
    att = me.color_attributes.new(name="vm", type="FLOAT_COLOR", domain="POINT")
    for i in range(len(me.vertices)):
        att.data[i].color = (cols[i, 0], cols[i, 1], cols[i, 2], 1.0)
    mat = bpy.data.materials.new("vm"); mat.use_nodes = True
    nt = mat.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emi = nt.nodes.new("ShaderNodeEmission"); emi.inputs["Strength"].default_value = 1.0
    vc = nt.nodes.new("ShaderNodeVertexColor"); vc.layer_name = "vm"
    nt.links.new(vc.outputs["Color"], emi.inputs["Color"])
    nt.links.new(emi.outputs["Emission"], out.inputs["Surface"])
    me.materials.clear(); me.materials.append(mat)

def bending_field(P, center, axes, surface_bias=0.7):
    """Normalized von-Mises-like field: hot at mid-span surface (3-point bending)."""
    loc = (P - center) @ axes
    xl = loc[:, 0]; yl = loc[:, 2]
    hx = np.abs(xl).max() or 1.0; hy = np.abs(yl).max() or 1.0
    span = 1.0 - np.abs(xl) / hx              # max at mid-span
    fibre = (1 - surface_bias) + surface_bias * np.abs(yl) / hy  # max at the surface
    f = span * fibre
    return f / (f.max() or 1.0)

# --- implant (placed) -----------------------------------------------------------------
impl = imp(spec["implant"])
Pi = world_verts(impl)
ci, ai = pca(Pi)
paint(impl, bending_field(Pi, ci, ai))

# --- bone -----------------------------------------------------------------------------
scene_pts = [Pi]
if spec.get("bone"):
    bone = imp(spec["bone"])
    Pb = world_verts(bone)
    cb, ab = pca(Pb)
    paint(bone, bending_field(Pb, cb, ab, surface_bias=0.55) * 0.78)  # bone load lower band
    scene_pts.append(Pb)

allp = np.vstack(scene_pts)
center = mathutils.Vector(allp.mean(0))
size = float(np.linalg.norm(allp.max(0) - allp.min(0))) or 100.0

cd = bpy.data.cameras.new("c"); cd.clip_end = 1e5
cam = bpy.data.objects.new("c", cd); bpy.context.collection.objects.link(cam)
view = mathutils.Vector((0.5, -0.82, 0.16)).normalized()
cam.location = center + view * size * 1.5
cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
bpy.context.scene.camera = cam
bpy.context.scene.render.resolution_x = 1200
bpy.context.scene.render.resolution_y = 760

if spec.get("out_blend"):
    bpy.ops.wm.save_as_mainfile(filepath=spec["out_blend"])
if spec.get("out_png"):
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.filepath = spec["out_png"]
    bpy.ops.render.render(write_still=True)
