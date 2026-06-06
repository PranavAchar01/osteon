"""Blender-side renderer for the dashboard stage images (run via `blender --background`).

Reads a small JSON spec: {"meshes":[{"path","kind"}], "out_png", "out_blend", "view"}.
kind = "bone" (matte ivory) | "implant" (steel blue). Frames on the implant if present,
else on everything. Used for the "implant alone" and "implant in femur" stage renders.
"""
import json
import sys

import bpy
import mathutils
import numpy as np

spec = json.load(open(sys.argv[sys.argv.index("--") + 1:][0]))

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

world = bpy.data.worlds.new("w")
bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.62, 0.64, 0.67, 1)
bg.inputs["Strength"].default_value = 1.0


def imp(path):
    before = set(bpy.context.scene.objects)
    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=path)
    else:
        bpy.ops.import_mesh.stl(filepath=path)
    return [o for o in bpy.context.scene.objects if o not in before]


def mat(name, color, metal, rough):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = color
    try:
        b.inputs["Metallic"].default_value = metal
        b.inputs["Roughness"].default_value = rough
    except Exception:
        pass
    return m


STYLE = {
    "bone": ((0.92, 0.89, 0.82, 1), 0.0, 0.8),
    "implant": ((0.30, 0.45, 0.95, 1), 0.7, 0.30),
}


def world_pts(objs):
    pts = []
    for o in objs:
        for v in o.data.vertices:
            w = o.matrix_world @ v.co
            pts.append((w.x, w.y, w.z))
    return np.array(pts) if pts else np.zeros((1, 3))


implant_objs, all_objs = [], []
for entry in spec["meshes"]:
    objs = imp(entry["path"])
    c, metal, rough = STYLE[entry["kind"]]
    for o in objs:
        o.data.materials.append(mat(entry["kind"], c, metal, rough))
    all_objs += objs
    if entry["kind"] == "implant":
        implant_objs += objs

focus = implant_objs or all_objs
fp = world_pts(focus)
center = mathutils.Vector(fp.mean(0))
size = float(np.linalg.norm(fp.max(0) - fp.min(0))) or 50.0

# face normal of the implant (PCA minor axis), so we look straight at the plate
c = fp - fp.mean(0)
_, vecs = np.linalg.eigh(np.cov(c.T))
normal = mathutils.Vector(vecs[:, 0])
allp = world_pts(all_objs)
scene_center = mathutils.Vector(allp.mean(0))
if normal.dot(center - scene_center) < 0:
    normal = -normal

for off, energy in [((1, -1, 2), 3.2), ((-1, 0.6, 1), 1.4)]:
    ld = bpy.data.lights.new("L", "SUN")
    ld.energy = energy
    lo = bpy.data.objects.new("L", ld)
    bpy.context.collection.objects.link(lo)
    lo.location = center + mathutils.Vector(off) * size

cd = bpy.data.cameras.new("c")
cd.clip_end = 1e5
cam = bpy.data.objects.new("c", cd)
bpy.context.collection.objects.link(cam)
view = (normal + (center - scene_center).normalized() * 0.25).normalized() if len(all_objs) > len(focus) \
    else mathutils.Vector((1.0, -0.4, 0.45)).normalized()
cam.location = center + view * size * 1.7
cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
bpy.context.scene.camera = cam

bpy.context.scene.render.resolution_x = 1100
bpy.context.scene.render.resolution_y = 760

if spec.get("out_blend"):
    bpy.ops.wm.save_as_mainfile(filepath=spec["out_blend"])
if spec.get("out_png"):
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.filepath = spec["out_png"]
    bpy.ops.render.render(write_still=True)
