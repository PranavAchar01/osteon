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
bg.inputs["Color"].default_value = (0.16, 0.17, 0.19, 1)   # dark bg for contrast (matches Stage C)
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
    "bone": ((0.94, 0.91, 0.85, 1), 0.0, 0.7),       # bright ivory so it reads on the dark bg
    "implant": ((0.38, 0.52, 0.96, 1), 0.55, 0.32),  # steel blue
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

# When a bone is present, frame the WHOLE scene (bone + plate) from a side angle so the
# plate is clearly seen seated on the shaft. Implant-only: frame the plate from a 3/4 view.
has_bone = len(all_objs) > len(implant_objs)
fp = world_pts(all_objs) if has_bone else world_pts(implant_objs or all_objs)
center = mathutils.Vector(fp.mean(0))
size = float(np.linalg.norm(fp.max(0) - fp.min(0))) or 50.0

for off, energy in [((1, -1, 2), 5.5), ((-1, 0.6, 1), 3.0), ((0, 0.3, -1), 1.5)]:
    ld = bpy.data.lights.new("L", "SUN")
    ld.energy = energy
    lo = bpy.data.objects.new("L", ld)
    bpy.context.collection.objects.link(lo)
    lo.location = center + mathutils.Vector(off) * size

cd = bpy.data.cameras.new("c")
cd.clip_end = 1e5
cam = bpy.data.objects.new("c", cd)
bpy.context.collection.objects.link(cam)
if has_bone:
    view = mathutils.Vector((0.55, -0.8, 0.18)).normalized()   # side, whole bone + plate
    dist = size * 1.7   # far enough to fit the full femur in frame
else:
    # look at the plate's BROAD face (PCA minor axis) so the screw holes are visible
    ip = world_pts(implant_objs or all_objs)
    cc = ip - ip.mean(0)
    _, vecs = np.linalg.eigh(np.cov(cc.T))
    face_n = mathutils.Vector(vecs[:, 0])
    view = (face_n * 0.9 + mathutils.Vector((0.18, -0.28, 0.12))).normalized()
    dist = size * 1.5
cam.location = center + view * dist
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
