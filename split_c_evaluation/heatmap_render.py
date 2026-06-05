"""Headless Blender renderer for the Split C stress heat map.

Reuses Split A's blender_render.py pattern (clear scene, STL import, bbox-based camera
framing, .blend save). Paints the implant surface with a per-vertex von Mises color
attribute and renders it FLAT (Workbench) so the colors read as a true stress contour.

Invoked as:
    blender --background --python heatmap_render.py -- \
        <mesh_stl> <data_json> <out_v0_png> <out_v1_png> <out_blend> [bone_stl]

<data_json> = {
    "implant": {"vertices": [[x,y,z],...], "rgba": [[r,g,b,a],...],
                "peak_xyz": [x,y,z], "marker_radius": float},
    "bone":    {"vertices": [[x,y,z],...], "rgba": [[r,g,b,a],...], "scale": float}  # optional
}
The implant (warm von-Mises colours) and the bone (cool load colours) are two disjoint
colour systems painted onto the two meshes in one scene. ``scale`` converts the imported
bone STL (Split A ships it in metres) to mm so it overlays the mm implant.
"""

import json
import os
import sys

import bpy
import mathutils
from mathutils import kdtree


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_stl(path, scale=1.0):
    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=path)
    else:
        bpy.ops.import_mesh.stl(filepath=path)
    obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else bpy.context.object
    if scale and scale != 1.0:
        obj.scale = (scale, scale, scale)
        bpy.context.view_layer.update()
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj


def paint_vertex_colors(obj, vertices, rgba):
    """Map each mesh vertex to the nearest provided point and write its RGBA."""
    mesh = obj.data
    kd = kdtree.KDTree(len(vertices))
    for i, v in enumerate(vertices):
        kd.insert(mathutils.Vector(v), i)
    kd.balance()
    attr = mesh.color_attributes.new(name="vm", type="FLOAT_COLOR", domain="POINT")
    for i, v in enumerate(mesh.vertices):
        _co, idx, _d = kd.find(v.co)
        c = rgba[idx]
        attr.data[i].color = (c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0)
    try:
        mesh.color_attributes.active_color = attr
        mesh.attributes.active_color = attr
    except Exception:
        pass


def uniform_color(obj, rgba):
    mesh = obj.data
    attr = mesh.color_attributes.new(name="vm", type="FLOAT_COLOR", domain="POINT")
    for i in range(len(mesh.vertices)):
        attr.data[i].color = rgba
    try:
        mesh.color_attributes.active_color = attr
        mesh.attributes.active_color = attr
    except Exception:
        pass


def scene_bounds():
    mins = mathutils.Vector((float("inf"),) * 3)
    maxs = mathutils.Vector((float("-inf"),) * 3)
    found = False
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        found = True
        for corner in obj.bound_box:
            world = obj.matrix_world @ mathutils.Vector(corner)
            mins = mathutils.Vector(min(a, b) for a, b in zip(mins, world))
            maxs = mathutils.Vector(max(a, b) for a, b in zip(maxs, world))
    if not found:
        return mathutils.Vector((-1, -1, -1)), mathutils.Vector((1, 1, 1))
    return mins, maxs


def setup_light():
    mins, maxs = scene_bounds()
    center = (mins + maxs) / 2
    diag = (maxs - mins).length or 1.0
    ld = bpy.data.lights.new(name="key", type="SUN")
    ld.energy = 3.0
    light = bpy.data.objects.new(name="key", object_data=ld)
    bpy.context.collection.objects.link(light)
    light.location = center + mathutils.Vector((1, -1, 2)) * diag
    return center, diag


def add_camera(center, diag, direction):
    cd = bpy.data.cameras.new(name="cam")
    cd.clip_end = max(1000.0, diag * 10)
    cam = bpy.data.objects.new(name="cam", object_data=cd)
    bpy.context.collection.objects.link(cam)
    cam.location = center + direction.normalized() * diag * 1.5
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam


def render_to(path):
    scn = bpy.context.scene
    scn.render.engine = "BLENDER_WORKBENCH"
    scn.display.shading.light = "FLAT"  # no shading gradient — flat contour
    scn.display.shading.color_type = "VERTEX"  # use the per-vertex color attribute
    scn.render.image_settings.file_format = "PNG"
    scn.render.resolution_x = 1100
    scn.render.resolution_y = 900
    scn.render.film_transparent = False
    scn.world = scn.world or bpy.data.worlds.new("w")
    scn.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main():
    argv = sys.argv
    i = (argv.index("--") + 1) if "--" in argv else len(argv)
    mesh_path, data_json, out_v0, out_v1, out_blend = argv[i : i + 5]
    bone_path = argv[i + 5] if len(argv) > i + 5 else ""

    with open(data_json) as f:
        data = json.load(f)

    # backward-compatible: old flat schema = the implant layer
    impl = data.get("implant", data)
    bone = data.get("bone")

    clear_scene()
    # bone first (under the plate): paint its OWN cool stress colours, scaled metres -> mm.
    if bone_path and os.path.exists(bone_path):
        if bone and bone.get("vertices") and bone.get("rgba"):
            bobj = import_stl(bone_path, scale=float(bone.get("scale", 1.0)))
            paint_vertex_colors(bobj, bone["vertices"], bone["rgba"])
        else:
            uniform_color(import_stl(bone_path), (0.82, 0.80, 0.74, 1.0))  # ivory context
    implant = import_stl(mesh_path)
    paint_vertex_colors(implant, impl["vertices"], impl["rgba"])

    # peak-stress marker (dark sphere) at the implant hot spot, like Split A's anchors
    if impl.get("peak_xyz"):
        r = impl.get("marker_radius", 2.0)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=tuple(impl["peak_xyz"]))
        uniform_color(bpy.context.object, (0.05, 0.05, 0.05, 1.0))

    center, diag = setup_light()
    cam = add_camera(center, diag, mathutils.Vector((1, -1, 0.6)))
    render_to(out_v0)
    # second (lateral) view so a hot spot can't hide behind the mesh
    cam.location = center + mathutils.Vector((0.2, 1, 0.6)).normalized() * diag * 1.5
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    render_to(out_v1)

    bpy.ops.wm.save_as_mainfile(filepath=out_blend)


if __name__ == "__main__":
    main()
