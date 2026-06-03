import bpy
import json
import sys
import os
import mathutils

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def import_stl(path):
    # bpy.ops.import_mesh.stl was removed in Blender 4.x in favor of wm.stl_import.
    before = set(bpy.context.scene.objects)
    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=path)
    else:
        bpy.ops.import_mesh.stl(filepath=path)
    return [o for o in bpy.context.scene.objects if o not in before]

def normalize_bone(objects):
    """Match _rung2's normalization so the mesh shares the plan's coordinate space:
    recenter geometry to the origin, then convert m -> mm (x1000). Edits vertex
    coordinates directly so it behaves identically headless and in the GUI."""
    for o in objects:
        if o.type != 'MESH':
            continue
        me = o.data
        if not me.vertices or not me.polygons:
            continue
        # Area-weighted face centroid == trimesh.centroid used by _rung2.
        acc = mathutils.Vector()
        total_area = 0.0
        for poly in me.polygons:
            acc += poly.center * poly.area
            total_area += poly.area
        center = acc / total_area if total_area else mathutils.Vector()
        for v in me.vertices:
            v.co = (v.co - center) * 1000.0
        me.update()

def create_material(name, color):
    mat = bpy.data.materials.new(name=name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = color
    return mat

def scene_bounds():
    """World-space (min, max) corners over all mesh objects, or a unit box if empty."""
    mins = mathutils.Vector((float("inf"),) * 3)
    maxs = mathutils.Vector((float("-inf"),) * 3)
    found = False
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        found = True
        for v in obj.data.vertices:
            world = obj.matrix_world @ v.co
            mins = mathutils.Vector(min(a, b) for a, b in zip(mins, world))
            maxs = mathutils.Vector(max(a, b) for a, b in zip(maxs, world))
    if not found:
        return mathutils.Vector((-1, -1, -1)), mathutils.Vector((1, 1, 1))
    return mins, maxs

def setup_camera_and_light():
    """Frame the whole scene from a fixed angle. view3d ops don't exist in --background."""
    mins, maxs = scene_bounds()
    center = (mins + maxs) / 2
    diagonal = (maxs - mins).length or 1.0

    light_data = bpy.data.lights.new(name="key", type='SUN')
    light_data.energy = 3.0
    light = bpy.data.objects.new(name="key", object_data=light_data)
    bpy.context.collection.objects.link(light)
    light.location = center + mathutils.Vector((1, -1, 2)) * diagonal

    cam_data = bpy.data.cameras.new(name="cam")
    cam_data.clip_end = max(1000.0, diagonal * 10)
    cam = bpy.data.objects.new(name="cam", object_data=cam_data)
    bpy.context.collection.objects.link(cam)
    direction = mathutils.Vector((1, -1, 0.6)).normalized()
    cam.location = center + direction * diagonal * 1.6
    cam.rotation_euler = (center - cam.location).to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = cam

def render_scene(output_path):
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

def _reference_scale(plan):
    """Characteristic size of the scene, used to size markers so they're visible
    against any bone. Prefer the loaded bone's diagonal; else the anchor spread."""
    mins, maxs = scene_bounds()
    diag = (maxs - mins).length
    if diag > 1e-6:
        return diag
    pts = [a["xyz"] for a in plan.get("anchor_points", [])]
    if pts:
        spans = [max(p[k] for p in pts) - min(p[k] for p in pts) for k in ("x", "y", "z")]
        if max(spans) > 1e-6:
            return max(spans)
    return 100.0

def build_scene(plan, bone_mesh_path):
    clear_scene()

    # Load bone mesh (normalized to match the plan's coordinate space)
    if bone_mesh_path and os.path.exists(bone_mesh_path):
        normalize_bone(import_stl(bone_mesh_path))

    ref = _reference_scale(plan)
    anchor_r = ref * 0.012
    defect_r = ref * 0.020
    axis_len = ref * 0.55
    axis_r = ref * 0.004

    green_mat = create_material("Green", (0, 1, 0, 1))
    red_mat = create_material("Red", (1, 0, 0, 1))

    # Anchors
    for anchor in plan.get("anchor_points", []):
        xyz = anchor["xyz"]
        bpy.ops.mesh.primitive_uv_sphere_add(radius=anchor_r, location=(xyz["x"], xyz["y"], xyz["z"]))
        bpy.context.object.data.materials.append(green_mat)

    # Defect
    defect_centroid = plan.get("defect_region", {}).get("centroid")
    if defect_centroid:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=defect_r, location=(defect_centroid["x"], defect_centroid["y"], defect_centroid["z"]))
        bpy.context.object.data.materials.append(red_mat)

    # Frame
    origin = plan.get("coordinate_frame", {}).get("origin", {"x":0, "y":0, "z":0})
    basis = plan.get("coordinate_frame", {}).get("basis", [[1,0,0],[0,1,0],[0,0,1]])

    for i, (axis, color) in enumerate(zip(basis, [(1,0,0,1), (0,1,0,1), (0,0,1,1)])):
        # Create a cylinder
        bpy.ops.mesh.primitive_cylinder_add(radius=axis_r, depth=axis_len, location=(0,0,0))
        cyl = bpy.context.object

        # Create a material and assign it
        mat = create_material(f"axis_{i}", color)
        cyl.data.materials.append(mat)

        # Align the cylinder with the axis vector
        axis_vec = mathutils.Vector(axis)
        up_vec = mathutils.Vector((0,0,1))
        quat = up_vec.rotation_difference(axis_vec)
        cyl.rotation_euler = quat.to_euler()

        # Move the cylinder to the origin
        cyl.location = mathutils.Vector((origin['x'], origin['y'], origin['z']))


def _frame_all_when_ready():
    """Frame the whole scene in the viewport once the GUI is up (view3d ops need a
    VIEW_3D context that doesn't exist while the startup script runs). Returns a
    retry interval until a viewport is available, then None to stop."""
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            region = next((r for r in area.regions if r.type == 'WINDOW'), None)
            with bpy.context.temp_override(window=window, area=area, region=region):
                # Frame only meshes — the camera/light sit far out and would
                # blow up the bounds, zooming everything to a speck.
                bpy.ops.object.select_all(action='DESELECT')
                for o in bpy.context.scene.objects:
                    if o.type == 'MESH':
                        o.select_set(True)
                bpy.ops.view3d.view_selected()
            return None
    return 0.25

def main():
    # argv: blender ... --python script -- <plan_json_path> <bone_mesh_path> <output_path>
    # output_path ".png" -> render image; ".blend" -> save 3D file; anything else -> just
    # build the scene (used to open the live 3D scene in the Blender GUI).
    argv = sys.argv
    try:
        index = argv.index("--") + 1
    except ValueError:
        index = len(argv)

    if len(argv) <= index + 2:
        print("Usage: blender [--background] --python blender_render.py -- <plan_json_path> <bone_mesh_path> <output.png|output.blend|view>")
        sys.exit(1)

    plan_path = argv[index]
    bone_mesh_path = argv[index+1]
    output_path = argv[index+2]

    with open(plan_path, 'r') as f:
        plan = json.load(f)

    build_scene(plan, bone_mesh_path)
    setup_camera_and_light()

    if output_path.endswith(".blend"):
        bpy.ops.wm.save_as_mainfile(filepath=output_path)
    elif output_path.endswith(".png"):
        render_scene(output_path)
    # else: leave the assembled scene for interactive GUI viewing

    if not bpy.app.background:
        bpy.app.timers.register(_frame_all_when_ready, first_interval=0.5)


if __name__ == "__main__":
    main()
