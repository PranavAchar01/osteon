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
    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=path)
    else:
        bpy.ops.import_mesh.stl(filepath=path)

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
        for corner in obj.bound_box:
            world = obj.matrix_world @ mathutils.Vector(corner)
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
    setup_camera_and_light()
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

def main():
    # argv is blender executable, --background, --python, script_path, json_string, output_path
    argv = sys.argv
    try:
        index = argv.index("--") + 1
    except ValueError:
        index = len(argv)

    if len(argv) <= index + 2:
        print("Usage: blender --background --python blender_render.py -- <plan_json_path> <bone_mesh_path> <output_png_path>")
        sys.exit(1)

    plan_path = argv[index]
    bone_mesh_path = argv[index+1]
    output_path = argv[index+2]

    with open(plan_path, 'r') as f:
        plan = json.load(f)

    clear_scene()

    # Load bone mesh
    if bone_mesh_path and os.path.exists(bone_mesh_path):
        import_stl(bone_mesh_path)

    green_mat = create_material("Green", (0, 1, 0, 1))
    red_mat = create_material("Red", (1, 0, 0, 1))

    # Anchors
    for anchor in plan.get("anchor_points", []):
        xyz = anchor["xyz"]
        bpy.ops.mesh.primitive_uv_sphere_add(radius=2, location=(xyz["x"], xyz["y"], xyz["z"]))
        bpy.context.object.data.materials.append(green_mat)

    # Defect
    defect_centroid = plan.get("defect_region", {}).get("centroid")
    if defect_centroid:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=5, location=(defect_centroid["x"], defect_centroid["y"], defect_centroid["z"]))
        bpy.context.object.data.materials.append(red_mat)
    
    # Frame
    origin = plan.get("coordinate_frame", {}).get("origin", {"x":0, "y":0, "z":0})
    basis = plan.get("coordinate_frame", {}).get("basis", [[1,0,0],[0,1,0],[0,0,1]])

    for i, (axis, color) in enumerate(zip(basis, [(1,0,0,1), (0,1,0,1), (0,0,1,1)])):
        # Create a cylinder
        bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=50, location=(0,0,0))
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


    
    render_scene(output_path)


if __name__ == "__main__":
    main()
