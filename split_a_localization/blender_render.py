
import bpy
import json
import sys
import os

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def create_material(name, color):
    mat = bpy.data.materials.new(name=name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = color
    return mat

def render_scene(output_path):
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

    if len(argv) <= index + 1:
        print("Usage: blender --background --python blender_render.py -- <plan_json_path> <bone_mesh_path> <output_png_path>")
        sys.exit(1)

    plan_path = argv[index]
    bone_mesh_path = argv[index+1]
    output_path = argv[index+2]

    with open(plan_path, 'r') as f:
        plan = json.load(f)

    clear_scene()

    # Load bone mesh
    if os.path.exists(bone_mesh_path):
        bpy.ops.import_mesh.stl(filepath=bone_mesh_path)

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
        bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=50, location=(origin['x'] + axis[0]*25, origin['y'] + axis[1]*25, origin['z'] + axis[2]*25))
        # align cylinder to axis
        # this is a bit tricky, skipping for now
    
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.view3d.camera_to_view_selected()

    render_scene(output_path)


if __name__ == "__main__":
    main()
