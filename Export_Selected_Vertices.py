bl_info = {
    "name": "Export Selected Vertices",
    "author": "Your Name",  # Change this!
    "version": (1, 0),
    "blender": (2, 80, 0),  # Minimum Blender version.  Works on later versions too.
    "location": "View3D > Sidebar > My Tools",
    "description": "Exports the coordinates of selected vertices to a file.",
    "warning": "",
    "doc_url": "",
    "category": "Mesh",
}

import bpy
import bmesh
import os
from typing import Generator

# --- Generator for vertex lines (as before) ---

def generate_vertex_lines(obj) -> Generator[str, None, None]:
    """Generates strings representing vertex coordinates (world space)."""
    if obj.mode == 'EDIT':
        bm = bmesh.from_edit_mesh(obj.data)
        selected_verts = [v for v in bm.select_history if isinstance(v, bmesh.types.BMVert)]
        if not selected_verts:
            selected_verts = (v for v in bm.verts if v.select)  # No real benefit being a generator.
    else:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        selected_verts = (v for v in bm.verts if v.select)

    for v in selected_verts:
        co = obj.matrix_world @ v.co
        yield f"{co.x} {co.y} {co.z}\n"
    if obj.mode != "EDIT":
        bm.free() #cleanup


# --- Operator to perform the export ---

class ExportSelectedVertices(bpy.types.Operator):
    """Exports coordinates of selected vertices to a file."""
    bl_idname = "mesh.export_selected_vertices"  # Unique ID
    bl_label = "Export Vertices"
    bl_options = {'REGISTER', 'UNDO'}  # Enable undo

    filepath: bpy.props.StringProperty(
        name="Output Path",
        description="Filepath for exporting vertex coordinates",
        default="", #empty by default
        subtype='FILE_PATH'  # Makes it a file selector
    )
    
    use_blend_file_dir: bpy.props.BoolProperty(
        name = "Use Blend file Directory",
        description="If enabled, use a default output file relative to .blend path",
        default=True
    )

    @classmethod
    def poll(cls, context):
        # Only active if in Edit Mode, an object is selected, the object type is Mesh and a vertex is selected
        return (context.mode == 'EDIT_MESH' and
                context.active_object is not None and
                context.active_object.type == 'MESH' and
                context.selected_objects)
                
    def invoke(self, context, event):
        if self.use_blend_file_dir:
           if context.blend_data.filepath:  # Check if the .blend file is saved.
              blend_dir = os.path.dirname(context.blend_data.filepath)
              self.filepath = os.path.join(blend_dir, "vertex_output.xyz") # Create an initial default file
           else:
              self.report({'WARNING'}, "Blend file must be saved to use a relative output path")
              # The dialog allows to override any way.

        #Open the file browser
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'} # Let the invoke run and open the dialog.
        

    def execute(self, context):
        obj = context.edit_object
        if not obj or obj.type != "MESH":
           self.report({'ERROR'}, 'Active object should be mesh and in edit mode!')
           return {'CANCELLED'}

        if not self.filepath:
            self.report({'ERROR'}, "No output file specified.")
            return {'CANCELLED'}
        try:
            with open(self.filepath, 'w') as file:
                 file.writelines(generate_vertex_lines(obj))
        except IOError as e:
            self.report({'ERROR'}, f"Error writing to file: {e}")
            return {'CANCELLED'}
        except Exception as e:
           self.report({'ERROR'}, f"Unexpected error!: {e}")

        self.report({'INFO'}, f"Vertex coordinates written to: {self.filepath}")
        return {'FINISHED'}



# --- UI Panel in the 3D View Sidebar ---

class MyAddonPanel(bpy.types.Panel):
    """Creates a Panel in the 3D View Sidebar."""
    bl_label = "My Tools"
    bl_idname = "VIEW3D_PT_my_tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "My Tools"  # Tab name

    def draw(self, context):
        layout = self.layout

        # Check if an object is selected and if it is a mesh.
        if context.active_object and context.active_object.type == "MESH":
           row = layout.row()
           row.prop(context.scene, "my_tool_use_relative_path")  # Link to the scene property
           row = layout.row() #next line
           row.operator("mesh.export_selected_vertices")
        else:
           layout.label(text="Select a Mesh to use tools")


# --- Registration (very important!) ---

def register():
    bpy.utils.register_class(ExportSelectedVertices)
    bpy.utils.register_class(MyAddonPanel)
     # Create a scene property to control output path relatively or not.
    bpy.types.Scene.my_tool_use_relative_path = bpy.props.BoolProperty(
        name="Relative Path",
        description="If True, exports in a folder relative to the .blend file path",
        default=True
    )



def unregister():
    bpy.utils.unregister_class(ExportSelectedVertices)
    bpy.utils.unregister_class(MyAddonPanel)
    del bpy.types.Scene.my_tool_use_relative_path


if __name__ == "__main__":
    register()