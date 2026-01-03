import open3d as o3d

# Load the mesh
mesh_path = 'G:\soybean_2dgs_outdoor\soybean_1/train\ours_8000/fuse_post.ply'
mesh = o3d.io.read_triangle_mesh(mesh_path)

if len(mesh.triangles) > 600000:
    decimated_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=600000)
else:
    decimated_mesh = mesh
o3d.io.write_triangle_mesh(mesh_path.replace('.ply', '_decimated.ply'), decimated_mesh)
print("Done decimating mesh")