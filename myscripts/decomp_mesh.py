"""
Decompose a convex mesh into smaller parts so that the Mujoco engine can simulate the concavity
(i.e., hollow part of a bowl)
"""

import coacd
import trimesh
import numpy as np
import os

MESH_DIR = "/home/khw/Projects/robosuite/robosuite/models/assets/objects/meshes"
OBJ_OUT_DIR = os.path.join(MESH_DIR, "bowl_decomp")
os.makedirs(OBJ_OUT_DIR, exist_ok=True)

# Reload and run decomposition
mesh = trimesh.load(os.path.join(MESH_DIR, "bowl.obj"), force="mesh")
mesh.fill_holes()
cmesh = coacd.Mesh(mesh.vertices, mesh.faces)
parts = coacd.run_coacd(cmesh, threshold=0.05, max_convex_hull=32)

# center/scale constants matching bowl.xml
SCALE = 0.001
REFPOS = np.array([73.82, 62.75, 21.56])

saved = []
for i, (verts, faces) in enumerate(parts):
    part_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    fname = f"bowl_part_{i:02d}.obj"
    part_mesh.export(os.path.join(OBJ_OUT_DIR, fname))
    saved.append(fname)
    print(f"Saved {fname}")

print(f"\nTotal parts: {len(saved)}")
