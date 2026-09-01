import sys
import numpy as np
import cv2

sys.path.insert(0, ".")
from depthwizard.viz.interactive_viewer import triangulate_polygon_earcut

# Test L-shaped polygon
pts = np.array([
    [0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]
], dtype=np.float32)

tri_indices = triangulate_polygon_earcut(pts)
valid_tris = []
for i0, i1, i2 in tri_indices:
    c_x = (pts[i0][0] + pts[i1][0] + pts[i2][0]) / 3.0
    c_y = (pts[i0][1] + pts[i1][1] + pts[i2][1]) / 3.0
    dist = cv2.pointPolygonTest(pts.reshape(-1, 1, 2).astype(np.int32), (c_x, c_y), False)
    if dist >= 0:
        valid_tris.append((i0, i1, i2))
        print(f"  Triangle ({i0},{i1},{i2}) centroid ({c_x:.1f},{c_y:.1f}) is INSIDE polygon (dist={dist:.1f})")
    else:
        print(f"  Triangle ({i0},{i1},{i2}) centroid ({c_x:.1f},{c_y:.1f}) is OUTSIDE polygon! DISCARDED.")

print(f"Total triangles: {len(tri_indices)}, Valid inside triangles: {len(valid_tris)}")
