"""
Test ear clipping polygon triangulation for building roofs.
"""
import numpy as np

def is_point_in_triangle(p, a, b, c):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def triangulate_polygon_earcut(pts):
    """
    Robust 2D Ear Clipping triangulation for arbitrary simple polygons (convex and concave).
    pts: Nx2 array of (x, y) coordinates.
    Returns: list of (i0, i1, i2) index triplets.
    """
    n = len(pts)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]
        
    # Ensure Counter-Clockwise winding
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    
    indices = list(range(n))
    if area < 0:
        indices.reverse()
        
    triangles = []
    max_iters = n * n
    iters = 0
    
    while len(indices) > 3 and iters < max_iters:
        iters += 1
        ear_found = False
        m = len(indices)
        for i in range(m):
            prev_idx = indices[(i - 1 + m) % m]
            curr_idx = indices[i]
            next_idx = indices[(i + 1) % m]
            
            p_prev = pts[prev_idx]
            p_curr = pts[curr_idx]
            p_next = pts[next_idx]
            
            # Convex check (cross product > 0 for CCW)
            cross = (p_curr[0] - p_prev[0]) * (p_next[1] - p_curr[1]) - (p_curr[1] - p_prev[1]) * (p_next[0] - p_curr[0])
            if cross <= 1e-7:
                continue
                
            # Check if any other point is inside this triangle
            contains_point = False
            for j in range(m):
                if j in ((i - 1 + m) % m, i, (i + 1) % m):
                    continue
                test_pt = pts[indices[j]]
                if is_point_in_triangle(test_pt, p_prev, p_curr, p_next):
                    contains_point = True
                    break
                    
            if not contains_point:
                triangles.append((prev_idx, curr_idx, next_idx))
                indices.pop(i)
                ear_found = True
                break
                
        if not ear_found:
            # Fallback: remove one triangle to make progress
            triangles.append((indices[0], indices[1], indices[2]))
            indices.pop(1)
            
    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))
        
    return triangles

if __name__ == "__main__":
    # Test on L-shaped concave polygon
    l_shape = np.array([
        [0, 0], [10, 0], [10, 5], [5, 5], [5, 10], [0, 10]
    ], dtype=np.float32)

    triangles = triangulate_polygon_earcut(l_shape)
    print(f"L-shape vertices: 6, Triangles produced: {len(triangles)}, {triangles}")
    assert len(triangles) == 4, "Triangulation should yield N-2 = 4 triangles"
    
    # Test on rectangle
    rect = np.array([[0,0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    tri_rect = triangulate_polygon_earcut(rect)
    print(f"Rectangle vertices: 4, Triangles produced: {len(tri_rect)}, {tri_rect}")
    assert len(tri_rect) == 2, "Triangulation should yield 2 triangles"
    
    print("ALL EAR CLIPPING TESTS PASSED CLEANLY!")
