import numpy as np
from skimage.draw import line  # requires scikit-image
import numba as nb

# Option 4: Apply parallelism to your original implementation
# @nb.njit(parallel=True)
# def relax_parallel(U, V, update_mask, h, w, max_iter=20000, tolerance=5e-6):
#     for it in range(max_iter):
#         U_old = U.copy()
#         V_old = V.copy()
        
#         for i in nb.prange(1, h - 1):
#             for j in range(1, w - 1):
#                 if update_mask[i, j]:
#                     U[i, j] = 0.25 * (U_old[i - 1, j] + U_old[i + 1, j] + U_old[i, j - 1] + U_old[i, j + 1])
#                     V[i, j] = 0.25 * (V_old[i - 1, j] + V_old[i + 1, j] + V_old[i, j - 1] + V_old[i, j + 1])
        
#         # Check convergence (simple max norm)
#         errU = np.max(np.abs(U - U_old))
#         errV = np.max(np.abs(V - V_old))
#         if max(errU, errV) < tolerance:
#             print(f"Converged after {it+1} iterations, error = {max(errU, errV)}.")
#             return it+1, U, V
    
#     print(f"Reached max iterations, error = {max(errU, errV)}.")
#     return max_iter, U, V

@nb.njit(parallel=True)
def relax_parallel(U, V, update_mask, h, w, max_iter=20000, tolerance=2e-6):
    """
    Jacobi relaxation solver for Laplace equation with resolution-independent discretization.
    
    Args:
        U, V         : (h, w) arrays, initial guesses
        update_mask  : (h, w) boolean array, True for interior points to update
        h, w         : grid size
        max_iter     : max iterations
        tolerance    : convergence threshold
    """
    # Physical grid spacing, assuming domain is [0,1]x[0,1]
    dx = 1.0 / (w - 1)
    dy = 1.0 / (h - 1)
    r = dx / dy

    for it in range(max_iter):
        U_old = U.copy()
        V_old = V.copy()

        for i in nb.prange(1, h - 1):
            for j in range(1, w - 1):
                if update_mask[i, j]:
                    # Resolution-aware Laplace update
                    U[i, j] = (U_old[i-1, j] + U_old[i+1, j] + 
                               r**2 * (U_old[i, j-1] + U_old[i, j+1])) / (2.0 * (1.0 + r**2))
                    
                    V[i, j] = (V_old[i-1, j] + V_old[i+1, j] + 
                               r**2 * (V_old[i, j-1] + V_old[i, j+1])) / (2.0 * (1.0 + r**2))

        # Check convergence
        errU = np.max(np.abs(U - U_old))
        errV = np.max(np.abs(V - V_old))
        if max(errU, errV) < tolerance:
            print(f"Converged after {it+1} iterations, error = {max(errU, errV)}.")
            return it+1, U, V

    print(f"Reached max iterations, error = {max(errU, errV)}.")
    return max_iter, U, V

def assign_boundary_value(x, left=False, right=False):

    """
    x: (n, 2) points
    """

    assert left != right

    if left:
        v0 = np.array([0.5, 0.0])
        v1 = np.array([0.0, 0.0])
        v2 = np.array([0.0, 1.0])
        v3 = np.array([0.5, 1.0])
    
    if right:
        v0 = np.array([0.5, 0.0])
        v1 = np.array([1.0, 0.0])
        v2 = np.array([1.0, 1.0])
        v3 = np.array([0.5, 1.0])
    
    L = len(x)
    
    bound_value = np.zeros((L, 2))
    bound_value[0:L//4] = np.linspace(v0, v1, L//4)
    bound_value[L//4:L//4*3] = np.linspace(v1, v2, L//4*2)
    bound_value[L//4*3:] = np.linspace(v2, v3, L - L//4*3)
    
    return bound_value

def rasterize_polygon(vertices, h, w, scale=True):
    """
    Rasterize the boundary of a polygon onto an array of shape [h, w].
    
    Parameters:
      vertices: NumPy array of shape [m,2] with polygon vertices in normalized [0,1] coordinates.
                vertices[:, 0] corresponds to x (column) and vertices[:, 1] to y (row).
      h, w: Height and width of the output image.
      
    Returns:
      boundary_img: A 2D NumPy array (shape [h, w]) with the polygon boundary pixels set to 1.
      boundary_pixels: A list of (row, col) tuples corresponding to the boundary pixels in the
                       sequence they are drawn.
    """
    # Map normalized coordinates to pixel indices.
    # x (vertices[:,0]) maps to column index in [0, w-1]
    # y (vertices[:,1]) maps to row index in [0, h-1]
    if scale:
        pts = np.round(np.column_stack((vertices[:, 1] * (h - 1),
                                        vertices[:, 0] * (w - 1)))).astype(int)
    else:
        pts = np.round(np.column_stack((vertices[:, 1],
                                        vertices[:, 0]))).astype(int)
    
    boundary_img = np.zeros((h, w), dtype=np.uint8)
    boundary_pixels = []
    m = pts.shape[0]
    
    # Loop over each edge (last vertex connects back to the first)
    for i in range(m-1):
        r0, c0 = pts[i]
        r1, c1 = pts[(i + 1) % m]
        rr, cc = line(r0, c0, r1, c1)
        for r, c in zip(rr, cc):
            boundary_img[r, c] = 1
            boundary_pixels.append((r, c))
            
    return boundary_img, boundary_pixels

# Example usage:
if __name__ == '__main__':
    # Define a square polygon in normalized coordinates.
    vertices = np.array([
        [0.2, 0.2],
        [0.8, 0.2],
        [0.8, 0.8],
        [0.2, 0.8]
    ])
    
    h, w = 100, 100  # Output image dimensions.
    img, pixel_seq = rasterize_polygon(vertices, h, w)

    import matplotlib.pyplot as plt
    plt.imshow(img, cmap='gray')
    plt.title('Rasterized Polygon Boundary')
    plt.axis('off')
    plt.show()
    
    print("Sequence of boundary pixels (row, col):")
    print(pixel_seq)
