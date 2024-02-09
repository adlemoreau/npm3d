#
#
#      0===========================================================0
#      |              TP4 Surface Reconstruction                   |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 15/01/2024
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from skimage import measure

import trimesh


# Hoppe surface reconstruction
def compute_hoppe(points, normals, scalar_field, grid_resolution, min_grid, size_voxel):
    """Compute the Hoppe implicit function"""
    
    # Create a regular grid of the volume space
    grid = np.mgrid[0:grid_resolution, 0:grid_resolution, 0:grid_resolution]
    d = grid.shape[0]
    grid = grid.reshape(d, -1).T
    grid = (grid * size_voxel + min_grid).astype(np.float32)

    # Compute the Hoppe function
    kdtree = KDTree(points)
    ind = kdtree.query(grid, return_distance=False).squeeze()

    volume = np.sum(normals[ind]*(grid-points[ind]),axis=1)
    scalar_field[:, :, :] = volume.reshape(grid_resolution, grid_resolution, grid_resolution)



# IMLS surface reconstruction
def compute_imls(points, normals, scalar_field, grid_resolution, min_grid, size_voxel, knn):
    
    # Nearest neighbor search
    kdtree = KDTree(points)
    h = 0.01
    dim = points.shape[1]
    coord = np.arange(grid_resolution)
    grid = np.stack(np.meshgrid(*dim*[coord], indexing='ij'), axis=-1) # attention mettre le * à la décompression pour avoir une liste sans espace mémoire partagé
    grid = (grid * size_voxel + min_grid).astype(np.float32).reshape(-1,dim)
    distance, neighbors_idx  = kdtree.query(grid, k=knn)
    theta_i = np.exp(-(distance**2)/h**2)
    inter = np.sum(normals[neighbors_idx]*(grid[:,np.newaxis]-points[neighbors_idx]),axis=-1)
    num = np.sum(inter*theta_i, axis=-1)
    denom = np.sum(theta_i)
    f = num / (denom + 1e-10)
    scalar_field[:,:,:] = f.reshape(*dim*[grid_resolution])



if __name__ == "__main__":

    t0 = time.time()

    # Path of the file
    file_path = "../data/bunny_normals.ply"

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data["x"], data["y"], data["z"])).T
    normals = np.vstack((data["nx"], data["ny"], data["nz"])).T

    # Compute the min and max of the data points
    min_grid = np.amin(points, axis=0)
    max_grid = np.amax(points, axis=0)

    # Increase the bounding box of data points by decreasing min_grid and inscreasing max_grid
    min_grid = min_grid - 0.10 * (max_grid - min_grid)
    max_grid = max_grid + 0.10 * (max_grid - min_grid)

    # grid_resolution is the number of voxels in the grid in x, y, z axis
    # grid_resolution = 16
    grid_resolution = 128
    size_voxel = max([(max_grid[0]-min_grid[0])/(grid_resolution-1),(max_grid[1]-min_grid[1])/(grid_resolution-1),(max_grid[2]-min_grid[2])/(grid_resolution-1)])

    # Create a volume grid to compute the scalar field for surface reconstruction
    scalar_field = np.zeros(
        (grid_resolution, grid_resolution, grid_resolution), dtype=np.float32
    )

    # Compute the scalar field in the grid
    #compute_hoppe(points, normals, scalar_field, grid_resolution, min_grid, size_voxel)
    compute_imls(points,normals,scalar_field,grid_resolution,min_grid,size_voxel,30)

    # Compute the mesh from the scalar field based on marching cubes algorithm
    verts, faces, normals_tri, values_tri = measure.marching_cubes(
        scalar_field, level=0.0, spacing=(size_voxel, size_voxel, size_voxel)
    )
    verts += min_grid

    # Export the mesh in ply using trimesh lib
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(file_obj="../bunny_mesh_imls_"+str(grid_resolution)+".ply", file_type="ply")

    print("Total time for surface reconstruction : ", time.time() - t0)
