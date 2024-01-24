#
#
#      0=============================0
#      |    TP4 Point Descriptors    |
#      0=============================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#

 
# Import numpy package and name it "np"
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def PCA(points):
    mu = np.mean(points, axis=0 )
    mean_points = points - mu
    N = points.shape[0]
    ## not use np.cov as it won't work when features > samples
    sigma = (mean_points.T @ mean_points)/N
    eigenvalues , eigenvectors = np.linalg.eigh(sigma)
    return eigenvalues, eigenvectors



def compute_local_PCA(query_points, cloud_points, radius):

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    kdtree = KDTree(cloud_points)
    neighborhoods_points = kdtree.query_radius(query_points, radius)
    all_eigenvalues = np.zeros((cloud.shape[0], 3))
    all_eigenvectors = np.zeros((cloud.shape[0], 3, 3))
    for i, idx in enumerate(neighborhoods_points):
        neigborhood = cloud_points[idx,:]
        eigenvalues, eigenvectors = PCA(neigborhood)
        all_eigenvalues[i] = eigenvalues
        all_eigenvectors[i] = eigenvectors
    return all_eigenvalues, all_eigenvectors





def compute_features(query_points, cloud_points, radius, eps=1e-8):
# Compute the features for all query points in the cloud
    eigenvalues, eigenvectors = compute_local_PCA(query_points, cloud_points, radius)
    
    # apply correction for 0 eigenvalues
    mask_eigenvalues_0 = np.where(eigenvalues==0)
    eigenvalues[mask_eigenvalues_0] = eps 

    # remember the convention of np.linalg.eigh
    lambda_1, lambda_2, lambda_3 = eigenvalues[:,2], eigenvalues[:,1], eigenvalues[:,0]

    # compute lineartiy, planarity and sphericity 
    linearity = 1 - lambda_2/lambda_1
    planarity = (lambda_2 - lambda_3)/lambda_1
    sphericity = lambda_3/lambda_1
    
    # Dip scalar field
    # Extract the sine of the angle corresponding to the third eigenvector (Z-component of the normal)
    sinus = eigenvectors[:, 2, 0]

    verticality = 2*np.arcsin(sinus)/np.pi
   

    return verticality, linearity, planarity, sphericity


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # PCA verification
    # ****************
    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = PCA(cloud)

        # Print your result
        print(eigenvalues)

        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050177 21.7893201  89.58924003]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)
        #

		
    # Normal computation
    # ******************
    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud, cloud, 0.50)
        normals = all_eigenvectors[:, :, 0]

        # Save cloud with normals
        write_ply('../data/Lille_street_small_normals.ply', (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])
		
    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute features 
        v, l, p, s = compute_features(cloud, cloud, 0.5)

        # Save cloud with features
        write_ply('../data/Lille_street_small_features.ply', [cloud, v, l, p, s], ['x','y','z','verticality','linearity','planarity','sphericity'])