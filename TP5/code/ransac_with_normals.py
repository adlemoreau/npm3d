#
#
#      0===========================================================0
#      |                      TP6 Modelisation                     |
#      0===========================================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Plane detection with RANSAC
#
#------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


#------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

## Additional imports
import random
from sklearn.neighbors import KDTree
from tqdm import tqdm

#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def compute_plane(points: np.ndarray):
    """
    Computes a plane (reference point + normal) from the given point cloud.

    If np.linalg.norm(normal) is too small we will get NaN values,
    which is not an issue because n_points_in_plane will be equal to 0.
    """
    point = points[0].reshape((3, 1))
    normal = np.cross(points[1] - point.T, points[2] - point.T).reshape((3, 1))

    return point, normal / np.linalg.norm(normal)


##### NORMALS RANSAC

######## Utils from TP3
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
    all_eigenvalues = np.zeros((cloud_points.shape[0], 3))
    all_eigenvectors = np.zeros((cloud_points.shape[0], 3, 3))
    for i, idx in enumerate(neighborhoods_points):
        neigborhood = cloud_points[idx,:]
        eigenvalues, eigenvectors = PCA(neigborhood)
        all_eigenvalues[i] = eigenvalues
        all_eigenvectors[i] = eigenvectors
    return all_eigenvalues, all_eigenvectors


def compute_local_PCA_knn(query_points, cloud_points, k):
    
    # k is the number of neighbors
    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    kdtree = KDTree(cloud_points, leaf_size=k)

    _, neighborhoods = kdtree.query(query_points, k)

    all_eigenvalues = np.zeros((cloud_points.shape[0], 3))
    all_eigenvectors = np.zeros((cloud_points.shape[0], 3, 3))
    for i, idx in tqdm(enumerate(neighborhoods)):
        eigenvalues, eigenvectors = PCA(cloud_points[idx,:])
        all_eigenvalues[i] = eigenvalues
        all_eigenvectors[i] = eigenvectors
    return all_eigenvalues, all_eigenvectors

def compute_normals(query_points, cloud_points, k=None, radius=None):
    if (k and radius):
        raise Exception('Please specify which method to use by fixing ONLY one parameter between k and radius')
    elif k :
        all_eigenvalues, all_eigenvectors = compute_local_PCA_knn(query_points, cloud_points, k)
    elif radius :
        all_eigenvalues, all_eigenvectors = compute_local_PCA(query_points, cloud_points, radius)
    else : 
        raise Exception('Please specify which method to use by fixing one parameter between k and radius')
    return all_eigenvectors[:,0]

def normals_in_plane(points, pt_plane, normal_plane, normals_ref, threshold_in=0.1, max_angle=0.1):
    
    indexes = np.zeros(len(points))
    ref = np.tile(pt_plane,(points.shape[0])).T
    mask_dist = np.zeros(len(points))
    dists = np.abs((points - pt_plane.T) @ normal_plane)
    mask_dist = (dists < threshold_in).squeeze()  
    # Calculate the dot product between the normals and the reference direction
    cosine_values = np.clip((normals_ref @ normal_plane).squeeze(), -1, 1)
    # Calculate the angles using the arccosine function
    angles = np.arccos(cosine_values)
    mask_normals = angles < max_angle
    mask = (mask_dist & mask_normals)
    
    return mask

def normals_RANSAC(points, normals, nb_draws=100, threshold_in=0.1, max_angle=0.1):
    
    best_vote = 3
    best_pt_plane = np.zeros((3,1))
    best_normal_plane = np.zeros((3,1))
    
    seq = list(np.arange(0,points.shape[0]))
    for i in range(nb_draws):
        indexes_drawn = random.sample(seq, k=3)
        points_drawn = points[indexes_drawn,:]
        point_plane, normal_plane = compute_plane(points=points_drawn)
        indexes_in_plane = normals_in_plane(points=points, 
                                    pt_plane= point_plane, 
                                    normal_plane = normal_plane,
                                    normals_ref = normals,
                                    threshold_in= threshold_in)
        vote = np.sum(indexes_in_plane)
        if vote > best_vote : 
            best_vote = vote 
            best_pt_plane = point_plane
            best_normal_plane = normal_plane
                
    return best_pt_plane, best_normal_plane, best_vote

def normals_recursive_RANSAC(points, nb_draws=100, threshold_in=0.1, max_angle=0.1, nb_planes=2, k=30, radius=None):
 
    """
    Runs the RANSAC algorithm iteratively to find the "best plane" among all the points and then the "second-best plane"
    among the remaining points, and so on until nb_planes planes are found.
    Adaptation of the previous function with an additional condition to include a point in a plane: it also needs its
    normal to form an angle small enough with the normal to the plane.
    """

    n_points = len(points)
    plane_indices = np.arange(0, 0)
    plane_labels = np.arange(0, 0)
    remaining_indices = np.arange(n_points)
    normals = compute_normals(points, points, k=k, radius=radius)

    for label in range(nb_planes):
        pt_plane, normal_plane, _ = normals_RANSAC(points[remaining_indices], 
                                        normals[remaining_indices], 
                                        nb_draws=nb_draws, 
                                        threshold_in=threshold_in, 
                                        max_angle=max_angle)
        pts_in_plane = normals_in_plane(points[remaining_indices],
                                        pt_plane, 
                                        normal_plane,
                                        normals[remaining_indices], 
                                        threshold_in=threshold_in, 
                                        max_angle=max_angle)

        plane_indices = np.append(plane_indices, remaining_indices[pts_in_plane])
        plane_labels = np.append(plane_labels, np.repeat(label, pts_in_plane.sum()))
        remaining_indices = remaining_indices[~pts_in_plane]
        print(f"{label + 1}-th plane found.")

    return plane_indices, remaining_indices, plane_labels


#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':


    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']
    nb_points = len(points)
    

    print("\n--- 5) ---\n")
    radius = None 
    k_list = [10, 30, 50]
    nb_draws = 400
    threshold_in = 0.10
    max_angle = 0.10
    nb_planes_list = [5,2,3,4,5]
    for k in k_list:
        for nb_planes in nb_planes_list:
            t0 = time.time()
            pts_in_plane_indices, remaining_pts_indices, pts_in_plane_labels = normals_recursive_RANSAC(
                points, 
                nb_draws=nb_draws, 
                threshold_in = threshold_in,
                max_angle = max_angle, 
                nb_planes = nb_planes,
                k=k, 
                radius=None)

            t1 = time.time()
            print('normal recursive RANSAC done in {:.3f} seconds'.format(t1 - t0))
            str_save_best_plane = "../best_planes_normals_"+str(nb_planes)+"neighbors_"+str(k)+".ply"
            write_ply(
            str_save_best_plane,
                [
                    points[pts_in_plane_indices],
                    colors[pts_in_plane_indices],
                    labels[pts_in_plane_indices],
                    pts_in_plane_labels.astype(np.int32),
                ],
                ["x", "y", "z", "red", "green", "blue", "label", "plane_label"],
            )
            str_save_remaining =  "../remaining_points_best_planes_normals_"+str(nb_planes)+"neighbors_"+str(k)+".ply"
            write_ply(str_save_remaining, 
                [
                    points[remaining_pts_indices],
                    colors[remaining_pts_indices],
                    labels[remaining_pts_indices],
                ],
                ["x", "y", "z", "red", "green", "blue", "label"],
            )
            print("Done!")
            
        
            