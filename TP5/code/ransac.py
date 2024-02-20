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
    If np.linalg.norm(normal) is too small we will get NaN values, which is not an issue because n_points_in_plane will be equal to 0.
    """
    point = points[0].reshape((3, 1))
    normal = np.cross(points[1] - point.T, points[2] - point.T).reshape((3, 1))
    
    return point, normal / np.linalg.norm(normal)



def in_plane(points, pt_plane, normal_plane, threshold_in=0.1):
    
    indexes = np.zeros(len(points))
    dists = np.abs((points - pt_plane.T) @ normal_plane)
    indexes = (dists < threshold_in).squeeze()
    
    return indexes


def RANSAC(points, nb_draws=100, threshold_in=0.1):
    
    best_vote = 3
    best_pt_plane = np.zeros((3,1))
    best_normal_plane = np.zeros((3,1))
    
    seq = list(np.arange(0,points.shape[0]))
    for i in range(nb_draws):
        indexes_drawn = random.sample(seq, k=3)
        points_drawn = points[indexes_drawn,:]
        point_plane, normal_plane = compute_plane(points=points_drawn)
        indexes_in_plane = in_plane(points=points, 
                                    pt_plane= point_plane, 
                                    normal_plane = normal_plane,
                                    threshold_in= threshold_in)
        vote = np.sum(indexes_in_plane)
        if vote > best_vote : 
            best_vote = vote 
            best_pt_plane = point_plane
            best_normal_plane = normal_plane
                
    return best_pt_plane, best_normal_plane, best_vote


def recursive_RANSAC(points, nb_draws=100, threshold_in=0.1, nb_planes=2):
    
    nb_points = len(points)
    plane_inds = np.arange(0,0)
    plane_labels = np.arange(0,0)
    remaining_inds = np.arange(0,nb_points)
    points_iter = points
    
    for i in range(nb_planes):
        print(f'iteration/plan {i}, remaining points {len(remaining_inds)}')
        best_pt_plane, best_normal_plane, best_vote = RANSAC(points_iter, nb_draws=nb_draws, threshold_in=threshold_in)
        indexes_in_plane = in_plane(points=points_iter, 
                                    pt_plane= best_pt_plane, 
                                    normal_plane = best_normal_plane,
                                    threshold_in= threshold_in)

        plane_inds = np.append(plane_inds, remaining_inds[indexes_in_plane])
        plane_labels = np.append(plane_labels, np.repeat(i, indexes_in_plane.sum()))
        remaining_inds = remaining_inds[~indexes_in_plane]
        points_iter = points[remaining_inds]
         
    return plane_inds, remaining_inds, plane_labels


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
    # file_path = '../data/Notre_Dame_Des_Champs_1.ply'
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']
    nb_points = len(points)
    

    # Computes the plane passing through 3 randomly chosen points
    # ************************
    #
    print('\n--- 1) and 2) ---\n')
    
    # Define parameter
    threshold_in = 0.10

    # Take randomly three points
    pts = points[np.random.randint(0, nb_points, size=3)]
    
    # Computes the plane passing through the 3 points
    t0 = time.time()
    pt_plane, normal_plane = compute_plane(pts)
    t1 = time.time()
    print('plane computation done in {:.3f} seconds'.format(t1 - t0))
    
    # Find points in the plane and others
    t0 = time.time()
    points_in_plane = in_plane(points, pt_plane, normal_plane, threshold_in)
    t1 = time.time()
    print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
    plane_inds = points_in_plane.nonzero()[0]
    remaining_inds = (1-points_in_plane).nonzero()[0]
    
    # Save extracted plane and remaining points
    write_ply('../plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('../remaining_points_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    

    # Computes the best plane fitting the point cloud
    # ***********************************
    #
    #
    
    print('\n--- 3) ---\n')

    # Define parameters of RANSAC
    nb_draws = 100
    threshold_in = 0.10

    # Find best plane by RANSAC
    t0 = time.time()
    best_pt_plane, best_normal_plane, best_vote = RANSAC(points, nb_draws, threshold_in)
    t1 = time.time()
    print('RANSAC done in {:.3f} seconds'.format(t1 - t0))
    
    # Find points in the plane and others
    points_in_plane = in_plane(points, best_pt_plane, best_normal_plane, threshold_in)
    plane_inds = points_in_plane.nonzero()[0]
    remaining_inds = (1-points_in_plane).nonzero()[0]
    
    # Save the best extracted plane and remaining points
    write_ply('../best_plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('../remaining_points_best_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    

    # Find "all planes" in the cloud
    # ***********************************
    #
    #
    
    print('\n--- 4) ---\n')
    
    # Define parameters of recursive_RANSAC
    nb_draws = 100
    threshold_in = 0.1
    nb_planes = 2
    
    # Recursively find best plane by RANSAC
    t0 = time.time()
    plane_inds, remaining_inds, plane_labels = recursive_RANSAC(points, nb_draws, threshold_in, nb_planes)
    t1 = time.time()
    print('recursive RANSAC done in {:.3f} seconds'.format(t1 - t0))
                
    # Save the best planes and remaining points
    write_ply('../best_planes.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
    # write_ply('../best_planes_notredame_5_05.ply', [points[plane_inds]], ['x', 'y', 'z'])
    write_ply('../remaining_points_best_planes.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    
    print("Done!")
    
   
    