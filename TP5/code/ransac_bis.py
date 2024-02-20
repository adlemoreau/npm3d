#
#
#      0===================0
#      |    6 Modelling    |
#      0===================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      First script of the practical session. Plane detection by RANSAC
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np
import os

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time


# ----------------------------------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def compute_plane(points):

    p1, p2, p3 = points
    normal = np.cross(p2 - p1, p3 - p1)

    return p1, normal


def in_plane(points, ref_pt, normal, threshold_in=0.1):

    norm = np.maximum(np.linalg.norm(normal), 1e-10)
    distance = np.abs((points - ref_pt) @ normal) / norm

    return np.flatnonzero(distance<threshold_in)


def RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1):

    best_ref_pt = np.zeros((3,1))
    best_normal = np.zeros((3,1))
    best_nb = 0

    for i in range(NB_RANDOM_DRAWS):
        # Select random set of 3 points
        random_indices = np.random.choice(len(points), 3, replace=False)
        random_points = points[random_indices]

        # Evaluate random plane
        ref_pt, normal = compute_plane(random_points)
        nb_points_in_plane = len(in_plane(points, ref_pt, normal, threshold_in))

        if nb_points_in_plane > best_nb :
            # Update best plane
            best_ref_pt = ref_pt
            best_normal = normal
            best_nb = nb_points_in_plane

    return best_ref_pt, best_normal, best_nb


def recursive_RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1, NB_PLANES=2):
    plane_labels = -np.ones(len(points), dtype=np.int32)

    for plane in range(NB_PLANES):
        print('Computing plane {:d}/{:d}...'.format(plane+1, NB_PLANES), end='\r')
        remaining_inds = np.flatnonzero(plane_labels==-1)

        ref_pt, normal, vote = RANSAC(points[remaining_inds], NB_RANDOM_DRAWS, threshold_in)

        plane_labels[in_plane(points, ref_pt, normal, threshold_in)] = plane

    plane_inds, remaining_inds = plane_labels>=0, plane_labels<0

    return plane_inds, remaining_inds, plane_labels[plane_inds]


# ----------------------------------------------------------------------------------------------------------------------
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
     
    #### with notre dame
    # Path of the file
    file_path_ndc = '../data/indoor_scan.ply'

    # Load point cloud
    data_ndc = read_ply(file_path_ndc)

    # Concatenate data
    points_ndc = np.vstack((data_ndc['x'], data_ndc['y'], data_ndc['z'])).T
    colors_ndc = np.vstack((data_ndc['red'], data_ndc['green'], data_ndc['blue'])).T
    labels_ndc = data_ndc['label']
    nb_points_ndc = len(points_ndc)
    
    ## run ransac 
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
    write_ply('../best_plane_ndc.ply', [points_ndc[plane_inds_ndc], colors_ndc[plane_inds_ndc], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('../remaining_points_best_plane_ndc.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    
    # Find "all planes" in the cloud
    # ***********************************
    #
    #
    
    print('\n--- 4) ---\n')
    
    # Define parameters of recursive_RANSAC
    nb_draws = 100
    threshold_in = 0.10
    nb_planes = 2
    
    # Recursively find best plane by RANSAC
    t0 = time.time()
    plane_inds, remaining_inds, plane_labels = recursive_RANSAC(points, nb_draws, threshold_in, nb_planes)
    t1 = time.time()
    print('recursive RANSAC done in {:.3f} seconds'.format(t1 - t0))
                
    # Save the best planes and remaining points
    write_ply('../best_planes.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
    write_ply('../remaining_points_best_planes.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    
    print("Done!")