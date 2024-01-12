#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
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

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def brute_force_spherical(queries, supports, radius):

    neighborhoods = []
    for q in queries:
        distances = np.linalg.norm(supports - q, axis=1)
        idx = np.where(distances < radius)[0]
        neighborhoods.append(idx)
    
    return neighborhoods


def brute_force_KNN(queries, supports, k):
    
    neighborhoods = []
    for q in queries:
        distances = np.linalg.norm(supports - q, axis=1)
        idx = np.argpartition(distances, k)[:k]
        neighborhoods.append(idx)

    return neighborhoods


# ------------------------------------------------------------------------------------------
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

    # Brute force neighborhoods
    # *************************
    #

    # If statement to skip this part if you want
    if True:

        # Define the search parameters
        neighbors_num = 100
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Search spherical
        t0 = time.time()
        neighborhoods = brute_force_spherical(queries, points, radius)
        t1 = time.time()

        # Search KNN      
        neighborhoods = brute_force_KNN(queries, points, neighbors_num)
        t2 = time.time()

        # Print timing results
        print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
        print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))

 



    # KDTree neighborhoods
    # ********************
    #

    # Question 4a
    # If statement to skip this part if wanted
    if False:

        # Define the search parameters
        num_queries = 1000

        leaf_sizes = [1, 10, 30, 50, 60, 70, 80, 85, 90, 100, 250, 500]
        times = []
        
        for leaf_size in leaf_sizes:
            tree = KDTree(points, leaf_size=leaf_size)
            t0 = time.time()
            tree.query_radius(points[:num_queries], r=0.2)
            t1 = time.time()
            times.append(t1 - t0)
        
        optimal_leaf_size = leaf_sizes[np.argmin(times)]
        print(f"Optimal leaf size: {optimal_leaf_size}") 
        
        plt.plot(leaf_sizes, times)
        plt.xlabel("Leaf size")
        plt.ylabel("Time (seconds)")
        plt.title("Time to do 1000 queries depending on the leaf size")
        plt.show()
     
    # Question 4b   
    if False :
        
        # Define the search parameters
        num_queries = 1000

        radius_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1, 2, 3, 5, 7, 10, 12, 15]
        times = []
        
        tree = KDTree(points, leaf_size=50) # optimal leaf size
        
        for r in radius_list:
            t0 = time.time()
            tree.query_radius(points[:num_queries], r=r)
            t1 = time.time()
            times.append(t1 - t0)
        
        total_time = points.shape[0] * times[1] / num_queries
        print(f"Estimated time to to search 20cm neighborhoods for all points in the cloud: {total_time} seconds")
        # print("Estimated time to to search 20cm neighborhoods for all points in the cloud: {:.0f} minutes".format(total_time / 60))
        
        plt.plot(radius_list, times)
        plt.xlabel("Radius")
        plt.ylabel("Time (seconds)")
        plt.title("Time to do 1000 queries depending on the radius")
        plt.show()