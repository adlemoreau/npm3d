#
#
#      0===================================0
#      |    TP2 Iterative Closest Point    |
#      0===================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
#------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 17/01/2018
#


#------------------------------------------------------------------------------------------
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
from visu import show_ICP

import sys


#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''
    R = np.eye(data.shape[0])
    T = np.zeros((data.shape[0],1))
    
    # Calculate barycenters p_m and p_m'
    p_m = np.mean(ref, axis=1).reshape(-1,1)
    p_m_prime = np.mean(data, axis=1).reshape(-1,1)
    
    # Compute centered clouds Q and Q'
    Q = ref - p_m
    Q_prime = data - p_m_prime
    
    # Get covariance matrix H
    H = Q_prime @ Q.T
    
    # Find the singular value decomposition USV.T of H
    U, S, V = np.linalg.svd(H)
    
    # Compute R and T
    R = V.T @ U.T
    
    if np.linalg.det(R) < 0:
        U[:,-1] = -U[:,-1]
        R = V.T @ U.T
        
    T = p_m - R @ p_m_prime
    
    return R, T


def icp_point_to_point(data, ref, max_iter, RMS_threshold):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''
    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []
    
    # Initiate RMS and iteration counter
    RMS = 0
    i = 0
    old_R = np.identity(len(data))
    old_T = np.zeros((len(data), 1))
    
    tree = KDTree(ref.T)
    
    # Loop until convergence
    while (i < max_iter and RMS > RMS_threshold) or (i == 0):   
             
        # Find the nearest neighbors between the data and the ref
        neighbors = tree.query(data_aligned.T, return_distance=False).squeeze()
        
        # Compute the transformation between the current data and the ref
        R_incr, T_incr = best_rigid_transform(data_aligned, ref[:, neighbors])
        
        R = old_R @ R_incr
        T = old_T + old_R @ T_incr
        
        R_list.append(R)
        T_list.append(T)
        
        old_R = R
        old_T = T
        
        # Apply the transformation to the data
        data_aligned = R_incr @ data_aligned + T_incr
        
        neighbors_list.append(neighbors)
        
        # Compute RMS
        distances2 = np.sum(np.power(data_aligned - ref[:, neighbors], 2), axis=0)
        RMS = np.sqrt(np.mean(distances2))
        RMS_list.append(RMS)
        
        # Update counter
        i += 1
    
    return data_aligned, R_list, T_list, neighbors_list, RMS_list


def icp_point_to_point_fast(data, ref, sampling_limit, max_iter, RMS_threshold):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        sampling_limit = int
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration 
    '''
    # Variable for aligned data
    data_aligned = np.copy(data)
    N_data = data.T.shape[0]
    index_all = np.arange(N_data)
    
    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []
    
    # Initiate RMS and iteration counter
    RMS = 0
    i = 0
    old_R = np.identity(len(data))
    old_T = np.zeros((len(data), 1))
    
    # Loop until convergence
    while ((i < max_iter) and (RMS > RMS_threshold)) or (i == 0):   
             
        # Find the nearest neighbors between the data and the ref
        tree = KDTree(ref.T)
        subset_data_aligned_index = np.random.choice(index_all, size=sampling_limit, replace=False)
        subset_data_aligned = data_aligned[:,subset_data_aligned_index]
        neighbors = tree.query(subset_data_aligned.T, return_distance=False).squeeze()
        neighbors_list.append(neighbors)
        
        # Compute the transformation between the current data and the ref
        R_incr, T_incr = best_rigid_transform(subset_data_aligned, ref[:, neighbors])
        
        R = old_R @ R_incr
        T = old_T + old_R @ T_incr
        
        R_list.append(R)
        T_list.append(T)
        
        old_R = R
        old_T = T
        
        # Apply the transformation to the data
        data_aligned = R_incr @ data_aligned + T_incr
        
        # Compute RMS
        distances2 = np.sum(np.power(subset_data_aligned - ref[:, neighbors], 2), axis=0)
        RMS = np.sqrt(np.mean(distances2))
        RMS_list.append(RMS)
        
        # Update counter
        i += 1
    
    return data_aligned, R_list, T_list, neighbors_list, RMS_list


#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#


if __name__ == '__main__':
   
    # Transformation estimation
    # *************************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_r_path = '../data/bunny_returned.ply'

		# Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_r_ply = read_ply(bunny_r_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_r = np.vstack((bunny_r_ply['x'], bunny_r_ply['y'], bunny_r_ply['z']))

        # Find the best transformation
        R, T = best_rigid_transform(bunny_r, bunny_o)

        # Apply the tranformation
        bunny_r_opt = R.dot(bunny_r) + T

        # Save cloud
        write_ply('../bunny_r_opt', [bunny_r_opt.T], ['x', 'y', 'z'])

        # Compute RMS
        distances2_before = np.sum(np.power(bunny_r - bunny_o, 2), axis=0)
        RMS_before = np.sqrt(np.mean(distances2_before))
        distances2_after = np.sum(np.power(bunny_r_opt - bunny_o, 2), axis=0)
        RMS_after = np.sqrt(np.mean(distances2_after))

        print('Average RMS between points :')
        print('Before = {:.3f}'.format(RMS_before))
        print(' After = {:.3f}'.format(RMS_after))
   

    # Test ICP and visualize
    # **********************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        ref2D_path = '../data/ref2D.ply'
        data2D_path = '../data/data2D.ply'
        
        # Load clouds
        ref2D_ply = read_ply(ref2D_path)
        data2D_ply = read_ply(data2D_path)
        ref2D = np.vstack((ref2D_ply['x'], ref2D_ply['y']))
        data2D = np.vstack((data2D_ply['x'], data2D_ply['y']))        

        # Apply ICP
        data2D_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(data2D, ref2D, 20, 1e-4)
        
        # Show ICP
        show_ICP(data2D, ref2D, R_list, T_list, neighbors_list)
        
        # Plot RMS
        plt.plot(RMS_list)
        plt.show()
        

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_p_path = '../data/bunny_perturbed.ply'
        
        # Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_p_ply = read_ply(bunny_p_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_p = np.vstack((bunny_p_ply['x'], bunny_p_ply['y'], bunny_p_ply['z']))

        # Apply ICP
        bunny_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(bunny_p, bunny_o, 30, 1e-4)
        
        # Show ICP
        show_ICP(bunny_p, bunny_o, R_list, T_list, neighbors_list)
        
        # Plot RMS
        plt.plot(RMS_list)
        plt.show()
        

    # Question bonus      
    if True:

            # Cloud paths
            ndc_o_path = '../data/Notre_Dame_Des_Champs_1.ply'
            ndc_p_path = '../data/Notre_Dame_Des_Champs_2.ply'
            
            # Load clouds
            ndc_o_ply = read_ply(ndc_o_path)
            ndc_p_ply = read_ply(ndc_p_path)
            ndc_o = np.vstack((ndc_o_ply['x'], ndc_o_ply['y'], ndc_o_ply['z']))
            ndc_p = np.vstack((ndc_p_ply['x'], ndc_p_ply['y'], ndc_p_ply['z']))

            # Apply ICP
            ndc_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point_fast(ndc_p, ndc_o, 1000, 100, 1e-4)
            ndc_p_opt2, R_list2, T_list2, neighbors_list2, RMS_list2 = icp_point_to_point_fast(ndc_p, ndc_o, 10000, 100, 1e-4)
            
            # Show ICP
            #show_ICP(ndc_p, ndc_o, R_list, T_list, neighbors_list)
            
            # Plot RMS
            plt.plot(RMS_list, label="1000 neighbors")
            plt.plot(RMS_list2, label="10000 neighbors")
            print(np.std(RMS_list))
            print(np.std(RMS_list2))
            plt.legend()
            plt.show()