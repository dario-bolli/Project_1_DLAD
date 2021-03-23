import numpy as np
from matplotlib import pyplot as plt
import os
from load_data import load_data
import cv2 
import data_utils
import os

## ------------------------FUNCTIONS-----------------------------------------##
## --------------------------------------------------------------------------##

def printCloud2image(xyz_velodyne, img):
    """
    :param point cloud and current image
    From velo -> cam0 -> img coordinates of cam2
    :return image with colored projected points on it in function of their depth
    """
    R, T = data_utils.calib_velo2cam('data/problem_4/calib_velo_to_cam.txt')
    # Get Extrinsic Transformation matrix from velo to cam (homogeneous coordinates)
    Trans_matrix = np.hstack((R,T))
    Trans_matrix = np.vstack((Trans_matrix,np.array([0, 0, 0, 1])))

    velodyne_homogeneous = np.hstack((xyz_velodyne, np.ones((xyz_velodyne.shape[0],1))))
    # indices as columns and 3D coordinates as rows
    velodyne_homogeneous = np.transpose(velodyne_homogeneous)
    # To cam0 coordinate frame
    extrin_calib = np.matmul(Trans_matrix,velodyne_homogeneous)
    # filter points with negative z in Cam0 coordinates
    indexes = np.argwhere(extrin_calib[2,:]>=0).flatten()
    extrin_calib_fltrd = extrin_calib[:, indexes]
    #compute depth estimation: sqrt(x^2 + z^2) in Cam0 frame
    depth = np.sqrt(extrin_calib_fltrd[0,:]**2 + extrin_calib_fltrd[2,:]**2)
    
    # Project on camera 2
    intrins_calib = data_utils.calib_cam2cam("data/problem_4/calib_cam_to_cam.txt", '02')
    proj_cloud = np.matmul(intrins_calib,extrin_calib_fltrd)/extrin_calib_fltrd[2,:] #normalization by Zc

    #color points in function of their depth and print velodyne points on cam2 image
    color=data_utils.depth_color(depth, np.amin(depth), np.amax(depth))
    img = data_utils.print_projection_plt(proj_cloud, color, img)
    return img

def getCorrected_pointCloud(point_cloud, delta_t, vel, ang, trigger_cam):
    """
    :param point cloud, time intervall between each point in the velodyne scan,
           velocity and angular interpolation
           time where image is taken on camera2
    :return corrected velodyne points without distorsion effect
    """
    M_p = np.zeros((4,4,point_cloud.shape[1])) # Rigid Body Transformation Matrix
    pointCloud_corrected = np.zeros((4, point_cloud.shape[1]))
    homogeneous = np.array([0,0,0,1]) # to homogeneous coordinates

    #------ Velodyne points taken BEFORE the image trigger on camera 2-------#
    for j in range(0,trigger_cam):
        #Compute translation using angular velocity and velocity
        pos_xy_p = -vel[0:2]*(trigger_cam-j)*delta_t
        #Compute rotation using pos and angular velocity
        theta_p = -ang[2]*(trigger_cam-j)*delta_t
        R = np.array([[np.cos(theta_p), -np.sin(theta_p), 0],
                    [np.sin(theta_p), np.cos(theta_p), 0],
                    [0,             0,     1]])
        T = np.array([[pos_xy_p[0]],
                    [pos_xy_p[1]],
                    [0]])
        
        R_T = np.hstack((R,T))
        M_p[:,:,j] = np.vstack((R_T, homogeneous))
        # Rigid Body transformation in homogeneous coordinates
        homg_coordinates= np.hstack((point_cloud[:,j], 1))
        # Remove distorsion
        pointCloud_corrected[:,j] = np.matmul(M_p[:,:,j], homg_coordinates) 
    #-------------------------------------------------------------------------#

    #------ Velodyne points taken AFTER the image trigger on camera 2---------#
    for j in range(trigger_cam, point_cloud.shape[1]):
        #Compute translation using angular velocity and velocity
        pos_xy_p = vel[0:2]*(j-trigger_cam)*delta_t
        #Compute rotation using pos and angular velocity
        theta_p = ang[2]*(j-trigger_cam)*delta_t
        R = np.array([[np.cos(theta_p), -np.sin(theta_p), 0],
                    [np.sin(theta_p), np.cos(theta_p), 0],
                    [0,             0,     1]])
        T = np.array([[pos_xy_p[0]],
                    [pos_xy_p[1]],
                    [0]])
        
        R_T = np.hstack((R,T))
        M_p[:,:,j] = np.vstack((R_T, homogeneous))
        # Rigid Body transformation in homogeneous coordinates
        homg_coordinates= np.hstack((point_cloud[:,j], 1))
        # Remove distorsion
        pointCloud_corrected[:,j] = np.matmul(M_p[:,:,j], homg_coordinates) 
        
    return pointCloud_corrected[:3,:]


def getSorted_PointCloud(point_cloud, angle_start_velo):
    """
    :param velodyne point cloud, starting scan angle of velodyne
    :returns: velodyne point cloud sorted in an ascending order from the starting horizontal 
              angle at timestamp_start to the end of the scan at timestart_end       
    """
    alpha = np.zeros(point_cloud.shape[0])
    #--- compute alpha angle i.e. azimuthal angle in the x, y plane ---#
    #----------- Sorted by ascending azimuthal angle ------------------#
    for j in range(point_cloud.shape[0]):
        hyp = np.sqrt(point_cloud[j,0]**2+point_cloud[j,1]**2)
        alpha[j] = np.arcsin(point_cloud[j,1]/hyp)
        cos = point_cloud[j,0]/hyp
        if cos < 0:
            alpha[j] = alpha[j] + np.pi
        if alpha[j] < 0:
            alpha[j] = 2*np.pi + alpha[j]
    # Ascending azimuthal angle in clockwise direction (liDAR scan direction)
    indices_sorted = np.argsort(2*np.pi-alpha) # in the clockwise direction
    alpha_sorted = np.sort(2*np.pi-alpha)
    # Find point cloud index where the velodyne scan starts #
    start_point_x = int(min(np.argwhere(alpha_sorted >= angle_start_velo))) 
    indices_1 = indices_sorted[start_point_x:]
    indices_2 = indices_sorted[0:start_point_x-1]
    # Array of indices in the LiDAR scan order (velodyne start angle -> end angle)
    indices_orderOfScan = np.append(indices_1 ,indices_2)
    # Point cloud index associated to where camera2 triggers
    index_CameraTrigger = len(indices_sorted[start_point_x:])
    
    return point_cloud[indices_orderOfScan,:], index_CameraTrigger
        
## --------------------------------------------------------------------------##
##---------------------------------------------------------------------------##


##--------------------MAIN PROGRAM-------------------------------------------##
##---------------------------------------------------------------------------##
dirname = os.path.dirname(os.path.abspath('Task4'))
data_path = os.path.join(dirname,'data', 'demo.p')
data = load_data(data_path)

#----------Timestamps Paths--------------------------------------#
time_start = 'data/problem_4/velodyne_points/timestamps_start.txt'
time_camera = 'data/problem_4/image_02/timestamps.txt'
time_end = 'data/problem_4/velodyne_points/timestamps_end.txt'

#-!!!!---------- Velodyne Bin number and image number -----------!!!!-#
binAndImage_file = 37 # Change HERE 
#-!!!!-----------------------------------------------------------!!!!-#

file_index = str(binAndImage_file)
str_0 = (10-len(file_index))*"0"
data_path = os.path.join('data/problem_4/oxts/data/', str_0 + file_index+ '.txt')
imgloc = os.path.join('data/problem_4/image_02/data/', str_0 + file_index+ '.png')

# Get timestamps in seconds
lidar_start = data_utils.compute_timestamps(time_start, binAndImage_file )
trigger_camera = data_utils.compute_timestamps(time_camera, binAndImage_file )
lidar_end = data_utils.compute_timestamps(time_end,binAndImage_file )

# Get GPS velocity and angular velocity
vel = data_utils.load_oxts_velocity(data_path)
ang = data_utils.load_oxts_angular_rate(data_path)

#Compute translation using the Car angular velocity and velocity
pos_xy_cam = -vel[0:2]*(trigger_camera-lidar_start)

#Compute car rotation between camera trigger and lidar_start
theta = -ang[2]*(trigger_camera-lidar_start)

## Velodyne angle correction between camera trigger and start ##
w_velodyne = 2*np.pi/(lidar_end-lidar_start)# velodyne angular velocity  
ang_no_corr = w_velodyne*(trigger_camera-lidar_start)
angle_start_velo = ang_no_corr + theta
        
### Extract Velodyne points ###
file_index = str(binAndImage_file)
str_0 = (10-len(file_index))*"0"
data_velodyne = os.path.join('data/problem_4/velodyne_points/data/', str_0 + file_index+ '.bin')
point_cloud = data_utils.load_from_bin(data_velodyne)
velodyne = data_utils.load_from_bin(data_velodyne)

#----------- Sort the velodyne points -----------------------------------------------#
point_cloud, index_CameraTrigger = getSorted_PointCloud(point_cloud, angle_start_velo)

#-----------------Transform Lidar points to IMU coordinates--------------------------#
R, T = data_utils.calib_imu2velo('data/problem_4/calib_imu_to_velo.txt')
Trans_matrix = np.hstack((R,T))
Trans_matrix = np.vstack((Trans_matrix,np.array([0, 0, 0, 1])))
Trans_matrix_inv = np.linalg.inv(Trans_matrix)#Inverse Transformation matrix
homogenous_points = np.hstack((point_cloud, np.ones((point_cloud.shape[0],1))))
point_cloud = Trans_matrix_inv@np.transpose(homogenous_points)

#--------Compute rectification matrices for each point by linear interpolation-------#
delta_t = (lidar_end-lidar_start)/point_cloud.shape[1] # Delta time between each point in the scan
#--------Corrected point Cloud i.e. distorsion removed-------------------------------#
pointCloud_corrected = getCorrected_pointCloud(point_cloud[:3,:], delta_t, vel, ang, index_CameraTrigger)

#Apply Transformation imu2velo
homogenous_points = np.vstack((pointCloud_corrected,np.ones((1,pointCloud_corrected.shape[1]))))
pointCloud_corrected = Trans_matrix@homogenous_points
pointCloud_corrected = np.transpose(pointCloud_corrected[:3,:])
#----------------------------------------------------------------------------#

#------------Print Corrected Point Cloud on Camera 2 image---------------------------------------#
img = cv2.imread(imgloc)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#----------- non corrected image-----------------#
img_proj_no_corr = printCloud2image(velodyne, img)
cv2.imshow('Direct projection WITHOUT removing motion distorsion',img_proj_no_corr)
#--------------corrected image-------------------#
img_proj = printCloud2image(pointCloud_corrected, img)
cv2.imshow('Projection after MOTION DISTORSION REMOVED',img_proj)

cv2.waitKey(0)
cv2.destroyAllWindows()