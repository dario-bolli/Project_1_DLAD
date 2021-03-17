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
    R, T = data_utils.calib_velo2cam('data/problem_4/calib_velo_to_cam.txt')
    Trans_matrix = np.hstack((R,T))
    # Get Extrinsic Transformation matrix from velo to cam
    Trans_matrix = np.vstack((Trans_matrix,np.array([0, 0, 0, 1])))
    velodyne_fltrd = []
    # Filter point cloud for positive x only 
    for i in range(xyz_velodyne.shape[0]):
        if xyz_velodyne[i, 0] >= 0:
            velodyne_fltrd.append(xyz_velodyne[i, :])
    velodyne_fltrd = np.array(velodyne_fltrd)
    # To homogeneous coordinates 
    velodyne_fltrd_inter = np.hstack((velodyne_fltrd, np.ones((velodyne_fltrd.shape[0],1))))
    velodyne_fltrd = np.transpose(velodyne_fltrd_inter)
    # From velo -> cam -> img coordinates
    extrin_calib = np.matmul(Trans_matrix,velodyne_fltrd)
    intrins_calib = data_utils.calib_cam2cam("data/problem_4/calib_cam_to_cam.txt", '02')
    proj_cloud = np.matmul(intrins_calib,extrin_calib)/extrin_calib[2,:] #normalization by Zc

    #compute depth
    depth = np.sqrt(velodyne_fltrd[0,:]**2 + velodyne_fltrd[1,:]**2)
    #r_proj = np.sqrt(velodyne_fltrd[0,:]**2 + velodyne_fltrd[1,:]**2)
    #depth = np.sqrt(r_proj**2 + velodyne_fltrd[3,:]**2)
    color=data_utils.depth_color(depth)
    img = data_utils.print_projection_plt(proj_cloud, color, img)
    return img

def getCorrected_pointCloud(point_cloud, delta_t, vel, ang):
    M_p = np.zeros((4,4,point_cloud.shape[1]))
    pointCloud_corrected = np.zeros((4, point_cloud.shape[1]))
    homogeneous = np.array([0,0,0,1]) # to homogeneous coordinates
    for j in range(point_cloud.shape[1]):
    #Compute translation using angular velocity and velocity
        pos_xy_p = -vel[0:2]*j*delta_t
        #Compute rotation using pos and angular velocity
        teta_p = ang[2]*j*delta_t
        R = np.array([[np.cos(teta_p), -np.sin(teta_p), 0],
                    [np.sin(teta_p), np.cos(teta_p), 0],
                    [0,             0,     1]])
        T = np.array([[pos_xy_p[0]],
                    [pos_xy_p[1]],
                    [0]])
        
        R_T = np.hstack((R,T))
        M_p[:,:,j] = np.vstack((R_T, homogeneous))
        # Rigid Body transformation in homogeneous coordinates
        homg_coordinates= np.hstack((point_cloud[:,j], 1))
        pointCloud_corrected[:,j] = np.matmul(M_p[:,:,j], homg_coordinates) 
        
    return pointCloud_corrected[:3,:]
## --------------------------------------------------------------------------##



##--------------------MAIN PROGRAM-------------------------------------------##
##---------------------------------------------------------------------------##
dirname = os.path.dirname(os.path.abspath('Task4'))
data_path = os.path.join(dirname,'data', 'demo.p')
data = load_data(data_path)

M = np.zeros((4,4))


#-------Velodyne Timestamps--------------------------------------#
time_start = 'data/problem_4/velodyne_points/timestamps_start.txt'
time_camera = 'data/problem_4/velodyne_points/timestamps.txt'
time_end = 'data/problem_4/velodyne_points/timestamps_end.txt'
# Velodyne .bin to extract and project to the corresponding image
num_bin = 311
file_index = str(num_bin)
str_0 = (10-len(file_index))*"0"
data_path = os.path.join('data/problem_4/oxts/data/', str_0 + file_index+ '.txt')

lidar_start = data_utils.compute_timestamps(time_start, num_bin)
trigger_camera = data_utils.compute_timestamps(time_camera, num_bin)
lidar_end = data_utils.compute_timestamps(time_end, num_bin)

# Get GPS velocity and angular velocity
vel = data_utils.load_oxts_velocity(data_path)
ang = data_utils.load_oxts_angular_rate(data_path)

#Compute translation using angular velocity and velocity
pos_xy_cam = -vel[0:2]*(trigger_camera-lidar_start)

#Compute rotation and translation using position and angular velocity
teta = -ang[2]*(trigger_camera-lidar_start)

## Velodyne angle correction between camera trigger and start ##
w_velodyne = 2*np.pi/(lidar_end-lidar_start)#velodyne angular velocity  
ang_no_corr = w_velodyne*(trigger_camera-lidar_start)
angle_start_velo = ang_no_corr + teta
        
### Extract Velodyne points ###
file_index = str(num_bin)
str_0 = (10-len(file_index))*"0"
data_velodyne = os.path.join('data/problem_4/velodyne_points/data/', str_0 + file_index+ '.bin')
point_cloud = data_utils.load_from_bin(data_velodyne)
velodyne = data_utils.load_from_bin(data_velodyne)
###-- Sort the velodyne points by ascending alpha angle (plane xy) --###
alpha = np.zeros(point_cloud.shape[0])

for j in range(point_cloud.shape[0]):
    hyp = np.sqrt(point_cloud[j,0]**2+point_cloud[j,1]**2)
    alpha[j] = np.arcsin(point_cloud[j,1]/hyp)
    cos = point_cloud[j,0]/hyp
    if cos < 0:
        alpha[j] = alpha[j] + np.pi
    if alpha[j] < 0:
        alpha[j] = 2*np.pi + alpha[j]

indices_sorted = np.argsort(2*np.pi-alpha) # in the clockwise direction
alpha_sorted = np.sort(2*np.pi-alpha)

start_point_x = int(min(np.argwhere(alpha_sorted >= angle_start_velo))) # same as in indices_sorted
indices_1 = indices_sorted[start_point_x:]
indices_2 = indices_sorted[0:start_point_x-1]
indices_orderOfScan = np.append(indices_1 ,indices_2)

point_cloud = point_cloud[indices_orderOfScan,:]
for j in range(point_cloud.shape[0]):
    hyp = np.sqrt(point_cloud[j,0]**2+point_cloud[j,1]**2)
    alpha[j] = np.arcsin(point_cloud[j,1]/hyp)
    cos = point_cloud[j,0]/hyp
    if cos < 0:
        alpha[j] = alpha[j] + np.pi
    if alpha[j] < 0:
        alpha[j] = 2*np.pi + alpha[j]
print("alpha sorted", 2*np.pi-alpha)

# # Until here alpha contains all angle of the points, all btw 0 and 2pi (works fine)
# #indices = np.argsort(alpha)
# # Apply angle distorsion offset
# alpha = alpha - angle_start_velo

# # Now alpha can also be negative, real start point of lidar taking into account distorsion
# # is the smallest positive alpha
# start_point = int(min(np.argwhere(alpha >= 0)))

# ## Point cloud velodyne sorted in scan order
# point_cloud_1 = point_cloud[start_point:,:]
# point_cloud_2 = point_cloud[0:start_point-1,:]
# point_cloud = np.vstack((point_cloud_1,point_cloud_2))

## Transform Lidar points to IMU coordinates
R, T = data_utils.calib_imu2velo('data/problem_4/calib_imu_to_velo.txt')
Trans_matrix = np.hstack((R,T))
Trans_matrix = np.vstack((Trans_matrix,np.array([0, 0, 0, 1])))
#Inverse Transformation matrix
Trans_matrix_inv = np.linalg.inv(Trans_matrix)
homogenous_points = np.hstack((point_cloud, np.ones((point_cloud.shape[0],1))))

point_cloud = Trans_matrix_inv@np.transpose(homogenous_points)

## Compute rectification matrices for each point by linear interpolation
# Delta time between each point in the scan
delta_t = (lidar_end-lidar_start)/point_cloud.shape[1]
# Corrected point Cloud i.e. Distorsion taken into account
pointCloud_corrected = getCorrected_pointCloud(point_cloud[:3,:], delta_t, vel, ang)

#Apply Transformation imu2velo
homogenous_points = np.vstack((pointCloud_corrected,np.ones((1,pointCloud_corrected.shape[1]))))

pointCloud_corrected = Trans_matrix@homogenous_points
pointCloud_corrected = np.transpose(pointCloud_corrected[:3,:])
#print(np.shape(pointCloud_corrected))
## Project corrected point cloud on image ##
imgloc = "data/problem_4/image_02/data/0000000311.png"
img = cv2.imread(imgloc)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img_proj = printCloud2image(pointCloud_corrected, img)

#img_proj = cv2.circle(img_proj,int(pointCloud_corrected[113632,0]),int(pointCloud_corrected[113632,1]), 10, (0, 0, 255), -1)
img_proj_no_corr = printCloud2image(velodyne, img)
cv2.imshow('Projection on image',img_proj)
cv2.imshow('Projection non corrected',img_proj_no_corr)
cv2.waitKey(0)
cv2.destroyAllWindows()