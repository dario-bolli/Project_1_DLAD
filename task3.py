import numpy as np
import math
from matplotlib import pyplot as plt
import os
from load_data import load_data
import cv2 
import data_utils

## ------------------------FUNCTIONS-----------------------------------------##
## --------------------------------------------------------------------------##
def getProjected_pointCloud(xyz_velodyne, laser_id):
    
    #Projection of point cloud in image 2 coordinates
    ones_array = np.ones((xyz_velodyne.shape[0],1))
    xyz_velodyne = np.hstack((xyz_velodyne, ones_array))

    # x,y,z as rows, point indexes as columns 
    xyz_velodyne = np.transpose(xyz_velodyne)
    extrin_calib = np.matmul(T,xyz_velodyne)

    # filter points with negative z in Cam0 coordinates
    indexes = np.argwhere(extrin_calib[2,:]>=0).flatten()
    extrin_calib_fltrd = extrin_calib[:, indexes]
    # get corresponding laser id
    laser_id_fltrd = laser_id[indexes]

    # project to image of camera2
    proj_cloud = np.matmul(P,extrin_calib_fltrd)/extrin_calib_fltrd[2,:] #normalization by Zc

    return proj_cloud, laser_id_fltrd
## --------------------------------------------------------------------------##



##--------------------MAIN PROGRAM-------------------------------------------##
##---------------------------------------------------------------------------##
###extract data
dirname = os.path.dirname(os.path.abspath('Task3'))
data_path = os.path.join(dirname,'data', 'demo.p')
data = load_data(data_path)
#point cloud
xyz_velodyne = data['velodyne'][:,0:3] #data from 0 to 3-1

#projection matrices
P = data['P_rect_20']
T = data['T_cam2_velo']
#image
image2 = data['image_2']

###compute elevation angle
epsilon = np.zeros(xyz_velodyne.shape[0])
laser_id = np.zeros(xyz_velodyne.shape[0])
for i in range(xyz_velodyne.shape[0]):
    pythagore = math.sqrt(xyz_velodyne[i, 0]**2 + xyz_velodyne[i, 1]**2)
    z = xyz_velodyne[i, 2]
    epsilon[i] = np.arctan(z/pythagore)

FOV = max(epsilon)-min(epsilon)
resolution = FOV/64
nb_bins = 65
laser_angle = np.zeros(nb_bins)
for i in range(nb_bins):
    laser_angle[i] = min(epsilon)+i*resolution

for i in range(xyz_velodyne.shape[0]):
    for j in range(nb_bins-1):
        if epsilon[i]>=laser_angle[j] and epsilon[i]<laser_angle[j+1]:
            laser_id[i] = j+1

# get projected filtered point cloud with associated laser id's
proj_cloud, laser_id_fltrd = getProjected_pointCloud(xyz_velodyne, laser_id)

#Draw laser ID color of the point cloud on image
img = image2.astype(np.uint8)

color=np.zeros(proj_cloud.shape[1])
for i in range(proj_cloud.shape[1]):
    label=laser_id_fltrd[i]
    color[i] = data_utils.line_color(label)
color = color + 135
image = data_utils.print_projection_plt(proj_cloud, color, img)

cv2.imwrite("Task_3.png", image) #save to current directory
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()