import numpy as np
import math
from matplotlib import pyplot as plt
import os
from load_data import load_data
import cv2 
import data_utils as du


dirname = os.path.dirname(os.path.abspath('Ex1'))
data_path = os.path.join(dirname,'data', 'demo.p')
data = load_data(data_path)

# extract data
#point cloud
xyz_velodyne = data['velodyne'][:,0:3] #data from 0 to 3-1

#projection matrices
P = data['P_rect_20']
T = data['T_cam2_velo']
#image
image2 = data['image_2']

epsilon = np.zeros(xyz_velodyne.shape[0])
laser_id = np.zeros(xyz_velodyne.shape[0])
for i in range(xyz_velodyne.shape[0]):
    pythagore = math.sqrt(xyz_velodyne[i, 0]**2 + xyz_velodyne[i, 1]**2)
    z = xyz_velodyne[i, 2]
    epsilon[i] = np.arctan(z/pythagore)*360/(2*math.pi)
    
FOV = max(epsilon)-min(epsilon)
resolution = FOV/64
laser_angle = np.zeros(64)
for i in range(64):
    laser_angle[i] = min(epsilon)+i*resolution
err = np.zeros(64)
for i in range(xyz_velodyne.shape[0]):
    for j in range(64):
        err[j] = abs(epsilon[i] - laser_angle[j])
    laser_id[i] = np.argmin(err)


color_map = {1: [0, 0, 255], 2:[255,0,0], 3:[0,255,0], 4:[30, 30, 255]}#division par 4 avec reste


#filter points with negative x
indexes = np.argwhere(xyz_velodyne[:, 0]>=0).flatten()
velodyne_fltrd = np.zeros((len(indexes), 3))
laser_id_fltrd = np.zeros(len(indexes))
for i in range(len(indexes)):
    velodyne_fltrd[i] = xyz_velodyne[indexes[i],:]
    laser_id_fltrd[i] = laser_id[indexes[i]]

velodyne_fltrd = np.array(velodyne_fltrd)
laser_id_fltrd = np.array(laser_id_fltrd)

#Projection of point cloud in image 2 coordinates
a = np.ones((velodyne_fltrd.shape[0],1))
velodyne_fltrd = np.hstack((velodyne_fltrd, a))
velodyne_fltrd = np.transpose(velodyne_fltrd)

extrin_calib = np.matmul(T,velodyne_fltrd)
proj_cloud = np.matmul(P,extrin_calib)/extrin_calib[2,:] #normalization by Zc

u,v,k = proj_cloud   #k is an array of ones

u = u.astype(np.int32)
v = v.astype(np.int32)

#Draw laser ID color of the point cloud on image
img = image2.astype(np.uint8)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
color=np.zeros(velodyne_fltrd.shape[1])
for i in range(velodyne_fltrd.shape[1]):
    label=laser_id_fltrd[i]
    #label = label%4
    #color = color_map.get(label)
    color[i] = du.line_color(label)
    # Draw a circle of corresponding color 
    #cv2.circle(img,(u[i],v[i]), 1, color, -1)
img = du.print_projection_plt(proj_cloud, color,img)
cv2.imshow('image2',img)
cv2.waitKey(0)
cv2.destroyAllWindows()