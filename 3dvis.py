# Deep Learning for Autonomous Driving
# Material for Problem 2 of Project 1
# For further questions contact Ozan Unal, ozan.unal@vision.ee.ethz.ch

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
import os
from load_data import load_data
import cv2

class Visualizer():
    def __init__(self):
        self.canvas = SceneCanvas(keys='interactive', show=True)
        self.grid = self.canvas.central_widget.add_grid()
        self.view = vispy.scene.widgets.ViewBox(border_color='white',
                        parent=self.canvas.scene)
        self.grid.add_widget(self.view, 0, 0)

        # Point Cloud Visualizer
        self.sem_vis = visuals.Markers()
        self.view.camera = vispy.scene.cameras.TurntableCamera(up='z', azimuth=90)
        self.view.add(self.sem_vis)
        visuals.XYZAxis(parent=self.view.scene)
        
        # Object Detection Visualizer
        self.obj_vis = visuals.Line()
        self.view.add(self.obj_vis)
        self.connect = np.asarray([[0,1],[0,3],[0,4],
                                   [2,1],[2,3],[2,6],
                                   [5,1],[5,4],[5,6],
                                   [7,3],[7,4],[7,6]])

    def update(self, points):
        '''
        :param points: point cloud data
                        shape (N, 3)          
        Task 2: Change this function such that each point
        is colored depending on its semantic label
        '''
        self.sem_vis.set_data(points, size=3)
    
    def update_boxes(self, corners):
        '''
        :param corners: corners of the bounding boxes
                        shape (N, 8, 3) for N boxes
        (8, 3) array of vertices for the 3D box in
        following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
        If you plan to use a different order, you can
        change self.connect accordinly.
        '''
        for i in range(corners.shape[0]):
            connect = np.concatenate((connect, self.connect+8*i), axis=0) \
                      if i>0 else self.connect
        self.obj_vis.set_data(corners.reshape(-1,3),
                              connect=connect,
                              width=2,
                              color=[0,1,0,1])

def z_rotation(theta):
    """
    Rotation about the z-axis. (y in cam0 coordinates)
    """
    c = np.cos(theta)
    s = np.sin(theta)
    """
    Rot= np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])
    Rot = coordCam0toVelo(Rot)
    """              
    Rot = np.array([[c, s, 0],
                     [-s, c, 0],
                     [0, 0, 1]])
                     
    return Rot

def coordCam0toVelo(points):
    """
    :param vector of points (x,y,z) in camera 0 coordinates
    return: vector of points (x,y,z) in the velodyne frame
    """
    points.shape = (3,-1) # vector form
    # Inverse transformation in homogenous coordinates
    T_inv = np.linalg.inv(data['T_cam0_velo'])
    homogenous_points = np.vstack((points, np.ones((1,points.shape[1]))))
    pointsLidar = T_inv @ homogenous_points
    return pointsLidar[:3,:]
    
def box_corner_coordinates(data):
      
    for i in range(len(data)):  
        # extract dimensions of the box
        height = data[i][8]
        width = data[i][9]
        length = data[i][10]
        
        # extract location in camera0 coordinates and project into Velodyne frame
        CenterBox_lidar = coordCam0toVelo(np.array((data[i][11], \
                                          data[i][12],data[i][13])))

        # Corners location in 3D in velodyne coordinates
        x = [length/2, length/2, -length/2, -length/2, length/2, length/2, -length/2, -length/2]
        y = [-width/2, width/2, width/2, -width/2,-width/2, width/2, width/2, -width/2]
        z = [height,height, height, height, 0, 0, 0, 0 ]
        # Dimensions of box_dim: 3 x 8 i.e. rows (x,y,z) and columns are the corners
        box_dim = np.vstack([x,y,z])

        #rotation of axe 
        Rot = z_rotation(data[i][14])
        #box_dim = Rot @ box_dim #rotation GIVES WRONG OUTPUT WHY?

        # Center the box around the object location in the Velodyne frame
        box_dim += CenterBox_lidar
        box_dim = box_dim.T # coordinates as columns and corner numbers as rows
      
        if i == 0: # first object
            corners = box_dim 
        else: # stack up object boxes coordinates in one array
            corners = np.dstack((box_dim, corners)) # Shape is 8 x 3 x N
    
    return np.transpose(corners, (2,0,1)) # correct format for drawing N x 8 x 3


if __name__ == '__main__':
    
    data = load_data('data/demo.p') # Change to data.p for your final submission 
    visualizer = Visualizer() 
    # Get corner coordinates in Velodyne frame
    corners = box_corner_coordinates(data['objects']) 
    visualizer.update_boxes(corners[:,:,:]) # Draw the boxes in 3D
    visualizer.update(data['velodyne'][:,:3]) # Point cloud
    '''
    Task 2: Compute all bounding box corners from given
    annotations. You can visualize the bounding boxes using
    visualizer.update_boxes(corners)
    '''
    vispy.app.run()




