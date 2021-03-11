import numpy as np
import os
from load_data import load_data
import cv2
import vispy
visual = __import__ ('3dvis')


#==================================#
# Rotation Matrices 
#==================================#
def z_rotation(theta):
    """
    Rotation about the -z-axis. 
    (y in cam0 coordinates)
    """
    c = np.cos(theta)
    s = np.sin(theta) 

    Rot = np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])
                     
    return Rot

def y_rotation(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    Rot= np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])
    return Rot

#==================================#
# Rigid Body Transformation and Projection tools
#==================================#

def rectification_Cam0toCam2(points):
    points.shape = (3,-1) # vector form
    homogenous_points = np.vstack((points, np.ones((1,points.shape[1]))))
    pointsCam2 = np.matmul(data['P_rect_20'],homogenous_points)/homogenous_points[2,:]
    return pointsCam2[:2, :]

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

def box_corner_coordinates(objects):
      
    for i in range(len(objects)):  
        # extract dimensions of the box
        height = objects[i][8]
        width = objects[i][9]
        length = objects[i][10]
        
        # extract location in camera0 coordinates and project into Velodyne frame
        CenterBox_lidar = coordCam0toVelo(np.array((objects[i][11], \
                                          objects[i][12],objects[i][13])))
    
        y = [length/2, length/2, -length/2, -length/2, length/2, length/2, -length/2, -length/2]
        x = [-width/2, width/2, width/2, -width/2,-width/2, width/2, width/2, -width/2]
        z = [height,height, height, height, 0, 0, 0, 0 ]
        # Dimensions of box_dim: 3 x 8 i.e. rows (x,y,z) and columns are the corners
        box_dim = np.vstack([x,y,z])

        #rotation of axe 
        Rot = z_rotation(objects[i][14])
        box_dim = Rot @ box_dim 

        # Center the box around the object location in the Velodyne frame
        box_dim += CenterBox_lidar
        box_dim = box_dim.T # coordinates as columns and corner numbers as rows
      
        if i == 0: # first object
            corners = box_dim 
        else: # stack up object boxes coordinates in one array
            corners = np.dstack((box_dim, corners)) # Shape is 8 x 3 x N
    return np.transpose(corners, (2,0,1)) # correct format for drawing N x 8 x 3

def box_in_cam2(objects):

    img = data['image_2']
    for i in range(len(objects)):  
        # extract dimensions of the box
        height = objects[i][8]
        width = objects[i][9]
        length = objects[i][10]

        CenterboxCam0 = np.array((objects[i][11], objects[i][12],objects[i][13]))
        # Corners location 3D in cam0 frame
        x = [length/2, length/2, -length/2, -length/2, length/2, length/2, -length/2, -length/2]
        y = [-height, -height, -height, -height, 0, 0, 0, 0]
        z = [width/2, -width/2, -width/2, width/2, width/2, -width/2, -width/2, width/2]
        
        # Dimensions of box_dim: 3 x 8 i.e. rows (x,y,z) and columns are the corners
        box_dim = np.vstack([x,y,z])

        #rotation around y
        Rot = y_rotation(objects[i][14])
        box_dim = Rot @ box_dim

        CenterboxCam0.shape = (3, -1)
        box_dim+= CenterboxCam0

        #project to cam2, in pixel coordinates 2 x 8
        pixel_coord = rectification_Cam0toCam2(box_dim)

        if i == 0: # first object
            boxPixels = pixel_coord
        else: # stack up object boxes coordinates in one array
            boxPixels = np.dstack((pixel_coord, boxPixels)) # Shape is 8 x 2 x N
            
    return np.transpose(boxPixels, (2,0,1)) # correct format for drawing N x 8 x 2


def get_cloud_pixel_coordinates(xyz_velodyne, sem_label, P, T):
    """
    #filter points with negative x
    indexes = np.argwhere(xyz_velodyne[:, 0]>=0).flatten()
    velodyne_fltrd = np.zeros((len(indexes), 3))
    sem_label_fltrd = np.zeros(len(indexes))
    for i in range(len(indexes)):
        velodyne_fltrd[i] = xyz_velodyne[indexes[i],:]
        sem_label_fltrd[i] = sem_label[indexes[i]]
    """   
    #filter points with negative x
    velodyne_fltrd = []
    sem_label_fltrd = []
    for i in range(xyz_velodyne.shape[0]):
        if xyz_velodyne[i, 0] >= 0:
            velodyne_fltrd.append(xyz_velodyne[i, :])
            sem_label_fltrd.append(sem_label[i])
    velodyne_fltrd = np.array(velodyne_fltrd)
    sem_label_fltrd = np.array(sem_label_fltrd)
    
    #Projection of point cloud in image 2 coordinates
    a = np.ones((velodyne_fltrd.shape[0],1))
    velodyne_fltrd = np.hstack((velodyne_fltrd, a))
    velodyne_fltrd = np.transpose(velodyne_fltrd)

    extrin_calib = np.matmul(T,velodyne_fltrd)
    proj_cloud = np.matmul(P,extrin_calib)/extrin_calib[2,:] #normalization by Zc

    u,v,k = proj_cloud   #k is an array of ones

    u = u.astype(np.int32)
    v = v.astype(np.int32)    
    return {'velodyne_fltrd':velodyne_fltrd, 'sem_label_fltrd':sem_label_fltrd, 'u': u, 'v':v}

#==================================#
# Drawing functions
#==================================#

def draw_points_cloud2image(img, u, v, sem_label_fltrd, velodyne_fltrd, color_map):
    #Draw color point cloud on image
    color=np.zeros(velodyne_fltrd.shape[1])
    for i in range(velodyne_fltrd.shape[1]):
        label=sem_label_fltrd[i][0]
        color = color_map.get(label)
        # Draw a circle of corresponding color 
        cv2.circle(img,(u[i],v[i]), 1, color, -1)  
    return img

def draw_box_image(img, pixel_coord):
    """"
    param: image, pixel coordinates N x 8 x 2
    N: number of boxes
    8: corners
    2: u, v
    """
    # Order of drawing (connect points as in 3dVision update boxes)
    indice = np.asarray([[0,1],[0,3],[0,4],
                [2,1],[2,3],[2,6],
                [5,1],[5,4],[5,6],
                [7,3],[7,4],[7,6]])
    # color of boxes
    green = (0,255,0)
    # iterate over number of boxes
    for i in range(pixel_coord.shape[0]):
        pixels= (pixel_coord[i,:,:]).astype(np.int32) # get the 8 pixels location 
        for i in range(indice.shape[0]): #connect the corners
            img = cv2.line(img, (pixels[0,indice[i][0]], pixels[1,indice[i][0]]), (pixels[0,indice[i][1]], pixels[1,indice[i][1]]), green, 2)
    return img


if __name__ == '__main__':
    
    dirname = os.path.dirname(os.path.abspath('Task2'))
    data_path = os.path.join(dirname,'data', 'demo.p')
    data = load_data(data_path) # Change to data.p for your final submission 

    #======= TASK 2.1 ======================#
    PixelCoord = get_cloud_pixel_coordinates(data['velodyne'][:,0:3], data['sem_label'], data['P_rect_20'], \
                                       data['T_cam2_velo'])
    velodyne_fltrd = PixelCoord['velodyne_fltrd']
    sem_label_fltrd = PixelCoord['sem_label_fltrd']
    u = PixelCoord['u']
    v = PixelCoord['v']
    #Draw color point cloud on image    
    image2 = data['image_2'].astype(np.uint8)
    img = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    img = draw_points_cloud2image(img, PixelCoord['u'], PixelCoord['v'], \
                                  PixelCoord['sem_label_fltrd'], PixelCoord['velodyne_fltrd'], data['color_map'])
    cv2.imshow('image2',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #======================================#

    #======= TASK 2.2 ======================#
    boxPixels = box_in_cam2(data['objects'])
    img = draw_box_image(img, boxPixels)
    cv2.imshow('image2',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #======================================#

    #======= TASK 2.3 ======================#
    v = visual.Visualizer() 
    # Get corner coordinates in Velodyne frame
    corners = box_corner_coordinates(data['objects']) 
    v.update_boxes(corners[:,:,:]) # Draw the boxes in 3D
    v.update(data['velodyne'][:,:3],data['sem_label'], data['color_map']) # Point cloud
    vispy.app.run()
    #======================================#