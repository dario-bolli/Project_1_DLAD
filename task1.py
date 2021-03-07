
import numpy as np
import os
from load_data import load_data
from PIL import Image

data = load_data('data/demo.p') #load data
# extract data
x_velodyne =  data['velodyne'][:,0]
y_velodyne = data['velodyne'][:,1]
r_velodyne = data['velodyne'][:,3]
res = 0.2 # resolution of 0.2m

# Transpose in pixel coordinates (0,0) top-left corner
bev_x = (-x_velodyne/res).astype(np.int32) #bins of 0.2 m resolution
bev_y = (-y_velodyne/res).astype(np.int32)
# translate w.r.t to (0,0) in image coordinates (no negative indices)
bev_x -= np.amin(bev_x) 
bev_y -= np.amin(bev_y)

# Create image array. Dimensions correspond to maximum (x,y) of velodyne cloud points
# in pixel coordinates
im = np.zeros([np.amax(bev_x)+1, np.amax(bev_y)+1], dtype = np.uint8) 

# Compute pixel values based on reflectance values
pixel_intens = (255*r_velodyne).astype(np.uint8) 
# assign pixel value to bins, based on the highest reflectance
for i in range(1, pixel_intens.size):
    if pixel_intens[i] > im[bev_x[i], bev_y[i]]:
        im[bev_x[i], bev_y[i]] = pixel_intens[i]
    
# Convert from numpy array to a PIL image
im = Image.fromarray(im)
im.show()