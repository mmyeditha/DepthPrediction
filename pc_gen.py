import json, sys, csv, cv2, os, shutil, argparse, re
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from numpy.linalg import inv
from multiprocessing import Pool
from scipy.io import loadmat
sys.path.insert(1, 'tensorflow/')
from predict import predict
from gen_data import remap, write_ply_file

def gen_point_cloud_robust(depth, image, intrinsics, extrinsics):
    """
    depth: depth data as an M x N numpy array
    image: the original rgb image (do I really need this?)
    intrinsics: 9-element numpy array (will be 3x3)
    extrinsics: 16-element numpy array (will be 4x4)
    """
    rot_x = np.reshape(np.array([1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1]), (4,4))
    intrinsics = np.reshape(intrinsics, (3,3))
    extrinsics = np.reshape(extrinsics, (4,4))
    # numpy is faster and takes up a lot less space than normal lists
    ptCloud = np.zeros(depth.shape[0]*depth.shape[1])
    height, width = depth.shape
    height_rgb, width_rgb, _ = cv2.imread(image).shape
    # Creating a threshold. Pixels with brightness above the 97th percentile
    # get cancelled. This is because SUNRGBD data has random bright spots that
    # mess with the normalization of depth values
    perc = np.percentile(depth, 97)
    ii, jj = np.meshgrid(np.arange(0,width,1), np.arange(0,height,1))
    # This step is important for FCRN images where the depth image isn't the same
    # resolution as the source RGB image
    iRemapped = (ii/width)*width_rgb
    jRemapped = (jj/height)*height_rgb
    vecs = np.stack((iRemapped,jRemapped,np.ones((530,730))), axis=2)
    s = vecs.reshape((530*730,3))
    normd = np.linalg.inv(intrinsics)@s.T
    newnorm = normd/(np.linalg.norm(normd,axis=0))
    projected = newnorm*depth.reshape((530*730))
    pcd = np.vstack((projected, np.ones((530*730))))
    return rot_x@(extrinsics@pcd)



j = remap('00005/rgb_00005.jpg', '00005/depth_00005.png')
pcd = gen_point_cloud_robust(j, '00005/rgb_00005.jpg', np.array([529.5000,0.000000,365.000000,0.000000,529.500000,265.000000,0.000000,0.000000,1.000000]), np.array([0.993447,0.114219,0.004112,0.000000,-0.114219,0.990867,0.071675,0.000000,0.004112,-0.071675,0.997420,0.000000,0,0,0,1]))
write_ply_file(pcd.T, 'matrixmath')