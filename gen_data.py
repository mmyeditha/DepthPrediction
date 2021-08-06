import json
import csv
import cv2
import os
import shutil
import argparse
import re
import math
import random
import numpy as np
import sys
sys.path.insert(1, 'tensorflow/')
from sklearn.decomposition import PCA
from scipy.io import loadmat
from multiprocessing import Pool
from predict import predict
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from numpy.linalg import inv

OPEN3D = True
try:
    import open3d as o3d
except:
    print("Open3D not installed properly, plane and point cloud\
            functionality will not work!")
    OPEN3D = False

DATA_PATH = "votenet/sunrgbd/sunrgbd_trainval"


def reorganize_data():
    """
    FOR THIS FUNCTION TO WORK, place the OFFICIAL_SUNRGBD data inside 
    of the sunrgbd folder along with the metadata matrices.
    """
    os.chdir('votenet/sunrgbd')
    data = loadmat(
        'OFFICIAL_SUNRGBD/SUNRGBDMeta2DBB_v2.mat')['SUNRGBDMeta2DBB']
    if 'sunrgbd_plane' in os.listdir():
        shutil.rmtree('sunrgbd_plane')
    os.mkdir('sunrgbd_plane')
    for i in tqdm(range(0, 10335), desc="Reorganizing Data"):
        os.mkdir(f'sunrgbd_plane/{i+1:05}')
        datapath = os.path.join('OFFICIAL_SUNRGBD', data[0][i][0][0])
        rgb_img = os.path.join(
            datapath, 'image', os.listdir(f'{datapath}/image')[0])
        depth_img = os.path.join(
            datapath, 'depth_bfx', os.listdir(f'{datapath}/depth_bfx')[0])
        extrinsics = os.path.join(
            datapath, 'extrinsics', os.listdir(f'{datapath}/extrinsics')[0])
        intrinsic = os.path.join(datapath, 'intrinsics.txt')
        seg = os.path.join(datapath, 'seg.mat')
        # Rename and move the files
        shutil.copyfile(rgb_img, f'sunrgbd_plane/{i+1:05}/rgb_{i+1:05}.jpg')
        shutil.copyfile(
            depth_img, f'sunrgbd_plane/{i+1:05}/depth_{i+1:05}.png')
        shutil.copyfile(
            extrinsics, f'sunrgbd_plane/{i+1:05}/extrinsic_{i+1:05}.txt')
        shutil.copyfile(
            intrinsic, f'sunrgbd_plane/{i+1:05}/intrinsics_{i+1:05}.txt')
        shutil.copyfile(seg, f'sunrgbd_plane/{i+1:05}/seg_{i+1:05}.mat')


def remap(rgb_img, truth_heatmap):
    """
    Takes the ground truth depthmaps and remaps them to match with the
    depth values of FCRN's prediction
    """
    heatmap = predict('tensorflow/models/NYU_FCRN.ckpt',
                      Image.open(rgb_img))[0]
    min_heatmap = np.min(heatmap)
    range_heatmap = np.max(heatmap)-min_heatmap
    gray_img = cv2.cvtColor(cv2.imread(truth_heatmap), cv2.COLOR_BGR2GRAY)
    norm_image = np.zeros(np.shape(gray_img))
    min_gray = np.min(gray_img)
    range_gray = np.max(gray_img)-min_gray
    print(np.shape(gray_img))
    with tqdm(total=gray_img.shape[0]*gray_img.shape[1], desc="mapping heatmap") as pbar:
        for i in range(np.shape(gray_img)[0]):
            for j in range(np.shape(gray_img)[1]):
                norm_image[i][j] = (
                    (gray_img[i][j]-min_gray)/range_gray)*range_heatmap+min_heatmap
                pbar.update(1)
    return norm_image


def gen_ply_file(image_id):
    data_path = 'votenet/sunrgbd/sunrgbd_plane'
    rgb_img = os.path.join(
        data_path, f'{image_id+1:05}/rgb_{image_id+1:05}.jpg')
    depth_img = os.path.join(
        data_path, f'{image_id+1:05}/depth_{image_id+1:05}.png')
    r = remap(rgb_img, depth_img)
    intrinsic_list = re.split(' |\n', open(os.path.join(
        data_path, f'{image_id+1:05}/intrinsics_{image_id+1:05}.txt')).read())[0:9]
    extrinsic_list = re.split(' |\n', open(os.path.join(
        data_path, f'{image_id+1:05}/extrinsic_{image_id+1:05}.txt')).read())[0:12]
    extrinsic_list.extend(['0', '0', '0', '1'])
    intrinsic = [float(x) for x in intrinsic_list]
    extrinsic = [float(x) for x in extrinsic_list]
    print(intrinsic)
    print(extrinsic)
    pc = gen_point_cloud(r, rgb_img, np.asarray(
        intrinsic), np.asarray(extrinsic))
    write_ply_file(pc, os.path.join(
        data_path, f'{image_id+1:05}/pcd_{image_id+1+1:05}'))


def gen_fcrn_ply(image_id):
    data_path = f'votenet/sunrgbd/sunrgbd_plane/{image_id+1:05}'
    rgb_img = f'{data_path}/rgb_{image_id+1:05}.jpg'
    heatmap = predict('tensorflow/models/NYU_FCRN.ckpt', Image.open(rgb_img))[0]
    heatmap = np.reshape(heatmap, (128,160))
    seg_new = seg_fcrn(f'{data_path}/seg_{image_id+1:05}')
    intrinsic_list = re.split(' |\n', open(os.path.join(
        data_path, f'intrinsics_{image_id+1:05}.txt')).read())[0:9]
    extrinsic_list = re.split(' |\n', open(os.path.join(
        data_path, f'extrinsic_{image_id+1:05}.txt')).read())[0:12]
    extrinsic_list.extend(['0', '0', '0', '1'])
    intrinsic = [float(x) for x in intrinsic_list]
    extrinsic = [float(x) for x in extrinsic_list]
    print(rgb_img)
    pc = gen_point_cloud(heatmap, rgb_img, np.asarray(
        intrinsic), np.asarray(extrinsic))
    write_ply_file(pc, os.path.join(
        data_path, f'fcrn_{image_id+1+1:05}'))


def predict_clean(image_path):
    return predict('models/NYU_FCRN.ckpt', Image.open(image_path))[0]


def segment_and_remove(pcd):
    plane, pts = pcd.segment_plane(
        distance_threshold=0.2, ransac_n=5000, num_iterations=1000)
    inlier_cloud = pcd.select_by_index(pts)
    outlier_cloud = pcd.select_by_index(pts, invert=True)
    o3d.visualization.draw_geometries([outlier_cloud])
    o3d.visualization.draw_geometries([inlier_cloud])
    return outlier_cloud, inlier_cloud


def segment_pt_cloud(cloud, show=False, noisy=False):
    # Read the point cloud in open3D
    if not OPEN3D:
        print('Open3D must be installed to segment the point cloud. Aborting')
        return None

    centers = []
    extents = []
    norms = []
    outlier_cloud = cloud.voxel_down_sample(voxel_size=0.05)
    # o3d.io.read_point_cloud(filename)
    if show:
        o3d.visualization.draw_geometries([outlier_cloud])
    coords = np.asarray(outlier_cloud.points)
    num_planes = 0
    plane_list = []
    while (len(coords) >= 300 and num_planes < 4):
        print("Enough Points for RANSAC Pass")
        plane, pts = outlier_cloud.segment_plane(distance_threshold=0.1,
                                            ransac_n=300,
                                            num_iterations=1000)

        inlier_cloud = outlier_cloud.select_by_index(pts)
        norms.append(plane[0:3])
        outlier_cloud = outlier_cloud.select_by_index(pts, invert=True)
        cl, ind = inlier_cloud.remove_statistical_outlier(
            nb_neighbors=75, std_ratio=.375)
        plane_list.append(cl)
        if show:
            o3d.visualization.draw_geometries([outlier_cloud])
            o3d.visualization.draw_geometries([cl])
        if noisy:
            o3d.io.write_point_cloud(
                f'outlier_{num_planes}.ply', outlier_cloud)
            o3d.io.write_point_cloud(f'inlier_{num_planes}.ply', cl)
        coords = np.asarray(outlier_cloud.points)
        min_bound = cl.get_min_bound()
        max_bound = cl.get_max_bound()
        extents.append([max_bound[0]-min_bound[0], max_bound[1] -
                       min_bound[1], max_bound[2]-min_bound[2]])
        centers.append([(max_bound[0]+min_bound[0])/2, (max_bound[1] +
                       min_bound[1])/2, (max_bound[2]+min_bound[2])/2])
        print("RANSAC Pass Complete")
        num_planes += 1

    print("All planes detected")
    if show:
        o3d.visualization.draw_geometries([outlier_cloud])

    return plane_list


def write_label_file(centers, extents, norms, image_id):
    if not 'label_gen' in os.listdir(DATA_PATH):
        os.mkdir(os.path.join(DATA_PATH, 'label_gen'))
    with open(f'{DATA_PATH}/label_gen/{image_id}.txt', 'w') as f:
        for i, _ in enumerate(centers):
                # pylint isn't mad, it's disappointed
                f.write(
                    f'plane, {centers[i][0]}, {centers[i][1]}, \
                            {centers[i][2]}, {extents[i][0]},\
                            {extents[i][1]}, {extents[i][2]},\
                            {norms[i][0]}, {norms[i][1]}, {norms[i][2]}\n')


def crop_to_square(image):
    width, height = image.size
    if width == height:
        return image
    offset = int(abs(height-width)/2)
    if width > height:
        image = image.crop([offset, 0, width-offset, height])
    else:
        image = image.crop([0, offset, width, height-offset])
    return image


def crop_to_FCRN(image):
    width, height = image.size
    offset = (height-height/1.25)/2
    image = image.crop([0, 0+offset, width, height-offset])
    return image


def gen_point_cloud_old(depth, image, intrinsics, extrinsics):
    """
    Old method for point cloud generation. Unbearably slow
    """
    intrinsics = np.reshape(intrinsics, (3, 3))
    print(intrinsics)
    ptCloud = []
    width, height = image.size
    with tqdm(total=np.shape(depth)[0]*np.shape(depth)[1], desc='processing cloud') as pbar:
        perc = np.percentile(depth, 97)
        for i in range(np.shape(depth)[0]):
            for j in range(np.shape(depth)[1]):
                # Remap to 4:3 with blank bars
                iRemapped = (i/np.shape(depth)[0])*height
                jRemapped = (j/np.shape(depth)[1])*width

                ptvec = np.matrix([jRemapped, iRemapped, 1]).T
                norm_val = np.linalg.norm(np.linalg.inv(intrinsics) @ ptvec)
                vec = (np.linalg.inv(intrinsics) @ ptvec)/norm_val
                # get rid of weird overlybright pictures in sungrbd depth data
                if depth[i][j] >= perc:
                    vec *= 0
                else:
                    vec *= depth[i][j]
                vec = vec.tolist()
                ting = np.matrix([-vec[0][0], vec[1][0], -vec[2][0], 1]).T
                # new = rotation * ting
                # print(new)
                matlist = np.array(ting.flatten()).tolist()
                ptCloud.append(matlist)
                pbar.update(1)

    # rotate the point cloud to align with the proper coordinate system
    extrinsics = np.reshape(extrinsics, (4, 4))
    rot_x = np.reshape(
        np.array([1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1]), (4, 4))
    pcd = np.reshape(np.asarray(ptCloud), (len(ptCloud), 4))
    pcd = rot_x@(extrinsics@pcd)
    return pcd


def write_ply_file(pc, name):
    """
    Take in a pointcloud as an array and write it

    pc: 3xN Numpy array containing points
    name: String representing the name of the ply file
    """
    if name[-4:] != '.ply':
        name += '.ply'
    with open(f'{name}', 'w') as f:
        f.write(
            f'ply\nformat ascii 1.0\nelement vertex {len(pc)}\nproperty double x\nproperty double y\nproperty double z\nend_header\n')
        for i, point in enumerate(pc):
            f.write(
                f'{round(point[0]*point[3],15)} {round(point[1]*point[3],15)} {round(point[2]*point[3],15)}\n')


#------------------------------------ new workflow -------------------------------------
def gen_point_cloud(depth, image, intrinsics, extrinsics):
    # intrinsics = np.asmatrix(np.reshape(metadata['intrinsics'], (3,3)).swapaxes(0,1))
    """
    Projects rgbd data into 3D space using camera intrinsics and extrinsics

    depth: depth data as an M x N numpy array
    image: the original rgb image (do I really need this?)
    intrinsics: 9-element numpy array (will be 3x3)
    extrinsics: 16-element numpy array (will be 4x4)
    """
    rot_x = np.reshape(
        np.array([1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1]), (4, 4))
    intrinsics = np.reshape(intrinsics, (3, 3))
    extrinsics = np.reshape(extrinsics, (4, 4))
    # numpy is faster and takes up a lot less space than normal lists
    ptCloud = np.zeros(depth.shape[0]*depth.shape[1])
    height, width = depth.shape
    height_rgb, width_rgb, _ = cv2.imread(image).shape
    # Creating a threshold. Pixels with brightness above the 97th percentile
    # get cancelled. This is because SUNRGBD depth_bfx data has random bright spots that
    # mess with the normalization of depth values
    perc = np.percentile(depth, 97)
    ii, jj = np.meshgrid(np.arange(0, width, 1), np.arange(0, height, 1))
    # This step is important for FCRN images where the depth image isn't the same
    # resolution as the source RGB image
    iRemapped = (ii/width)*width_rgb
    jRemapped = (jj/height)*height_rgb
    vecs = np.stack((iRemapped, jRemapped, np.ones(
        (height, width))), axis=2)
    s = vecs.reshape((height*width, 3))
    normd = np.linalg.inv(intrinsics)@s.T
    newnorm = normd/(np.linalg.norm(normd, axis=0))
    projected = newnorm*depth.reshape((height*width))
    pcd = np.vstack((projected, np.ones((height*width))))
    return (rot_x@(extrinsics@pcd)).T


def extract_objects(mat, pcd_points, fcrn=False, mat_new=None, name_list=None):
    """
    args:
        mat: String referencing the path to the seg.mat file from dataset,
        classifies each pixel as a certain object and defines each class as a number,
        also has a dictionary to tell us what number
        means what object.
        pcd_points: ply file representing a point cloud

    returns a dictionary mapping objects to the pointcloud indices of points that 
    make up that object. Ex. a key will be 'wall_0', and the value will be a list 
    of indices that make up the points that fall within that wall. To visualize,
    open the point cloud with o3d, then run o3d.select_by_points(<point list>) 
    and this should show you all the points that make up that object.
    """
    pcd = o3d.io.read_point_cloud(pcd_points)
    if fcrn:
        seglabel = mat_new
        names = name_list
        surfaces = {}
    else:
        segs = loadmat(mat)
        seglabel = segs['seglabel']
        names = segs['names'][0]
        surfaces = {}
    whitelist = ['bed', 'table', 'sofa', 'chair', 'toilet',
        'desk', 'dresser', 'night_stand', 'bookshelf', 'bathtub']
    walls = [x+1 for x in range(len(names)) if names[x][0] == 'wall']
    ceilings = [x+1 for x in range(len(names)) if names[x][0] == 'ceiling']
    floor = [x+1 for x in range(len(names)) if names[x][0] == 'floor']
    other = [x+1 for x in range(len(names)) if names[x][0] in whitelist]
    for i in range(seglabel.shape[0]):
        for j in range(seglabel.shape[1]):
            if seglabel[i][j] in walls:
                # Sometimes walls aren't defined separately. Maybe look int
                # running RANSAC here to separate out walls definitively
                if f'wall_{seglabel[i][j]}' in surfaces.keys():
                    surfaces[f'wall_{seglabel[i][j]}'].append(i*seglabel.shape[1]+j)
                else:
                    surfaces[f'wall_{seglabel[i][j]}'] = [i*seglabel.shape[1]+j]
            elif seglabel[i][j] in ceilings:
                if f'ceiling_{seglabel[i][j]}' in surfaces.keys():
                    surfaces[f'ceiling_{seglabel[i][j]}'].append(i*seglabel.shape[1]+j)
                else:
                    surfaces[f'ceiling_{seglabel[i][j]}'] = [i*seglabel.shape[1]+j]
            elif seglabel[i][j] in floor:
                if f'floor_{seglabel[i][j]}' in surfaces.keys():
                    surfaces[f'floor_{seglabel[i][j]}'].append(i*seglabel.shape[1]+j)
                else:
                    surfaces[f'floor_{seglabel[i][j]}'] = [i*seglabel.shape[1]+j]
            elif seglabel[i][j] in other:
                if f'other_{seglabel[i][j]}' in surfaces.keys():
                    surfaces[f'other_{seglabel[i][j]}'].append(i*seglabel.shape[1]+j)
                else:
                    surfaces[f'other_{seglabel[i][j]}'] = [i*seglabel.shape[1]+j]
    return surfaces


def get_heading_angle(bbox, centroid, obj_centroid):
    """
    Given the bbox o3d object, returns the heading angle relative to
    the +x direction (right)

    args:
        bbox: bounding box object from open3d

    returns a 2 element array.
    """
    # First, run PCA on the corner pts of the 3DBB
    pts = np.asarray(bbox.get_box_points())
    # vector between centroid and scene centroid
    vec = centroid-obj_centroid
    pca = PCA(n_components=3)
    pca.fit(pts)
    best_dot = 0
    for i, component in enumerate(pca.components_):
        if abs(np.dot(component, vec)) > best_dot:
            # The principle component pointing most towards the center of the scene
            # (dot product of the component and vec is greater) corresponds to the
            # heading angle we want
            best_dot = abs(np.dot(component, vec))
            # Get rid of the +Z component of the vector (should be close to zero anyway)
            heading_vector = component[0:2]
    return heading_vector


def gen_label_line(classname, bbox, pcd, debug=False):
    """
    Returns the line of a label file corresponding to one box

    args:
        classname: String representing the name of the object
        bbox: open3d bbox object
    """
    coords = np.asarray(bbox.get_box_points())
    center = [np.average(coords[:, 0]), np.average(
        coords[:, 1]), np.average(coords[:, 2])]
    # Heading angle calculation done in a separate function :D
    heading = get_heading_angle(bbox, center, np.mean(np.asarray(pcd.points), axis=0))
    centroid, extent, xmin, ymin, xmax, ymax = rotate_points(bbox, heading)
    line = f'{classname} {round(xmin,3)} {round(ymin,3)} {round(xmax,3)} {round(ymax,3)} {round(centroid[0],3)} {round(float(centroid[1]),3)} {round(float(centroid[2]),3)} {round(float(extent[0]),3)} {round(float(extent[1]),3)} {round(float(extent[2]),3)} {round(heading[0],3)} {round(heading[1],3)}\n' 
    if debug:
        print(coords)
    return line


def gen_label(pts, pcd, imageid, debug=False):
    """
    Generates label text for a pointcloud

    args:
        pts: Dictionary given from extract_walls
        pcd: open3d pointcloud object
        imageid: int from 1 to 10335
    """
    f=open(f'votenet/sunrgbd/sunrgbd_plane/label/label_{imageid:05}.txt', 'w')
    for obj in pts.keys():
        obj_3d=pcd.select_by_index(pts[obj])
        # Remove outliers from geometry before generating bounding box
        obj_3d, ind=obj_3d.remove_statistical_outlier(
            nb_neighbors=300, std_ratio=.75)
        if np.asarray(obj_3d.points).size > 100:
            try:
                bbox=obj_3d.get_oriented_bounding_box()
                f.write(gen_label_line(obj[0:obj.index('_')], bbox, pcd, debug))
            except:
                print('Issue getting bbox')


def process_depth():
    """
    One-time script to take all of the depth.mat files from the sunrgbd original data
    and turn them into ply files
    """
    files=os.listdir(f'{DATA_PATH}/depth')
    if not 'depth_ply' in os.listdir(DATA_PATH):
        os.mkdir(f'{DATA_PATH}/depth_ply')

    pool=Pool()

    def write_file(file):
        depth_data=loadmat(f'{DATA_PATH}/depth/{file}')['instance']
        with open(f'{DATA_PATH}/depth_ply/{file[:-3]}ply', 'w') as f:
            f.write(
                f'ply\nformat ascii 1.0\nelement vertex {len(depth_data)}\nproperty double x\nproperty double y\nproperty double z\nend_header\n')
            for i, point in enumerate(depth_data):
                f.write(
                    f'{round(point[0],15)} {round(point[1],15)} {round(point[2],15)}\n')

    for i, _ in enumerate(pool.imap_unordered(write_file, files), 1):
        print(f'{i}/{len(files)} done')

    pool.close()
    pool.join()
    pool.close()


def rotate_points(bbox, head):
    """
    Given eight bounding box points, rotates them to have heading of <1,0>

    When bounding box points are rotated at an angle, l, w, and h extents
    do not translate to a fitted bounding box. So, for the purposes of the
    model (yawn), we have to put box centroid and extents in terms of a
    rotated bbox that has a heading vector in the direction of <1,0> or +x.

    args:
        bbox: The open3d boundingbox class corresponding to a bound box in
        a scene

    returns the center and l, w, h extents of the box with a heading of <1,0>
    """
    angle = math.acos(np.dot(head, np.array([1,0]))/(np.linalg.norm(head)))
    rot = np.array([[math.cos(angle), -math.sin(angle),0],
        [math.sin(angle),math.cos(angle),0],[0,0,1]])
    coords = bbox.get_box_points()
    new_pts = rot@(np.asarray(coords).T)
    max_bound = np.array([new_pts[0,:].max(),new_pts[1,:].max(),new_pts[2,:].max()])
    min_bound = np.array([new_pts[0,:].min(),new_pts[1,:].min(),new_pts[2,:].min()])
    center = (max_bound+min_bound)/2
    xmin = min_bound[0]
    ymin = min_bound[1]
    xmax = max_bound[0]-min_bound[0]
    ymax = max_bound[1]-min_bound[1]
    extent = np.array([(max_bound[0]-min_bound[0])/2,(max_bound[1]-min_bound[1])/2,
        (max_bound[2]-min_bound[2])/2])
    return center, extent, xmin, ymin, xmax, ymax


def seg_fcrn(seg):
    """
    Takes segmentation data for original image and maps it to FCRN size

    args:
        seg: the seg.mat file corresponding to the fcrn image to segment

    returns a 128x160 array of segments for each FCRN pixel and names defining
    what each number mean as an object.
    """
    segs = loadmat(seg)
    seglabel = segs['seglabel']
    names = segs['names'][0]
    height = seglabel.shape[0]
    width = (seglabel.shape[0]*5)//4
    # can i get a woot woot for numpy?
    X, Y = np.meshgrid(np.arange(0, 128, 1), np.arange(0, 160, 1))
    X_remapped = ((X/128)*height).astype('uint32')
    Y_remapped = ((Y/160)*width+(seglabel.shape[1]-width)//2).astype('uint32')
    # x and y are defined kinda weirdly here. The image is 730x530 where
    # 730 is the horizontal axis, I'm just defining 730 as y_remapped beacuse
    # it makes more sense to me to index with seglabel[x,y]
    seg_remapped = seglabel[X_remapped,Y_remapped]
    return seg_remapped.T, names


def gen_label_data(i):
    pcd = o3d.io.read_point_cloud(f'votenet/sunrgbd/sunrgbd_plane/{i+1:05}/fcrn_{i+2:05}.ply')
    seg_remapped, names = seg_fcrn(
                    f'votenet/sunrgbd/sunrgbd_plane/{i+1:05}/seg_{i+1:05}.mat')
    segs = extract_objects('s',
                    f'votenet/sunrgbd/sunrgbd_plane/{i+1:05}/fcrn_{i+2:05}.ply',
                    fcrn=True, mat_new = seg_remapped, name_list = names)
    gen_label(segs, pcd, i)


def check_file_system():
    os.chdir('votenet/sunrgbd/sunrgbd_plane')
    folders = os.listdir()
    missing_folders = []
    incomplete_folders = []
    for i in range(0,10335):
        # Check that all folders exist
        if f'{i+1:05}' not in folders:
            print(f'Error! Folder {i} not found!')
            missing_folders.append(i+1)
        else:
            contents = os.listdir(f'{i+1:05}')
            # Check that core files exist
            if f'rgb_{i+1:05}.jpg' not in contents or f'seg_{i+1:05}.mat' not in contents:
                print("Core dataset files missing")
                missing_folders.append(i+1)    
            # Check that FCRN cloud was made for all folders
            if f'fcrn_{i+2:05}.ply' not in contents:
                print("FCRN Cloud Missing")
                incomplete_folders.append(i+1)
            else:
                with open(f'{i+1:05}/fcrn_{i+2:05}.ply') as f:
                    text = f.readlines()
                    size = len(text)
                    if size != 20487:
                        print('ruh roh. the fcrn file wasnt generated right!')
                        incomplete_folders.append(i+2)
    return missing_folders, incomplete_folders

#---------------------------------Parser Arguments---------------------------------
if __name__ == '__main__':
    # half of these arguments don't even do anything anymore :/
    parser=argparse.ArgumentParser()
    parser.add_argument('--ply', action='store_true',
                        help='Generate ply files from extracted data')
    parser.add_argument('--test', action='store_true',
                        help='writes ply file for every object in scene 00005')
    parser.add_argument('--second_test', action='store_true')
    parser.add_argument('--test_label', action='store_true')
    parser.add_argument('--demo', action='store_true', help='Show functionality on random images')
    parser.add_argument('--gen_data', action='store_true', help='Generate the data my god i really didnt think i would get here my god its happening')
    parser.add_argument('--gen_fcrn_ply', action='store_true')
    parser.add_argument('--fcrn_demo', action='store_true', help='blah')
    parser.add_argument('--gen_label', action='store_true', help='Generate labels for allll the files')
    parser.add_argument('--check_files', action='store_true',
            help="Check that files within sunrgbd_plane all exist and aren't corrupted")
    parser.add_argument('--org_data', action='store_true', help='Reorganize data')
    args=parser.parse_args()
    
    if args.ply:
        data_path='votenet/sunrgbd/sunrgbd_plane'
        pool=Pool()
        for i, _ in enumerate(pool.imap_unordered(gen_ply_file, range(10335)), 1):
            print(f'{i}/10335 done')
        pool.close()
        pool.join()
        pool.close()

    if args.demo:
        indices = random.sample(range(0,10335),15)
        with tqdm(total=len(indices)) as pbar:
            for indX in indices:
                pcd = o3d.io.read_point_cloud(f'votenet/sunrgbd/sunrgbd_plane/{indX:05}/pcd_{indX+1:05}.ply')
                segs = extract_objects(
                        f'votenet/sunrgbd/sunrgbd_plane/{indX:05}/seg_{indX:05}.mat',
                        f'votenet/sunrgbd/sunrgbd_plane/{indX:05}/pcd_{indX+1:05}.ply')
                gen_label(segs, pcd, indX)
                pbar.update(1)

    if args.gen_label:
        pool = Pool()
        for i, _ in enumerate(pool.imap_unordered(gen_label_data, range(0,10335)), 1):
            print(f'{i}/10335 done')

    if args.fcrn_demo:
         indices = random.sample(range(0,10335),10)
         with tqdm(total=len(indices)) as pbar:
             for indX in indices:
                 pcd = o3d.io.read_point_cloud(
                         f'votenet/sunrgbd/sunrgbd_plane/{indX:05}/fcrn_{indX+1:05}.ply')
                 seg_remapped, names = seg_fcrn(
                         f'votenet/sunrgbd/sunrgbd_plane/{indX:05}/seg_{indX:05}.mat')
                 segs = extract_objects('s',
                         f'votenet/sunrgbd/sunrgbd_plane/{indX:05}/fcrn_{indX+1:05}.ply',
                         fcrn=True, mat_new = seg_remapped, name_list = names)
                 gen_label(segs, pcd, indX)
                 pbar.update(1)

    if args.check_early:
        m, inc = check_file_system()
        print('missing folders:', m)
        print('incomplete folders:', inc)
        if len(m) == 0 and len(inc) == 0:
            print('file system looks good')
        else:
            print('we gon fix it')

    if args.org_data:
        reorganize_data()

    if args.gen_fcrn_ply:
        pool = Pool()
        for i in range(52):
            pool = Pool()
            if i*200 == 10200:
                for i, _ in enumerate(pool.imap_unordered(gen_fcrn_ply, range(i*200, 10335)), 1):  
                    print(f'{i+(200*i)}/10335 done')
                pool.close()
                pool.join()
                pool.close()
            else:
                for i, _ in enumerate(pool.imap_unordered(gen_fcrn_ply, range(i*200,i*200+200)),1):
                    print(f'{i}/10335 done')
                pool.close()
                pool.join
                pool.close
