import json
import sys
import numpy as np
import csv
import cv2
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from numpy.linalg import inv
sys.path.insert(1,'tensorflow/')
from predict import predict
from multiprocessing import Pool
from scipy.io import loadmat
import open3d as o3d

DATA_PATH = "votenet/sunrgbd/sunrgbd_trainval"

def segment_and_remove(pcd):
    plane, pts = pcd.segment_plane(distance_threshold=0.2, ransac_n=5000, num_iterations=1000)
    inlier_cloud = pcd.select_by_index(pts)
    outlier_cloud = pcd.select_by_index(pts, invert=True)
    o3d.visualization.draw_geometries([outlier_cloud])
    return outlier_cloud

def segment_pt_cloud(cloud, show=False):
    # Read the point cloud in open3D
    centers = []
    extents = []
    norms = []
    outlier_cloud = cloud.voxel_down_sample(voxel_size=0.05)
    #o3d.io.read_point_cloud(filename)
    if show:
        o3d.visualization.draw_geometries([outlier_cloud])
    coords = np.asarray(outlier_cloud.points)
    num_planes = 0
    while (len(coords) >= 3000 and num_planes<4):
        print("Enough Points for RANSAC Pass")
        plane, pts = outlier_cloud.segment_plane(distance_threshold=0.05,
                                            ransac_n=3000,
                                            num_iterations=1000)
        
        inlier_cloud = outlier_cloud.select_by_index(pts)
        norms.append(plane[0:3])
        outlier_cloud = outlier_cloud.select_by_index(pts, invert=True)
        cl, ind = inlier_cloud.remove_statistical_outlier(nb_neighbors=75, std_ratio=.375)
        if show:
            o3d.visualization.draw_geometries([outlier_cloud])
            o3d.visualization.draw_geometries([cl])
        coords = np.asarray(outlier_cloud.points)
        min_bound = cl.get_min_bound()
        max_bound = cl.get_max_bound()
        print(max_bound, min_bound)
        extents.append([max_bound[0]-min_bound[0], max_bound[1]-min_bound[1], max_bound[2]-min_bound[2]])
        centers.append([(max_bound[0]+min_bound[0])/2, (max_bound[1]+min_bound[1])/2, (max_bound[2]+min_bound[2])/2])
        print("RANSAC Pass Complete")
        num_planes += 1
    
    print("All planes detected")
    if show:
        o3d.visualization.draw_geometries([outlier_cloud])
    

    return centers, extents, norms

def gen_coords(imgfile):
    # Rotate image for FCRN to be good
    img = cv2.imread(imgfile)
    #img = np.transpose(img, (1,0,2))
    #thing = np.transpose(img, (1,0,2))
    #ballz = np.transpose(thing, (1,0,2))
    #img = np.transpose(img, (1,0,2))
    #img = np.transpose(img, (1,0,2))
    cv2.imshow('wha',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # new = crop_to_square(rotated_img)
    # new_2 = crop_to_FCRN(new)
    # new_2.show()
    # Use FCRN to get a heatmap
    image = Image.fromarray(img)
    s = predict('tensorflow/models/NYU_FCRN.ckpt', image, silent=True)
    gen_point_cloud(s, js, image)

def write_label_file(centers, extents, norms, image_id):
    if not 'label_gen' in os.listdir(DATA_PATH):
        os.mkdir(os.path.join(DATA_PATH, 'label_gen'))
    with open(f'{DATA_PATH}/label_gen/{image_id}.txt', 'w') as f:
        for i, _ in enumerate(centers):
                # pylint isn't mad, it's disappointed
                f.write(f'plane, {centers[i][0]}, {centers[i][1]}, {centers[i][2]}, {extents[i][0]}, {extents[i][1]}, {extents[i][2]}, {norms[i][0]}, {norms[i][1]}, {norms[i][2]}\n')

def crop_to_square(image):
    width, height = image.size
    if width == height:
        return image
    offset  = int(abs(height-width)/2)
    if width>height:
        image = image.crop([offset,0,width-offset,height])
    else:
        image = image.crop([0,offset,width,height-offset])
    return image

def crop_to_FCRN(image):
    width, height = image.size
    offset = (height-height/1.25)/2
    image = image.crop([0,0+offset, width, height-offset])
    return image

def gen_point_cloud(depth, image, intrinsics):
    # intrinsics = np.asmatrix(np.reshape(metadata['intrinsics'], (3,3)).swapaxes(0,1))
    intrinsics = np.reshape(intrinsics, (3,3))
    ptCloud = []
    width, height = image.size
    with tqdm(total = 128*170, desc = 'processing cloud') as pbar:
        for i in range(0, 128): 
            for j in range(0, 170):
                # Remap to 4:3 with blank bars
                iRemapped = (i/128)*height
                jRemapped = (j/170)*width

                ptvec = np.matrix([iRemapped, jRemapped, 1]).T
                norm_val = np.linalg.norm(np.linalg.inv(intrinsics) @ ptvec)
                vec = (np.linalg.inv(intrinsics) @ ptvec)/norm_val

                if j > 4 and j < 165:
                    vec *= depth[0][i][j-5][0]
                else:
                    vec *= 0
                vec = vec.tolist()
                ting = np.matrix([-vec[0][0], vec[1][0], -vec[2][0], 1]).T
                # new = rotation * ting
                # print(new)
                matlist = np.array(ting.flatten()).tolist()
                ptCloud.append(matlist)
                pbar.update(1)
    return ptCloud
    # write to csv
    """
    with open(f'cloud.csv', 'w', newline="") as f:
            writer = csv.writer(f)
            image = Image.open('frame.jpg')
            resized = ImageOps.fit(image, (128,170), Image.ANTIALIAS).convert('RGB') 
            cv_resized = np.array(resized).reshape((128*170, 3))
            cv2.imshow('hi',cv_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            for i in range(0, len(ptCloud)):
                point = ptCloud[i]
                color = cv_resized[i].tolist()
                point.extend(color)
                writer.writerow(point)
                """
        
    
# gen_coords('framemetadata.json', 'frame.jpg')

def parse_planes(json):
    f = open('transform.csv', 'w', newline="")
    p = open('plane.csv', 'w', newline="")
    write_transform = csv.writer(f)
    write_plane = csv.writer(p)

    for i, plane in enumerate(json['planes']):
        data = plane['center']
        data.extend(plane['extent'])
        write_plane.writerow(data)
        write_transform.writerow(plane['transform'])

js = json.load(open('data/framemetadata.json'))

def generate_v1_data():
    # Extract intrinsic matrices from text file
    intrinsic_list = open(os.path.join(DATA_PATH, 'intrinsics.txt')).read().split('\n')
    for image_id in range(1, 10335):
        # Extract single intrinsic as numpy matrix
        intrinsic = np.asarray([float(i) for i in intrinsic_list[image_id].split()[1:]])
        # Generate a point cloud for the image_id
        # first, open the original image in PILLOW
        id_full = '%06d'%(int('000000')+image_id)
        img_path = os.path.join(DATA_PATH, f'image/{id_full}.jpg')
        rgb_img = Image.open(img_path)
        # run the image through the network
        depth_img = predict('tensorflow/models/NYU_FCRN.ckpt', rgb_img)
        # generate a point cloud
        pt_cloud = gen_point_cloud(depth_img, rgb_img, intrinsic)
        # generate ply file from FCRN pointcloud
        with open(f'cloud_{id}.ply', 'w') as f:
            # I can feel pylint's disappointment in this line's length
            f.write(f'ply\nformat ascii 1.0\nelement vertex {len(pt_cloud)}\nproperty double x\nproperty double y\nproperty double z\nend_header\n')
            for i, point in enumerate(pt_cloud):
                # and this one :((
                f.write(f'{round(point[0][0]*point[0][3],15)} {round(point[0][1]*point[0][3],15)} {round(point[0][2]*point[0][3],15)}\n')

        # get ground truth pointcloud and generate planes

        centers, extents, norms = segment_pt_cloud(f"cloud_{id_full}.ply")
        # write to text file
        print(os.listdir())
        with open(f'{DATA_PATH}/label_gen/{id_full}.txt', 'w') as f:
            for i, _ in enumerate(centers):
                # pylint isn't mad, it's disappointed
                f.write(f'plane, {centers[i][0]}, {centers[i][1]}, {centers[i][2]}, {extents[i][0]}, {extents[i][1]}, {extents[i][2]}, {norms[i][0]}, {norms[i][1]}, {norms[i][2]}\n')
        
def process_depth():
    files = os.listdir(f'{DATA_PATH}/depth')
    if not 'depth_ply' in os.listdir(DATA_PATH):
        os.mkdir(f'{DATA_PATH}/depth_ply')

    pool = Pool()

    def write_file(file):
        depth_data = loadmat(f'{DATA_PATH}/depth/{file}')['instance']
        with open(f'{DATA_PATH}/depth_ply/{file[:-3]}ply', 'w') as f:
            f.write(f'ply\nformat ascii 1.0\nelement vertex {len(depth_data)}\nproperty double x\nproperty double y\nproperty double z\nend_header\n')
            for i, point in enumerate(depth_data):
                f.write(f'{round(point[0],15)} {round(point[1],15)} {round(point[2],15)}\n')


    for i, _ in enumerate(pool.imap_unordered(write_file, files),1):
        print(f'{i}/{len(files)} done')
        

    pool.close()
    pool.join()
    pool.close()






"""
with tqdm(total = len(files), desc="Writing cloud") as pbar:
    for file in files:
        depth_data = loadmat(f'{DATA_PATH}/depth/{file}')['instance']
        with open(f'{DATA_PATH}/depth_ply/{file[:-3]}ply', 'w') as f:
            f.write(f'ply\nformat ascii 1.0\nelement vertex {len(depth_data)}\nproperty double x\nproperty double y\nproperty double z\nend_header\n')
            with tqdm(total=len(depth_data), desc="Writing points") as pbar2:
                for i, point in enumerate(depth_data):
                    f.write(f'{round(point[0],15)} {round(point[1],15)} {round(point[2],15)}\n')
                    pbar2.update(1)
        pbar.update(1)
        """

    