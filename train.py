import json
import sys
import numpy as np
import csv
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from numpy.linalg import inv
sys.path.insert(1,'tensorflow/')
from predict import predict
import open3d as o3d

def segment_pt_cloud(filename, show=False):
    # Read the point cloud in open3D
    planedata = []
    points = []
    planes = []
    pcd = o3d.io.read_point_cloud("data/object3d.ply")    
    plane, pts = pcd.segment_plane(distance_threshold=0.1,
                                         ransac_n=3,
                                         num_iterations=1000)
    print(pts)
    planedata.append(plane.tolist())
    points.append(pts)
    planes.extend(pts.get_min_bound().tolist())
    planes.extend(pts.get_max_bounds().tolist())
    inlier_cloud = pcd.select_by_index(pts)
    outlier_cloud = pcd.select_by_index(pts, invert=True)
    plane, pts = outlier_cloud.segment_plane(distance_threshold=0.05,
                                         ransac_n=3,
                                         num_iterations=1000)
    planedata.append(plane.tolist())
    points.append(pts)
    inlier_2 = outlier_cloud.select_by_index(pts)
    outlier_cloud = outlier_cloud.select_by_index(pts, invert=True)
    plane, pts = outlier_cloud.segment_plane(distance_threshold=0.05,
                                         ransac_n=3,
                                         num_iterations=1000)
    planedata.append(plane.tolist())
    points.append(pts)
    planes.extend(pts.get_min_bound().tolist())
    planes.extend(pts.get_max_bounds().tolist())
    inlier_3 = outlier_cloud.select_by_index(pts)
    outlier_cloud = outlier_cloud.select_by_index(pts, invert=True)
    plane, pts = outlier_cloud.segment_plane(distance_threshold=0.05,
                                         ransac_n=3,
                                         num_iterations=1000)
    planedata.append(plane.tolist())
    points.append(pts)
    planes.extend(pts.get_min_bound().tolist())
    planes.extend(pts.get_max_bounds().tolist())
    inlier_4 = outlier_cloud.select_by_index(pts)
    outlier_cloud = outlier_cloud.select_by_index(pts, invert=True)
    plane, pts = outlier_cloud.segment_plane(distance_threshold=0.05,
                                         ransac_n=3,
                                         num_iterations=1000)
    planedata.append(plane.tolist())
    points.append(pts)
    planes.extend(pts.get_min_bound().tolist())
    planes.extend(pts.get_max_bounds().tolist())

    if show:
        o3d.visualization.draw_geometries([inlier_cloud, inlier_2, inlier_3, inlier_4])
    

    return planedata, points, planes

def gen_coords(filename, imgfile):
    
    file = open(filename)
    js = json.load(file)

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
    s = predict('tensorflow/models/NYU_FCRN.ckpt', image)
    gen_point_cloud(s, js, image)

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

def gen_point_cloud(depth, metadata, image):
    intrinsics = np.asmatrix(np.reshape(metadata['intrinsics'], (3,3)).swapaxes(0,1))
    transform = np.asmatrix(np.reshape(metadata['pose'], (4,4)).swapaxes(0,1))

    # Adjusting for coordinate system
    rotation = np.matrix('0, -1, 0, 0; 1, 0, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1')
    
    ptCloud = []
    width, height = image.size
    with tqdm(total = 128*170, desc = 'processing cloud') as pbar:
        for i in range(0, 128): 
            for j in range(0, 170):
                # Remap to 4:3 with blank bars
                iRemapped = (i/128)*height
                jRemapped = (j/170)*width

                ptvec = np.matrix([iRemapped, jRemapped, 1]).T
                norm_val = np.linalg.norm(intrinsics.getI() * ptvec)
                vec = (intrinsics.getI() * ptvec)/norm_val

                if j > 4 and j < 165:
                    vec *= depth[0][i][j-5][0]
                else:
                    vec *= 0
                vec = vec.tolist()
                ting = np.matrix([-vec[0][0], vec[1][0], -vec[2][0], 1]).T
                # new = rotation * ting
                # print(new)
                matlist = np.array((rotation*ting).T*transform.T).flatten().tolist()
                ptCloud.append(matlist)
                pbar.update(1)

    # write to csv
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
        
    
# gen_coords('framemetadata.json', 'frame.jpg')

def write_to_ply(depthData):
    """
    Takes the depthData from metadata json format and 
    converts to ply
    """
    # Multiply unit vector vals by length
    with open('object.ply', 'w') as f:
        # I can feel pylint's disappointment in this line's length
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(depthData)}\nproperty double x\nproperty double y\nproperty double z\nend_header\n')
        for i, point in enumerate(depthData):
            f.write(f'{round(point[0]*point[3],15)} {round(point[1]*point[3],15)} {round(point[2]*point[3],15)}\n')

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


