import json, sys, csv, cv2, os, shutil, argparse, re
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from numpy.linalg import inv
sys.path.insert(1,'tensorflow/')
from predict import predict
from multiprocessing import Pool
from scipy.io import loadmat
OPEN3D = True
try:
    import open3d as o3d
except:
    print("Open3D not installed properly, plane and point cloud functionality will not work!")
    OPEN3D = False

DATA_PATH = "votenet/sunrgbd/sunrgbd_trainval"

def reorganize_data():
    """
    FOR THIS FUNCTION TO WORK, place the OFFICIAL_SUNRGBD data inside of the sunrgbd folder
    along with the metadata matrices.
    """
    os.chdir('votenet/sunrgbd')
    data = loadmat('OFFICIAL_SUNRGBD/SUNRGBDMeta2DBB_v2.mat')['SUNRGBDMeta2DBB']
    if 'sunrgbd_plane' in os.listdir():
        shutil.rmtree('sunrgbd_plane')
    os.mkdir('sunrgbd_plane')
    for i in tqdm(range(0, 10335), desc="Reorganizing Data"):
        os.mkdir(f'sunrgbd_plane/{i+1:05}')
        datapath = os.path.join('OFFICIAL_SUNRGBD', data[0][i][0][0])
        rgb_img = os.path.join(datapath, 'image', os.listdir(f'{datapath}/image')[0])
        depth_img = os.path.join(datapath, 'depth_bfx', os.listdir(f'{datapath}/depth_bfx')[0])
        extrinsics = os.path.join(datapath, 'extrinsics', os.listdir(f'{datapath}/extrinsics')[0])
        intrinsic = os.path.join(datapath, 'intrinsics.txt')
        seg = os.path.join(datapath, 'seg.mat')
        # Rename and move the files
        shutil.copyfile(rgb_img, f'sunrgbd_plane/{i+1:05}/rgb_{i+1:05}.jpg')
        shutil.copyfile(depth_img, f'sunrgbd_plane/{i+1:05}/depth_{i+1:05}.png')
        shutil.copyfile(extrinsics, f'sunrgbd_plane/{i+1:05}/extrinsic_{i+1:05}.txt')
        shutil.copyfile(intrinsic, f'sunrgbd_plane/{i+1:05}/intrinsics_{i+1:05}.txt')
        shutil.copyfile(seg, f'sunrgbd_plane/{i+1:05}/seg_{i+1:05}.mat')

def remap(rgb_img, truth_heatmap):
    """
    Takes the ground truth depthmaps and remaps them to match with the depth values of
    FCRN's prediction
    """
    heatmap = predict('tensorflow/models/NYU_FCRN.ckpt', Image.open(rgb_img), silent=True)[0]
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
                norm_image[i][j] = ((gray_img[i][j]-min_gray)/range_gray)*range_heatmap+min_heatmap
                pbar.update(1)
    return norm_image

def gen_ply_file(image_id):
    data_path = 'votenet/sunrgbd/sunrgbd_plane'
    rgb_img = os.path.join(data_path, f'{image_id+1:05}/rgb_{image_id+1:05}.jpg')
    depth_img = os.path.join(data_path, f'{image_id+1:05}/depth_{image_id+1:05}.png')
    r = remap(rgb_img, depth_img)
    intrinsic_list = re.split(' |\n', open(os.path.join(data_path, f'{image_id+1:05}/intrinsics_{image_id+1:05}.txt')).read())[0:9]
    extrinsic_list = re.split(' |\n', open(os.path.join(data_path, f'{image_id+1:05}/extrinsic_{image_id+1:05}.txt')).read())[0:12]
    extrinsic_list.extend(['0','0','0','1'])
    intrinsic = [float(x) for x in intrinsic_list]
    extrinsic = [float(x) for x in extrinsic_list]
    print(intrinsic)
    print(extrinsic)
    pc = gen_point_cloud(r, rgb_img, np.asarray(intrinsic), np.asarray(extrinsic))
    write_ply_file(pc, os.path.join(data_path, f'{image_id+1:05}/pcd_{image_id+1+1:05}'))

def predict_clean(image_path):
    return predict('models/NYU_FCRN.ckpt', Image.open(image_path))[0]

def segment_and_remove(pcd):
    plane, pts = pcd.segment_plane(distance_threshold=0.2, ransac_n=5000, num_iterations=1000)
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
        if noisy:
            o3d.io.write_point_cloud(f'outlier_{num_planes}.ply', outlier_cloud)
            o3d.io.write_point_cloud(f'inlier_{num_planes}.ply', cl)
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

def gen_point_cloud_old(depth, image, intrinsics, extrinsics):
    # intrinsics = np.asmatrix(np.reshape(metadata['intrinsics'], (3,3)).swapaxes(0,1))
    """
    Old method for point cloud generation. Unbearably slow
    """
    intrinsics = np.reshape(intrinsics, (3,3))
    print(intrinsics)
    ptCloud = []
    width, height = image.size
    with tqdm(total = np.shape(depth)[0]*np.shape(depth)[1], desc = 'processing cloud') as pbar:
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
                """
                if j > 4 and j < 165:
                    vec *= depth[i][j-5]
                else:
                    vec *= 0
                """
                vec = vec.tolist()
                ting = np.matrix([-vec[0][0], vec[1][0], -vec[2][0], 1]).T
                # new = rotation * ting
                # print(new)
                matlist = np.array(ting.flatten()).tolist()
                ptCloud.append(matlist)
                pbar.update(1)

    # rotate the point cloud to align with the proper coordinate system
    extrinsics = np.reshape(extrinsics, (4,4))
    rot_x = np.reshape(np.array([1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1]), (4,4))   
    pcd = np.reshape(np.asarray(ptCloud), (len(ptCloud),4))
    pcd = rot_x@(extrinsics@pcd)
    return pcd
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

def write_ply_file(pc, name):
    """
    Take in a pointcloud as an array and write it
    """
    with open(f'{name}.ply', 'w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(pc)}\nproperty double x\nproperty double y\nproperty double z\nend_header\n')
        for i, point in enumerate(pc):
            f.write(f'{round(point[0]*point[3],15)} {round(point[1]*point[3],15)} {round(point[2]*point[3],15)}\n')


#----------------------------- new workflow --------------------------------------------
def gen_point_cloud(depth, image, intrinsics, extrinsics):
    # intrinsics = np.asmatrix(np.reshape(metadata['intrinsics'], (3,3)).swapaxes(0,1))
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
    vecs = np.stack((iRemapped,jRemapped,np.ones((height_rgb,width_rgb))), axis=2)
    s = vecs.reshape((height_rgb*width_rgb,3))
    normd = np.linalg.inv(intrinsics)@s.T
    newnorm = normd/(np.linalg.norm(normd,axis=0))
    projected = newnorm*depth.reshape((height_rgb*width_rgb))
    pcd = np.vstack((projected, np.ones((height_rgb*width_rgb))))
    return (rot_x@(extrinsics@pcd)).T

def extract_walls(mat):
    """
    args:
        mat: String referencing the path to the seg.mat file from dataset, 
        classifies each pixel as a certain object and defines each class as a number,
        also has a dictionary to tell us what number
        means what object. 

    returns a dictionary mapping objects to the pointcloud indices of points that make up that object.
    Ex. a key will be 'wall_0', and the value will be a list of indices that make up the points that fall
    within that wall. To visualize, open the point cloud with o3d, then run o3d.select_by_points(<point list>)
    and this should show you all the points that make up that object.
    """
    segs = loadmat(mat)
    seglabel = segs['seglabel']
    print(seglabel)
    names = segs['names'][0]
    print(names)
    surfaces = {}
    whitelist = ['bed', 'table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']
    walls= [x+1 for x in range(len(names)) if names[x][0] == 'wall']
    ceilings = [x+1 for x in range(len(names)) if names[x][0] == 'ceiling']
    floor = [x+1 for x in range(len(names)) if names[x][0] == 'floor']
    other = [x+1 for x in range(len(names)) if names[x][0] in whitelist]
    # Eventually change these to be dependent on list length, not just only for SUNGRB data
    for i in range(530):
        for j in range(730):
            if seglabel[i][j] in walls:
                if f'wall_{seglabel[i][j]}' in surfaces.keys():
                    surfaces[f'wall_{seglabel[i][j]}'].append(i*730+j)
                else:
                    surfaces[f'wall_{seglabel[i][j]}'] = [i*730+j]
            elif seglabel[i][j] in ceilings:
                if f'ceiling_{seglabel[i][j]}' in surfaces.keys():
                    surfaces[f'ceiling_{seglabel[i][j]}'].append(i*730+j)
                else:
                    surfaces[f'ceiling_{seglabel[i][j]}'] = [i*730+j]
            elif seglabel[i][j] in floor:
                if f'floor_{seglabel[i][j]}' in surfaces.keys():
                    surfaces[f'floor_{seglabel[i][j]}'].append(i*730+j)
                else:
                    surfaces[f'floor_{seglabel[i][j]}'] = [i*730+j]
            elif seglabel[i][j] in other:
                if f'other_{seglabel[i][j]}' in surfaces.keys():
                    surfaces[f'other_{seglabel[i][j]}'].append(i*730+j)
                else:
                    surfaces[f'other_{seglabel[i][j]}'] = [i*730+j]

    return surfaces

def get_heading_angle(bbox):
    """
    Given the bbox o3d object, returns the heading angle relative to
    the +x direction (right)

    args:
        bbox: bounding box object from open3d
    
    returns a 2 element array.
    """
    # First, run PCA on the pts of the 3DBB

    # The heading angle that is more in the direction of <1,1> (dot product
    # of the vector and <1,1> is greater) corresponds to the heading angle we want
    
    # Get rid of the +Z component of the vector (should be close to zero)
    return None

def gen_label(pts, class_name, pcd):
    """
    Generates label text for a set of points in a pointcloud

    args:
        pts: a list of the indices of every point
        class_name: classification of the object contained within the points
        pcd: open3d pointcloud
    """
    obj = pcd.select_by_index(pts)

    

def parse_planes(json):
    """
    Parses planes from firebase framemetadata json. Writes as csv into center, extent, and
    transform. We no longer use framemetadata for planes, but keeping it here in case it becomes
    useful in the future

    EVENTUALLY, add the list of centers, extents, and transforms and return those

    args:
        json: the json file parsed and opened with json.load
    """
    f = open('transform.csv', 'w', newline="")
    p = open('plane.csv', 'w', newline="")
    write_transform = csv.writer(f)
    write_plane = csv.writer(p)

    for i, plane in enumerate(json['planes']):
        data = plane['center']
        data.extend(plane['extent'])
        write_plane.writerow(data)
        write_transform.writerow(plane['transform'])

def generate_v1_data():
    """
    Automated script to run through all of the sunrgbd_trainval data and generates
    labels and planes. 

    DEPRECATED NOW, will soon turn this into a script to pull from the raw OFFICIAL_SUNRGBD
    and convert it the ~new~ way. 
    """
    # Extract intrinsic matrices from text file
    intrinsic_list = open(os.path.join(DATA_PATH, 'intrinsics.txt')).read().split('\n')
    for image_id in tqdm(range(1, 10335)):
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
    """
    One-time script to take all of the depth.mat files from the sunrgbd original data
    and turn them into ply files
    """
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


def parse_annotation(json):
    labels = json.load(open(json))
    label_list = []




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

    
#quick script
#j = remap('demo_files/second_test/0000041.jpg', 'demo_files/second_test/0000041.png')
#pc = gen_point_cloud(j, Image.open('demo_files/second_test/0000041.jpg'), np.array([529.500000, 0.000000, 365.000000, 0.000000, 529.500000 ,265.000000, 0, 0, 1]))
#write_ply_file(pc, 'demo_files/second_test/0000041.ply')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply', action='store_true', help='Generate ply files from extracted data')
    args = parser.parse_args()

    if args.ply:
        data_path = 'votenet/sunrgbd/sunrgbd_plane'
        pool = Pool()
        for i, _ in enumerate(pool.imap_unordered(gen_ply_file, range(1810,5000)),1):
            print(f'{i}/10335 done')
        pool.close()
        pool.join()
        pool.close()

