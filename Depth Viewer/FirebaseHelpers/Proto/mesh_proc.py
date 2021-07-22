"""
Author: Hwei-Shin Harriman
Altered by: Neel Dhulipala
Original program: https://github.com/occamLab/augmented-reality-tools/blob/plane-visualization/plane-visualization/meshes/meshes.py
Process AR mesh data from Depth Viewer
"""
import json
import numpy as np
import os, sys
from pathlib import Path
import Mesh_pb2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def read_protobuf(filepath):
    """
    Open and read protobuf files to be parsed.

    Args:
        filepath: A String with the path to the location of the protobuf file.
    Returns:
        A matrix including the data from the protobuf.
    """
    f = open(filepath, 'rb')
    m = Mesh_pb2.MeshesProto()
    # Read the mesh and return the output matrix
    m2 = m.FromString(f.read())
    return m2

# Read the raycasts from the metadata JSON file, given String of filepath
def get_raycasts_from_metadata(filepath):
    """
    Extract raycast data from a metadata JSON file.

    Args:
        filepath: A String with the path to the location of the metadata file.
    Returns:
        A list containing the raycast data.
    """
    f = open(filepath)
    # after opening filepath, get it parsed
    metadata = json.load(f)
    # return section of metadata file with the raycast data
    return metadata['raycast']

def get_raycasts_from_firebase(frame_number):
    """
    Find the names of all the paths which include the raycast data requested.

    Args:
        frame_number: An integer of the frame where you want the raycast data from
    Returns:
        None.
    """
    # NOTE: meshes data has to be downloaded first, which means
    # get_meshes_from_firebase must be called before this function
    raycast_array = []
    try:
        file_name = "{:04d}".format(frame_number)
        raycast_array = get_raycasts_from_metadata(f"meshes/{file_name}/framemetadata.json")
    except FileNotFoundError:
        print(f"No metadata file in frame {num_meshes}. Try again.")
        quit()
    except:
        print("Error: something went wrong with the raycasts.")
        quit()

    print("Raycasts received...")
    return raycast_array

def get_meshes_from_firebase(trial_path):
    """
    Find the names of all the paths which include the mesh files requested.

    Args:
        trial_path: A string containing the trial where you want meshes from
    Returns:
        None.
    """
    if trial_path != "same":
        # removes the meshes directory, if it already exists
        os.system('rm -r meshes')
        # replaces it with a new meshes directory, which will contain the data
        # from this particular trial (iphone/trial_name)
        os.system('mkdir meshes')
        os.system(f'gsutil -m rsync -r gs://clew-sandbox.appspot.com/{trial_path}/ meshes')
    # Count how many files there are in the directory, read protobufs in
    # reverse order
    num_meshes = len(os.listdir('meshes'))
    mesh_dict = []
    for i in range(num_meshes-1, 0, -1):
        try:
            file_name = "{:04d}".format(i)
            mat = read_protobuf(f"meshes/{file_name}/meshes.pb")
            mesh_dict += parse_meshes(mat)
        except FileNotFoundError:
            print(f"No mesh file in frame {i}. Continuing...")
        except:
            print("Error: something went wrong with the meshes.")


    print("Meshes received...")
    return mesh_dict

def get_vertices(mat, index):
    """
    Extracts vertices from data from a protobuf file.

    Args:
        mat: a dictionary of mesh data from a protobuf file
        index: an integer representing the index of the protobuf file
    Returns:
        A list of lists containing the x,y,z points of all vertices
    """
    # since transform is a 4x4, add a 1.0 at the end of the list
    return list(map(lambda x: np.array([x.x, x.y, x.z, 1.0]), mat.meshes[index].vertices))

def get_transform(mat, index):
    """
    Extracts transform matrix from mesh data from a protobuf file.

    Args:
        mat: a dictionary of mesh data from a protobuf file
        index: an integer representing the index of the protobuf file
    Returns:
        A list of lists containing the columns of the transform matrix
    """
    mesh_transform = mat.meshes[index].transform
    # Formatting the transform matrix that can be read
    real_transform = [
        [mesh_transform.c1.x, mesh_transform.c1.y, mesh_transform.c1.z,
            mesh_transform.c1.w],
        [mesh_transform.c2.x, mesh_transform.c2.y, mesh_transform.c2.z,
            mesh_transform.c2.w],
        [mesh_transform.c3.x, mesh_transform.c3.y, mesh_transform.c3.z,
            mesh_transform.c3.w],
        [mesh_transform.c4.x, mesh_transform.c4.y, mesh_transform.c4.z,
            mesh_transform.c4.w]
    ]
    return real_transform

def parse_meshes(mat):
    """
    Update dictionary so that values are Python objects instead of strings.

    Args:
        mat: A matrix of the raw Protobuf data read directly from the .pb file
    Returns:
        A list of dicts containing parsed version of input data
    """
    data_dicts = []
    id_list = []
    for mesh_index in range(len(mat.meshes)):
        # Create a dict which has vertices and transform of mesh defined
        face = {}
        #check if mesh ID already has been crossed, otherwise add it to a list
        if mat.meshes[mesh_index].id not in id_list:
            id_list.append(mat.meshes[mesh_index].id)
            #parse transform
            face["transform"] = get_transform(mat, mesh_index)
            #parse geometry array
            face["vertices"] = get_vertices(mat, mesh_index)
            # Append this dict to data_dicts
            data_dicts.append(face)
    return data_dicts


def loc2glob(meshes):
    """
    Given info about planes in local coordinate system, convert to global
    coordinate system

    Args:
        meshes: A list of dicts containing parsed version of input data
    Returns:
        A list of global vertices
    """
    res = []
    # Go through all the meshes in list
    for mesh in meshes:
        transform = np.array(mesh["transform"])
        vertices = np.array(mesh["vertices"])
        #check if vertices mesh is empty, skips if True
        if vertices.size == 0:
            continue
        #calculate vertices in the global coordinate system and store in dict
        global_array = vertices @ transform
        #convert from np.ndarray back to list
        res += global_array.tolist()

    print("Meshes were localized...")
    return res

def write_to_ply(meshes, ply_file_name):
    """
    Takes data from list of transformed vertices and writes them into ply file

    Args:
        meshes: A list of dicts containing parsed mesh data from Firebase
        ply_file_name: A string with the desired name for the ply file
    Returns:
        None.
    """
    # Multiply unit vector vals by length
    with open(f"mesh_plys/{ply_file_name}.ply", "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(meshes)}\nproperty double x\nproperty double y\nproperty double z\nend_header\n")
        for verts in meshes:
            f.write(f'{verts[0]} {verts[1]} {verts[2]}\n')
    print("PLY file written...")

def write_to_raycast_ply(raycasts):
    """
    Takes data from raycasts and writes it to a ply file

    Args:
        raycasts: An array of floats containing raycast data extracted from
            metadata JSON
    Returns:
        None.
    """
    # Multiply unit vector vals by length
    with open(f"mesh_plys/raycast_heatmap.ply", "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(raycasts)}\nproperty double x\nproperty double y\nproperty double z\nend_header\n")
        for ray in raycasts:
            if len(ray) == 5:
                f.write(f'{ray[0]} {ray[1]} {ray[4]}\n')
            else:
                f.write(f'{ray[0]} {ray[1]} {ray[2]}\n')
    print("Raycasts heatmap PLY file written...")
    # Point cloud file for raycasts
    pcrays = []
    for ray in raycasts:
        if len(ray) == 5:
            pcrays.append(ray)
    with open(f"mesh_plys/raycast_pointcloud.ply", "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pcrays)}\nproperty double x\nproperty double y\nproperty double z\nend_header\n")
        for ray in pcrays:
            f.write(f'{ray[2]} {ray[3]} {ray[4]}\n')
    print("Raycasts pointcloud PLY file written...")


def main(file_prefix, trial_path, frame_number):
    """
    Pulls latest mesh data from Firebase, parses and stores into PLY file.
    Also pulls raycast data and stores that as two PLY files, one with heatmap
    data and one with pointcloud data.

    Args:
        file_prefix: A string with desired prefix for the saved file
        trial_path: A string containing name of the trial where you want to
            pull meshes from
        frame_number: An integer of the number of the frame from where raycasts
            should be
        extracted
    Returns:
        A dict of parsed mesh data
    """
    # retrieve data from Firebase
    # If you do not want to redownload the files in a trial, type "same"
    meshes = get_meshes_from_firebase(trial_path)
    raycasts = get_raycasts_from_firebase(int(frame_number))
    # localize meshes in global coordinate system
    loc_meshes = loc2glob(meshes)
    # create a ply file of the meshes
    write_to_ply(loc_meshes, file_prefix)
    write_to_raycast_ply(raycasts)
    print("Execution successful!")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
