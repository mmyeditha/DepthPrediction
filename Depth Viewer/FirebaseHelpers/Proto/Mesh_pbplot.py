import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import Mesh_pb2

# Read a protobuf file, given a String with the path info
def read_protobuf(filepath):
    f = open(filepath, 'rb')
    m = Mesh_pb2.MeshesProto()
    # Read the mesh and return the output matrix
    m2 = m.FromString(f.read())
    return m2

def extract_transform(m, index):
    mesh_transform = m.meshes[index].transform
    # Formatting the transform matrix that can be read
    real_transform = [
        [mesh_transform.c1.x, mesh_transform.c1.y, mesh_transform.c1.z, mesh_transform.c1.w],
        [mesh_transform.c2.x, mesh_transform.c2.y, mesh_transform.c2.z, mesh_transform.c2.w],
        [mesh_transform.c3.x, mesh_transform.c3.y, mesh_transform.c3.z, mesh_transform.c3.w],
        [mesh_transform.c4.x, mesh_transform.c4.y, mesh_transform.c4.z, mesh_transform.c4.w]
    ]
    return np.array(real_transform)


def print_verts(m, index):
    # Compile all the vertices from index'th column in m
    all_verts = np.array(list((map(lambda x: np.array([x.x, x.y, x.z]), m.meshes[index].vertices))))
    # Plot these vertices
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(all_verts[:,0], all_verts[:,1], all_verts[:,2])

def print_all_verts(m):
    # Create an empty np.array()
    all_verts = []
    for i in range(len(m.meshes)):
        # Compile all vertices for the i'th index in m
        i_vert = np.array(list((map(lambda x: np.array([x.x, x.y, x.z, 1.0]), m.meshes[i].vertices))))
        # Get the transform matrix for that i'th index
        i_transform = extract_transform(m, i)
        # Multiply the two together
        i_transformed_vert = i_vert @ i_transform
        # Turn into a list
        i_list = i_transformed_vert.tolist()
        # Append to all_verts
        all_verts += i_list
    #print(all_verts)
    # Change it back into an array
    all_verts = np.array(all_verts)
    # Plot these vertices
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(all_verts[:,0], all_verts[:,1], all_verts[:,2])
