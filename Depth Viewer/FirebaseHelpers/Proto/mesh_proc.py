"""
Author: Hwei-Shin Harriman
Altered by: Neel Dhulipala
Original program: https://github.com/occamLab/augmented-reality-tools/blob/plane-visualization/plane-visualization/meshes/meshes.py
Process AR mesh data from Depth Viewer
"""
import pickle
import numpy as np
import os, sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import SharedUtils

MESH_CLRS = {
    "": "rgb(117,117,117)", #gray
    "Wall": "rgb(51, 190, 255)",    #blue
    "Ceiling": "rgb(145, 255, 145)",   #light green
    "Floor": "rgb(255, 56, 38)",    #red
    "Table": "rgb(149, 71, 204)",   # light purple
    "Seat": "rgb(219, 189, 255)",   #lavender
    "Window": "rgb(252, 242, 101)", #light yellow
    "Door": "rgb(245, 137, 227)",   #pink
    "Unknown": "rgb(245, 150, 49)" #bright orange
}


class MeshHandler(SharedUtils):
    def __init__(self):
        super().__init__()

    def get_vertices(self, mat, index):
        """
        Extracts vertices from data from a protobuf file.
        Args:
            mat: a dictionary of mesh data from a protobuf file
            index: an integer representing the index of the protobuf file
        Returns:
            A list of lists containing the x,y,z points of all vertices
        """
        # Add 1.0 to end since transform is a 4x4
        return list(map(lambda x: np.array([x.x, x.y, x.z, 1.0]), mat.meshes[index].vertices))

    def get_transform(self, mat, index):
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

    def parse_meshes(self, mat):
        """
        Update dictionary so that values are Python objects instead of strings.
        :param mat: (dict?) raw Protobuf data read directly from the .pb file
        :returns: (list of dicts) containing parsed version of input data
        """
        data_dicts = []
        for mesh_index in range(len(mat.meshes)):
            # Create a dict which has vertices and transform of mesh defined
            face = {}
            #parse transform
            face["transform"] = self.get_transform(mat, mesh_index)
            #parse geometry array
            face["vertices"] = self.get_vertices(mat, mesh_index)
            # Append this dict to data_dicts
            data_dicts.append(face)
        return data_dicts


    def loc2glob(self, meshes):
        """
        Given info about planes in local coordinate system, convert to global coordinate system
        :param meshes: (list of dicts) contains parsed version of input data
        :returns: (list of dicts) with added dict including 'global' matrix
        """
        # Go through all the meshes in list
        for mesh in meshes:
            vertices = np.array(mesh["vertices"])
            transform = np.array(mesh["transform"])
            #calculate vertices in the global coordinate system and store in dict
            mesh["global"] = vertices @ transform
        return meshes

    def make_3d_file(self, meshes, prefix):
        """
        Build CSV files of meshes that is compatible with Plotly 3dMesh. Creates one CSV file per possible
        mesh classifciation
        :param meshes: (list of dicts) containing parsed mesh data from Firebase
        :param prefix: (str) prefix used for saving the CSVs
        :returns: None, saves the CSVs locally
        """
        frames = {}
        to_str = lambda co: ",".join([str(pt) for pt in co])
        #needs to be separated by classification
        for face in meshes:
            #convert corners to string representations for hashability
            str_corners = [to_str(co) for co in face["g_corners"]]
            cl = face["class"]
            #add new entry for each possible classification
            if not frames.get(cl, None):
                frames[cl] = {"index": 0, "frame": {}}  #index tracks "row number" for each entry, "frame" contains dicts, one for each vertex in the mesh


            for co in str_corners:
                if not frames[cl]["frame"].get(co, None):
                    #add the new mesh vertex to frame with matching classification
                    frames[cl]["frame"][co] = {
                        "index": frames[cl]["index"],
                        "i": str_corners[0],
                        "j": str_corners[1],
                        "k": str_corners[2],
                        "facecolor": MESH_CLRS[cl]
                    }
                    frames[cl]["index"] += 1
        #build csvs
        csvs = {}
        for points in meshes:
            verts = points["global"]
            #x,y,z = (x,y,z) coordinates of a single vertex in a mesh
            #facecolor = "rgb(a,b,c)" color of the face of the mesh
            #i,j,k = INDICES (aka the ROW of the csv not including the header row) of each of the 3 corners of a mesh
            csvs[cl] = '"x","y","z"\n'  #the header of the CSVs
            for pt, info in verts.items():
                #i,j,k need to represent the INDEX (in the CSV) of each corner of a single triangle mesh
                #so we update their values by looking up the points they correspond to and replacing with their indices
                i, j, k = verts[info["i"]]["index"], verts[info["j"]]["index"], verts[info["k"]]["index"]
                pt = pt.split(",")
                csvs[cl] += f'"{pt[0]}","{pt[1]}","{pt[2]}","{info["facecolor"]}","{i}","{j}","{k}"\n'

        Path(f"3d_files/{prefix}").mkdir(exist_ok=True)
        for cl in csvs.keys():
            save_path = f"3d_files/{prefix}/{cl}-ply.csv"
            print("saving csv at: ", save_path)
            with open(save_path, "w") as f:
                f.write(csvs[cl])
                print("saving csv at: ", save_path)


    def main(self, file_prefix, full_path, should_pull=True, make_csv=True):
        """
        Pulls latest mesh data from Firebase, parses and stores as a pickle file.
        :param file_prefix: (str) desired prefix for the saved file
        :param should_pull: (bool) True if you want to pull new file from Firebase, otherwise
            opens an existing file with matching `file_prefix`
        :returns: (dict) parsed mesh data
        """
        Path(f"../files/").mkdir(exist_ok=True)
        file_name = f"../files/{file_prefix}_meshes.pckl"
        if should_pull:
            #TODO currently all uploaded meshes are overwriting meshes_1.json, update when this is fixed from Clew side
            data_dict = self.fetch_from_firebase(full_path)

            #convert string representations of vectors to np arrays
            meshes = self.parse_meshes(data_dict)

            #add information about coordinates in global frame to dict
            meshes = self.loc2glob(meshes)

            #store updated results
            with open(file_name, 'wb') as f:
                pickle.dump(meshes, f)
        else:
            with open(file_name, 'rb') as f:
                meshes = pickle.load(f)
        if make_csv:
            self.make_3d_file(meshes, file_prefix)

        return meshes

if __name__ == "__main__":
    handler = MeshHandler()
    meshes = handler.main("apartment", "2021-06-17T19:31:30-07:00_meshes", should_pull=True, make_csv=True)
