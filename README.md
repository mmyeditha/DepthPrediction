# Pointnet Training
## Merwan Yeditha

Retraining the pointnet branch using lidar scans and images from firebase.

This does most of the processing in python, including
- Process images in FCRN and get heatmaps
- Convert heatmaps to point clouds
- Pull transform matrix and lidar point cloud from metadata
- Generate planes from lidar data
(coming soon)
- Pull data from firebase automatically
- Apply planes to relevant FCRN data
- Run SUN_RGB scripts to package into training data
- Train the facebook votenet model

We did our testing on an Ubuntu 18.04 machine with a GTX 1060 3GB (GP107).
We recommend using venv as a virtual environment manager.
To download all dependencies, run `pip install -r requirements.txt`

Detailed installation instructions/description of our work available in the Wiki
