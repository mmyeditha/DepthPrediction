# FastDepth Model Variant
### By: Neel Dhulipala, Mario Gergis, and Merwan Yeditha

An implementation of the FCRN CoreML library created by [tucan9389](https://github.com/tucan9389/DepthPrediction-CoreML) and adapted by [JustinFincher.](https://github.com/JustinFincher/FastDepth-CoreML)

This app displays the live camera output and a heatmap produced by a matrix of distances computed by the FCRN neural network. The network takes the camera output as an input and, using the CoreML library (see below), estimates the distance between every point in the image and the camera. Our additions to this app include:

- Generating point clouds based on these approximated distances
- Updating code from using UIKit to SwiftUI

![](/Depth\ Viewer/Pictures/rendering_pointcloud.png)
*Figure 1: Rendering a point cloud based on an image taken from the app*

This app is a work in progress, with the eventual goal of being able to detect obstacles within a designated range.

[mlmodel download](https://drive.google.com/file/d/16NV8gUvvrlmhgFT9hrrEkAetp-BAKGlG/view?usp=sharing)
