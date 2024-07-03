## Object Detector

This 3D object detector ROS package uses 3D point cloud and image data to create bounding boxes around detected objects. It utilizes the KITTI 3D object detection dataset and is deployed using ROS Noetic. The model is based on the GraphRCNN voxel image-based model. The environment details are as follows:

- Python: 3.8.12
- CUDA: 11.1
- cumm-cu111: 0.2.9
- spconv-cu111: 2.1.25
- tensorboardX: 2.6.2.2
- torch: 1.10.1+cu111
- torch-scatter: 2.0.9
- torchaudio: 0.10.1+cu111
- torchvision: 0.11.2+cu111