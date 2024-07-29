# KITTI 3D Object Detection Dataset

The KITTI 3D object detection dataset consists of:
- 7481 training samples
- 7518 test samples

The training samples are further divided into:
- Training set: 3712 samples
- Validation set: 3769 samples

[Dataset Research Paper](https://www.cvlibs.net/publications/Geiger2013IJRR.pdf)

## LiDAR and Camera Orientation

```
                     Z   
                     |   Y
                     |  /
    RIGHT            | /                   LEFT
                     |/
                     *------------>X
```

## Dataset Download

Download left color images, calibration matrices, and velodyne data for object detection from the [KITTI 3D Object Detection Benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

## Calibration

KITTI calibration files include extrinsic, intrinsic, and rectification matrices.

![Object Detector](https://github.com/rajeev-gupta-bashrc/OBJECT_DETECTOR/blob/master/images/Kitti_transforms.png)

### LiDAR to Camera Transform

```python
Tr_velo_to_cam.shape = (3, 4)
Tr_velo_to_cam = [R, T]
```

This transformation matrix converts (x, y, z) points from the LiDAR frame to the P0 camera frame.

### Camera Calibration

- Left color camera image corresponds to the P2 calibration matrix
- P0 corresponds to the black and white image

```python
P2.shape = (3, 4)
P2 = [[fx, 0, 0, -tx*fx],
      [0, fy, 0, -ty*fy],
      [0,  0, 0,      1]]
```

Where:
- R0 is the rotation from P0 to P2
- tx, ty (in P2 matrix) are transformations from P0 to P2 camera
- This projection matrix converts [x, y, z, 1] to [u, v, depth, 1]
  - (x, y, z) are in the camera frame
  - (u, v) are pixel coordinates of the image

## General Camera Frame

For conversion from camera frame to pixel coordinates:

```
                        Z       __ camera facing direction
                       /         /|
                      /         /            
                     /
                    *────────────>X
                    |
                    |
                    |
                    |
                    Y
```

This representation shows the standard orientation of the camera frame used for converting 3D points in the camera coordinate system to 2D pixel coordinates in the image plane.