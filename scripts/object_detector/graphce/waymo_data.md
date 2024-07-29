# Waymo Open Dataset

The Waymo Open Dataset is a comprehensive autonomous driving dataset comprising:
- 798 scenes for training
- 202 scenes for validation

## Evaluation Protocol
- Average Precision (AP)
- Average Precision weighted by Heading (APH)

### Difficulty Levels
1. LEVEL 1: Objects with more than 5 points
2. LEVEL 2: Objects with at least 1 point

[Dataset Research Paper](https://arxiv.org/pdf/1912.04838)

## LiDAR and Camera Orientation

```
                            FRONT
            FRONT_LEFT                   FRONT_RIGHT
                        Z   
                        |   Y
                        |  /
           SIDE_LEFT    | /               SIDE_RIGHT
                        |/
                        *------------>X
```

## Dataset Download
[Download Perception Dataset v1.1](https://waymo.com/open/download/)

The dataset contains `.tfrecord` files, each including 196 continuous frames of a particular scene.

## Data Extraction
While Waymo provides tutorials for extracting images, point clouds, and calibration data, they don't offer complete data extraction methods (e.g., intensities and other point-cloud features). A Python notebook has been created to extract the data and store it in KITTI dataset format.

![Object Detector](https://github.com/rajeev-gupta-bashrc/OBJECT_DETECTOR/blob/main/images/waymo_transforms.png)

## Calibration Notes
- In the Waymo tfrecord, the rotation from LiDAR to camera frame is actually from camera to LiDAR, while the translational data is correct.
- Transformation matrices from LiDAR to camera are calculated internally.
- Camera coordinates in the tfrecord are not in the general camera coordinate orientation; they are internally rotated to the general orientation.

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

## Data Conversion: Waymo to KITTI Format

### Directory Structure
```
data
    ├──raw_data
        ├──*.tfrecord files
    ├── waymo
```

### Setup Instructions

1. Open the Python notebook
2. Select the Python kernel with TensorFlow and Waymo package installed
3. Use a Python 3.10 conda environment to run the notebook

To install the conda environment:

```bash
cd scripts/object_detector/graphce
conda env create -f environment.yaml
```

You can now create the Waymo data using this conda environment.