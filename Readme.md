# Object Detector

This is a real-time object detection and visualization package for autonomous driving, integrated with ROS Noetic. It uses pretrained models GraphCE and GraphVoI for real-time object detection, creating bounding boxes around detected objects.

## Model Outputs

Both models output an array of bounding boxes in the format:
```
[centre_X, centre_Y, centre_Z, length, breadth, height, Yaw]
```

Visualization:
- 3D bounding boxes on point-cloud data in Rviz
- 2D projected bounding boxes on images

## Directory Structure
```
obj_det_ws
├── src
    ├── data
        ├── kitti
            ├── testing
                ├── calib
                    ├── 000000.txt 
                    ├── ...
                ├── image
                    ├── 000000.png 
                    ├── ...
                ├── velodyne
                    ├── 000000.bin 
                    ├── ...
            ├── training
                ├── same as testing ...
        ├── waymo
            ├── segment-*
                ├── calib
                    ├── 000000.txt 
                    ├── ...
                ├── image
                    ├── 000000.png 
                    ├── ...
                ├── velodyne
                    ├── 000000.bin 
                    ├── ...
            ├── segment-* ...


    ├── object_detector
        ├── launch
        ├── Readme.md
        ├── src
            ├── object_detector
                ├── __init__.py
                ├── import_me.py
                ├── __pycache__
                    ├── import_me.cpython-38.pyc
        ├── CMakeLists.txt
        ├── msg
            ├── ImagePointCloudCalib.msg
        ├── .vscode
            ├── c_cpp_properties.json
            ├── settings.json
        ├── setup.py
        ├── rviz
            ├── graphce_rviz.rviz
        ├── pip_list.txt
        ├── scripts
            ├── object_detector
                ├── __init__.py
                ├── graphce
                ├── graphvoi
                ├── detector.py
                ├── graphce_detector.py
                ├── graphvoi_detector.py
                ├── fake_kitti_publisher.py
                ├── fake_waymo_publisher.py
        ├── package.xml
        ├── .gitignore
```

## Demo:

![Object Detector](https://github.com/rajeev-gupta-bashrc/OBJECT_DETECTOR/blob/master/images/graphce_demo.gif)

Please go to [videos/graphce_demo_video.mp4](https://github.com/rajeev-gupta-bashrc/OBJECT_DETECTOR/blob/master/videos/graphce_demo_video.mp4) for full sequence video.

## Prerequisites

- Ubuntu 20.04
- Python version 3.8.10 (default in Ubuntu 20)
- [ROS Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu)
- CUDA version 11.1

## Package Structure

The package follows a general ROS structure with scripts in the `scripts` folder. The model name is specified in the script name.

### GraphCE

- [GitHub Repository](https://github.com/Nightmare-n/GraphRCNN)
- Dataset: Waymo [Readme](https://github.com/rajeev-gupta-bashrc/OBJECT_DETECTOR/blob/master/scripts/object_detector/graphce/waymo_data.md)
- Input: PointCloud (PCD) only
- Output: BBOX, visualization on Image and PCD in Rviz
- Detection Classes: `['VEHICLE', 'PEDESTRIAN', 'CYCLIST']` <==> `[0, 1, 2]`
- Continuous Frames: Yes

### GraphVoI

- [GitHub Repository](https://github.com/Nightmare-n/GD-MAE)
- Dataset: KITTI [Readme](https://github.com/rajeev-gupta-bashrc/OBJECT_DETECTOR/blob/master/scripts/object_detector/graphvoi/kitti_data.md)
- Input: PointCloud (PCD) + Image
- Output: BBOX, visualization on Image and PCD in Rviz
- Detection Classes: `['VEHICLE']` <==> `[0]`
- Continuous Frames: No

## Installation

```bash
cd ~
mkdir -p obj_det_ws/src
cd obj_det_ws/src
git clone https://github.com/rajeev-gupta-bashrc/object_detector.git
cd ../
catkin_make

# Install Python packages
cd src/object_detector
pip install -r requirements.txt 

# Build GraphCE
cd scripts/object_detector/graphce
python setup.py develop --user

# Build GraphVoI
cd ../graphvoi
python setup.py develop --user
cd ops/dcn && python setup.py develop --user
```

## Usage

Source the workspace:

```bash
cd ~
source ~/obj_det_ws/devel/setup.bash
```

Or add to `.bashrc`:

```bash
echo "source ~/obj_det_ws/devel/setup.bash" >> ~/.bashrc
```

Restart the terminal, then:

Terminal 1:
```bash
roslaunch object_detector launch_rviz.launch
```

Terminal 2:
```bash
roslaunch object_detector publish_data_kitti.launch
```
or
```bash
roslaunch object_detector publish_data_waymo.launch
```

Terminal 3:
For KITTI:
```bash
roslaunch object_detector graphvoi_detector.launch to_cpu:=False
```
For Waymo:
```bash
roslaunch object_detector graphce_detector.launch to_cpu:=False
```

In Rviz, choose the available topics for the respective models. You should see the PCD, image, and bounding box markers on both the point-cloud and the image.

## Node-Topic Structure

The node-topic structure can be visualized using `rqt_graph`.

![Object Detector](https://github.com/rajeev-gupta-bashrc/OBJECT_DETECTOR/blob/master/images/rqt_graph.png)


- `fake_waymo_publisher` reads data from a local path and combines point-cloud, image, and calibration data into a custom msg type, then publishes it.
- The detector node subscribes to this data and processes the PCD, image, and calibration data.
- The detector node publishes PCD data and annotated image data (via model predictions).

Various parameters like fps, num_cameras, etc., can be changed.

The structure for the KITTI detector is similar.

## References:
[GraphRCNN: Towards Accurate 3D Object Detection with Semantic-Decorated Local Graph](https://arxiv.org/pdf/2208.03624)

## Acknowledgement
This project is mainly based on the following codebases. Thanks for their great works!

[GraphRCNN](https://github.com/Nightmare-n/GraphRCNN)
[GD-MAE/GraphVoI](https://github.com/Nightmare-n/GD-MAE)