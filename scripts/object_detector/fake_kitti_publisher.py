#!/usr/bin/env python3
import rospy
from object_detector.msg import ImagePointCloudCalib
from sensor_msgs.msg import Image
import numpy as np
import cv2, os
from cv_bridge import CvBridge

## pkg path added to path to detect the modules to be loaded while ros node execution
pkg_path = '/home/rajeev-gupta/sensyn_ws/src/object_detector'
module_path = pkg_path + '/scripts/object_detector'
# sys.path.append(module_path)
print('object_detector path added to path ', module_path)

from graphvoi.dataset import calibration_kitti 


class FakeKittiData():
    def __init__(self, data_path, num_cameras=None) -> None:
        self.data_path = data_path
        self.cam_map = ['']
        self.num_cameras = num_cameras if num_cameras else len(self.cam_map)

    def get_calib_dict(self, calib_path):
        calib_dict = calibration_kitti.get_calib_from_file(calib_path)
        return calib_dict

    def get_image(self, image_path):
        image = cv2.imread(image_path).astype(np.uint8)
        return image

    def get_lidar(self, lidar_path):
        lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        return lidar

    def create_kitti_message(self, points, image_all, calib_dict):
        bridge = CvBridge()
        msg = ImagePointCloudCalib()
        '''
        # ImagePointCloudCalib.msg

        float32[] points
        uint32 num_point_features
        sensor_msgs/Image[] image_all

        float32[] projection_all
        uint32 projection_rows 
        uint32 projection_cols 

        float32[] transformation_all
        uint32 transformation_rows 
        uint32 transformation_cols 

        float32[] rectification_all
        uint32 rectification_rows 
        uint32 rectification_cols 

        uint32 num_cameras
        '''
        msg.points = points.flatten()
        msg.num_point_features = points.shape[1]

        msg.image_all = [bridge.cv2_to_imgmsg(image, encoding="bgr8") for image in image_all]
        # msg.image_all = bridge.cv2_to_imgmsg(image_all[0], encoding="bgr8")

        msg.projection_all = np.array(calib_dict['P2']).flatten().astype(np.float32)
        msg.transformation_all = np.array(calib_dict['Tr_velo2cam']).flatten().astype(np.float32)
        msg.rectification_all = np.array(calib_dict['R0']).flatten().astype(np.float32)
        msg.projection_rows = 3
        msg.projection_cols = 4
        msg.transformation_rows = 3
        msg.transformation_cols = 4
        msg.rectification_rows = 3
        msg.rectification_cols = 3
        
        msg.num_cameras = self.num_cameras
        return msg

    def generate_data_from_index(self, index):
        calib_path = os.path.join(self.data_path, 'calib', f'{index:06d}.txt')
        calib_dict = self.get_calib_dict(calib_path)
        lidar_path = os.path.join(self.data_path, 'velodyne', f'{index:06d}.bin')
        points = self.get_lidar(lidar_path)
        image_all = []
        for camera in self.cam_map[:self.num_cameras]:
            image_path = os.path.join(self.data_path, 'image_2', camera, f'{index:06d}.png')
            image_all.append(self.get_image(image_path))
        return points, image_all, calib_dict
    
    
    
if __name__ == '__main__':
    rospy.init_node('fake_kitti_message_publisher')
    pub = rospy.Publisher('kitti_data_topic', ImagePointCloudCalib, queue_size=10)
    fps = 2
    rate = rospy.Rate(fps) 
    data_path = '/media/rajeev-gupta/Drive250/SENSYN_/data/kitti/'
    segment_folders = ['testing', 'training']
    
    ## Dataset
    which_segment = 0
    kitti_data = FakeKittiData(
        data_path=data_path+segment_folders[which_segment]
    )
        
    ##generate all segment data
    num_frames = len(os.listdir(os.path.join(data_path, segment_folders[which_segment], 'calib')))
    while not rospy.is_shutdown():
        for frame_index in range(num_frames):
            rospy.loginfo('Publishing frame index: %d' % frame_index)
            points, image_all, calib_dict = kitti_data.generate_data_from_index(index=frame_index)
            msg = kitti_data.create_kitti_message(points=points, image_all=image_all, calib_dict=calib_dict)
            pub.publish(msg)
            rate.sleep()
        
        
