#!/usr/bin/env python3
import rospy 

import sys, os, time
import torch
import numpy as np
import logging
import threading

## pkg path added to path to detect the modules to be loaded while ros node execution
pkg_path = '/home/rajeev-gupta/sensyn_ws/src/object_detector'
module_path = pkg_path + '/scripts/object_detector'
# sys.path.append(module_path)
# print('object_detector path added to path ', module_path)

from graphce.det3d.torchie import Config
# from graphce.det3d.datasets import build_dataset
from graphce.det3d.models import build_detector
from graphce.det3d.torchie.apis import (
    batch_processor
)
from graphce.det3d.torchie.trainer import load_checkpoint

from object_detector.msg import ImagePointCloudCalib
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg


from graphce.tools import calibration_waymo 
from graphce.tools import bbox_points_utils 


import copy
from det3d.utils import build_from_cfg
from det3d.datasets.dataset_wrappers import ConcatDataset, RepeatDataset
from det3d.datasets.waymo.waymo import WaymoDataset
from det3d.datasets.registry import DATASETS
def build_dataset(cfg, default_args=None):
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg["type"] == "RepeatDataset":
        dataset = RepeatDataset(
            build_dataset(cfg["dataset"], default_args), cfg["times"]
        )
    # elif isinstance(cfg['ann_file'], (list, tuple)):
    #     dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset




class GraphCE_Detector():
    def __init__(self, to_cpu=False, 
                    cfg_path = os.path.join(module_path, 'graphce/configs/waymo_centerpoint_voxelnet_graphrcnn_6epoch_freeze_copy.py')
                 ) -> None:
        self.cfg_path = cfg_path
        self.cfg = self.get_cfg()
        self.logger = self.get_logger()
        self.dataset = self.init_dataset()
        self.to_cpu = to_cpu
        self.ckpt = None
        self.model = self.init_model()
        
        self.point_cloud = None
        self.bbox_markers = []
        self.bbox_lifetime = 0.100                  #seconds
        self.pc_and_image_lifetime = 1
        self.image_with_boxes = []
        self.lock = threading.Lock()
        
        rospy.init_node('graphce_object_detector')
        self.bbox_pub = rospy.Publisher('graphce_bbox_publisher', Marker, queue_size=10)
        self.image_pub = rospy.Publisher('graphce_annotated_image', Image, queue_size=5)
        self.point_cloud_pub = rospy.Publisher('point_cloud', PointCloud2, queue_size=5)
        
        self.publisher_thread_all = [
            threading.Thread(target=self.publish_markers),
            threading.Thread(target=self.publish_pc_and_image),
            ]
        self.bbox_pub_rate = rospy.Rate(1/self.bbox_lifetime)
        self.pc_and_image_pub_rate = rospy.Rate(1/self.pc_and_image_lifetime)
        
        
    def publish_markers(self):
        while not rospy.is_shutdown():
            self.lock.acquire()
            if len(self.bbox_markers) != 0:
                self.marker_pub_start_time = time.time()
                for i in range(len(self.bbox_markers)):
                    self.bbox_pub.publish(self.bbox_markers[i])
                self.marker_pub_end_time = time.time()
                rospy.loginfo('Marker published at %f, elapsed time: %f ' % (self.marker_pub_start_time,
                                                                             self.marker_pub_end_time-self.marker_pub_start_time) )
            else:
                rospy.loginfo('BBOX is empty')
            self.lock.release()
            self.bbox_pub_rate.sleep()
        
    def publish_pc_and_image(self):
        while not rospy.is_shutdown():
            self.lock.acquire()
            if self.point_cloud is not None:
                self.pc_pub_start_time = time.time()
                self.point_cloud_pub.publish(self.point_cloud)
                self.pc_pub_end_time = time.time()
                rospy.loginfo('PC published at %f, elapsed time: %f ' % (self.pc_pub_start_time,
                                                                             self.pc_pub_end_time-self.pc_pub_start_time) )
                # rospy.loginfo('sending PC')
            else:
                rospy.loginfo('Point cloud is None')
            if len(self.image_with_boxes) != 0:
                # rospy.loginfo('sending images')
                self.image_pub_start_time = time.time()
                for i in range(len(self.image_with_boxes)):
                    self.image_pub.publish(self.image_with_boxes[i])
                self.image_pub_end_time = time.time()
                rospy.loginfo('Image published at %f, elapsed time: %f ' % (self.image_pub_start_time,
                                                                             self.image_pub_end_time-self.image_pub_start_time) )
            else:
                rospy.loginfo('Images are empty')
            self.lock.release()
            self.bbox_pub_rate.sleep()
        
    def start_threading_publishers(self):
        for threads in self.publisher_thread_all:
            threads.start()
        
        
    def start_subscribing(self):
        self.sub_waymo_data = rospy.Subscriber('waymo_data_topic', ImagePointCloudCalib, detector.waymo_message_callback)
        
        
    def init_model(self):
        with torch.no_grad():
            model = build_detector(self.cfg.model, train_cfg=None, test_cfg=self.cfg.test_cfg)
            device = 'cpu' if ( self.to_cpu and torch.cuda.is_available() ) else 'cuda:0'
            model.to(device=device)
            model.eval()
            self.ckpt = load_checkpoint(model, self.cfg.load_from, device, strict=False, logger=self.logger)
        return  model

    def init_dataset(self):
        dataset = build_dataset(self.cfg.data.val)
        return dataset

    def get_logger(self):
        logger = logging.getLogger()
        return logger

    def get_cfg(self, local_rank=0):
        cfg = Config.fromfile(self.cfg_path)
        cfg.local_rank = local_rank
        return cfg

    def get_outputs(self, data, local_rank=0, train_mode=False):
        start_time = time.time()
        try:
            outputs = batch_processor(
                        self.model, data, train_mode=train_mode, local_rank=local_rank,
                    )
        except Exception as E:
            rospy.loginfo('Unable to get predictions, here is the issue')
            rospy.loginfo(E)
            del self.model, self.ckpt
            self.model = None
            self.ckpt = None
            outputs = [
                {
                'box3d_lidar': np.array([[0, 0, 0, 0, 0, 0, 0]]),
                'scores': np.array([0]),
                'label_preds': np.array([0])
                }
            ]
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            end_time = time.time()
            pred_boxes = outputs[0]['box3d_lidar'].cpu().detach().numpy()
            scores = outputs[0]['scores'].cpu().detach().numpy()
            label_preds = outputs[0]['label_preds'].cpu().detach().numpy()
            del outputs
        except:
            end_time = time.time()
            pred_boxes = outputs[0]['box3d_lidar']
            scores = outputs[0]['scores']
            label_preds = outputs[0]['label_preds']   
        self.logger.info('Completed prediction in %f seconds' % (end_time-start_time))
        return pred_boxes, scores, label_preds

    def generate_feedable_data(self, points):
        data = self.dataset.__getitem__(0, points)
        data['metadata'] = {'image_prefix': '', 'num_point_features': 5, 'token': ''}
        data['points'] = [torch.Tensor(data['points'])]
        data['metadata'] = [data['metadata']]
        return data
    
    def get_markers_from_preds(self, pred_boxes):
        bbox8_all = [bbox_points_utils.get_8_point_bbox(bbox7) for bbox7 in pred_boxes]
        return [bbox_points_utils.create_bounding_box_marker(
                                bbox8, id, namespace = 'graphce_pred_boxes', duration=self.bbox_lifetime * 1.5
                            ) for id, bbox8 in enumerate(bbox8_all)]
    
    def annotate_image_all(self, pred_boxes, image_all, calib_obj_all):
        bridge = CvBridge()
        for i, image in enumerate(image_all):
            cam_to_img = calib_obj_all[i].P2
            pred_boxes_cam = bbox_points_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib_obj_all[i])
            # convert cv2 image to sensor_msgs/Image
            image_all[i] = bridge.cv2_to_imgmsg(
                                bbox_points_utils.draw_boxes(image, pred_boxes_cam, cam_to_img), 
                                encoding="bgr8")
        return image_all
    
    def generate_point_cloud(self, points):
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        point_cloud = pc2.create_cloud_xyz32(header, points[:, :3])
        return point_cloud
    
    def waymo_message_callback(self, msg):
        '''
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
        # empty markers
        # self.lock.acquire()
        # self.bbox_markers = []
        # self.lock.release()
        self.data_callback_start_time = time.time()
        points = np.array(msg.points, dtype=np.float32).reshape(-1, msg.num_point_features)
        # rospy.loginfo(f"Received points: {points.shape}")

        # Process image
        bridge = CvBridge()
        image_all = []
        try:
            image_all = [bridge.imgmsg_to_cv2(image, desired_encoding="bgr8") for image in msg.image_all]
            # rospy.loginfo(f"Received images with shape: {image_all[0].shape}")
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert image: {e}")

        # Process calibration matrix
        projection_matrices = np.array(msg.projection_all, dtype=np.float32).reshape(-1, msg.projection_rows, msg.projection_cols)
        transformation_matrices = np.array(msg.transformation_all, dtype=np.float32).reshape(-1, msg.transformation_rows, msg.transformation_cols)
        rectification_matrices = np.array(msg.rectification_all, dtype=np.float32).reshape(-1, msg.rectification_rows, msg.rectification_cols)
        
        assert projection_matrices.shape[0] == transformation_matrices.shape[0]
        calib_obj_all = [calibration_waymo.Calibration(
                            {
                                'P0': projection_matrices[i], 
                                'R0': rectification_matrices[i],
                                'Tr_velo_to_cam': transformation_matrices[i]
                            }
                        ) for i in range(msg.num_cameras)]
        
        # get preds
        self.model_processing_start_time = time.time()
        with torch.no_grad():
            pred_boxes, scores, label_preds = self.get_outputs(self.generate_feedable_data(points=points))
        self.model_processing_end_time = time.time()
        # update bbox markers
        self.lock.acquire()
        rospy.loginfo('Predictions Completed')
        self.bbox_markers = self.get_markers_from_preds(pred_boxes)
        self.image_with_boxes = self.annotate_image_all(calib_obj_all=calib_obj_all, image_all=image_all, pred_boxes=pred_boxes)
        self.point_cloud = self.generate_point_cloud(points=points)
        self.lock.release()
        self.data_callback_end_time = time.time()
        rospy.loginfo('Data callback at %f with the Processing time %f, whereas model took : %f seconds' % (
            self.data_callback_start_time, 
            self.data_callback_end_time-self.data_callback_start_time, 
            self.model_processing_end_time - self.model_processing_start_time
        ))


if __name__=='__main__':
    detector = GraphCE_Detector(to_cpu=False)
    detector.start_subscribing()
    detector.start_threading_publishers()
    rospy.spin()
    