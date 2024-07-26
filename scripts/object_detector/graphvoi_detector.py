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


## modules
from graphvoi.dataset.custom_dataset import CustomKittiDataset
from graphvoi.dataset import calibration_kitti
from graphvoi.model.graph_rcnn import GraphRCNN
from graphvoi.config import cfg, cfg_from_yaml_file
from graphvoi.utils.common_utils import create_logger
from graphvoi.model import load_data_to_gpu


from object_detector.msg import ImagePointCloudCalib
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg


from graphce.tools import bbox_points_utils 


class GraphVoI_Detector():
    def __init__(self, to_cpu=False, 
                    cfg_path = os.path.join(module_path, 'graphvoi/config/kitti_models/graph_rcnn_voi.yaml')
                 ) -> None:
        self.cfg_path = cfg_path
        self.cfg = self.get_cfg()
        self.log_path = './test_logs.txt'
        self.logger = self.get_logger()
        self.dataset = self.init_dataset()
        self.to_cpu = to_cpu
        self.ckpt_path = os.path.join(module_path, 'graphvoi/config/ckpts/graph_rcnn_voi_kitti.pth')
        self.model = self.init_model()
        
        self.point_cloud = None
        self.bbox_markers = []
        self.bbox_lifetime = 0.100                  #seconds
        self.pc_and_image_lifetime = 1
        self.image_with_boxes = []
        self.lock = threading.Lock()
        
        rospy.init_node('graphvoi_object_detector')
        self.bbox_pub = rospy.Publisher('graphvoi_bbox_publisher', Marker, queue_size=10)
        self.image_pub = rospy.Publisher('graphvoi_annotated_image', Image, queue_size=5)
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
        self.sub_kitti_data = rospy.Subscriber('kitti_data_topic', ImagePointCloudCalib, detector.kitti_message_callback)
        
        
    def init_model(self):
        model = GraphRCNN(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.dataset, logger=self.logger)
        with torch.no_grad():
            model.load_params_from_file(filename=self.ckpt_path, logger=self.logger, to_cpu=self.to_cpu)
            model.cuda()
            model.eval()    
            torch.cuda.synchronize()
        # model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params}")
        self.logger.info(f"Trainable parameters: {trainable_params}")
        self.logger.info(f"Non-trainable parameters: {total_params - trainable_params}")
        return  model

    def init_dataset(self):
        dataset = CustomKittiDataset(
            dataset_cfg=self.cfg.DATA_CONFIG,
            class_names=self.cfg.CLASS_NAMES,
            root_path=None,
            training=False,
            logger=self.logger,
        )
        return dataset

    def get_logger(self):
        logger = create_logger(self.log_path)
        return logger

    def get_cfg(self, local_rank=0):
        cfg_ = cfg_from_yaml_file(self.cfg_path, cfg)
        return cfg_

    def get_outputs(self, data, local_rank=0, train_mode=False):
        start_time = time.time()
        try:
            load_data_to_gpu(data)
            with torch.no_grad():
                outputs, ret_dict = self.model(data)
            torch.cuda.synchronize()
        except Exception as E:
            rospy.loginfo('Unable to get predictions, here is the issue')
            rospy.loginfo(E)
            del self.model
            self.model = None
            outputs = [
                {
                'pred_boxes': np.array([[0, 0, 0, 0, 0, 0, 0]]),
                'pred_scores': np.array([0]),
                'pred_labels': np.array([0])
                }
            ]
        try:
            torch.cuda.empty_cache()
            end_time = time.time()
            pred_boxes = outputs[0]['pred_boxes'].cpu().detach().numpy()
            scores = outputs[0]['pred_scores'].cpu().detach().numpy()
            label_preds = outputs[0]['pred_labels'].cpu().detach().numpy()
            del outputs
        except:
            end_time = time.time()
            pred_boxes = outputs[0]['pred_boxes']
            scores = outputs[0]['pred_scores']
            label_preds = outputs[0]['pred_labels']   
        self.logger.info('Completed prediction in %f seconds' % (end_time-start_time))
        return pred_boxes, scores, label_preds

    def generate_feedable_data(self, points, image_all, calib_obj_all):
        data = self.dataset.__getitem__(points, image_all, calib_obj_all)
        return data
    
    def get_markers_from_preds(self, pred_boxes):
        bbox8_all = [bbox_points_utils.get_8_point_bbox(bbox7) for bbox7 in pred_boxes]
        return [bbox_points_utils.create_bounding_box_marker(
                                bbox8, id, namespace = 'graphvoi_pred_boxes', duration=self.bbox_lifetime * 1.5
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
    
    def kitti_message_callback(self, msg):
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
        calib_obj_all = [calibration_kitti.Calibration(
                            {
                                'P2': projection_matrices[i], 
                                'R0': rectification_matrices[i],
                                'Tr_velo2cam': transformation_matrices[i]
                            }
                        ) for i in range(msg.num_cameras)]
        
        # get preds
        self.model_processing_start_time = time.time()
        pred_boxes, scores, label_preds = self.get_outputs(self.generate_feedable_data(points=points, image_all=image_all, calib_obj_all=calib_obj_all))
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
    detector = GraphVoI_Detector(to_cpu=False)
    detector.start_subscribing()
    detector.start_threading_publishers()
    rospy.spin()
    