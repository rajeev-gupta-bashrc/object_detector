#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion


import numpy as np
import os, cv2


def get_8_point_bbox(point7):
    try:
        bbox7 = point7.cpu().numpy()
    except Exception:
        bbox7 = point7
    theta = -float(bbox7[6])
    cx, cy, cz = float(bbox7[0]), float(bbox7[1]), float(bbox7[2])
    dx, dy, dz = float(bbox7[3]), float(bbox7[4]), float(bbox7[5])
    # theta = 0
    # print(len(bbox7))
    # print(theta)
    hx, hy, hz = dx / 2, dy / 2, dz / 2
    corners = np.array([
        [hx, hy, hz],
        [hx, hy, -hz],
        [hx, -hy, -hz],
        [hx, -hy, hz],
        [-hx, hy, hz],
        [-hx, hy, -hz],
        [-hx, -hy, -hz],
        [-hx, -hy, hz],
    ])    
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    rotated_corners = np.dot(corners, rotation_matrix.T)
    global_corners = rotated_corners + np.array([cx, cy, cz])
    return global_corners


def create_bounding_box_marker(bbox_point, id, rgb = None, namespace = "", duration=2):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = namespace
    marker.id = id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD

    edges = [
        # (0, 1), (1, 2), (2, 3), (3, 0),  # front face YZ
        # (4, 5), (5, 6), (6, 7), (7, 4),  # rear face YZ
        # (0, 4), (1, 5), (2, 6), (3, 7),  # Horizontal lines X
        # (0, 2), (1, 3)                   # Cross in front face
        ##for waymo GraphCE
        (3, 2), (2, 6), (6, 7), (7, 3),  # front face YZ
        (0, 1), (1, 5), (5, 4), (4, 0),  # rear face YZ
        (0, 3), (1, 2), (5, 6), (4, 7),  # Horizontal lines X
        (7, 2), (6, 3)                   # Cross in front face
    ]

    for start, end in edges:
        p1 = Point()
        p1.x, p1.y, p1.z = bbox_point[start]
        marker.points.append(p1)
        p2 = Point()
        p2.x, p2.y, p2.z = bbox_point[end]
        marker.points.append(p2)

    marker.pose.orientation = Quaternion()
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    
    marker.scale.x = 0.05  # Line width
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    if rgb is not None and len(rgb) == 3:
        marker.color.r = rgb[0]
        marker.color.g = rgb[1]
        marker.color.b = rgb[2]
    _duration = rospy.rostime.Duration(duration)
    marker.lifetime = _duration
    return marker


def boxes3d_lidar_to_kitti_camera(boxes3d_lidar, calib):
    """
    :param boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    :param calib:
    :return:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    """
    import copy
    boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
    xyz_lidar = boxes3d_lidar_copy[:, 0:3]
    l, w, h = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:, 4:5], boxes3d_lidar_copy[:, 5:6]
    r = boxes3d_lidar_copy[:, 6:7]

    xyz_lidar[:, 2] -= h.reshape(-1) / 2
    xyz_cam = calib.lidar_to_rect(xyz_lidar)
    # xyz_cam[:, 1] += h.reshape(-1) / 2
    return np.concatenate([xyz_cam, l, h, w, r], axis=-1)



def draw_boxes(image, pred_boxes_cam, cam_to_img):
    import copy
    image_with_boxes = copy.deepcopy(image)
    scope_h, scope_k = image.shape[:2]
    # print('image_scope: ', scope_h, scope_k)
    for line in pred_boxes_cam[:]:
    # for line in [pred_boxes_cam[3]]:
        dims   = np.asarray([float(number) for number in line[3:6]])
        ## swap x, y, only required when reading from GD-MAE test - txt results
        tmp = dims[1]
        dims[1]=dims[0]
        dims[0]=tmp
        center = np.asarray([float(number) for number in line[0:3]])
        # rot_y  = float(line[3]) + np.arctan(center[0]/center[2])
        # rot_y  = float(line[3]) + float(line[6]) + np.arctan(center[0]/center[2])
        # rot_y = float(line[6]) + 1.57
        rot_y = -float(line[6])
        # rot_y = 0
        box_3d = []
        is_bbox_inside_image_scope = True
        for i in [1,-1]:
            for j in [1,-1]:
                for k in [0,1]:
                    point = np.copy(center)
                    point[0] = center[0] + i * dims[1]/2 * np.cos(-rot_y+np.pi/2) + (j*i) * dims[2]/2 * np.cos(-rot_y)
                    point[2] = center[2] + i * dims[1]/2 * np.sin(-rot_y+np.pi/2) + (j*i) * dims[2]/2 * np.sin(-rot_y)                  
                    point[1] = center[1] - k * dims[0]
                    point = np.append(point, 1)
                    point = np.dot(cam_to_img, point)
                    point = point[:2]/point[2]
                    point = point.astype(np.int16)
                    # if point[0]>=scope_h or point[1]>=scope_k or point[0]<0 or point[1]<0:
                    if point[0]<0 or point[1]<0:
                        is_bbox_inside_image_scope = False
                    box_3d.append(point)
        if not is_bbox_inside_image_scope:
            continue
        for i in range(4):
            point_1_ = box_3d[2*i]
            point_2_ = box_3d[2*i+1]
            cv2.line(image_with_boxes, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (0,255,0), 2)
        for i in range(8):
            point_1_ = box_3d[i]
            point_2_ = box_3d[(i+2)%8]
            cv2.line(image_with_boxes, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (0,255,0), 2)
    return image_with_boxes


