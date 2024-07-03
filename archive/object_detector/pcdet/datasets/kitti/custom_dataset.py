import cv2
from pathlib import Path
import torch.utils.data as torch_data
import pickle
import numpy as np

from pcdet.utils.calibration_kitti import Calibration
from pcdet.datasets.kitti.kitti_utils import calib_to_matricies
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder


class CustomKittiDataset(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = Path(root_path) if root_path is not None else Path(self.dataset_cfg.DATA_PATH)

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )
        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self.cur_epoch = 0
        self._merge_all_iters_to_one_epoch = False

    @property
    def mode(self):
        return 'train' if self.training else 'test'
        
    def __getitem__(self, index):
        info_path = '/media/rajeev-gupta/Drive250/data/kitti/kitti_infos_test.pkl'
        with open(info_path, 'rb') as i_file:
            i_dict = pickle.load(i_file)
        info = i_dict[index]
        
        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            input_dict['points'] = points

        if "image" in get_item_list:
            input_dict['image'] = self.get_image(sample_idx)

        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = calib_to_matricies(calib)

        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape
        # converting to the expected format: bypass of the dataloader
        for key, val in data_dict.items():
            print(key)
            if type(val) == tuple:
                data_dict[key] = list(val)
            elif key == 'points':
                # add a zero column
                n = val.shape[0]
                z_col = np.zeros((n, 1), dtype=float)
                data_dict[key] = np.concatenate((z_col, val), axis = 1)
                continue
            elif key == 'image':
                # transpose (384, 1280, 3) to (3, 384, 1280)
                val_transposed = np.transpose(val, (2, 0, 1))
                data_dict[key] = val_transposed
                # print(data_dict[key].shape)
            elif key == 'transformation_2d_list' or key == 'transformation_2d_params':
                data_dict[key] = [val]
                continue
            data_dict[key] = np.array([data_dict[key]])
        data_dict['batch_size'] = 1
        return data_dict
    
    def get_calib(self, idx):
        calib_file = self.root_path / 'testing' / 'calib' / ('%s.txt' % idx)
        return Calibration(calib_file)

    def get_lidar(self, idx):
        lidar_file = self.root_path / 'testing' / 'velodyne' / ('%s.bin' % idx)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    
    def get_image(self, idx):
        img_file = self.root_path / 'testing' / 'image_2' / ('%s.png' % idx)
        return cv2.imread(str(img_file), cv2.IMREAD_COLOR)
    
    def prepare_data(self, data_dict):
        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )
        return data_dict

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag