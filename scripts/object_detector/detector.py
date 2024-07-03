#!/usr/bin/env python3
import rospy 

import sys, os, time
import torch
import numpy as np

## pkg path added to path to detect the modules to be loaded while ros node execution
pkg_path = '/home/rajeev-gupta/sensyn_ws/src/object_detector'
module_path = pkg_path + '/scripts/object_detector'
print('object_detector path added to path ', module_path)
sys.path.append(module_path)

## modules
# from graphvoi.module import call
# call()
from graphvoi.dataset.custom_dataset import CustomKittiDataset
from graphvoi.model.graph_rcnn import GraphRCNN
from graphvoi.config import cfg, cfg_from_yaml_file
from graphvoi.utils.common_utils import create_logger
from graphvoi.model import load_data_to_gpu

# from pcdet.datasets.kitti.custom_dataset import CustomKittiDataset
# from pcdet.models.detectors.graph_rcnn import GraphRCNN
# from pcdet.config import cfg, cfg_from_yaml_file
# from pcdet.utils.common_utils import create_logger
# from pcdet.models import load_data_to_gpu


def prepare_model(model, ckpt_path_full, logger, to_cpu):   
    with torch.no_grad():
        model.load_params_from_file(filename=ckpt_path_full, logger=logger, to_cpu=to_cpu)
        model.cuda()
        model.eval()    
        torch.cuda.synchronize()

def test_output(model, data_input, logger):
    torch.cuda.synchronize()
    start_time = time.time()
    load_data_to_gpu(data_input)
    with torch.no_grad():
        pred_dicts, ret_dict = model(data_input)
    torch.cuda.synchronize()
    end_time = time.time()
    logger.info('Inference Time: %f' % (end_time-start_time))
    logger.info('Predicted dicts')
    logger.info(pred_dicts)
    logger.info('ret_dicts')
    logger.info(ret_dict)

if __name__=='__main__':
    rospy.init_node('object_detector')
    rospy.loginfo('object_detector Node is running')
    ## directories
    log_path = './test_logs.txt'
    # relative paths wrt tools
    cfg_file  = '/graphvoi/config/kitti_models/graph_rcnn_voi.yaml'
    ckpt_path = '/graphvoi/config/ckpts/graph_rcnn_voi_kitti.pth'
    to_cpu = False

    # create logger
    logger = create_logger(log_path)

    # load config to the cfg object 
    cfg_from_yaml_file(module_path+cfg_file, cfg)

    # create dataset object 
    dataset = CustomKittiDataset(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            root_path='/home/rajeev-gupta/sensyn_ws/src/GD-MAE/data/kitti',     #only to use direct data, 
            training=False,
            logger=logger,
        )

    # create model object
    model = GraphRCNN(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset, logger=logger)

    # get a test object
    data_input = dataset.__getitem__(3)
    prepare_model(model, module_path+ckpt_path, logger, to_cpu)
    test_output(model, data_input, logger)