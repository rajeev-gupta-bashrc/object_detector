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

def print_model_info(model, logger):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params}")
    
    # Optionally, print more detailed information
    # print("\nLayer-wise details:")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: {param.numel()} parameters")



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
    pred_info = {'inference_time' : end_time-start_time}
    return pred_dicts, ret_dict, pred_info

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
    prepare_model(model, module_path+ckpt_path, logger, to_cpu)
    print_model_info(model, logger)
    
    ## single test 
    # data_input = dataset.__getitem__(3)
    # test_output(model, data_input, logger)
    
    ## user commanded test
    # while(1):
    #     try:
    #         index = int(input('\n\nGive an index of data input, or neg decimal to break: '))
    #     except Exception as E:
    #         print('Unable to fetch index, try again....')
    #         continue
    #     if index < 0: break
    #     data_input = dataset.__getitem__(index)
    #     test_output(model, data_input, logger)
      
    ## random 10 test for avg time  
    import random
    random_numbers = [random.randint(0, 50) for _ in range(25)]
    inference_time = []
    processing_time = []
    for index in random_numbers:
        data_input = dataset.__getitem__(index)
        processing_time.append(data_input['processing_time'])
        pred_dict, ret_dict, pred_info = test_output(model, data_input, logger)
        inference_time.append(pred_info['inference_time'])
    # print(inference_time)
    avg_inf = sum(inference_time)/len(inference_time)
    avg_prs = sum(processing_time)/len(processing_time)
    logger.info('average inference time: %f' % avg_inf)
    logger.info('average processing time: %f' % avg_prs)

    