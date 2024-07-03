import torch
import numpy as np


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib', 'image_shape', 'image_pad_shape', 'image_rescale_shape']:
            continue
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()
