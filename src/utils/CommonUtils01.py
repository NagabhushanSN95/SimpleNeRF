# Shree KRISHNAya Namaha
# Common Utility Functions
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

from pathlib import Path
from typing import Union

import torch

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def get_device(device):
    """
    Returns torch device object
    :param device: None//0/[0,],[0,1]. If multiple gpus are specified, first one is chosen
    :return:
    """
    if (device is None) or (device == '') or (not torch.cuda.is_available()):
        device = torch.device('cpu')
    else:
        device0 = device[0] if isinstance(device, list) else device
        device = torch.device(f'cuda:{device0}')
    return device


def move_to_device(tensor_data: Union[torch.Tensor, list, dict], device):
    if isinstance(tensor_data, torch.Tensor):
        moved_tensor_data = tensor_data.to(device, non_blocking=True)
    elif isinstance(tensor_data, list):
        moved_tensor_data = []
        for tensor_elem in tensor_data:
            moved_tensor_data.append(move_to_device(tensor_elem, device))
    elif isinstance(tensor_data, dict):
        moved_tensor_data = {}
        for key in tensor_data:
            moved_tensor_data[key] = move_to_device(tensor_data[key], device)
    else:
        moved_tensor_data = tensor_data
    return moved_tensor_data


def reproject(
        points_to_reproject,
        poses_to_reproject_to,
        intrinsics,
):
    """

    Args:
        points_to_reproject: (num_rays, )
        poses_to_reproject_to: (num_poses, 4, 4)
        intrinsics: (num_poses, 3, 3)

    Returns:

    """
    other_views_origins = poses_to_reproject_to[:, :3, 3]
    other_views_rotations = poses_to_reproject_to[:, :3, :3]
    reprojected_rays_d = points_to_reproject - other_views_origins

    # for changing coordinate system conventions
    permuter = torch.eye(3).to(points_to_reproject.device)
    permuter[1:] *= -1
    intrinsics = intrinsics[:1]  # TODO: Do not hard-code. Take intrinsic corresponding to each ray

    pos_2 = (intrinsics @ permuter[None] @ other_views_rotations.transpose(1, 2) @ reprojected_rays_d[..., None]).squeeze()
    pos_2 = pos_2[:, :2] / pos_2[:, 2:]
    return pos_2
