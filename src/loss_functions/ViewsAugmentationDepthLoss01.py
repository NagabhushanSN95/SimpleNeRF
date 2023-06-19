# Shree KRISHNAya Namaha
# Depth MSE loss function between Main Coarse NeRF and Views Augmentation.
# Author: Nagabhushan S N
# Last Modified: 15/06/2023

import torch
from torch import Tensor
from pathlib import Path

from loss_functions import LossUtils01
from loss_functions.LossFunctionParent01 import LossFunctionParent

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class ViewsAugmentationDepthLoss(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.coarse_mlp_needed = 'coarse_mlp' in self.configs['model']
        self.fine_mlp_needed = 'fine_mlp' in self.configs['model']
        self.augmented_coarse_mlp_needed = 'coarse_mlp' in self.configs['model']['views_augmentation']
        self.augmented_fine_mlp_needed = 'fine_mlp' in self.configs['model']['views_augmentation']
        return

    def compute_loss(self, input_dict: dict, output_dict: dict, return_loss_maps: bool = False):
        total_loss = torch.tensor(0).to(input_dict['rays_o'])
        loss_maps = {}

        if self.coarse_mlp_needed and self.augmented_coarse_mlp_needed:
            depth_coarse_main = output_dict['depth_coarse']
            depth_coarse_aug = output_dict['views_augmentation_depth_coarse']
            loss_coarse = self.compute_depth_loss(depth_coarse_main, depth_coarse_aug, return_loss_maps)
            total_loss += loss_coarse['loss_value']
            if return_loss_maps:
                loss_maps = LossUtils01.update_loss_map_dict(loss_maps, loss_coarse['loss_maps'], suffix='coarse')

        if self.fine_mlp_needed and self.augmented_fine_mlp_needed:
            depth_fine_main = output_dict['depth_fine']
            depth_fine_aug = output_dict['views_augmentation_depth_fine']
            loss_fine = self.compute_depth_loss(depth_fine_main, depth_fine_aug, return_loss_maps)
            total_loss += loss_fine['loss_value']
            if return_loss_maps:
                loss_maps = LossUtils01.update_loss_map_dict(loss_maps, loss_fine['loss_maps'], suffix='fine')

        loss_dict = {
            'loss_value': total_loss,
        }

        if return_loss_maps:
            loss_dict['loss_maps'] = {
                this_filename: loss_maps,
            }
        return loss_dict

    @classmethod
    def compute_depth_loss(cls, main_model_value: Tensor, augmented_model_value: Tensor, return_loss_maps: bool) -> dict:
        total_loss, loss_map = cls.compute_mse(main_model_value, augmented_model_value)
        loss_dict = {
            'loss_value': total_loss,
        }
        if return_loss_maps:
            loss_dict['loss_maps'] = {
                this_filename: loss_map,
            }
        return loss_dict

    @staticmethod
    def compute_mse(value_1: Tensor, value_2: Tensor) -> tuple[Tensor, Tensor]:
        error = value_1 - value_2
        loss_map = torch.square(error)
        mse = torch.mean(loss_map)
        return mse, loss_map
