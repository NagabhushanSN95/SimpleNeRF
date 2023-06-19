# Shree KRISHNAya Namaha
# Depth MSE loss function between Main Coarse NeRF and Main Fine NeRF. Reprojection error (patch-wise) is employed
# to determine the more accurate depth estimate.
# Author: Nagabhushan S N
# Last Modified: 15/06/2023

import torch
from torch import Tensor
from pathlib import Path
import torch.nn.functional as F

from loss_functions.LossFunctionParent01 import LossFunctionParent
from utils import CommonUtils02 as CommonUtils

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class CoarseFineConsistencyLoss(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.coarse_mlp_needed = 'coarse_mlp' in self.configs['model']
        self.fine_mlp_needed = 'fine_mlp' in self.configs['model']
        self.sparse_depth_needed = 'sparse_depth' in self.configs['data_loader']
        self.px, self.py = self.loss_configs['patch_size']
        self.hpx, self.hpy = [x // 2 for x in self.loss_configs['patch_size']]
        self.rmse_threshold = self.loss_configs['rmse_threshold']
        return

    def compute_loss(self, input_dict: dict, output_dict: dict, return_loss_maps: bool = False):
        total_loss = torch.tensor(0).to(input_dict['rays_o'])

        # Loss requires both coarse and fine models
        if not self.coarse_mlp_needed or not self.fine_mlp_needed:
            loss_dict = {'loss_value': total_loss}
            return loss_dict

        rays_o = input_dict['rays_o']
        rays_d = input_dict['rays_d']
        depths_coarse = output_dict['depth_coarse']
        depths_fine = output_dict['depth_fine']
        gt_poses = input_dict['common_data']['poses']  # (num_views, 4, 4)
        gt_images = input_dict['common_data']['images']
        intrinsics = input_dict['common_data']['intrinsics']
        resolution = input_dict['common_data']['resolution']
        indices_mask_nerf = input_dict['indices_mask_nerf']
        indices_mask_sd = input_dict.get('indices_mask_sparse_depth', None)
        pixel_ids = input_dict['pixel_id'].long()

        loss = self.compute_depth_loss(depths_coarse, depths_fine, indices_mask_nerf, indices_mask_sd,
                                       rays_o, rays_d, gt_poses, gt_images,
                                       pixel_ids, intrinsics, resolution, return_loss_maps)
        total_loss += loss['loss_value']
        if return_loss_maps:
            loss_maps = loss['loss_maps']

        loss_dict = {
            'loss_value': total_loss,
        }

        if return_loss_maps:
            loss_dict['loss_maps'] = loss_maps
        return loss_dict

    def compute_depth_loss(self, depth_coarse, depth_fine, indices_mask_nerf, indices_mask_sd,
                           rays_o, rays_d, gt_poses, gt_images, pixel_ids, intrinsics, resolution,
                           return_loss_maps: bool) -> dict:
        total_loss = 0

        loss_nerf, loss_map_nerf_coarse, loss_map_nerf_fine = self.compute_loss_nerf(
            depth_coarse, depth_fine, indices_mask_nerf,
            rays_o, rays_d, gt_poses, gt_images, pixel_ids, intrinsics, resolution
        )
        total_loss += loss_nerf

        if self.sparse_depth_needed:
            loss_sd, loss_map_sd_coarse = self.compute_loss_sd(depth_coarse, depth_fine, indices_mask_sd)
            total_loss += loss_sd

        loss_dict = {
            'loss_value': total_loss
        }
        if return_loss_maps:
            loss_dict['loss_maps'] = {
                f'{this_filename}_coarse': loss_map_nerf_coarse,
                f'{this_filename}_fine': loss_map_nerf_fine,
            }
            if self.sparse_depth_needed:
                loss_dict['loss_maps'][f'{this_filename}_coarse'] += loss_map_sd_coarse
        return loss_dict

    def compute_loss_nerf(self, depth1, depth2, indices_mask_nerf, rays_o, rays_d, gt_poses, gt_images,
                          pixel_ids, intrinsics, resolution) -> tuple[Tensor, Tensor, Tensor]:
        """
        Computes the loss for nerf samples (and not for sparse_depth or any other samples)

        Naming convention
        1, 2 -> refers to two different models
        a, b -> refers to source view and the other reprojection view

        :param depth1:
        :param depth2:
        :param indices_mask_nerf:
        :param rays_o:
        :param rays_d:
        :param gt_poses:
        :param gt_images:
        :param pixel_ids:
        :param intrinsics:
        :param resolution:
        :return:
        """
        h, w = resolution
        image_ids = pixel_ids[:, 0]

        rays_o = rays_o[indices_mask_nerf]
        rays_d = rays_d[indices_mask_nerf]
        depth1 = depth1[indices_mask_nerf]
        depth2 = depth2[indices_mask_nerf]

        gt_origins = gt_poses[:, :3, 3]
        distances = torch.sqrt(torch.sum(torch.square(gt_origins[image_ids].unsqueeze(1).repeat([1, gt_origins.shape[0], 1]) - gt_origins), dim=2))
        # Taking second smallest value as the smallest distance will always be with the same view at 0.0. Kth value
        # by default randomly returns one index if two distances are the same. Which works for our use-case.
        closest_image_ids = torch.kthvalue(distances, 2, dim=1)[1]

        image_ids_a = image_ids[indices_mask_nerf]
        pixel_ids_a = pixel_ids[indices_mask_nerf]
        image_ids_b = closest_image_ids[indices_mask_nerf]

        poses_b = gt_poses[image_ids_b]
        points1a = rays_o + rays_d * depth1.unsqueeze(-1)
        points2a = rays_o + rays_d * depth2.unsqueeze(-1)

        pos1b = CommonUtils.reproject(points1a.detach(), poses_b, intrinsics).round().long()
        pos2b = CommonUtils.reproject(points2a.detach(), poses_b, intrinsics).round().long()

        x_a, y_a = pixel_ids_a[:, 1], pixel_ids_a[:, 2]
        x1b, y1b = pos1b[:, 0], pos1b[:, 1]
        x2b, y2b = pos2b[:, 0], pos2b[:, 1]

        # Ignore reprojections that were set outside the image
        valid_mask_a = (x_a >= self.hpx) & (x_a < w - self.hpx) & (y_a >= self.hpy) & (y_a < h - self.hpy)
        valid_mask_1b = (x1b >= self.hpx) & (x1b < w - self.hpx) & (y1b >= self.hpy) & (y1b < h - self.hpy)
        valid_mask_2b = (x2b >= self.hpx) & (x2b < w - self.hpx) & (y2b >= self.hpy) & (y2b < h - self.hpy)

        x1b1, y1b1 = torch.clip(x1b, 0, w - 1).long(), torch.clip(y1b, 0, h - 1).long()
        x2b1, y2b1 = torch.clip(x2b, 0, w - 1).long(), torch.clip(y2b, 0, h - 1).long()
        patches_a = torch.zeros(image_ids_a.shape[0], self.py, self.px, gt_images.shape[3]).to(image_ids_a.device)  # (nr, py, px, 3)
        patches1b = torch.zeros(image_ids_b.shape[0], self.py, self.px, gt_images.shape[3]).to(image_ids_b.device)  # (nr, py, px, 3)
        patches2b = torch.zeros(image_ids_b.shape[0], self.py, self.px, gt_images.shape[3]).to(image_ids_b.device)  # (nr, py, px, 3)
        gt_images_padded = F.pad(gt_images, (0, 0, 0, self.hpy, 0, self.hpx), mode='constant', value=0)
        for i, y_offset in enumerate(range(-self.hpy, self.hpy + 1)):  # y_offset: [-2, -1, 0, 1, 2]
            for j, x_offset in enumerate(range(-self.hpx, self.hpx + 1)):
                patches_a[:, i, j, :] = gt_images_padded[image_ids_a, y_a + y_offset, x_a + x_offset]
                patches1b[:, i, j, :] = gt_images_padded[image_ids_b, y1b1 + y_offset, x1b1 + x_offset]
                patches2b[:, i, j, :] = gt_images_padded[image_ids_b, y2b1 + y_offset, x2b1 + x_offset]

        rmse1 = self.compute_patch_rmse(patches_a, patches1b)
        rmse2 = self.compute_patch_rmse(patches_a, patches2b)

        # mask1 is true wherever model1 is more accurate
        mask1 = ((rmse1 < rmse2) | (~valid_mask_2b)) & (rmse1 < self.rmse_threshold) & valid_mask_1b & valid_mask_a
        # mask2 is true wherever model2 is more accurate
        mask2 = ((rmse2 < rmse1) | (~valid_mask_1b)) & (rmse2 < self.rmse_threshold) & valid_mask_2b & valid_mask_a

        # depth_mse1 is loss on depth1; depth_mse2 is loss on depth2
        depth_mse1, depth_mse_map1 = self.compute_depth_mse(depth1, depth2.detach(), mask2)  # (nr, )
        depth_mse2, depth_mse_map2 = self.compute_depth_mse(depth2, depth1.detach(), mask1)
        loss = depth_mse1 + depth_mse2
        return loss, depth_mse_map1, depth_mse_map2

    def compute_loss_sd(self, depths_coarse: Tensor, depths_fine: Tensor, indices_mask_sd: Tensor)\
            -> tuple[Tensor, Tensor]:
        """
        Since the fine model is supervised by sparse depth, we assume fine depth is correct and use it to supervise coarse depth
        :param depths_coarse:
        :param depths_fine:
        :param indices_mask_sd:
        :return:
        """
        if indices_mask_sd is None:
            return 0, 0
        depths_coarse = depths_coarse[indices_mask_sd]
        depths_fine = depths_fine[indices_mask_sd]

        loss, loss_map = self.compute_depth_mse(depths_coarse, depths_fine.detach())
        return loss, loss_map

    @classmethod
    def compute_patch_rmse(cls, patch1: Tensor, patch2: Tensor) -> Tensor:
        """

        Args:
            patch1: (num_rays, patch_size, patch_size, 3)
            patch2: (num_rays, patch_size, patch_size, 3)

        Returns:
            rmse: (num_rays, )

        """
        rmse = torch.sqrt(torch.mean(torch.square(patch1 - patch2), dim=(1, 2, 3)))
        return rmse

    @classmethod
    def compute_depth_mse(cls, pred_depth: Tensor, gt_depth: Tensor, mask: Tensor = None) -> tuple[Tensor, Tensor]:
        """

        Args:
            pred_depth: (num_rays, )
            gt_depth: (num_rays, )
            mask: (num_rays, ); Loss is computed only where mask is True

        Returns:

        """
        zero_tensor = torch.tensor(0).to(pred_depth)
        if mask is not None:
            pred_depth[~mask] = 0
            gt_depth[~mask] = 0
        loss_map = torch.square(pred_depth - gt_depth)
        mse = torch.mean(loss_map) if pred_depth.numel() > 0 else zero_tensor
        return mse, loss_map
