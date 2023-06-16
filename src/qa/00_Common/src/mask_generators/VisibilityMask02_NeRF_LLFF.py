# Shree KRISHNAya Namaha
# Computes visibility mask for test frames based on PoseWarping. Indicates whether a pixel in test view is visible in
# each of the training view or not. Uses depth given by dense input NeRF
# Authors: Nagabhushan S N
# Last Modified: 15/06/2023

import json
import time
import datetime
import traceback
from typing import Optional, Tuple

import numpy
import simplejson
import skimage.io
import skvideo.io
import pandas

from pathlib import Path

from deepdiff import DeepDiff
from tqdm import tqdm
from matplotlib import pyplot

from MaskComputer01 import MaskComputer

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class VisibilityMaskComputer:
    def __init__(self, configs: dict):
        self.configs = configs
        self.mask_computer = MaskComputer(self.configs)
        return

    def compute_mask(self, frame_train, depth_train, depth_test, extrinsic_train, extrinsic_test, intrinsic_train, intrinsic_test):
        mask = self.mask_computer.compute_mask(frame_train, depth_train, depth_test, extrinsic_train, extrinsic_test, intrinsic_train, intrinsic_test)
        return mask

    @staticmethod
    def read_image(path: Path) -> numpy.ndarray:
        if path.suffix in ['.jpg', '.png', '.bmp']:
            image = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            image = numpy.load(path.as_posix())
        else:
            raise RuntimeError(f'Unknown image format: {path.as_posix()}')
        return image

    @staticmethod
    def read_depth(path: Path) -> numpy.ndarray:
        if path.suffix == '.png':
            depth = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            depth = numpy.load(path.as_posix())
        elif path.suffix == '.npz':
            with numpy.load(path.as_posix()) as depth_data:
                depth = depth_data['depth']
        elif path.suffix == '.exr':
            import Imath
            import OpenEXR

            exr_file = OpenEXR.InputFile(path.as_posix())
            raw_bytes = exr_file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
            depth_vector = numpy.frombuffer(raw_bytes, dtype=numpy.float32)
            height = exr_file.header()['displayWindow'].max.y + 1 - exr_file.header()['displayWindow'].min.y
            width = exr_file.header()['displayWindow'].max.x + 1 - exr_file.header()['displayWindow'].min.x
            depth = numpy.reshape(depth_vector, (height, width))
        else:
            raise RuntimeError(f'Unknown depth format: {path.as_posix()}')
        return depth

    @staticmethod
    def save_mask(path: Path, mask: numpy.ndarray, as_image: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        mask_image = mask.astype('uint8') * 255
        if path.suffix == '.png':
            skimage.io.imsave(path.as_posix(), mask_image)
        elif path.suffix == '.npy':
            numpy.save(path.as_posix(), mask)
            if as_image:
                path1 = path.parent / f'{path.stem}.png'
                skimage.io.imsave(path1.as_posix(), mask_image)
        else:
            raise RuntimeError(f'Unknown format: {path.as_posix()}')
        return


def save_configs(output_dirpath: Path, configs: dict):
    configs_path = output_dirpath / 'Configs.json'
    if configs_path.exists():
        with open(configs_path.as_posix(), 'r') as configs_file:
            old_configs = simplejson.load(configs_file)
        for key in old_configs.keys():
            if key not in configs.keys():
                configs[key] = old_configs[key]
        if configs != old_configs:
            raise RuntimeError(f'Configs mismatch while resuming testing: {DeepDiff(old_configs, configs)}')
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)
    return


def start_generation(gen_configs: dict):
    root_dirpath = Path('../../../../../')
    database_dirpath = root_dirpath / 'data/databases' / gen_configs['database_dirpath']
    dense_nerf_dirpath = root_dirpath / gen_configs['dense_nerf_dirpath']
    depth_dirpath = root_dirpath / gen_configs['depth_dirpath']

    output_dirpath = database_dirpath / f"all/visibility_masks/VM{gen_configs['gen_num']:02}"
    output_dirpath.mkdir(parents=True, exist_ok=True)
    save_configs(output_dirpath, gen_configs)

    set_num = gen_configs['gen_set_num']
    train_video_datapath = database_dirpath / f'train_test_sets/set{set_num:02}/TrainVideosData.csv'
    test_video_datapath = database_dirpath / f'train_test_sets/set{set_num:02}/TestVideosData.csv'
    train_video_data = pandas.read_csv(train_video_datapath)
    test_video_data = pandas.read_csv(test_video_datapath)
    scene_names = numpy.unique(train_video_data['scene_name'].to_numpy())
    suffix = gen_configs['resolution_suffix']

    mask_computer = VisibilityMaskComputer(gen_configs)

    for scene_name in tqdm(scene_names):

        train_frame_nums = train_video_data.loc[train_video_data['scene_name'] == scene_name]['pred_frame_num'].to_numpy()
        test_frame_nums = test_video_data.loc[test_video_data['scene_name'] == scene_name]['pred_frame_num'].to_numpy()
        
        extrinsics_path = database_dirpath / f'all/database_data/{scene_name}/CameraExtrinsics.csv'
        intrinsics_path = database_dirpath / f'all/database_data/{scene_name}/CameraIntrinsics{suffix}.csv'
        extrinsics = numpy.loadtxt(extrinsics_path.as_posix(), delimiter=',').reshape((-1, 4, 4))
        intrinsics = numpy.loadtxt(intrinsics_path.as_posix(), delimiter=',').reshape((-1, 3, 3))

        model_configs_path = dense_nerf_dirpath / f'{scene_name}/ModelConfigs.json'
        with open(model_configs_path.as_posix(), 'r') as model_configs_file:
            model_configs = json.load(model_configs_file)
            translation_scale = model_configs['translation_scale']
        
        for train_frame_num in train_frame_nums:
            for test_frame_num in test_frame_nums:
                mask_output_path = output_dirpath / f'{scene_name}/visibility_masks/{test_frame_num:04}_{train_frame_num:04}.npy'
                if mask_output_path.exists():
                    continue
        
                train_frame_path = database_dirpath / f'all/database_data/{scene_name}/rgb{suffix}/{train_frame_num:04}.png'
                # test_frame_path = database_dirpath / f'all/database_data/{scene_name}/rgb{suffix}/{test_frame_num:04}.png'
                train_depth_path = depth_dirpath / f'{scene_name}/predicted_depths/{train_frame_num:04}.npy'
                test_depth_path = depth_dirpath / f'{scene_name}/predicted_depths/{test_frame_num:04}.npy'
        
                frame_train = mask_computer.read_image(train_frame_path)
                # frame_test = mask_computer.read_image(test_frame_path)
                depth_train = mask_computer.read_depth(train_depth_path)
                depth_test = mask_computer.read_depth(test_depth_path)
                depth_train /= translation_scale
                depth_test /= translation_scale
                extrinsic_train = extrinsics[train_frame_num]
                extrinsic_test = extrinsics[test_frame_num]
                intrinsic_train = intrinsics[train_frame_num]
                intrinsic_test = intrinsics[test_frame_num]
        
                mask = mask_computer.compute_mask(frame_train, depth_train, depth_test, extrinsic_train, extrinsic_test, intrinsic_train, intrinsic_test)
                
                mask_computer.save_mask(mask_output_path, mask, as_image=True)
    return


def demo1():
    gen_configs = {
        'generator': this_filename,
        'gen_num': 2,
        'gen_set_num': 2,
        'database_name': 'NeRF_LLFF',
        'database_dirpath': 'NeRF_LLFF/data',
        'resolution_suffix': '_down4',
        'dense_nerf_dirpath': 'data/DenseNeRF/runs/training/train1001',
        'depth_dirpath': 'data/DenseNeRF/runs/testing/test1001',
        'depth_error_threshold': 0.05,
    }
    start_generation(gen_configs)

    gen_configs = {
        'generator': this_filename,
        'gen_num': 3,
        'gen_set_num': 3,
        'database_name': 'NeRF_LLFF',
        'database_dirpath': 'NeRF_LLFF/data',
        'resolution_suffix': '_down4',
        'dense_nerf_dirpath': 'data/DenseNeRF/runs/training/train1001',
        'depth_dirpath': 'data/DenseNeRF/runs/testing/test1001',
        'depth_error_threshold': 0.05,
    }
    start_generation(gen_configs)

    gen_configs = {
        'generator': this_filename,
        'gen_num': 4,
        'gen_set_num': 4,
        'database_name': 'NeRF_LLFF',
        'database_dirpath': 'NeRF_LLFF/data',
        'resolution_suffix': '_down4',
        'dense_nerf_dirpath': 'data/DenseNeRF/runs/training/train1001',
        'depth_dirpath': 'data/DenseNeRF/runs/testing/test1001',
        'depth_error_threshold': 0.05,
    }
    start_generation(gen_configs)
    return


def main():
    demo1()
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = 'Error: ' + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
