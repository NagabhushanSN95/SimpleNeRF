# Shree KRISHNAya Namaha
# SROCC measure between predicted depths and depths from dense input NeRF
# Author: Nagabhushan S N
# Last Modified: 15/06/2023

import argparse
import datetime
import json
import time
import traceback
from pathlib import Path

import numpy
import pandas
import simplejson
import skimage.io
import skimage.transform
from scipy.stats import spearmanr
from tqdm import tqdm

this_filepath = Path(__file__)
this_filename = Path(__file__).stem
this_metric_name = this_filename[:-10]


class DepthSROCC:
    def __init__(self, frames_data: pandas.DataFrame, verbose_log: bool = True) -> None:
        super().__init__()
        self.frames_data = frames_data
        self.verbose_log = verbose_log
        return

    @staticmethod
    def compute_depth_srocc(gt_depth: numpy.ndarray, eval_depth: numpy.ndarray):
        gt_depth = gt_depth.astype('float')
        eval_depth = eval_depth.astype('float')
        srocc = spearmanr(gt_depth.ravel(), eval_depth.ravel()).correlation
        return srocc

    def compute_avg_srocc(self, old_data: pandas.DataFrame, gt_depth_dirpath: Path, pred_videos_dirpath: Path,
                         pred_depths_dirname: str, resolution_suffix: str, downsampling_factor: int):
        """

        :param old_data:
        :param gt_depth_dirpath: Should be path to databases/NeRF_LLFF/data
        :param pred_videos_dirpath:
        :param pred_depths_dirname:
        :param resolution_suffix:
        :param downsampling_factor:
        :return:
        """
        qa_scores = []
        for i, frame_data in tqdm(self.frames_data.iterrows(), total=self.frames_data.shape[0], leave=self.verbose_log):
            scene_name, pred_frame_num = frame_data
            if old_data is not None and old_data.loc[
                (old_data['scene_name'] == scene_name) & (old_data['pred_frame_num'] == pred_frame_num)
            ].size > 0:
                continue
            gt_depth_path = gt_depth_dirpath / f'{scene_name}/{pred_depths_dirname}/{pred_frame_num:04}.npy'
            pred_depth_path = pred_videos_dirpath / f'{scene_name}/{pred_depths_dirname}/{pred_frame_num:04}.npy'
            if not pred_depth_path.exists():
                continue
            gt_depth = self.read_depth(gt_depth_path)
            pred_depth = self.read_depth(pred_depth_path)
            if (downsampling_factor > 1) and (gt_depth.shape != pred_depth.shape):
                gt_depth = self.downsample_depth(gt_depth, downsampling_factor)
            gt_scale = self.get_depth_scale(gt_depth_dirpath, scene_name)
            pred_scale = self.get_depth_scale(pred_videos_dirpath, scene_name)
            gt_depth = gt_depth * gt_scale
            pred_depth = pred_depth * pred_scale
            qa_score = self.compute_depth_srocc(gt_depth, pred_depth)
            qa_scores.append([scene_name, pred_frame_num, qa_score])
        qa_scores_data = pandas.DataFrame(qa_scores, columns=['scene_name', 'pred_frame_num', this_metric_name])

        merged_data = self.update_qa_frame_data(old_data, qa_scores_data)
        merged_data = merged_data.round({this_metric_name: 4, })

        avg_srocc = numpy.mean(merged_data[this_metric_name])
        if isinstance(avg_srocc, numpy.ndarray):
            avg_srocc = avg_srocc.item()
        avg_srocc = numpy.round(avg_srocc, 4)
        return avg_srocc, merged_data

    @staticmethod
    def update_qa_frame_data(old_data: pandas.DataFrame, new_data: pandas.DataFrame):
        if (old_data is not None) and (new_data.size > 0):
            old_data = old_data.copy()
            new_data = new_data.copy()
            old_data.set_index(['scene_name', 'pred_frame_num'], inplace=True)
            new_data.set_index(['scene_name', 'pred_frame_num'], inplace=True)
            merged_data = old_data.combine_first(new_data).reset_index()
        elif old_data is not None:
            merged_data = old_data
        else:
            merged_data = new_data
        return merged_data

    @staticmethod
    def read_depth(path: Path):
        depth = numpy.load(path.as_posix())
        return depth

    @staticmethod
    def downsample_depth(depth: numpy.ndarray, downsampling_factor: int):
        downsampled_depth = skimage.transform.rescale(depth, scale=1 / downsampling_factor, preserve_range=True,
                                                      multichannel=False, anti_aliasing=True)
        return downsampled_depth

    @staticmethod
    def get_depth_scale(test_dirpath, scene_name):
        test_configs_path = test_dirpath / 'Configs.json'
        with open(test_configs_path.as_posix(), 'r') as test_configs_file:
            test_configs = json.load(test_configs_file)
        train_num = test_configs['train_num']
        train_dirpath = test_dirpath.parent.parent / f'training/train{train_num:04}'
        model_configs_path = train_dirpath / f'{scene_name}/ModelConfigs.json'
        with open(model_configs_path.as_posix(), 'r') as model_configs_file:
            model_configs = json.load(model_configs_file)
        scale = 1 / model_configs['translation_scale']
        return scale


# noinspection PyUnusedLocal
def start_qa(pred_videos_dirpath: Path, gt_depth_dirpath: Path, frames_datapath: Path, pred_depths_dirname: str, 
             resolution_suffix, downsampling_factor: int):
    if not pred_videos_dirpath.exists():
        print(f'Skipping QA of folder: {pred_videos_dirpath.stem}. Reason: pred_videos_dirpath does not exist')
        return

    if not gt_depth_dirpath.exists():
        print(f'Skipping QA of folder: {pred_videos_dirpath.stem}. Reason: gt_depth_dirpath does not exist')
        return

    qa_scores_filepath = pred_videos_dirpath / 'QA_Scores.json'
    srocc_data_path = pred_videos_dirpath / f'QA_Scores/{pred_depths_dirname}/{this_metric_name}_FrameWise.csv'
    if qa_scores_filepath.exists():
        with open(qa_scores_filepath.as_posix(), 'r') as qa_scores_file:
            qa_scores = json.load(qa_scores_file)
    else:
        qa_scores = {}

    if pred_depths_dirname in qa_scores:
        if this_metric_name in qa_scores[pred_depths_dirname]:
            avg_srocc = qa_scores[pred_depths_dirname][this_metric_name]
            print(f'Average {this_metric_name}: {pred_videos_dirpath.as_posix()} - {pred_depths_dirname}: {avg_srocc}')
            print('Running QA again.')
    else:
        qa_scores[pred_depths_dirname] = {}

    if srocc_data_path.exists():
        srocc_data = pandas.read_csv(srocc_data_path)
    else:
        srocc_data = None

    frames_data = pandas.read_csv(frames_datapath)[['scene_name', 'pred_frame_num']]

    mse_computer = DepthSROCC(frames_data)
    avg_srocc, srocc_data = mse_computer.compute_avg_srocc(srocc_data, gt_depth_dirpath, pred_videos_dirpath,
                                                        pred_depths_dirname, resolution_suffix, downsampling_factor)
    if numpy.isfinite(avg_srocc):
        qa_scores[pred_depths_dirname][this_metric_name] = avg_srocc
        print(f'Average {this_metric_name}: {pred_videos_dirpath.as_posix()} - {pred_depths_dirname}: {avg_srocc}')
        with open(qa_scores_filepath.as_posix(), 'w') as qa_scores_file:
            simplejson.dump(qa_scores, qa_scores_file, indent=4)
        srocc_data_path.parent.mkdir(parents=True, exist_ok=True)
        srocc_data.to_csv(srocc_data_path, index=False)
    return avg_srocc


def demo1():
    root_dirpath = Path('../../../../')
    pred_videos_dirpath = root_dirpath / 'runs/testing/test0011'
    database_dirpath = root_dirpath / 'data/databases/NeRF_LLFF/data'
    gt_depth_dirpath = root_dirpath / 'data/DenseNeRF/runs/testing/test1001'
    frames_data_path = database_dirpath / 'train_test_sets/set02/TestVideosData.csv'
    pred_depths_dirname = 'predicted_depths'
    resolution_suffix = '_down4'
    downsampling_factor = 1
    avg_srocc = start_qa(pred_videos_dirpath, gt_depth_dirpath, frames_data_path, pred_depths_dirname, resolution_suffix,
                         downsampling_factor)
    return avg_srocc


def demo2(args: dict):
    pred_videos_dirpath = args['pred_videos_dirpath']
    if pred_videos_dirpath is None:
        raise RuntimeError(f'Please provide pred_videos_dirpath')
    pred_videos_dirpath = Path(pred_videos_dirpath)

    gt_depth_dirpath = args['gt_depth_dirpath']
    if gt_depth_dirpath is None:
        raise RuntimeError(f'Please provide gt_depth_dirpath')
    gt_depth_dirpath = Path(gt_depth_dirpath)

    frames_datapath = args['frames_datapath']
    if frames_datapath is None:
        raise RuntimeError(f'Please provide frames_datapath')
    frames_datapath = Path(frames_datapath)

    pred_depths_dirname = args['pred_depths_dirname']
    resolution_suffix = args['resolution_suffix']
    downsampling_factor = args['downsampling_factor']

    avg_srocc = start_qa(pred_videos_dirpath, gt_depth_dirpath, frames_datapath, pred_depths_dirname, resolution_suffix,
                        downsampling_factor)
    return avg_srocc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_function_name', default='demo1')
    parser.add_argument('--pred_videos_dirpath')
    parser.add_argument('--gt_depth_dirpath')
    parser.add_argument('--frames_datapath')
    parser.add_argument('--pred_depths_dirname', default='predicted_depths')
    parser.add_argument('--resolution_suffix', default='_down4')
    parser.add_argument('--downsampling_factor', type=int, default=1)
    args = parser.parse_args()

    args_dict = {
        'demo_function_name': args.demo_function_name,
        'pred_videos_dirpath': args.pred_videos_dirpath,
        'gt_depth_dirpath': args.gt_depth_dirpath,
        'frames_datapath': args.frames_datapath,
        'pred_depths_dirname': args.pred_depths_dirname,
        'resolution_suffix': args.resolution_suffix,
        'downsampling_factor': args.downsampling_factor,
    }
    return args_dict


def main(args: dict):
    if args['demo_function_name'] == 'demo1':
        avg_srocc = demo1()
    elif args['demo_function_name'] == 'demo2':
        avg_srocc = demo2(args)
    else:
        raise RuntimeError(f'Unknown demo function: {args["demo_function_name"]}')
    return avg_srocc


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    args = parse_args()
    try:
        output_score = main(args)
        run_result = f'Program completed successfully!\nAverage {this_metric_name}: {output_score}'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = "Error: " + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
