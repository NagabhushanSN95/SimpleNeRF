# Shree KRISHNAya Namaha
# Downloads the video and extracts RGB frames, camera intrinsics and extrinsics.
# Author: Nagabhushan S N
# Last Modified: 23/10/2023

import datetime
import os
import shutil
import time
import traceback
from enum import Enum
from pathlib import Path

import numpy
import skimage.io

this_filepath = Path(__file__)
this_filename = this_filepath.stem


# ------------- Enums for easier data passing ---------- #
class DataFeatures(Enum):
    FRAME = 'frame'
    INTRINSIC = 'intrinsic'
    EXTRINSIC = 'extrinsic'


class DataExtractor:
    def __init__(self, database_dirpath, timestamps, num_frames_per_scene, step_size):
        self.database_dirpath = database_dirpath
        self.timestamps = timestamps
        self.num_frames_per_scene = num_frames_per_scene
        self.step_size = step_size
        return

    def extract_data(self, features, resolution):
        """
        Downloads the videos for the scenes in database_dirpath and extracts the data specified by features.
        :param features:
        :param resolution:
        :return:
        """
        for scene_num in self.timestamps:
            scene_dirpath = self.database_dirpath / f'{scene_num:05}'
            start_timestamp = self.timestamps[scene_num]

            scene_datapath = scene_dirpath / 'CameraData.txt'
            with open(scene_datapath.as_posix(), 'r') as scene_data_file:
                scene_data = [line.strip() for line in scene_data_file.readlines()]
            url = scene_data[0]
            scene_data = [line.split(' ') for line in scene_data[1:]]
            scene_data = numpy.array(scene_data)
            start_line_num = numpy.where(scene_data[:, 0].astype('int') == start_timestamp)[0][0]
            end_line_num = start_line_num + self.num_frames_per_scene * self.step_size
            frames_data = scene_data[start_line_num:end_line_num:self.step_size]

            frames, extrinsics, intrinsics = None, None, None
            if DataFeatures.FRAME in features:
                timestamps = frames_data[:, 0].astype('int')
                frames = self.download_frames(scene_dirpath, url, timestamps, resolution)
                if frames is None:
                    continue
            if DataFeatures.EXTRINSIC in features:
                extrinsics = self.compute_extrinsic_matrices(frames_data[:, 7:19])
            if DataFeatures.INTRINSIC in features:
                intrinsics = self.compute_intrinsic_matrices(frames_data[:, 1:5], resolution)

            self.save_data(scene_dirpath, features, frames, intrinsics, extrinsics)
        return

    @staticmethod
    def download_frames(scene_dirpath: Path, url: str, timestamps: numpy.ndarray, resolution):
        video_filepath = scene_dirpath / 'video.mp4'
        cmd = f'youtube-dl -o {video_filepath.absolute().as_posix()} {url}'
        return_code = os.system(cmd)

        if return_code == 0:
            # Get the saved video path
            video_filepath = list(scene_dirpath.rglob('video.*'))[0]
            tmp_dirpath = scene_dirpath / 'tmp'
            tmp_dirpath.mkdir(parents=True, exist_ok=False)
            # Extract required frames
            frames = []
            for timestamp in timestamps:
                timestamp = int(timestamp / 1000)
                str_hour = str(int(timestamp / 3600000)).zfill(2)
                str_min = str(int(int(timestamp % 3600000) / 60000)).zfill(2)
                str_sec = str(int(int(int(timestamp % 3600000) % 60000) / 1000)).zfill(2)
                str_mill = str(int(int(int(timestamp % 3600000) % 60000) % 1000)).zfill(3)
                str_timestamp = str_hour + ":" + str_min + ":" + str_sec + "." + str_mill
                orig_frame_filepath = tmp_dirpath / f'{timestamp}.png'
                cmd = f'ffmpeg -loglevel quiet -ss {str_timestamp} -i {video_filepath.as_posix()} -vframes 1 -f image2 {orig_frame_filepath.as_posix()}'
                return_code = os.system(cmd)
                if return_code == 0:
                    frame = skimage.io.imread(orig_frame_filepath.as_posix())
                    resized_frame = skimage.transform.resize(frame, output_shape=resolution, preserve_range=True, anti_aliasing=True)
                    frame = numpy.round(resized_frame).astype('uint8')
                    frames.append(frame)
                else:
                    frames = None
                    break
            frames = numpy.stack(frames) if frames is not None else None
            shutil.rmtree(tmp_dirpath)
        else:
            print(f'Unable to download scene: {scene_dirpath.stem}')
            frames = None

        return frames

    def compute_intrinsic_matrices(self, intrinsics_data: numpy.ndarray, resolution):
        intrinsics_data = intrinsics_data.astype('float32')
        h, w = resolution
        num_frames = intrinsics_data.shape[0]
        intrinsic_matrices = numpy.zeros(shape=(num_frames, 9), dtype=numpy.float32)
        fx, fy, px, py = [x.squeeze() for x in numpy.split(intrinsics_data, 4, axis=1)]
        intrinsic_matrices[:, 0] = w * fx
        intrinsic_matrices[:, 4] = h * fy
        intrinsic_matrices[:, 2] = w * px
        intrinsic_matrices[:, 5] = h * py
        intrinsic_matrices[:, 8] = 1
        return intrinsic_matrices

    @staticmethod
    def compute_extrinsic_matrices(extrinsics_data: numpy.ndarray):
        extrinsics_data = extrinsics_data.astype('float32')
        num_frames = extrinsics_data.shape[0]
        last_row = numpy.zeros(shape=(num_frames, 4), dtype=numpy.float32)
        last_row[:, 3] = 1
        extrinsic_matrices = numpy.concatenate([extrinsics_data, last_row], axis=1)
        return extrinsic_matrices

    def save_data(self, scene_dirpath: Path, features, frames, intrinsics, extrinsics):
        if DataFeatures.FRAME in features:
            self.save_frames(scene_dirpath, frames)

        if DataFeatures.EXTRINSIC in features:
            self.save_extrinsics(scene_dirpath, extrinsics)

        if DataFeatures.INTRINSIC in features:
            self.save_intrinsics(scene_dirpath, intrinsics)
        return

    @staticmethod
    def save_frames(scene_dirpath: Path, frames: numpy.ndarray, frame_nums: numpy.ndarray = None):
        rgb_dirpath = scene_dirpath / 'rgb'
        rgb_dirpath.mkdir(parents=True, exist_ok=False)
        if frame_nums is None:
            frame_nums = numpy.arange(frames.shape[0])
        for frame_num, frame in zip(frame_nums, frames):
            frame_path = rgb_dirpath / f'{frame_num:04}.png'
            skimage.io.imsave(frame_path.as_posix(), frame)
        return

    @staticmethod
    def save_intrinsics(scene_dirpath: Path, intrinsics: numpy.ndarray):
        intrinsics_path = scene_dirpath / 'CameraIntrinsics.csv'
        numpy.savetxt(intrinsics_path.as_posix(), intrinsics, delimiter=',')
        return

    @staticmethod
    def save_extrinsics(scene_dirpath: Path, extrinsics: numpy.ndarray):
        extrinsics_path = scene_dirpath / 'CameraExtrinsics.csv'
        numpy.savetxt(extrinsics_path.as_posix(), extrinsics, delimiter=',')
        return


def demo1():
    features = [DataFeatures.FRAME, DataFeatures.EXTRINSIC, DataFeatures.INTRINSIC]
    resolution = (576, 1024)
    timestamps = {
        0: 53453400,
        1: 227894333,
        3: 46012000,
        4: 61144000,
        6: 54387667
    }
    num_frames_per_scene = 50
    step_size = 1

    root_dirpath = Path('../../../../data/databases/RealEstate10K/')
    database_dirpath = root_dirpath / 'data/test/database_data'

    data_extractor = DataExtractor(database_dirpath, timestamps, num_frames_per_scene, step_size)
    data_extractor.extract_data(features, resolution)
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
        run_result = str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
