# Shree KRISHNAya Namaha
# Creates a video to compare predictions of two models
# For SIGGRAPH Asia 2023 submission.
# Modified from R13/12_VideoComparisonMaker.py. Merging is done on a single frame as opposed to video
# Author: Nagabhushan S N
# Last Modified: 07/05/2023

import time
import datetime
import traceback
import numpy
import skimage.io
import skvideo.io

from pathlib import Path

from PIL import ImageFont, Image, ImageDraw
from tqdm import tqdm
from matplotlib import pyplot

this_filepath = Path(__file__)
this_filename = this_filepath.stem
this_filenum = int(this_filename[:2])


class VideoMaker:
    color_black = numpy.array([0, 0, 0])
    color_white = numpy.array([255, 255, 255])
    color_red = numpy.array([255, 0, 0])
    color_green = numpy.array([0, 255, 0])
    color_blue = numpy.array([0, 0, 255])

    def __init__(self, configs):
        self.configs = configs
        return

    def make_video(self, models_data: list, sample_data):
        scene_name = sample_data['scene_name']
        num_unmerged_loops = self.configs['num_unmerged_loops']
        merging_frame_num = sample_data['merging_frame_num']
        num_merging_frames = self.configs['num_merging_frames']
        num_merged_loops = self.configs['num_merged_loops']
        num_static_comparison_frames_per_loop = self.configs['num_static_comparison_frames_per_loop']
        num_static_comparison_loops = self.configs['num_static_comparison_loops']
        static_comparison_frame_num = sample_data['static_comparison_frame_num']
        demarcation_line_width = self.configs['demarcation_line_width']
        demarcation_line_color = self.configs['demarcation_line_color']
        background_color = self.configs['background_color']
        annotation_height = self.configs['annotation_height']
        annotation_font_size = self.configs['annotation_font_size']

        num_models = len(models_data)
        frame_nums = sample_data.get('frame_nums', None)
        first_model_dirpath = Path(models_data[0]['model_dirpath'].format(scene_name=scene_name))
        if frame_nums is None:
            frame_nums = sorted([int(frame_path.stem) for frame_path in first_model_dirpath.iterdir()])
        h, w, c = self.read_image(first_model_dirpath / f'{frame_nums[0]:04}.png').shape
        demarcation_line = demarcation_line_color[None, None] * numpy.ones(shape=(h, demarcation_line_width, c),
                                                                           dtype='uint8')

        video_frames = []
        # Unmerged videos
        for i in range(num_unmerged_loops):
            for frame_num in tqdm(frame_nums, desc=f'Unmerged loop: {i + 1}/{num_unmerged_loops}'):
                current_frames = []
                for model_data in models_data:
                    model_dirpath = Path(model_data['model_dirpath'].format(scene_name=scene_name))
                    model_frame_path = model_dirpath / f'{frame_num:04}.png'
                    model_frame = self.read_image(model_frame_path)
                    current_frames.append(model_frame)
                    current_frames.append(demarcation_line)
                current_frames = current_frames[:-1]
                current_frame = numpy.concatenate(current_frames, axis=1)
                video_frames.append(current_frame)

        # Unmerged video before reaching the merging frame
        for j, frame_num in enumerate(tqdm(frame_nums, desc=f'Pre merging')):
            if j > frame_nums.index(merging_frame_num):
                break
            current_frames = []
            for model_data in models_data:
                model_dirpath = Path(model_data['model_dirpath'].format(scene_name=scene_name))
                model_frame_path = model_dirpath / f'{frame_num:04}.png'
                model_frame = self.read_image(model_frame_path)
                current_frames.append(model_frame)
                current_frames.append(demarcation_line)
            current_frames = current_frames[:-1]
            current_frame = numpy.concatenate(current_frames, axis=1)
            video_frames.append(current_frame)

        # Merging
        unmerged_frame_width = w * num_models + demarcation_line_width * (num_models - 1)
        cropped_frame_width_final = (w - demarcation_line_width * (num_models - 1)) // num_models
        for i in tqdm(range(num_merging_frames), desc='Merging'):
            progress = (i + 1) / num_merging_frames
            cropped_frame_width = int(round(progress * cropped_frame_width_final + (1 - progress) * w))
            current_frames = []
            for k, model_data in enumerate(models_data):
                model_dirpath = Path(model_data['model_dirpath'].format(scene_name=scene_name))
                model_frame_path = model_dirpath / f'{merging_frame_num:04}.png'
                model_frame = self.read_image(model_frame_path)
                if k == 0:
                    start = 0
                elif k == num_models - 1:
                    start = w - cropped_frame_width
                else:
                    start = (w - cropped_frame_width) // 2
                cropped_model_frame = model_frame[:, start: start + cropped_frame_width]
                current_frames.append(cropped_model_frame)
                current_frames.append(demarcation_line)
            current_frames = current_frames[:-1]
            current_frame = numpy.concatenate(current_frames, axis=1)
            padded_current_frame = self.pad_frame(current_frame, unmerged_frame_width, background_color)
            video_frames.append(padded_current_frame)

        # Merged video after merging
        for j, frame_num in enumerate(tqdm(frame_nums, desc=f'Post merging')):
            if (frame_nums.index(merging_frame_num) == 0) or (j < frame_nums.index(merging_frame_num)):
                continue
            cropped_frame_width = cropped_frame_width_final
            current_frames = []
            for k, model_data in enumerate(models_data):
                model_dirpath = Path(model_data['model_dirpath'].format(scene_name=scene_name))
                model_frame_path = model_dirpath / f'{frame_num:04}.png'
                model_frame = self.read_image(model_frame_path)
                if k == 0:
                    start = 0
                elif k == num_models - 1:
                    start = w - cropped_frame_width
                else:
                    start = (w - cropped_frame_width) // 2
                cropped_model_frame = model_frame[:, start: start + cropped_frame_width]
                current_frames.append(cropped_model_frame)
                current_frames.append(demarcation_line)
            current_frames = current_frames[:-1]
            current_frame = numpy.concatenate(current_frames, axis=1)
            padded_current_frame = self.pad_frame(current_frame, unmerged_frame_width, background_color)
            video_frames.append(padded_current_frame)

        # Merged loops
        for i in range(num_merged_loops):
            for frame_num in tqdm(frame_nums, desc=f'Merged loop: {i + 1}/{num_merged_loops}'):
                cropped_frame_width = cropped_frame_width_final
                current_frames = []
                for k, model_data in enumerate(models_data):
                    model_dirpath = Path(model_data['model_dirpath'].format(scene_name=scene_name))
                    model_frame_path = model_dirpath / f'{frame_num:04}.png'
                    model_frame = self.read_image(model_frame_path)
                    if k == 0:
                        start = 0
                    elif k == num_models - 1:
                        start = w - cropped_frame_width
                    else:
                        start = (w - cropped_frame_width) // 2
                    cropped_model_frame = model_frame[:, start: start + cropped_frame_width]
                    current_frames.append(cropped_model_frame)
                    current_frames.append(demarcation_line)
                current_frames = current_frames[:-1]
                current_frame = numpy.concatenate(current_frames, axis=1)
                padded_current_frame = self.pad_frame(current_frame, unmerged_frame_width, background_color)
                video_frames.append(padded_current_frame)

        # Merged video before reaching static frame
        for j, frame_num in enumerate(tqdm(frame_nums, desc=f'Pre static frame')):
            if j > frame_nums.index(static_comparison_frame_num):
                break
            cropped_frame_width = cropped_frame_width_final
            current_frames = []
            for k, model_data in enumerate(models_data):
                model_dirpath = Path(model_data['model_dirpath'].format(scene_name=scene_name))
                model_frame_path = model_dirpath / f'{frame_num:04}.png'
                model_frame = self.read_image(model_frame_path)
                if k == 0:
                    start = 0
                elif k == num_models - 1:
                    start = w - cropped_frame_width
                else:
                    start = (w - cropped_frame_width) // 2
                cropped_model_frame = model_frame[:, start: start + cropped_frame_width]
                current_frames.append(cropped_model_frame)
                current_frames.append(demarcation_line)
            current_frames = current_frames[:-1]
            current_frame = numpy.concatenate(current_frames, axis=1)
            padded_current_frame = self.pad_frame(current_frame, unmerged_frame_width, background_color)
            video_frames.append(padded_current_frame)

        # Static frame comparison
        for i in range(num_static_comparison_loops):
            for j in tqdm(range(num_static_comparison_frames_per_loop),
                          desc=f'Static frame loop: {i + 1}/{num_static_comparison_loops}'):
                progress = (j + 1) / num_static_comparison_frames_per_loop
                demarcation_line_progress = (0.5 + 2 * progress) * (progress <= 0.25) + \
                                            (1 - 2 * (progress - 0.25)) * (0.25 < progress <= 0.75) + \
                                            (2 * (progress - 0.75)) * (progress > 0.75)
                first_frame_max_width = (w - (num_models - 1) * demarcation_line_width) / (num_models - 1)
                first_frame_end = int(demarcation_line_progress * first_frame_max_width)
                cropped_frame_width = int(round(progress * cropped_frame_width_final + (1 - progress) * w))
                current_frames = []
                for k, model_data in enumerate(models_data):
                    model_dirpath = Path(model_data['model_dirpath'].format(scene_name=scene_name))
                    model_frame_path = model_dirpath / f'{static_comparison_frame_num:04}.png'
                    model_frame = self.read_image(model_frame_path)
                    if k == 0:
                        start = 0
                        end = first_frame_end
                    elif k == num_models - 1:
                        start = first_frame_end + demarcation_line_width + (
                                    demarcation_line_width * cropped_frame_width) * (k - 1)
                        end = w
                    else:
                        start = first_frame_end + demarcation_line_width + (
                                    demarcation_line_width * cropped_frame_width) * (k - 1)
                        end = start + cropped_frame_width
                    cropped_model_frame = model_frame[:, start: end]
                    current_frames.append(cropped_model_frame)
                    current_frames.append(demarcation_line)
                current_frames = current_frames[:-1]
                current_frame = numpy.concatenate(current_frames, axis=1)
                padded_current_frame = self.pad_frame(current_frame, unmerged_frame_width, background_color)
                video_frames.append(padded_current_frame)

        # Merged video after static frame comparison
        for j, frame_num in enumerate(tqdm(frame_nums, desc='Post static frame')):
            if (frame_nums.index(static_comparison_frame_num) == 0) or (
                    j < frame_nums.index(static_comparison_frame_num)):
                continue
            cropped_frame_width = cropped_frame_width_final
            current_frames = []
            for k, model_data in enumerate(models_data):
                model_dirpath = Path(model_data['model_dirpath'].format(scene_name=scene_name))
                model_frame_path = model_dirpath / f'{frame_num:04}.png'
                model_frame = self.read_image(model_frame_path)
                if k == 0:
                    start = 0
                elif k == num_models - 1:
                    start = w - cropped_frame_width
                else:
                    start = (w - cropped_frame_width) // 2
                cropped_model_frame = model_frame[:, start: start + cropped_frame_width]
                current_frames.append(cropped_model_frame)
                current_frames.append(demarcation_line)
            current_frames = current_frames[:-1]
            current_frame = numpy.concatenate(current_frames, axis=1)
            padded_current_frame = self.pad_frame(current_frame, unmerged_frame_width, background_color)
            video_frames.append(padded_current_frame)

        # Merged video before unmerging
        for j, frame_num in enumerate(tqdm(frame_nums, desc='Pre unmerging')):
            if j > frame_nums.index(merging_frame_num):
                break
            cropped_frame_width = cropped_frame_width_final
            current_frames = []
            for k, model_data in enumerate(models_data):
                model_dirpath = Path(model_data['model_dirpath'].format(scene_name=scene_name))
                model_frame_path = model_dirpath / f'{frame_num:04}.png'
                model_frame = self.read_image(model_frame_path)
                if k == 0:
                    start = 0
                elif k == num_models - 1:
                    start = w - cropped_frame_width
                else:
                    start = (w - cropped_frame_width) // 2
                cropped_model_frame = model_frame[:, start: start + cropped_frame_width]
                current_frames.append(cropped_model_frame)
                current_frames.append(demarcation_line)
            current_frames = current_frames[:-1]
            current_frame = numpy.concatenate(current_frames, axis=1)
            padded_current_frame = self.pad_frame(current_frame, unmerged_frame_width, background_color)
            video_frames.append(padded_current_frame)

        # Unmerging
        for i in tqdm(range(num_merging_frames), desc='Unmerging'):
            progress = (i + 1) / num_merging_frames
            cropped_frame_width = int(round((1 - progress) * cropped_frame_width_final + progress * w))
            current_frames = []
            for k, model_data in enumerate(models_data):
                model_dirpath = Path(model_data['model_dirpath'].format(scene_name=scene_name))
                model_frame_path = model_dirpath / f'{merging_frame_num:04}.png'
                model_frame = self.read_image(model_frame_path)
                if k == 0:
                    start = 0
                elif k == num_models - 1:
                    start = w - cropped_frame_width
                else:
                    start = (w - cropped_frame_width) // 2
                cropped_model_frame = model_frame[:, start: start + cropped_frame_width]
                current_frames.append(cropped_model_frame)
                current_frames.append(demarcation_line)
            current_frames = current_frames[:-1]
            current_frame = numpy.concatenate(current_frames, axis=1)
            padded_current_frame = self.pad_frame(current_frame, unmerged_frame_width, background_color)
            video_frames.append(padded_current_frame)

        # Unmerged video after unmerging
        for j, frame_num in enumerate(tqdm(frame_nums, desc=f'Post unmerging')):
            if (frame_nums.index(merging_frame_num) == 0) or (j < frame_nums.index(merging_frame_num)):
                continue
            current_frames = []
            for model_data in models_data:
                model_dirpath = Path(model_data['model_dirpath'].format(scene_name=scene_name))
                model_frame_path = model_dirpath / f'{frame_num:04}.png'
                model_frame = self.read_image(model_frame_path)
                current_frames.append(model_frame)
                current_frames.append(demarcation_line)
            current_frames = current_frames[:-1]
            current_frame = numpy.concatenate(current_frames, axis=1)
            video_frames.append(current_frame)

        video = numpy.stack(video_frames, axis=0)
        del video_frames

        model_names = [model_data['model_name'] for model_data in models_data]
        annotation = background_color[None, None] * numpy.ones(shape=(annotation_height, unmerged_frame_width, c))
        for i, model_name in enumerate(model_names):
            model_annotation = self.get_annotation(anno_shape=(annotation_height, w), title=model_name,
                                                   font_size=annotation_font_size)
            annotation[:, i * (w + demarcation_line_width): i * (w + demarcation_line_width) + w] = model_annotation
        annotation_video = numpy.stack([annotation] * video.shape[0], axis=0)
        video = numpy.concatenate([annotation_video, video], axis=1)
        return video

    @staticmethod
    def get_annotation(anno_shape: tuple, title: str, font_size: int = 50):
        frame_height, frame_width = anno_shape[:2]
        times_font = ImageFont.truetype('../res/fonts/times-new-roman.ttf', font_size)
        text_image = Image.new('RGB', (frame_width, frame_height), (255, 255, 255))
        drawer = ImageDraw.Draw(text_image)
        w, h = drawer.textsize(title, font=times_font)
        drawer.text(((frame_width - w) / 2, (frame_height - h) / 2), text=title, fill=(0, 0, 0), font=times_font,
                    align='center')
        annotation = numpy.array(text_image)
        return annotation

    @staticmethod
    def pad_frame(frame, final_width, color):
        h, w, c = frame.shape
        w1 = (final_width - w) // 2
        padded_frame = color[None, None] * numpy.ones(shape=(h, final_width, c), dtype='uint8')
        padded_frame[:, w1: w1 + w] = frame
        return padded_frame

    @staticmethod
    def read_image(path: Path) -> numpy.ndarray:
        if path.suffix in ['.jpg', '.png', '.bmp']:
            image = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            image = numpy.load(path.as_posix())
        else:
            raise RuntimeError(f'Unknown image format: {path.as_posix()}')
        return image


def demo1a():
    """
    NeRF v/s SimpleNeRF
    :return:
    """
    configs = {
        'num_unmerged_loops': 1,
        'num_merging_frames': 25,
        'num_merged_loops': 1,
        'num_static_comparison_frames_per_loop': 100,
        'num_static_comparison_loops': 1,
        'demarcation_line_width': 10,
        'demarcation_line_color': VideoMaker.color_white,
        'annotation_height': 70,
        'annotation_font_size': 50,
        'background_color': VideoMaker.color_white,
    }
    models_data = [
        {
            'model_name': 'NeRF',
            'model_dirpath': '../../../view_synthesis/literature/020_NeRFs_Simplified/runs/testing/test0002/{scene_name}_video01/predicted_frames',
        },
        {
            'model_name': 'ViP-NeRF',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test0231/{scene_name}_video01/predicted_frames',
        },
    ]
    samples_data = [
        # {
        #     'scene_name': '00000',
        #     'merging_frame_num': 0,
        #     'static_comparison_frame_num': 0,
        #     'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        # },
        {
            'scene_name': '00001',
            'merging_frame_num': 0,
            'static_comparison_frame_num': 0,
            'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        },
        {
            'scene_name': '00003',
            'merging_frame_num': 0,
            'static_comparison_frame_num': 15,
            'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        },
        # {
        #     'scene_name': '00004',
        #     'merging_frame_num': 0,
        #     'static_comparison_frame_num': 0,
        #     'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        # },
        # {
        #     'scene_name': '00006',
        #     'merging_frame_num': 0,
        #     'static_comparison_frame_num': 0,
        #     'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        # },
    ]
    frame_rate = 25

    video_maker = VideoMaker(configs)
    for sample_data in samples_data:
        scene_name = sample_data['scene_name']
        video = video_maker.make_video(models_data, sample_data)
        output_path = Path(f'../data/outputs/{this_filenum:02}_VideoComparisonMaker01a_{scene_name}.mp4')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        skvideo.io.vwrite(output_path.as_posix(), video,
                          inputdict={'-r': str(frame_rate)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p', '-profile:v': 'main'})
        del video
    return


def demo1b():
    """
    InfoNeRF v/s SimpleNeRF
    :return:
    """
    configs = {
        'num_unmerged_loops': 1,
        'num_merging_frames': 25,
        'num_merged_loops': 1,
        'num_static_comparison_frames_per_loop': 100,
        'num_static_comparison_loops': 1,
        'demarcation_line_width': 10,
        'demarcation_line_color': VideoMaker.color_white,
        'annotation_height': 70,
        'annotation_font_size': 50,
        'background_color': VideoMaker.color_white,
    }
    models_data = [
        {
            'model_name': 'InfoNeRF',
            'model_dirpath': '../../../view_synthesis/literature/012_InfoNeRF/runs/testing/test0002/{scene_name}_video01/predicted_frames',
        },
        {
            'model_name': 'ViP-NeRF',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test0231/{scene_name}_video01/predicted_frames',
        },
    ]
    samples_data = [
        # {
        #     'scene_name': '00000',
        #     'merging_frame_num': 0,
        #     'static_comparison_frame_num': 0,
        #     'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        # },
        {
            'scene_name': '00001',
            'merging_frame_num': 0,
            'static_comparison_frame_num': 0,
            'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        },
        {
            'scene_name': '00003',
            'merging_frame_num': 0,
            'static_comparison_frame_num': 15,
            'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        },
        # {
        #     'scene_name': '00004',
        #     'merging_frame_num': 0,
        #     'static_comparison_frame_num': 0,
        #     'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        # },
        # {
        #     'scene_name': '00006',
        #     'merging_frame_num': 0,
        #     'static_comparison_frame_num': 0,
        #     'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        # },
    ]
    frame_rate = 25

    video_maker = VideoMaker(configs)
    for sample_data in samples_data:
        scene_name = sample_data['scene_name']
        video = video_maker.make_video(models_data, sample_data)
        output_path = Path(f'../data/outputs/{this_filenum:02}_VideoComparisonMaker01b_{scene_name}.mp4')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        skvideo.io.vwrite(output_path.as_posix(), video,
                          inputdict={'-r': str(frame_rate)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p', '-profile:v': 'main'})
        del video
    return


def demo1c():
    """
    DietNeRF v/s SimpleNeRF
    :return:
    """
    configs = {
        'num_unmerged_loops': 1,
        'num_merging_frames': 25,
        'num_merged_loops': 1,
        'num_static_comparison_frames_per_loop': 100,
        'num_static_comparison_loops': 1,
        'demarcation_line_width': 10,
        'demarcation_line_color': VideoMaker.color_white,
        'annotation_height': 70,
        'annotation_font_size': 50,
        'background_color': VideoMaker.color_white,
    }
    models_data = [
        {
            'model_name': 'DietNeRF',
            'model_dirpath': '../../../view_synthesis/literature/021_DietNeRF/runs/testing/test0002/{scene_name}_video01/predicted_frames',
        },
        {
            'model_name': 'ViP-NeRF',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test0231/{scene_name}_video01/predicted_frames',
        },
    ]
    samples_data = [
        # {
        #     'scene_name': '00000',
        #     'merging_frame_num': 0,
        #     'static_comparison_frame_num': 0,
        #     'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        # },
        {
            'scene_name': '00001',
            'merging_frame_num': 0,
            'static_comparison_frame_num': 0,
            'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        },
        {
            'scene_name': '00003',
            'merging_frame_num': 0,
            'static_comparison_frame_num': 15,
            'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        },
        # {
        #     'scene_name': '00004',
        #     'merging_frame_num': 0,
        #     'static_comparison_frame_num': 0,
        #     'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        # },
        # {
        #     'scene_name': '00006',
        #     'merging_frame_num': 0,
        #     'static_comparison_frame_num': 0,
        #     'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        # },
    ]
    frame_rate = 25

    video_maker = VideoMaker(configs)
    for sample_data in samples_data:
        scene_name = sample_data['scene_name']
        video = video_maker.make_video(models_data, sample_data)
        output_path = Path(f'../data/outputs/{this_filenum:02}_VideoComparisonMaker01c_{scene_name}.mp4')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        skvideo.io.vwrite(output_path.as_posix(), video,
                          inputdict={'-r': str(frame_rate)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p', '-profile:v': 'main'})
        del video
    return


def demo1d():
    """
    RegNeRF v/s SimpleNeRF
    :return:
    """
    configs = {
        'num_unmerged_loops': 1,
        'num_merging_frames': 25,
        'num_merged_loops': 1,
        'num_static_comparison_frames_per_loop': 100,
        'num_static_comparison_loops': 1,
        'demarcation_line_width': 10,
        'demarcation_line_color': VideoMaker.color_white,
        'annotation_height': 70,
        'annotation_font_size': 50,
        'background_color': VideoMaker.color_white,
    }
    models_data = [
        {
            'model_name': 'RegNeRF',
            'model_dirpath': '../../../view_synthesis/literature/013_RegNeRF/runs/testing/test0002/{scene_name}_video01/predicted_frames',
        },
        {
            'model_name': 'ViP-NeRF',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test0231/{scene_name}_video01/predicted_frames',
        },
    ]
    samples_data = [
        {
            'scene_name': '00000',
            'merging_frame_num': 0,
            'static_comparison_frame_num': 15,
            'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        },
        # {
        #     'scene_name': '00001',
        #     'merging_frame_num': 0,
        #     'static_comparison_frame_num': 0,
        #     'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        # },
        # {
        #     'scene_name': '00003',
        #     'merging_frame_num': 0,
        #     'static_comparison_frame_num': 15,
        #     'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        # },
        # {
        #     'scene_name': '00004',
        #     'merging_frame_num': 0,
        #     'static_comparison_frame_num': 0,
        #     'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        # },
        # {
        #     'scene_name': '00006',
        #     'merging_frame_num': 0,
        #     'static_comparison_frame_num': 0,
        #     'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        # },
    ]
    frame_rate = 25

    video_maker = VideoMaker(configs)
    for sample_data in samples_data:
        scene_name = sample_data['scene_name']
        video = video_maker.make_video(models_data, sample_data)
        output_path = Path(f'../data/outputs/{this_filenum:02}_VideoComparisonMaker01d_{scene_name}.mp4')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        skvideo.io.vwrite(output_path.as_posix(), video,
                          inputdict={'-r': str(frame_rate)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p', '-profile:v': 'main'})
        del video
    return


def demo1e1():
    """
    DS-NeRF v/s SimpleNeRF
    :return:
    """
    configs = {
        'num_unmerged_loops': 1,
        'num_merging_frames': 25,
        'num_merged_loops': 1,
        'num_static_comparison_frames_per_loop': 100,
        'num_static_comparison_loops': 1,
        'demarcation_line_width': 10,
        'demarcation_line_color': VideoMaker.color_white,
        'annotation_height': 70,
        'annotation_font_size': 50,
        'background_color': VideoMaker.color_white,
    }
    models_data = [
        {
            'model_name': 'DS-NeRF',
            'model_dirpath': '../../../view_synthesis/literature/009_DS_NeRF/runs/testing/test0012/{scene_name}/predicted_frames',
        },
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test0012/{scene_name}/predicted_frames',
        },
    ]
    samples_data = [
        # {
        #     'scene_name': '00000',
        #     'merging_frame_num': 0,
        #     'static_comparison_frame_num': 0,
        #     'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        # },
        # {
        #     'scene_name': '00001',
        #     'merging_frame_num': 0,
        #     'static_comparison_frame_num': 0,
        #     'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        # },
        # {
        #     'scene_name': '00003',
        #     'merging_frame_num': 0,
        #     'static_comparison_frame_num': 15,
        #     'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        # },
        {
            'scene_name': '00004',
            'merging_frame_num': 15,
            'static_comparison_frame_num': 15,
            'frame_nums': list(range(5, 26, 1)) + list(range(25, 4, -1))
        },
        {
            'scene_name': '00006',
            'merging_frame_num': 15,
            'static_comparison_frame_num': 15,
            'frame_nums': list(range(5, 26, 1)) + list(range(25, 4, -1))
        },
    ]
    frame_rate = 25

    video_maker = VideoMaker(configs)
    for sample_data in samples_data:
        scene_name = sample_data['scene_name']
        video = video_maker.make_video(models_data, sample_data)
        output_path = Path(f'../data/outputs/{this_filenum:02}_VideoComparisonMaker01e1_{scene_name}.mp4')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        skvideo.io.vwrite(output_path.as_posix(), video,
                          inputdict={'-r': str(frame_rate)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p', '-profile:v': 'main'})
        del video
    return


def demo1f1():
    """
    DDP-NeRF v/s SimpleNeRF
    :return:
    """
    configs = {
        'num_unmerged_loops': 1,
        'num_merging_frames': 25,
        'num_merged_loops': 1,
        'num_static_comparison_frames_per_loop': 100,
        'num_static_comparison_loops': 1,
        'demarcation_line_width': 10,
        'demarcation_line_color': VideoMaker.color_white,
        'annotation_height': 70,
        'annotation_font_size': 50,
        'background_color': VideoMaker.color_white,
    }
    models_data = [
        {
            'model_name': 'DDP-NeRF',
            'model_dirpath': '../../../view_synthesis/literature/010_DDP_NeRF/runs/testing/test0012/{scene_name}/predicted_frames',
        },
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test0012/{scene_name}/predicted_frames',
        },
    ]
    samples_data = [
        {
            'scene_name': '00000',
            'merging_frame_num': 22,
            'static_comparison_frame_num': 22,
            'frame_nums': list(range(5, 26, 1)) + list(range(25, 4, -1))
        },
        {
            'scene_name': '00003',
            'merging_frame_num': 15,
            'static_comparison_frame_num': 15,
            'frame_nums': list(range(5, 26, 1)) + list(range(25, 4, -1))
        },
        # {
        #     'scene_name': '00004',
        #     'merging_frame_num': 0,
        #     'static_comparison_frame_num': 0,
        #     'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        # },
        # {
        #     'scene_name': '00006',
        #     'merging_frame_num': 0,
        #     'static_comparison_frame_num': 0,
        #     'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        # },
    ]
    frame_rate = 25

    video_maker = VideoMaker(configs)
    for sample_data in samples_data:
        scene_name = sample_data['scene_name']
        video = video_maker.make_video(models_data, sample_data)
        output_path = Path(f'../data/outputs/{this_filenum:02}_VideoComparisonMaker01f1_{scene_name}.mp4')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        skvideo.io.vwrite(output_path.as_posix(), video,
                          inputdict={'-r': str(frame_rate)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p', '-profile:v': 'main'})
        del video
    return


def demo1h1():
    """
    ViP-NeRF v/s SimpleNeRF
    :return:
    """
    configs = {
        'num_unmerged_loops': 1,
        'num_merging_frames': 25,
        'num_merged_loops': 1,
        'num_static_comparison_frames_per_loop': 100,
        'num_static_comparison_loops': 1,
        'demarcation_line_width': 10,
        'demarcation_line_color': VideoMaker.color_white,
        'annotation_height': 70,
        'annotation_font_size': 50,
        'background_color': VideoMaker.color_white,
    }
    models_data = [
        {
            'model_name': 'ViP-NeRF',
            'model_dirpath': '../../../view_synthesis/literature/018_ViP_NeRF/runs/testing/test0012/{scene_name}/predicted_frames',
        },
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test0012/{scene_name}/predicted_frames',
        },
    ]
    samples_data = [
        # {
        #     'scene_name': '00000',
        #     'merging_frame_num': 0,
        #     'static_comparison_frame_num': 0,
        #     'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        # },
        {
            'scene_name': '00001',
            'merging_frame_num': 15,
            'static_comparison_frame_num': 15,
            'frame_nums': list(range(5, 26, 1)) + list(range(25, 4, -1))
        },
        # {
        #     'scene_name': '00003',
        #     'merging_frame_num': 0,
        #     'static_comparison_frame_num': 15,
        #     'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        # },
        # {
        #     'scene_name': '00004',
        #     'merging_frame_num': 0,
        #     'static_comparison_frame_num': 0,
        #     'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        # },
        # {
        #     'scene_name': '00006',
        #     'merging_frame_num': 0,
        #     'static_comparison_frame_num': 0,
        #     'frame_nums': list(range(0, 30, 1)) + list(range(29, -1, -1))
        # },
    ]
    frame_rate = 25

    video_maker = VideoMaker(configs)
    for sample_data in samples_data:
        scene_name = sample_data['scene_name']
        video = video_maker.make_video(models_data, sample_data)
        output_path = Path(f'../data/outputs/{this_filenum:02}_VideoComparisonMaker01h1_{scene_name}.mp4')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        skvideo.io.vwrite(output_path.as_posix(), video,
                          inputdict={'-r': str(frame_rate)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p', '-profile:v': 'main'})
        del video
    return


def demo2d1():
    """
    RegNeRF v/s SimpleNeRF
    :return:
    """
    configs = {
        'num_unmerged_loops': 1,
        'num_merging_frames': 30,
        'num_merged_loops': 0,
        'num_static_comparison_frames_per_loop': 120,
        'num_static_comparison_loops': 1,
        'demarcation_line_width': 10,
        'demarcation_line_color': VideoMaker.color_white,
        'annotation_height': 70,
        'annotation_font_size': 50,
        'background_color': VideoMaker.color_white,
    }
    models_data = [
        {
            'model_name': 'RegNeRF',
            'model_dirpath': '../../../view_synthesis/literature/011_RegNeRF/runs/testing/test1021/{scene_name}_Video02/predicted_frames',
        },
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test1142/{scene_name}_video01/predicted_frames',
        },
    ]
    samples_data = [
        {
            'scene_name': 'flower',
            'merging_frame_num': 19,
            'static_comparison_frame_num': 19,
        },
        {
            'scene_name': 'orchids',
            'merging_frame_num': 17,
            'static_comparison_frame_num': 17,
        },
    ]
    frame_rate = 30

    video_maker = VideoMaker(configs)
    for sample_data in samples_data:
        scene_name = sample_data['scene_name']
        video = video_maker.make_video(models_data, sample_data)
        output_path = Path(f'../data/outputs/{this_filenum:02}_VideoComparisonMaker02d1_{scene_name}.mp4')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        skvideo.io.vwrite(output_path.as_posix(), video,
                          inputdict={'-r': str(frame_rate)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p', '-profile:v': 'main'})
        del video
    return


def demo2e1():
    """
    DS-NeRF vs SimpleNeRF
    :return:
    """
    configs = {
        'num_unmerged_loops': 1,
        'num_merging_frames': 30,
        'num_merged_loops': 0,
        'num_static_comparison_frames_per_loop': 120,
        'num_static_comparison_loops': 1,
        'demarcation_line_width': 10,
        'demarcation_line_color': VideoMaker.color_white,
        'annotation_height': 70,
        'annotation_font_size': 50,
        'background_color': VideoMaker.color_white,
    }
    models_data = [
        {
            'model_name': 'DS-NeRF',
            'model_dirpath': '../../../view_synthesis/literature/009_DS_NeRF/runs/testing/test1012/{scene_name}_Video02/PredictedFrames',
        },
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test1142/{scene_name}_video01/predicted_frames',
        },
    ]
    samples_data = [
        # {
        #     'scene_name': 'fortress',
        #     'merging_frame_num': 19,
        #     'static_comparison_frame_num': 19,
        #     # 'merging_frame_num': 66,
        #     # 'static_comparison_frame_num': 66,
        # },
        {
            'scene_name': 'flower',
            'merging_frame_num': 20,
            'static_comparison_frame_num': 20,
        },
        # {
        #     'scene_name': 'trex',
        #     'merging_frame_num': 39,
        #     'static_comparison_frame_num': 39,
        # },
    ]
    frame_rate = 30

    video_maker = VideoMaker(configs)
    for sample_data in samples_data:
        scene_name = sample_data['scene_name']
        video = video_maker.make_video(models_data, sample_data)
        output_path = Path(f'../data/outputs/{this_filenum:02}_VideoComparisonMaker02e1_{scene_name}.mp4')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        skvideo.io.vwrite(output_path.as_posix(), video,
                          inputdict={'-r': str(frame_rate)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p', '-profile:v': 'main'})
        del video
    return


def demo2e2():
    """
    DS-NeRF vs SimpleNeRF static camera
    :return:
    """
    configs = {
        'num_unmerged_loops': 1,
        'num_merging_frames': 30,
        'num_merged_loops': 0,
        'num_static_comparison_frames_per_loop': 120,
        'num_static_comparison_loops': 1,
        'demarcation_line_width': 10,
        'demarcation_line_color': VideoMaker.color_white,
        'annotation_height': 70,
        'annotation_font_size': 50,
        'background_color': VideoMaker.color_white,
    }
    models_data = [
        {
            'model_name': 'DS-NeRF',
            'model_dirpath': '../../../view_synthesis/literature/009_DS_NeRF/runs/testing/test1012/{scene_name}_video04_static_camera/predicted_frames',
        },
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test1061/{scene_name}_video04_static_camera/predicted_frames',
        },
    ]
    samples_data = [
        # {
        #     'scene_name': 'fortress',
        #     'merging_frame_num': 19,
        #     'static_comparison_frame_num': 19,
        #     # 'merging_frame_num': 66,
        #     # 'static_comparison_frame_num': 66,
        # },
        {
            'scene_name': 'room',
            'merging_frame_num': 15,
            'static_comparison_frame_num': 15,
        },
        {
            'scene_name': 'trex',
            'merging_frame_num': 39,
            'static_comparison_frame_num': 39,
        },
    ]
    frame_rate = 30

    video_maker = VideoMaker(configs)
    for sample_data in samples_data:
        scene_name = sample_data['scene_name']
        video = video_maker.make_video(models_data, sample_data)
        output_path = Path(f'../data/outputs/{this_filenum:02}_VideoComparisonMaker02e2_{scene_name}.mp4')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        skvideo.io.vwrite(output_path.as_posix(), video,
                          inputdict={'-r': str(frame_rate)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p', '-profile:v': 'main'})
        del video
    return


def demo2f1():
    """
    DDP-NeRF v/s SimpleNeRF
    :return:
    """
    configs = {
        'num_unmerged_loops': 1,
        'num_merging_frames': 30,
        'num_merged_loops': 0,
        'num_static_comparison_frames_per_loop': 120,
        'num_static_comparison_loops': 1,
        'demarcation_line_width': 10,
        'demarcation_line_color': VideoMaker.color_white,
        'annotation_height': 70,
        'annotation_font_size': 50,
        'background_color': VideoMaker.color_white,
    }
    models_data = [
        {
            'model_name': 'DDP-NeRF',
            'model_dirpath': '../../../view_synthesis/literature/010_DDP_NeRF/runs/testing/test1022/{scene_name}_Video02/PredictedFrames',
        },
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test1142/{scene_name}_video01/predicted_frames',
        },
    ]
    samples_data = [
        {
            'scene_name': 'room',
            'merging_frame_num': 35,
            'static_comparison_frame_num': 35,
        },
        # {
        #     'scene_name': 'trex',
        #     'merging_frame_num': 86,
        #     'static_comparison_frame_num': 16,
        # },
    ]
    frame_rate = 30

    video_maker = VideoMaker(configs)
    for sample_data in samples_data:
        scene_name = sample_data['scene_name']
        video = video_maker.make_video(models_data, sample_data)
        output_path = Path(f'../data/outputs/{this_filenum:02}_VideoComparisonMaker02f1_{scene_name}.mp4')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        skvideo.io.vwrite(output_path.as_posix(), video,
                          inputdict={'-r': str(frame_rate)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p', '-profile:v': 'main'})
        del video
    return


def demo2f2():
    """
    DDP-NeRF v/s SimpleNeRF
    :return:
    """
    configs = {
        'num_unmerged_loops': 1,
        'num_merging_frames': 30,
        'num_merged_loops': 0,
        'num_static_comparison_frames_per_loop': 120,
        'num_static_comparison_loops': 1,
        'demarcation_line_width': 10,
        'demarcation_line_color': VideoMaker.color_white,
        'annotation_height': 70,
        'annotation_font_size': 50,
        'background_color': VideoMaker.color_white,
    }
    models_data = [
        {
            'model_name': 'DDP-NeRF',
            'model_dirpath': '../../../view_synthesis/literature/010_DDP_NeRF/runs/testing/test1012/{scene_name}_video04_static_camera/predicted_frames',
        },
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test1061/{scene_name}_video04_static_camera/predicted_frames',
        },
    ]
    samples_data = [
        {
            'scene_name': 'room',
            'merging_frame_num': 25,
            'static_comparison_frame_num': 25,
        },
        # {
        #     'scene_name': 'trex',
        #     'merging_frame_num': 86,
        #     'static_comparison_frame_num': 16,
        # },
    ]
    frame_rate = 30

    video_maker = VideoMaker(configs)
    for sample_data in samples_data:
        scene_name = sample_data['scene_name']
        video = video_maker.make_video(models_data, sample_data)
        output_path = Path(f'../data/outputs/{this_filenum:02}_VideoComparisonMaker02f2_{scene_name}.mp4')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        skvideo.io.vwrite(output_path.as_posix(), video,
                          inputdict={'-r': str(frame_rate)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p', '-profile:v': 'main'})
        del video
    return


def demo2g1():
    """
    FreeNeRF vs SimpleNeRF
    :return:
    """
    configs = {
        'num_unmerged_loops': 1,
        'num_merging_frames': 30,
        'num_merged_loops': 0,
        'num_static_comparison_frames_per_loop': 120,
        'num_static_comparison_loops': 1,
        'demarcation_line_width': 10,
        'demarcation_line_color': VideoMaker.color_white,
        'annotation_height': 70,
        'annotation_font_size': 50,
        'background_color': VideoMaker.color_white,
    }
    models_data = [
        {
            'model_name': 'FreeNeRF',
            'model_dirpath': '../../../view_synthesis/literature/017_FreeNeRF/runs/testing/test1012/{scene_name}_Video02/PredictedFrames',
        },
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test1061/{scene_name}_video01/predicted_frames',
        },
    ]
    samples_data = [
        {
            'scene_name': 'fortress',
            'merging_frame_num': 21,
            'static_comparison_frame_num': 21,
        },
        # {
        #     'scene_name': 'trex',
        #     'merging_frame_num': 23,
        #     'static_comparison_frame_num': 23,
        # },
    ]
    frame_rate = 30

    video_maker = VideoMaker(configs)
    for sample_data in samples_data:
        scene_name = sample_data['scene_name']
        video = video_maker.make_video(models_data, sample_data)
        output_path = Path(f'../data/outputs/{this_filenum:02}_VideoComparisonMaker02g1_{scene_name}.mp4')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        skvideo.io.vwrite(output_path.as_posix(), video,
                          inputdict={'-r': str(frame_rate)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p', '-profile:v': 'main'})
        del video
    return


def demo2g2():
    """
    FreeNeRF vs SimpleNeRF - static camera
    :return:
    """
    configs = {
        'num_unmerged_loops': 1,
        'num_merging_frames': 30,
        'num_merged_loops': 0,
        'num_static_comparison_frames_per_loop': 120,
        'num_static_comparison_loops': 1,
        'demarcation_line_width': 10,
        'demarcation_line_color': VideoMaker.color_white,
        'annotation_height': 70,
        'annotation_font_size': 50,
        'background_color': VideoMaker.color_white,
    }
    models_data = [
        {
            'model_name': 'FreeNeRF',
            'model_dirpath': '../../../view_synthesis/literature/017_FreeNeRF/runs/testing/test1012/{scene_name}_video04_static_camera/predicted_frames',
        },
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test1061/{scene_name}_video04_static_camera/predicted_frames',
        },
    ]
    samples_data = [
        {
            'scene_name': 'room',
            'merging_frame_num': 62,
            'static_comparison_frame_num': 62,
        },
        {
            'scene_name': 'trex',
            'merging_frame_num': 23,
            'static_comparison_frame_num': 23,
        },
    ]
    frame_rate = 30

    video_maker = VideoMaker(configs)
    for sample_data in samples_data:
        scene_name = sample_data['scene_name']
        video = video_maker.make_video(models_data, sample_data)
        output_path = Path(f'../data/outputs/{this_filenum:02}_VideoComparisonMaker02g2_{scene_name}.mp4')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        skvideo.io.vwrite(output_path.as_posix(), video,
                          inputdict={'-r': str(frame_rate)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p', '-profile:v': 'main'})
        del video
    return


def demo2h1():
    """
    ViP-NeRF vs SimpleNeRF
    :return:
    """
    configs = {
        'num_unmerged_loops': 1,
        'num_merging_frames': 30,
        'num_merged_loops': 0,
        'num_static_comparison_frames_per_loop': 120,
        'num_static_comparison_loops': 1,
        'demarcation_line_width': 10,
        'demarcation_line_color': VideoMaker.color_white,
        'annotation_height': 70,
        'annotation_font_size': 50,
        'background_color': VideoMaker.color_white,
    }
    models_data = [
        {
            'model_name': 'ViP-NeRF',
            'model_dirpath': '../../../view_synthesis/literature/018_ViP_NeRF/runs/testing/test1032/{scene_name}_Video02/PredictedFrames',
        },
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test1143/{scene_name}_video01/predicted_frames',
        },
    ]
    samples_data = [
        {
            'scene_name': 'horns',
            'merging_frame_num': 35,
            'static_comparison_frame_num': 35,
        },
        # {
        #     'scene_name': 'trex',
        #     'merging_frame_num': 23,
        #     'static_comparison_frame_num': 23,
        # },
    ]
    frame_rate = 30

    video_maker = VideoMaker(configs)
    for sample_data in samples_data:
        scene_name = sample_data['scene_name']
        video = video_maker.make_video(models_data, sample_data)
        output_path = Path(f'../data/outputs/{this_filenum:02}_VideoComparisonMaker02h1_{scene_name}.mp4')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        skvideo.io.vwrite(output_path.as_posix(), video,
                          inputdict={'-r': str(frame_rate)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p', '-profile:v': 'main'})
        del video
    return


def demo2h2():
    """
    ViP-NeRF vs SimpleNeRF - static camera
    :return:
    """
    configs = {
        'num_unmerged_loops': 1,
        'num_merging_frames': 30,
        'num_merged_loops': 0,
        'num_static_comparison_frames_per_loop': 120,
        'num_static_comparison_loops': 1,
        'demarcation_line_width': 10,
        'demarcation_line_color': VideoMaker.color_white,
        'annotation_height': 70,
        'annotation_font_size': 50,
        'background_color': VideoMaker.color_white,
    }
    models_data = [
        {
            'model_name': 'ViP-NeRF',
            'model_dirpath': '../../../view_synthesis/literature/018_ViP_NeRF/runs/testing/test1012/{scene_name}_video04_static_camera/predicted_frames',
        },
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test1061/{scene_name}_video04_static_camera/predicted_frames',
        },
    ]
    samples_data = [
        {
            'scene_name': 'room',
            'merging_frame_num': 62,
            'static_comparison_frame_num': 62,
        },
        {
            'scene_name': 'trex',
            'merging_frame_num': 23,
            'static_comparison_frame_num': 23,
        },
    ]
    frame_rate = 30

    video_maker = VideoMaker(configs)
    for sample_data in samples_data:
        scene_name = sample_data['scene_name']
        video = video_maker.make_video(models_data, sample_data)
        output_path = Path(f'../data/outputs/{this_filenum:02}_VideoComparisonMaker02h2_{scene_name}.mp4')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        skvideo.io.vwrite(output_path.as_posix(), video,
                          inputdict={'-r': str(frame_rate)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p', '-profile:v': 'main'})
        del video
    return


def demo4a():
    """
    w/o Points Augmentation
    :return:
    """
    configs = {
        'num_unmerged_loops': 1,
        'num_merging_frames': 30,
        'num_merged_loops': 0,
        'num_static_comparison_frames_per_loop': 120,
        'num_static_comparison_loops': 1,
        'demarcation_line_width': 10,
        'demarcation_line_color': VideoMaker.color_white,
        'annotation_height': 70,
        'annotation_font_size': 50,
        'background_color': VideoMaker.color_white,
    }
    models_data = [
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test1061/{scene_name}_video01/predicted_frames',
        },
        {
            'model_name': 'w/o Points Augmentation',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test1146/{scene_name}_video01/predicted_frames',
        },
    ]
    samples_data = [
        {
            'scene_name': 'fortress',
            'merging_frame_num': 26,
            'static_comparison_frame_num': 26,
        },
        # {
        #     'scene_name': 'trex',
        #     'merging_frame_num': 23,
        #     'static_comparison_frame_num': 23,
        # },
    ]
    frame_rate = 30

    video_maker = VideoMaker(configs)
    for sample_data in samples_data:
        scene_name = sample_data['scene_name']
        video = video_maker.make_video(models_data, sample_data)
        output_path = Path(f'../data/outputs/{this_filenum:02}_VideoComparisonMaker04a_{scene_name}.mp4')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        skvideo.io.vwrite(output_path.as_posix(), video,
                          inputdict={'-r': str(frame_rate)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p', '-profile:v': 'main'})
        del video
    return


def demo4b():
    """
    w/o Views Augmentation
    :return:
    """
    configs = {
        'num_unmerged_loops': 1,
        'num_merging_frames': 30,
        'num_merged_loops': 0,
        'num_static_comparison_frames_per_loop': 120,
        'num_static_comparison_loops': 1,
        'demarcation_line_width': 10,
        'demarcation_line_color': VideoMaker.color_white,
        'annotation_height': 70,
        'annotation_font_size': 50,
        'background_color': VideoMaker.color_white,
    }
    models_data = [
        {
            'model_name': 'w/o Views Augmentation',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test1147/{scene_name}_video04_static_camera/predicted_frames',
        },
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test1061/{scene_name}_video04_static_camera/predicted_frames',
        },
    ]
    samples_data = [
        {
            'scene_name': 'room',
            'merging_frame_num': 66,
            'static_comparison_frame_num': 66,
        },
        # {
        #     'scene_name': 'trex',
        #     'merging_frame_num': 23,
        #     'static_comparison_frame_num': 23,
        # },
    ]
    frame_rate = 30

    video_maker = VideoMaker(configs)
    for sample_data in samples_data:
        scene_name = sample_data['scene_name']
        video = video_maker.make_video(models_data, sample_data)
        output_path = Path(f'../data/outputs/{this_filenum:02}_VideoComparisonMaker04b_{scene_name}.mp4')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        skvideo.io.vwrite(output_path.as_posix(), video,
                          inputdict={'-r': str(frame_rate)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p', '-profile:v': 'main'})
        del video
    return


def demo4c1():
    """
    w/ Identical Augmentations
    :return:
    """
    configs = {
        'num_unmerged_loops': 1,
        'num_merging_frames': 30,
        'num_merged_loops': 0,
        'num_static_comparison_frames_per_loop': 120,
        'num_static_comparison_loops': 1,
        'demarcation_line_width': 10,
        'demarcation_line_color': VideoMaker.color_white,
        'annotation_height': 70,
        'annotation_font_size': 50,
        'background_color': VideoMaker.color_white,
    }
    models_data = [
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test1061/{scene_name}_video01/predicted_frames',
        },
        {
            'model_name': 'w/ Identical Augmentations',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test1154/{scene_name}_video01/predicted_frames',
        },
    ]
    samples_data = [
        {
            'scene_name': 'fortress',
            'merging_frame_num': 26,
            'static_comparison_frame_num': 26,
        },
        # {
        #     'scene_name': 'trex',
        #     'merging_frame_num': 23,
        #     'static_comparison_frame_num': 23,
        # },
    ]
    frame_rate = 30

    video_maker = VideoMaker(configs)
    for sample_data in samples_data:
        scene_name = sample_data['scene_name']
        video = video_maker.make_video(models_data, sample_data)
        output_path = Path(f'../data/outputs/{this_filenum:02}_VideoComparisonMaker04c1_{scene_name}.mp4')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        skvideo.io.vwrite(output_path.as_posix(), video,
                          inputdict={'-r': str(frame_rate)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p', '-profile:v': 'main'})
        del video
    return


def demo4c2():
    """
    w/ Identical Augmentations
    :return:
    """
    configs = {
        'num_unmerged_loops': 1,
        'num_merging_frames': 30,
        'num_merged_loops': 0,
        'num_static_comparison_frames_per_loop': 120,
        'num_static_comparison_loops': 1,
        'demarcation_line_width': 10,
        'demarcation_line_color': VideoMaker.color_white,
        'annotation_height': 70,
        'annotation_font_size': 50,
        'background_color': VideoMaker.color_white,
    }
    models_data = [
        {
            'model_name': 'w/ Identical Augmentations',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test1154/{scene_name}_video04_static_camera/predicted_frames',
        },
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': '../../../view_synthesis/research/001_NeRF/runs/testing/test1061/{scene_name}_video04_static_camera/predicted_frames',
        },
    ]
    samples_data = [
        {
            'scene_name': 'room',
            'merging_frame_num': 66,
            'static_comparison_frame_num': 66,
        },
        # {
        #     'scene_name': 'trex',
        #     'merging_frame_num': 23,
        #     'static_comparison_frame_num': 23,
        # },
    ]
    frame_rate = 30

    video_maker = VideoMaker(configs)
    for sample_data in samples_data:
        scene_name = sample_data['scene_name']
        video = video_maker.make_video(models_data, sample_data)
        output_path = Path(f'../data/outputs/{this_filenum:02}_VideoComparisonMaker04c2_{scene_name}.mp4')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        skvideo.io.vwrite(output_path.as_posix(), video,
                          inputdict={'-r': str(frame_rate)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p', '-profile:v': 'main'})
        del video
    return


def main():
    demo1e1()  # DS-NeRF
    demo1f1()  # DDP-NeRF
    demo1h1()  # ViP-NeRF
    demo2d1()  # RegNeRF
    demo2e1()  # DS-NeRF
    demo2e2()
    demo2f1()  # DDP-NeRF
    demo2f2()
    demo2g1()  # FreeNeRF
    demo2h1()  # ViP-NeRF
    demo4a()  # w/o points augmentation
    demo4b()  # w/o views augmentation
    demo4c1()  # w/ identical augmentation
    demo4c2()  # w/ identical augmentation
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
