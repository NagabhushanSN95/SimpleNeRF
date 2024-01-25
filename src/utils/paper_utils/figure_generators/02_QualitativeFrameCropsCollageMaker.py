# Shree KRISHNAya Namaha
# Makes collage for qualitative comparisons. Multiple crops of a full frame.
# For SIGGRAPH 2023 submission.
# Author: Nagabhushan S N
# Last Modified: 03/05/2023

import shutil
import time
import datetime
import traceback
from typing import List, Optional, Tuple

import cv2
import numpy
import skimage.io
import skimage.transform

from pathlib import Path

import sympy
from PIL import ImageFont, Image, ImageDraw
from matplotlib import pyplot

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class CollageMaker:
    def __init__(self, configs: dict):
        self.configs = configs
        self.frame_height, self.frame_width = configs['frame_size']
        self.crop_size = configs['crop_size']
        self.full_frame_model_name = configs['full_frame_model_name']

        self.color_black = numpy.array([0, 0, 0])
        self.color_white = numpy.array([255, 255, 255])
        self.color_red = numpy.array([255, 0, 0])
        self.color_green = numpy.array([0, 255, 0])
        self.color_blue = numpy.array([0, 0, 255])

        self.tmp_dirpath = Path('../tmp/')
        if self.tmp_dirpath.exists():
            shutil.rmtree(self.tmp_dirpath.as_posix())
        self.tmp_dirpath.mkdir(parents=True, exist_ok=True)
        return

    def create_collage(self, collage_data: list, models_data: list):
        rows = []
        white_row = None
        crop_white_row = None
        annotation_done = False

        for sample_collage_data in collage_data:
            scene_name = sample_collage_data['scene_name']
            frame_num = sample_collage_data['frame_num']

            full_frame = None
            frames, model_names = [], []
            for i, model_data in enumerate(models_data):
                model_name = model_data['model_name']
                model_dirpath = model_data['model_dirpath']
                scene_dir_suffix = model_data.get('scene_dir_suffix', '')
                frames_dirname = model_data['frames_dirname']

                frame_path = model_dirpath / f'{scene_name}{scene_dir_suffix}/{frames_dirname}/{frame_num:04}.png'
                frame = self.read_image(frame_path)

                if model_name == self.full_frame_model_name:
                    full_frame = frame

                crops = []
                for j, crop_data in enumerate(sample_collage_data['crops']):
                    y1, x1, ph, pw, bb_color = crop_data
                    y2, x2 = y1 + ph, x1 + pw
                    crop = frame[y1:y2, x1:x2]
                    resized_crop = self.resize_image(crop, *self.crop_size)
                    resized_crop = cv2.rectangle(resized_crop.copy(), (0, 0), self.crop_size[::-1], color=bb_color.tolist(), thickness=self.configs['bounding_box_thickness'])
                    # Arrows
                    if ('arrows' in sample_collage_data) and (sample_collage_data['arrows'][j] is not None) and (sample_collage_data['arrows'][j][i]):
                        pt1, pt2, arrow_color, arrow_thickness = sample_collage_data['arrows'][j][i]
                        resized_crop = cv2.arrowedLine(resized_crop.copy(), pt1, pt2, arrow_color, arrow_thickness)
                    crops.append(resized_crop)
                    if crop_white_row is None:
                        crop_white_row = self.color_white[None, None] * numpy.ones(shape=(self.configs['white_row_height'] // len(sample_collage_data['crops']), crops[0].shape[1], 3), dtype=numpy.uint8)
                    crops.append(crop_white_row)

                    if model_name == self.full_frame_model_name:
                        full_frame = cv2.rectangle(full_frame.copy(), (x1, y1), (x2, y2), color=bb_color.tolist(), thickness=self.configs['bounding_box_thickness'])
                crops_column = numpy.concatenate(crops[:-1], axis=0)
                frames.append(crops_column)
                model_names.append(model_name)

            if not annotation_done:
                annotations = [(model_name, self.is_latex_annotation_required(model_name)) for model_name in model_names]
                annotation_done = True
            else:
                annotations = None

            if sample_collage_data.get('full_frame_crop', None) is not None:
                y1, x1, ph, pw = sample_collage_data.get('full_frame_crop', None)
                y2, x2 = y1 + ph, x1 + pw
                full_frame = full_frame[y1:y2, x1:x2]
            frames.insert(0, full_frame)
            if annotations is not None:
                annotations.insert(0, (self.full_frame_model_name, self.is_latex_annotation_required(self.full_frame_model_name)))

            row = self.create_sample_collage(frames, annotations, output_height=self.frame_height)
            rows.append(row)
            if white_row is None:
                white_row = self.color_white[None, None] * numpy.ones(shape=(self.configs['white_row_height'], rows[0].shape[1], 3), dtype=numpy.uint8)
            rows.append(white_row)

        rows = rows[:-1]
        collage = numpy.concatenate(rows, axis=0)
        return collage

    def create_sample_collage(self, images: List[numpy.ndarray], annotations: List[Tuple[str, bool]], output_height: int):
        if annotations is None:
            row_height = output_height
        else:
            row_height = output_height + self.configs['annotation_height']
        white_column = 255 * numpy.ones(shape=(row_height, self.configs['white_column_width'], 3), dtype=numpy.uint8)

        resized_images = []

        for i in range(len(images)):
            resized_image = self.resize_image(images[i], output_height)
            if annotations is not None:
                resized_image = self.annotate_frame(resized_image, annotations[i][0], self.configs['annotation_height'],
                                                    self.configs['annotation_font_size'], latex=annotations[i][1])

            resized_images.append(resized_image)
            resized_images.append(white_column)

        collage = numpy.concatenate(resized_images, axis=1)
        return collage

    def annotate_frame(self, frame: numpy.ndarray, text: str, annotation_height: int, font_size: int,
                       latex: bool = False):
        if not latex:
            annotation = self.get_annotation((annotation_height, frame.shape[1]), text, font_size)
        else:
            annotation = self.get_latex_annotation((annotation_height, frame.shape[1]), text, font_size)
        annotated_frame = numpy.concatenate([annotation, frame], axis=0)
        return annotated_frame

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

    def get_latex_annotation(self, anno_shape: tuple, title: str, font_size: int = 50):
        annotation_path = self.tmp_dirpath / 'Annotation.png'
        sympy.preview(title, dvioptions=["-T", "tight", "-z", "0", "--truecolor", f"-D {16 * font_size}"], viewer='file',
                      filename=annotation_path.as_posix(), euler=False)
        annotation1 = skimage.io.imread(annotation_path.as_posix())
        annotation1 = self.resize_image(annotation1, annotation1.shape[0] // 3)
        h, w = anno_shape
        h1, w1 = annotation1.shape[:2]
        ph = (h - h1) // 2
        pw = (w - w1) // 2

        annotation = 255 * numpy.ones((h, w, 3), dtype=numpy.uint8)
        annotation[ph:ph+h1, pw:pw+w1] = annotation1
        return annotation

    @staticmethod
    def is_latex_annotation_required(annotation_text: str) -> bool:
        return '$' in annotation_text

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
    def resize_image(image: numpy.ndarray, height: int = None, width: int = None):
        if (width is None) and (height is None):
            return image

        h, w = image.shape[:2]
        if width is None:
            width = int(round(height * w / h))
        if height is None:
            height = int(round(width * h / w))
        resized_image = skimage.transform.resize(image, output_shape=(height, width), preserve_range=True,
                                                 anti_aliasing=True).astype('uint8')
        return resized_image

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
    def depth2image(depth: numpy.ndarray):
        norm_depth = numpy.round(depth / depth.max() * 255).astype('uint8')
        depth_image = numpy.stack([norm_depth] * 3, axis=2)
        return depth_image


def demo1a():
    configs = {
        'frame_size': (576, 1024),
        'crop_size': (100, 100),
        'full_frame_model_name': 'SimpleNeRF',
        'bounding_box_thickness': 5,
        'annotation_height': 50,
        'annotation_font_size': 30,
        'white_column_width': 10,
        'white_row_height': 10,
        'final_width': None,
    }
    collage_maker = CollageMaker(configs)
    models_data = [
        # {
        #     'model_name': 'InfoNeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/012_InfoNeRF/runs/testing/test0011'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        # {
        #     'model_name': 'DietNeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/008_DietNeRF/runs/testing/test0011'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        {
            'model_name': 'RegNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/011_RegNeRF/runs/testing/test0012'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'DS-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/009_DS_NeRF/runs/testing/test0012'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'DDP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/010_DDP_NeRF/runs/testing/test0012'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'FreeNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/017_FreeNeRF/runs/testing/test0012'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'ViP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/018_ViP_NeRF/runs/testing/test0012'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0012'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'Ground Truth',
            'model_dirpath': Path('../../../../databases/RealEstate10K/data/test/database_data/'),
            'frames_dirname': 'rgb',
            # 'depths_dirname': None,
        },
    ]
    collage_data = [
        {
            'scene_name': '00006',
            'frame_num': 25,
            'crops': [(453, 193, 100, 100, collage_maker.color_red), (258, 269, 100, 100, collage_maker.color_green)],
        },
        # {
        #     'scene_name': '00000',
        #     'frame_num': 32,
        #     'crops': [()],
        # },
    ]
    collage = collage_maker.create_collage(collage_data, models_data)
    output_path = Path('../data/outputs/02_QualitativeComparisons01a.jpg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(output_path.as_posix(), collage)
    return


def demo1b():
    configs = {
        'frame_size': (576, 1024),
        'crop_size': (100, 100),
        'full_frame_model_name': 'SimpleNeRF',
        'bounding_box_thickness': 5,
        'annotation_height': 70,
        'annotation_font_size': 45,
        'white_column_width': 10,
        'white_row_height': 10,
        'final_width': None,
    }
    collage_maker = CollageMaker(configs)
    models_data = [
        # {
        #     'model_name': 'InfoNeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/012_InfoNeRF/runs/testing/test0011'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        # {
        #     'model_name': 'DietNeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/008_DietNeRF/runs/testing/test0011'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        {
            'model_name': 'RegNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/011_RegNeRF/runs/testing/test0022'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'DS-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/009_DS_NeRF/runs/testing/test0022'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'DDP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/010_DDP_NeRF/runs/testing/test0022'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'FreeNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/017_FreeNeRF/runs/testing/test0022'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'ViP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/018_ViP_NeRF/runs/testing/test0022'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0042'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'Ground Truth',
            'model_dirpath': Path('../../../../databases/RealEstate10K/data/test/database_data/'),
            'frames_dirname': 'rgb',
            # 'depths_dirname': None,
        },
    ]
    collage_data = [
        {
            'scene_name': '00001',
            'frame_num': 24,
            'crops': [(62, 893, 100, 100, collage_maker.color_red), (375, 893, 100, 100, collage_maker.color_green)],
        },
        # {
        #     'scene_name': '00000',
        #     'frame_num': 32,
        #     'crops': [()],
        # },
    ]
    collage = collage_maker.create_collage(collage_data, models_data)
    output_path = Path('../data/outputs/02_QualitativeComparisons01b.jpg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(output_path.as_posix(), collage)
    return


def demo1c():
    configs = {
        'frame_size': (576, 1024),
        'crop_size': (100, 100),
        'full_frame_model_name': 'SimpleNeRF',
        'bounding_box_thickness': 5,
        'annotation_height': 50,
        'annotation_font_size': 30,
        'white_column_width': 10,
        'white_row_height': 10,
        'final_width': None,
    }
    collage_maker = CollageMaker(configs)
    models_data = [
        # {
        #     'model_name': 'InfoNeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/012_InfoNeRF/runs/testing/test0031'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        # {
        #     'model_name': 'DietNeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/008_DietNeRF/runs/testing/test0031'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        {
            'model_name': 'RegNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/011_RegNeRF/runs/testing/test0032'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'DS-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/009_DS_NeRF/runs/testing/test0032'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'DDP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/010_DDP_NeRF/runs/testing/test0032'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'FreeNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/017_FreeNeRF/runs/testing/test0032'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'ViP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/018_ViP_NeRF/runs/testing/test0032'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0043'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'Ground Truth',
            'model_dirpath': Path('../../../../databases/RealEstate10K/data/test/database_data/'),
            'frames_dirname': 'rgb',
            # 'depths_dirname': None,
        },
    ]
    collage_data = [
        # {
        #     'scene_name': '00003',
        #     'frame_num': 5,
        #     'crops': [(128, 768, 128, 128, collage_maker.color_red), (60, 250, 50, 50, collage_maker.color_green)],
        #     'arrows': [
        #         [None, None, None, None, None, None, None],
        #         [None, None, None, None, ((10, 50), (45, 50), (255, 0, 255), 2), ((10, 50), (45, 50), (255, 0, 255), 2), None],
        #     ]
        # },
        {
            'scene_name': '00000',
            'frame_num': 6,
            'crops': [(384, 384, 128, 128, collage_maker.color_red), (280, 435, 100, 100, collage_maker.color_green), (349, 334, 50, 50, collage_maker.color_blue)],
            'arrows': [
                [None, None, None, None, ((75, 25), (40, 70), (255, 0, 255), 2), None, None],
                [None, None, None, None, None, None, None],
                [None, None, None, None, ((10, 50), (65, 50), (255, 0, 255), 2), None, None],
            ]
        },
    ]
    collage = collage_maker.create_collage(collage_data, models_data)
    output_path = Path('../data/outputs/02_QualitativeComparisons01c.jpg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(output_path.as_posix(), collage)
    return


def demo2a():
    configs = {
        'frame_size': (756, 1008),
        'crop_size': (100, 100),
        'full_frame_model_name': 'SimpleNeRF',
        'bounding_box_thickness': 5,
        'annotation_height': 60,
        'annotation_font_size': 40,
        'white_column_width': 10,
        'white_row_height': 10,
        'final_width': None,
    }
    collage_maker = CollageMaker(configs)
    models_data = [
        # {
        #     'model_name': 'InfoNeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/012_InfoNeRF/runs/testing/test1011'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        # {
        #     'model_name': 'DietNeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/008_DietNeRF/runs/testing/test1011'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        {
            'model_name': 'RegNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/011_RegNeRF/runs/testing/test1012'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'DS-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/009_DS_NeRF/runs/testing/test1012'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'DDP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/010_DDP_NeRF/runs/testing/test1012'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'FreeNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/017_FreeNeRF/runs/testing/test1012'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'ViP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/018_ViP_NeRF/runs/testing/test1012'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1061'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'Ground Truth',
            'model_dirpath': Path('../../../../databases/NeRF_LLFF/data/all/database_data/'),
            'frames_dirname': 'rgb_down4',
            # 'depths_dirname': None,
        },
    ]
    collage_data = [
        # {
        #     'scene_name': 'flower',
        #     'frame_num': 24,
        #     'crops': [(596, 222, 150, 150, collage_maker.color_red), (637, 397, 100, 100, collage_maker.color_green)],
        #     # 'arrows': [
        #     #     [None, None, None, ((160, 190), (50, 150), (0, 0, 255), 4), ((10, 75), (120, 150), (0, 0, 255), 4), None, None],
        #     #     [None, None, ((10, 190), (80, 110), (255, 0, 255), 4), None, ((40, 160), (70, 60), (0, 255, 255), 4), None, None],
        #     # ]
        # },
        {
            'scene_name': 'horns',
            'frame_num': 32,
            'crops': [(363, 808, 100, 100, collage_maker.color_red), (384, 726, 100, 100, collage_maker.color_green)],
            # 'crops': [(363, 808, 100, 100, collage_maker.color_red), (384, 726, 100, 100, collage_maker.color_green), (213, 482, 100, 100, collage_maker.color_blue)],
            # 'arrows': [
            #     [None, None, ((60, 90), (140, 60), (255, 0, 255), 4), ((40, 120), (60, 40), (0, 0, 255), 4), ((40, 110), (60, 30), (0, 0, 255), 4), None, None],
            #     [None, None, ((20, 180), (100, 130), (255, 0, 255), 4), None, ((20, 180), (90, 120), (0, 255, 255), 4), None, None],
            # ]
        },
    ]
    collage = collage_maker.create_collage(collage_data, models_data)
    output_path = Path('../data/outputs/02_QualitativeComparisons02a.jpg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(output_path.as_posix(), collage)
    return


def demo2b():
    configs = {
        'frame_size': (756, 1008),
        'crop_size': (200, 200),
        'full_frame_model_name': 'SimpleNeRF',
        'bounding_box_thickness': 5,
        'annotation_height': 60,
        'annotation_font_size': 40,
        'white_column_width': 10,
        'white_row_height': 10,
        'final_width': None,
    }
    collage_maker = CollageMaker(configs)
    models_data = [
        # {
        #     'model_name': 'InfoNeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/012_InfoNeRF/runs/testing/test1021'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        # {
        #     'model_name': 'DietNeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/008_DietNeRF/runs/testing/test1021'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        {
            'model_name': 'RegNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/011_RegNeRF/runs/testing/test1022'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'DS-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/009_DS_NeRF/runs/testing/test1022'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'DDP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/010_DDP_NeRF/runs/testing/test1022'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'FreeNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/017_FreeNeRF/runs/testing/test1022'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'ViP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/018_ViP_NeRF/runs/testing/test1022'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1142'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'Ground Truth',
            'model_dirpath': Path('../../../../databases/NeRF_LLFF/data/all/database_data/'),
            'frames_dirname': 'rgb_down4',
            # 'depths_dirname': None,
        },
    ]
    collage_data = [
        # {
        #     'scene_name': 'trex',
        #     'frame_num': 48,
        #     # 'full_frame_crop': [0, 88, 756, 920],
        #     'crops': [(129, 405, 125, 125, collage_maker.color_red), (538, 289, 50, 50, collage_maker.color_green), (102, 296, 50, 50, collage_maker.color_blue)],
        #     # 'arrows': [
        #     #     [None, None, None, ((160, 190), (50, 150), (0, 0, 255), 4), ((10, 75), (120, 150), (0, 0, 255), 4), None, None],
        #     #     [None, None, ((10, 190), (80, 110), (255, 0, 255), 4), None, ((40, 160), (70, 60), (0, 255, 255), 4), None, None],
        #     # ]
        # },
        {
            'scene_name': 'orchids',
            'frame_num': 0,
            # 'full_frame_crop': [0, 88, 756, 920],
            'crops': [(544, 520, 100, 100, collage_maker.color_red), (252, 565, 50, 50, collage_maker.color_green)],
            # 'arrows': [
            #     [None, None, None, ((160, 190), (50, 150), (0, 0, 255), 4), ((10, 75), (120, 150), (0, 0, 255), 4), None, None],
            #     [None, None, ((10, 190), (80, 110), (255, 0, 255), 4), None, ((40, 160), (70, 60), (0, 255, 255), 4), None, None],
            # ]
        },
    ]
    collage = collage_maker.create_collage(collage_data, models_data)
    output_path = Path('../data/outputs/02_QualitativeComparisons02b.jpg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(output_path.as_posix(), collage)
    return


def demo2c():
    configs = {
        'frame_size': (756, 1008),
        'crop_size': (100, 100),
        'full_frame_model_name': 'SimpleNeRF',
        'bounding_box_thickness': 5,
        'annotation_height': 60,
        'annotation_font_size': 40,
        'white_column_width': 10,
        'white_row_height': 10,
        'final_width': None,
    }
    collage_maker = CollageMaker(configs)
    models_data = [
        # {
        #     'model_name': 'InfoNeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/012_InfoNeRF/runs/testing/test1031'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        # {
        #     'model_name': 'DietNeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/008_DietNeRF/runs/testing/test1031'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        {
            'model_name': 'RegNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/011_RegNeRF/runs/testing/test1032'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'DS-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/009_DS_NeRF/runs/testing/test1032'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'DDP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/010_DDP_NeRF/runs/testing/test1032'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'FreeNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/017_FreeNeRF/runs/testing/test1032'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'ViP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/018_ViP_NeRF/runs/testing/test1032'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1143'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'Ground Truth',
            'model_dirpath': Path('../../../../databases/NeRF_LLFF/data/all/database_data/'),
            'frames_dirname': 'rgb_down4',
            # 'depths_dirname': None,
        },
    ]
    collage_data = [
        {
            'scene_name': 'flower',
            'frame_num': 32,
            'full_frame_crop': [0, 88, 756, 920],
            'crops': [(194, 283, 75, 75, collage_maker.color_red), (4, 563, 100, 100, collage_maker.color_green)],
            # 'arrows': [
            #     [None, None, None, ((160, 190), (50, 150), (0, 0, 255), 4), ((10, 75), (120, 150), (0, 0, 255), 4), None, None],
            #     [None, None, ((10, 190), (80, 110), (255, 0, 255), 4), None, ((40, 160), (70, 60), (0, 255, 255), 4), None, None],
            # ]
        },
        # {
        #     'scene_name': 'orchids',
        #     'frame_num': 0,
        #     # 'full_frame_crop': [0, 0, 756, 920],
        #     'crops': [(102, 638, 75, 75, collage_maker.color_red), (392, 378, 75, 75, collage_maker.color_green)],
        #     # 'arrows': [
        #     #     [None, None, ((60, 90), (140, 60), (255, 0, 255), 4), ((40, 120), (60, 40), (0, 0, 255), 4), ((40, 110), (60, 30), (0, 0, 255), 4), None, None],
        #     #     [None, None, ((20, 180), (100, 130), (255, 0, 255), 4), None, ((20, 180), (90, 120), (0, 255, 255), 4), None, None],
        #     # ]
        # },
    ]
    collage = collage_maker.create_collage(collage_data, models_data)
    output_path = Path('../data/outputs/02_QualitativeComparisons02c.jpg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(output_path.as_posix(), collage)
    return


def demo2d():
    configs = {
        'frame_size': (756, 1008),
        'crop_size': (100, 100),
        'full_frame_model_name': '4 i/p views',
        'bounding_box_thickness': 5,
        'annotation_height': 100,
        'annotation_font_size': 80,
        'white_column_width': 10,
        'white_row_height': 10,
        'final_width': None,
    }
    collage_maker = CollageMaker(configs)
    models_data = [
        {
            'model_name': '2 i/p views',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1061'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': '3 i/p views',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1142'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': '4 i/p views',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1143'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'Ground Truth',
            'model_dirpath': Path('../../../../databases/NeRF_LLFF/data/all/database_data/'),
            'frames_dirname': 'rgb_down4',
            # 'depths_dirname': None,
        },
    ]
    collage_data = [
        {
            'scene_name': 'trex',
            'frame_num': 48,
            # 'full_frame_crop': [0, 88, 756, 920],
            'crops': [(53, 384, 100, 100, collage_maker.color_red)],
            # 'arrows': [
            #     [None, None, None, ((160, 190), (50, 150), (0, 0, 255), 4), ((10, 75), (120, 150), (0, 0, 255), 4), None, None],
            #     [None, None, ((10, 190), (80, 110), (255, 0, 255), 4), None, ((40, 160), (70, 60), (0, 255, 255), 4), None, None],
            # ]
        },
        # {
        #     'scene_name': 'orchids',
        #     'frame_num': 0,
        #     # 'full_frame_crop': [0, 0, 756, 920],
        #     'crops': [(102, 638, 75, 75, collage_maker.color_red), (392, 378, 75, 75, collage_maker.color_green)],
        #     # 'arrows': [
        #     #     [None, None, ((60, 90), (140, 60), (255, 0, 255), 4), ((40, 120), (60, 40), (0, 0, 255), 4), ((40, 110), (60, 30), (0, 0, 255), 4), None, None],
        #     #     [None, None, ((20, 180), (100, 130), (255, 0, 255), 4), None, ((20, 180), (90, 120), (0, 255, 255), 4), None, None],
        #     # ]
        # },
    ]
    collage = collage_maker.create_collage(collage_data, models_data)
    output_path = Path('../data/outputs/02_QualitativeComparisons02d.jpg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(output_path.as_posix(), collage)
    return


def demo2e():
    configs = {
        'frame_size': (756, 1008),
        'crop_size': (200, 200),
        'full_frame_model_name': 'SimpleNeRF',
        'bounding_box_thickness': 10,
        'annotation_height': 100,
        'annotation_font_size': 80,
        'white_column_width': 10,
        'white_row_height': 10,
        'final_width': None,
    }
    collage_maker = CollageMaker(configs)
    models_data = [
        # {
        #     'model_name': 'InfoNeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/012_InfoNeRF/runs/testing/test1011'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        # {
        #     'model_name': 'DietNeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/008_DietNeRF/runs/testing/test1011'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        # {
        #     'model_name': 'RegNeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/011_RegNeRF/runs/testing/test1012'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        {
            'model_name': 'DS-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/009_DS_NeRF/runs/testing/test1012'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
            'scene_dir_suffix': '_video04_static_camera',
        },
        # {
        #     'model_name': 'DDP-NeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/010_DDP_NeRF/runs/testing/test1012'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        # {
        #     'model_name': 'FreeNeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/017_FreeNeRF/runs/testing/test1012'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        # {
        #     'model_name': 'ViP-NeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/018_ViP_NeRF/runs/testing/test1012'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1061'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
            'scene_dir_suffix': '_video04_static_camera',
        },
        {
            'model_name': 'Dense-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/007_NeRFs_Simplified/runs/testing/test1001'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
            'scene_dir_suffix': '_video04_static_camera',
        },
        # {
        #     'model_name': 'Ground Truth',
        #     'model_dirpath': Path('../../../../databases/NeRF_LLFF/data/all/database_data/'),
        #     'frames_dirname': 'rgb_down4',
        #     # 'depths_dirname': None,
        # },
    ]
    collage_data = [
        {
            'scene_name': 'room',
            'frame_num': 15,
            'crops': [(453, 424, 250, 250, collage_maker.color_red)],
            # 'arrows': [
            #     [None, None, None, ((160, 190), (50, 150), (0, 0, 255), 4), ((10, 75), (120, 150), (0, 0, 255), 4), None, None],
            #     [None, None, ((10, 190), (80, 110), (255, 0, 255), 4), None, ((40, 160), (70, 60), (0, 255, 255), 4), None, None],
            # ]
        },
        # {
        #     'scene_name': 'horns',
        #     'frame_num': 32,
        #     'crops': [(0, 588, 150, 150, collage_maker.color_red), (83, 443, 75, 75, collage_maker.color_green)],
        #     'arrows': [
        #         [None, None, ((60, 90), (140, 60), (255, 0, 255), 4), ((40, 120), (60, 40), (0, 0, 255), 4), ((40, 110), (60, 30), (0, 0, 255), 4), None, None],
        #         [None, None, ((20, 180), (100, 130), (255, 0, 255), 4), None, ((20, 180), (90, 120), (0, 255, 255), 4), None, None],
        #     ]
        # },
    ]
    collage = collage_maker.create_collage(collage_data, models_data)
    output_path = Path('../data/outputs/02_QualitativeComparisons02e.jpg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(output_path.as_posix(), collage)
    return


def demo2f():
    configs = {
        'frame_size': (756, 1008),
        'crop_size': (200, 200),
        'full_frame_model_name': 'SimpleNeRF',
        'bounding_box_thickness': 10,
        'annotation_height': 100,
        'annotation_font_size': 80,
        'white_column_width': 10,
        'white_row_height': 10,
        'final_width': None,
    }
    collage_maker = CollageMaker(configs)
    models_data = [
        # {
        #     'model_name': 'InfoNeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/012_InfoNeRF/runs/testing/test1011'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        # {
        #     'model_name': 'DietNeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/008_DietNeRF/runs/testing/test1011'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        # {
        #     'model_name': 'RegNeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/011_RegNeRF/runs/testing/test1012'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        {
            'model_name': 'DS-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/009_DS_NeRF/runs/testing/test1012'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        # {
        #     'model_name': 'DDP-NeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/010_DDP_NeRF/runs/testing/test1012'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        # {
        #     'model_name': 'FreeNeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/017_FreeNeRF/runs/testing/test1012'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        # {
        #     'model_name': 'ViP-NeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/018_ViP_NeRF/runs/testing/test1012'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1061'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'Dense-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/007_NeRFs_Simplified/runs/testing/test1001'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        # {
        #     'model_name': 'Ground Truth',
        #     'model_dirpath': Path('../../../../databases/NeRF_LLFF/data/all/database_data/'),
        #     'frames_dirname': 'rgb_down4',
        #     'depths_dirname': None,
        # },
    ]
    collage_data = [
        {
            'scene_name': 'horns',
            'frame_num': 40,
            'crops': [(221, 439, 150, 150, collage_maker.color_red)],
            # 'arrows': [
            #     [None, None, None, ((160, 190), (50, 150), (0, 0, 255), 4), ((10, 75), (120, 150), (0, 0, 255), 4), None, None],
            #     [None, None, ((10, 190), (80, 110), (255, 0, 255), 4), None, ((40, 160), (70, 60), (0, 255, 255), 4), None, None],
            # ]
        },
        # {
        #     'scene_name': 'horns',
        #     'frame_num': 32,
        #     'crops': [(0, 588, 150, 150, collage_maker.color_red), (83, 443, 75, 75, collage_maker.color_green)],
        #     'arrows': [
        #         [None, None, ((60, 90), (140, 60), (255, 0, 255), 4), ((40, 120), (60, 40), (0, 0, 255), 4), ((40, 110), (60, 30), (0, 0, 255), 4), None, None],
        #         [None, None, ((20, 180), (100, 130), (255, 0, 255), 4), None, ((20, 180), (90, 120), (0, 255, 255), 4), None, None],
        #     ]
        # },
    ]
    collage = collage_maker.create_collage(collage_data, models_data)
    output_path = Path('../data/outputs/02_QualitativeComparisons02f.jpg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(output_path.as_posix(), collage)
    return


def demo2g():
    """
    Ablation w/o coarse-fine consistency
    :return:
    """
    configs = {
        'frame_size': (756, 1008),
        'crop_size': (200, 200),
        'full_frame_model_name': 'SimpleNeRF',
        'bounding_box_thickness': 10,
        'annotation_height': 100,
        'annotation_font_size': 80,
        'white_column_width': 10,
        'white_row_height': 10,
        'final_width': None,
    }
    collage_maker = CollageMaker(configs)
    models_data = [
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1061'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': r'w/o Loss $\mathcal{L}_{\text{cfc}}$',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1113'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'Dense-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/007_NeRFs_Simplified/runs/testing/test1001'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        # {
        #     'model_name': 'Ground Truth',
        #     'model_dirpath': Path('../../../../databases/NeRF_LLFF/data/all/database_data/'),
        #     'frames_dirname': 'rgb_down4',
        #     'depths_dirname': None,
        # },
    ]
    collage_data = [
        {
            'scene_name': 'horns',
            'frame_num': 40,
            'crops': [(221, 439, 150, 150, collage_maker.color_red)],
            # 'arrows': [
            #     [None, None, None, ((160, 190), (50, 150), (0, 0, 255), 4), ((10, 75), (120, 150), (0, 0, 255), 4), None, None],
            #     [None, None, ((10, 190), (80, 110), (255, 0, 255), 4), None, ((40, 160), (70, 60), (0, 255, 255), 4), None, None],
            # ]
        },
        # {
        #     'scene_name': 'horns',
        #     'frame_num': 32,
        #     'crops': [(0, 588, 150, 150, collage_maker.color_red), (83, 443, 75, 75, collage_maker.color_green)],
        #     'arrows': [
        #         [None, None, ((60, 90), (140, 60), (255, 0, 255), 4), ((40, 120), (60, 40), (0, 0, 255), 4), ((40, 110), (60, 30), (0, 0, 255), 4), None, None],
        #         [None, None, ((20, 180), (100, 130), (255, 0, 255), 4), None, ((20, 180), (90, 120), (0, 255, 255), 4), None, None],
        #     ]
        # },
    ]
    collage = collage_maker.create_collage(collage_data, models_data)
    output_path = Path('../data/outputs/02_QualitativeComparisons02g.jpg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(output_path.as_posix(), collage)
    return


def demo2h():
    """
    Ablation w/o views augmentation
    :return:
    """
    configs = {
        'frame_size': (756, 1008),
        'crop_size': (200, 200),
        'full_frame_model_name': 'SimpleNeRF',
        'bounding_box_thickness': 10,
        'annotation_height': 100,
        'annotation_font_size': 80,
        'white_column_width': 10,
        'white_row_height': 10,
        'final_width': None,
    }
    collage_maker = CollageMaker(configs)
    models_data = [
        {
            'model_name': 'SimpleNeRF',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1061'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
            'scene_dir_suffix': '_video04_static_camera',
        },
        {
            'model_name': 'w/o Views Aug',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1147'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
            'scene_dir_suffix': '_video04_static_camera',
        },
        {
            'model_name': 'Dense-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/007_NeRFs_Simplified/runs/testing/test1001'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
            'scene_dir_suffix': '_video04_static_camera',
        },
        # {
        #     'model_name': 'Ground Truth',
        #     'model_dirpath': Path('../../../../databases/NeRF_LLFF/data/all/database_data/'),
        #     'frames_dirname': 'rgb_down4',
        #     # 'depths_dirname': None,
        # },
    ]
    collage_data = [
        {
            'scene_name': 'room',
            'frame_num': 15,
            'crops': [(478, 171, 250, 250, collage_maker.color_red)],
            # 'arrows': [
            #     [None, None, None, ((160, 190), (50, 150), (0, 0, 255), 4), ((10, 75), (120, 150), (0, 0, 255), 4), None, None],
            #     [None, None, ((10, 190), (80, 110), (255, 0, 255), 4), None, ((40, 160), (70, 60), (0, 255, 255), 4), None, None],
            # ]
        },
        # {
        #     'scene_name': 'horns',
        #     'frame_num': 32,
        #     'crops': [(0, 588, 150, 150, collage_maker.color_red), (83, 443, 75, 75, collage_maker.color_green)],
        #     'arrows': [
        #         [None, None, ((60, 90), (140, 60), (255, 0, 255), 4), ((40, 120), (60, 40), (0, 0, 255), 4), ((40, 110), (60, 30), (0, 0, 255), 4), None, None],
        #         [None, None, ((20, 180), (100, 130), (255, 0, 255), 4), None, ((20, 180), (90, 120), (0, 255, 255), 4), None, None],
        #     ]
        # },
    ]
    collage = collage_maker.create_collage(collage_data, models_data)
    output_path = Path('../data/outputs/02_QualitativeComparisons02h.jpg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(output_path.as_posix(), collage)
    return


def main():
    demo1a()
    demo1b()
    demo1c()
    demo2a()
    demo2b()
    demo2c()
    demo2d()
    demo2e()
    demo2f()
    demo2g()
    demo2h()
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
