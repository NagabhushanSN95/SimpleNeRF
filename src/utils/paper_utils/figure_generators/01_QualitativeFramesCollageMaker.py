# Shree KRISHNAya Namaha
# Makes collage for qualitative comparisons. Full frames in two rows.
# For SIGGRAPH Asia 2023 submission.
# Author: Nagabhushan S N
# Last Modified: 02/05/2023

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

from PIL import ImageFont, Image, ImageDraw
from matplotlib import pyplot

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class CollageMaker:
    def __init__(self, configs: dict):
        self.configs = configs
        self.frame_height, self.frame_width = configs['frame_size']
        self.crop_needed = 'crop_size' in configs
        self.crop_height, self.crop_width = configs.get('crop_size', (None, None))
        self.full_frame_model_name = configs.get('full_frame_model_name', None)
        self.num_rows = configs.get('num_rows', 2)

        self.color_black = numpy.array([0, 0, 0])
        self.color_white = numpy.array([255, 255, 255])
        self.color_red = (255, 0, 0)
        self.color_green = numpy.array([0, 255, 0])
        return

    def create_collage(self, collage_data: list, models_data: list):
        rows = []
        white_row = None
        # annotation_done = False

        for sample_collage_data in collage_data:
            scene_name = sample_collage_data['scene_name']
            frame_num = sample_collage_data['frame_num']

            frames, model_names = [], []
            for i, model_data in enumerate(models_data):
                model_name = model_data['model_name']
                model_dirpath = model_data['model_dirpath']
                frames_dirname = model_data['frames_dirname']

                if 'Input' not in model_name:
                    frame_path = model_dirpath / f'{scene_name}/{frames_dirname}/{frame_num:04}.png'
                else:
                    input_frame_num = sample_collage_data['input_frame_nums'][int(model_name[6:]) - 1]
                    frame_path = model_dirpath / f'{scene_name}/{frames_dirname}/{input_frame_num:04}.png'
                frame = self.read_image(frame_path)

                if self.crop_needed:
                    ph, pw = sample_collage_data['ph'], sample_collage_data['pw']
                    y1, x1 = sample_collage_data['y1'], sample_collage_data['x1']
                    y2, x2 = y1 + ph, x1 + pw
                    if model_name == self.full_frame_model_name:
                        frame = cv2.rectangle(frame.copy(), (x1, y1), (x2, y2), color=self.color_red, thickness=self.configs['bounding_box_thickness'])
                    else:
                        frame = frame[y1:y2, x1:x2]
                frames.append(frame)
                model_names.append(model_name)

            # if not annotation_done:
            #     annotations = [(model_name, False) for model_name in model_names]
            #     annotation_done = True
            # else:
            #     annotations = None
            annotations = [(model_name, False) for model_name in model_names]

            row = self.create_sample_collage(frames, annotations, output_height=self.crop_height if self.crop_needed else self.frame_height)
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

        rows = []
        for j in range(self.num_rows):
            row_images = [resized_images[i] for i in range(len(resized_images)) if (i // 2) % self.num_rows == j]
            row_image = numpy.concatenate(row_images[:-1], axis=1)
            rows.append(row_image)
        collage = numpy.concatenate(rows, axis=0)
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
        sympy.preview(title, dvioptions=["-T", "tight", "-z", "0", "--truecolor", "-D 600"], viewer='file',
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
        # 'crop_size': (576, 1024),
        # 'bounding_box_thickness': 10,
        'annotation_height': 80,
        'annotation_font_size': 50,
        'white_column_width': 20,
        'white_row_height': 20,
        'final_width': None,
    }
    models_data = [
        {
            'model_name': 'Input 1',
            'model_dirpath': Path('../../../../databases/RealEstate10K/data/test/database_data/'),
            'frames_dirname': 'rgb',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'Input 2',
            'model_dirpath': Path('../../../../databases/RealEstate10K/data/test/database_data/'),
            'frames_dirname': 'rgb',
            # 'depths_dirname': 'predicted_depths',
        },
        # {
        #     'model_name': 'InfoNeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/012_InfoNeRF/runs/testing/test0011'),
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
            'scene_name': '00001',
            'frame_num': 1,
            'input_frame_nums': [10, 20],
            # 'ph': 576,
            # 'pw': 1024,
            # 'y1': 0,
            # 'x1': 0,
        },
    ]
    collage_maker = CollageMaker(configs)
    collage = collage_maker.create_collage(collage_data, models_data)
    output_path = Path('../data/outputs/01_QualitativeComparisons01a.jpg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(output_path.as_posix(), collage)
    return


def demo2a():
    configs = {
        'frame_size': (300, 400),
        'crop_size': (300, 400),
        'bounding_box_thickness': 10,
        'annotation_height': 50,
        'annotation_font_size': 30,
        'white_column_width': 10,
        'white_row_height': 10,
        'final_width': None,
    }
    models_data = [
        # {
        #     'model_name': 'DietNeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/021_DietNeRF/runs/testing/test0042'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        # {
        #     'model_name': 'DS-NeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/011_DS_NeRF/runs/testing/test0142'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        {
            'model_name': 'DDP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/015_DDP_NeRF/runs/testing/test0042'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'InfoNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/012_InfoNeRF/runs/testing/test0042'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        # {
        #     'model_name': 'RegNeRF',
        #     'model_dirpath': Path('../../../view_synthesis/literature/013_RegNeRF/runs/testing/test0042'),
        #     'frames_dirname': 'predicted_frames',
        #     # 'depths_dirname': 'predicted_depths',
        # },
        {
            'model_name': 'VC-NeRF',
            'model_dirpath': Path('../../../view_synthesis/Research/006_NeRF/runs/testing/test4007'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'Ground Truth',
            'model_dirpath': Path('../../../../databases/DTU/data/all/database_data/'),
            'frames_dirname': 'rgb',
            # 'depths_dirname': None,
        },
    ]
    collage_data = [
        {
            'scene_name': '00045',
            'frame_num': 26,
            'ph': 300,
            'pw': 400,
            'y1': 0,
            'x1': 0,
        },
        {
            'scene_name': '00110',
            'frame_num': 26,
            'ph': 300,
            'pw': 400,
            'y1': 0,
            'x1': 0,
        },
        {
            'scene_name': '00030',
            'frame_num': 26,
            'ph': 300,
            'pw': 400,
            'y1': 0,
            'x1': 0,
        },
    ]
    collage_maker = CollageMaker(configs)
    collage = collage_maker.create_collage01(collage_data, models_data, collage_maker.create_sample_collage01)
    output_path = Path('../data/Outputs/01_QualitativeComparisons02a.jpg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(output_path.as_posix(), collage)
    return


def demo3a():
    configs = {
        'frame_size': (756, 1008),
        'crop_size': (189, 252),
        'bounding_box_thickness': 10,
        'annotation_height': 50,
        'annotation_font_size': 30,
        'white_column_width': 10,
        'white_row_height': 10,
        'final_width': None,
    }
    models_data = [
        {
            'model_name': 'InfoNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/012_InfoNeRF/runs/testing/test0012'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'RegNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/013_RegNeRF/runs/testing/test0012'),
            'frames_dirname': 'predicted_frames',
            # 'depths_dirname': 'predicted_depths',
        },
        {
            'model_name': 'VC-NeRF',
            'model_dirpath': Path('../../../view_synthesis/Research/006_NeRF/runs/testing/test1152'),
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
            'ph': 189,
            'pw': 252,
            'y1': 520,
            'x1': 300,
        },
        {
            'scene_name': 'horns',
            'frame_num': 24,
            'ph': 189,
            'pw': 252,
            'y1': 0,
            'x1': 189,
        },
    ]
    collage_maker = CollageMaker(configs)
    collage = collage_maker.create_collage01(collage_data, models_data, collage_maker.create_sample_collage01)
    output_path = Path('../data/Outputs/01_QualitativeComparisons03a.jpg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(output_path.as_posix(), collage)
    return


def main():
    demo1a()
    # demo1b()
    # demo1c()
    # demo1d()
    # demo1e()
    # demo2a()
    # demo3a()
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
