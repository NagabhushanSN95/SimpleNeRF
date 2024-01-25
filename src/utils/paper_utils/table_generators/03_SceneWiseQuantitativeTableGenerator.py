# Shree KRISHNAya Namaha
# Generates latex table for scene-wise comparison
# Author: Nagabhushan S N
# Last Modified: 11/05/23

import json
import re
import time
import datetime
import traceback
import numpy
import simplejson
import skimage.io
import skvideo.io
import pandas

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def read_qa_data(models_data: list, metrics_data: dict):
    qa_data_dict = {}
    for metric_name in metrics_data.keys():
        qa_data = None
        for model_data in models_data:
            model_name = model_data['name']
            model_dirpath = model_data['model_dirpath']
            if not model_dirpath.exists():
                continue
            qa_filepath = next(model_dirpath.glob(f'QA_Scores/*/{metric_name}*_SceneWise.csv'))
            # if not qa_filepath.exists():
            #     continue
            model_qa_data = pandas.read_csv(qa_filepath)
            scene_id_name = list(model_qa_data.columns)[0]
            model_qa_data.columns = [scene_id_name, model_name]
            qa_filepath = model_dirpath / 'QA_Scores.json'
            with open(qa_filepath.as_posix(), 'r') as qa_file:
                qa_scores = json.load(qa_file)
            avg_qa_score = None
            for pred_type in qa_scores.keys():
                for key in qa_scores[pred_type].keys():
                    if key.startswith(metric_name):
                        avg_qa_score = qa_scores[pred_type][key]
            model_qa_data.loc[model_qa_data.shape[0]] = ['average', avg_qa_score]
            if qa_data is None:
                qa_data = model_qa_data
            else:
                qa_data = qa_data.merge(model_qa_data, on=scene_id_name)
        qa_data_dict[metric_name] = qa_data
    return qa_data_dict


def process_qa_data(qa_data: dict, metrics_data: dict):
    scene_id_name = list(qa_data[list(qa_data.keys())[0]].columns)[0]
    scene_names = qa_data[list(qa_data.keys())[0]][scene_id_name].to_list()
    num_scenes = len(scene_names)
    processed_qa_data = {}
    for metric_name in qa_data:
        qa_data1 = qa_data[metric_name].set_index(qa_data[metric_name].columns[0])
        qa_data2 = qa_data1.to_numpy()
        if metrics_data[metric_name]['best'] == 'max':
            best_col_idx = numpy.argmax(qa_data2, axis=1)
        else:
            best_col_idx = numpy.argmin(qa_data2, axis=1)
        best_qa_data = numpy.zeros_like(qa_data2)
        best_qa_data[numpy.arange(num_scenes), best_col_idx] = 1
        best_qa_data = best_qa_data.astype(bool)
        best_qa_data = pandas.DataFrame(data=best_qa_data, index=qa_data1.index, columns=qa_data1.columns)
        qa_data1 = qa_data1.reset_index(level=0)
        best_qa_data = best_qa_data.reset_index(level=0)
        processed_qa_data[metric_name] = {
            'qa_scores': qa_data1,
            'best_qa_flag': best_qa_data,
        }
    return processed_qa_data


def generate_latex_table(qa_data: dict, metrics_data: dict, model_length = 12, num_leading_spaces: int = 12):
    leading_spaces = ''.join([' '] * num_leading_spaces)

    sample_qa_data = qa_data[list(qa_data.keys())[0]]
    sample_qa_data = sample_qa_data[list(sample_qa_data.keys())[0]]
    scene_id_name = list(sample_qa_data.columns)[0]
    scene_names = sample_qa_data[scene_id_name].to_numpy()
    model_names = list(sample_qa_data.columns)[1:]
    metric_names = list(qa_data.keys())
    for model_name in model_names:
        num_model_trailing_spaces = model_length - len(model_name)
        model_trailing_spaces = ''.join([' '] * num_model_trailing_spaces)
        model_text = leading_spaces + str(model_name) + model_trailing_spaces
        for scene_name in scene_names:
            model_text = model_text + ' & \makecell{'
            for metric_name in metric_names:
                if metric_name.startswith('Masked'):
                    continue

                metric_name1 = metric_name
                if re.search('\w+\d\d$', metric_name):
                    metric_name1 = metric_name[:-2]

                if ('DepthMAE' in metric_name) and (('RegNeRF' in model_name) or ('FreeNeRF' in model_name)):
                    model_text = model_text + r' -- \\'
                    continue

                # Add the masked scores
                masked_metric_name = next(filter(lambda x: x.startswith(f'Masked{metric_name1}'), metric_names))
                model_qa_data = qa_data[masked_metric_name]['qa_scores'][[scene_id_name, model_name]]
                model_qa_flag = qa_data[masked_metric_name]['best_qa_flag'][[scene_id_name, model_name]]
                qa_score = model_qa_data[model_qa_data[scene_id_name] == scene_name][model_name].values[0]
                is_best_score = model_qa_flag[model_qa_flag[scene_id_name] == scene_name][model_name].values[0]
                metric_format = metrics_data[masked_metric_name]['format']
                if is_best_score:
                    model_text = model_text + r' \textbf{' + '{qa_score:{metric_format}}'.format(qa_score=qa_score, metric_format=metric_format) + r'}'
                else:
                    model_text = model_text + ' {qa_score:{metric_format}}'.format(qa_score=qa_score, metric_format=metric_format) + r''

                # Unmasked scores
                model_qa_data = qa_data[metric_name]['qa_scores'][[scene_id_name, model_name]]
                model_qa_flag = qa_data[metric_name]['best_qa_flag'][[scene_id_name, model_name]]
                qa_score = model_qa_data[model_qa_data[scene_id_name] == scene_name][model_name].values[0]
                is_best_score = model_qa_flag[model_qa_flag[scene_id_name] == scene_name][model_name].values[0]
                metric_format = metrics_data[metric_name]['format']
                if is_best_score:
                    model_text = model_text + r'(\textbf{' + '{qa_score:{metric_format}}'.format(qa_score=qa_score, metric_format=metric_format) + r'})'
                else:
                    model_text = model_text + '({qa_score:{metric_format}}'.format(qa_score=qa_score, metric_format=metric_format) + r')'

                model_text = model_text + r'\\'
            model_text = model_text[:-2]
            model_text = model_text + '}'
        model_text = model_text + r' \\'
        print(model_text)
        print(leading_spaces + '\hline')
    return


def demo1a():
    models_data = [
        {
            'name': 'InfoNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/012_InfoNeRF/runs/testing/test0011'),
        },
        {
            'name': 'DietNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/008_DietNeRF/runs/testing/test0011'),
        },
        {
            'name': 'RegNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/011_RegNeRF/runs/testing/test0012'),
        },
        {
            'name': 'DS-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/009_DS_NeRF/runs/testing/test0012'),
        },
        {
            'name': 'DDP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/010_DDP_NeRF/runs/testing/test0012'),
        },
        {
            'name': 'FreeNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/017_FreeNeRF/runs/testing/test0012'),
        },
        {
            'name': 'ViP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/018_ViP_NeRF/runs/testing/test0012'),
        },
        {
            'name': 'SimpleNeRF',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0012'),
            # 'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0041'),
        },
    ]
    metrics_data = {
        'LPIPS': {'format': '0.04f', 'best': 'min'},
        'SSIM': {'format': '0.04f', 'best': 'max'},
        'PSNR': {'format': '0.02f', 'best': 'max'},
        'DepthMAE01': {'format': '0.04f', 'best': 'min'},
        'DepthSROCC': {'format': '0.04f', 'best': 'max'},
        'MaskedLPIPS11': {'format': '0.04f', 'best': 'min'},
        'MaskedSSIM11': {'format': '0.04f', 'best': 'max'},
        'MaskedPSNR11': {'format': '0.02f', 'best': 'max'},
        'MaskedDepthMAE11': {'format': '0.04f', 'best': 'min'},
        'MaskedDepthSROCC11': {'format': '0.04f', 'best': 'max'},
    }
    qa_data = read_qa_data(models_data, metrics_data)
    qa_data = process_qa_data(qa_data, metrics_data)
    generate_latex_table(qa_data, metrics_data)
    return


def demo1b():
    models_data = [
        {
            'name': 'InfoNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/012_InfoNeRF/runs/testing/test0021'),
        },
        {
            'name': 'DietNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/008_DietNeRF/runs/testing/test0021'),
        },
        {
            'name': 'RegNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/011_RegNeRF/runs/testing/test0022'),
        },
        {
            'name': 'DS-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/009_DS_NeRF/runs/testing/test0022'),
        },
        {
            'name': 'DDP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/010_DDP_NeRF/runs/testing/test0022'),
        },
        {
            'name': 'FreeNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/017_FreeNeRF/runs/testing/test0022'),
        },
        {
            'name': 'ViP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/018_ViP_NeRF/runs/testing/test0022'),
        },
        {
            'name': 'SimpleNeRF',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0042'),
        },
        # {
        #     'name': r'\makecell[l]{SimpleNeRF w/o \\ reliable depth}',
        #     'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0025'),
        # },
    ]
    metrics_data = {
        'LPIPS': {'format': '0.04f', 'best': 'min'},
        'SSIM': {'format': '0.04f', 'best': 'max'},
        'PSNR': {'format': '0.02f', 'best': 'max'},
        'DepthMAE01': {'format': '0.04f', 'best': 'min'},
        'DepthSROCC': {'format': '0.04f', 'best': 'max'},
        'MaskedLPIPS11': {'format': '0.04f', 'best': 'min'},
        'MaskedSSIM11': {'format': '0.04f', 'best': 'max'},
        'MaskedPSNR11': {'format': '0.02f', 'best': 'max'},
        'MaskedDepthMAE11': {'format': '0.04f', 'best': 'min'},
        'MaskedDepthSROCC11': {'format': '0.04f', 'best': 'max'},
    }
    qa_data = read_qa_data(models_data, metrics_data)
    qa_data = process_qa_data(qa_data, metrics_data)
    generate_latex_table(qa_data, metrics_data)
    return


def demo1c():
    models_data = [
        {
            'name': 'InfoNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/012_InfoNeRF/runs/testing/test0031'),
        },
        {
            'name': 'DietNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/008_DietNeRF/runs/testing/test0031'),
        },
        {
            'name': 'RegNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/011_RegNeRF/runs/testing/test0032'),
        },
        {
            'name': 'DS-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/009_DS_NeRF/runs/testing/test0032'),
        },
        {
            'name': 'DDP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/010_DDP_NeRF/runs/testing/test0032'),
        },
        {
            'name': 'FreeNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/017_FreeNeRF/runs/testing/test0032'),
        },
        {
            'name': 'ViP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/018_ViP_NeRF/runs/testing/test0032'),
        },
        {
            'name': 'SimpleNeRF',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0043'),
        },
    ]
    metrics_data = {
        'LPIPS': {'format': '0.04f', 'best': 'min'},
        'SSIM': {'format': '0.04f', 'best': 'max'},
        'PSNR': {'format': '0.02f', 'best': 'max'},
        'DepthMAE01': {'format': '0.04f', 'best': 'min'},
        'DepthSROCC': {'format': '0.04f', 'best': 'max'},
        'MaskedLPIPS11': {'format': '0.04f', 'best': 'min'},
        'MaskedSSIM11': {'format': '0.04f', 'best': 'max'},
        'MaskedPSNR11': {'format': '0.02f', 'best': 'max'},
        'MaskedDepthMAE11': {'format': '0.04f', 'best': 'min'},
        'MaskedDepthSROCC11': {'format': '0.04f', 'best': 'max'},
    }
    qa_data = read_qa_data(models_data, metrics_data)
    qa_data = process_qa_data(qa_data, metrics_data)
    generate_latex_table(qa_data, metrics_data)
    return


def demo1d():
    models_data = [
        {
            'name': 'SimpleNeRF',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0012'),
            # 'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0041'),
        },
        {
            'name': r'\makecell[l]{SimpleNeRF w/o \\ Points Augmentation}',
            # 'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0031'),
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0046'),
        },
        {
            'name': r'\makecell[l]{SimpleNeRF w/o \\ Views Augmentation}',
            # 'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0032'),
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0047'),
        },
        {
            'name': r'\makecell[l]{SimpleNeRF w/o \\ Coarse-fine \\ Consistency}',
            # 'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0033'),
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0048'),
        },
        {
            'name': r'\makecell[l]{SimpleNeRF w/o \\ reliable depth}',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0050'),
        },
        {
            'name': r'\makecell[l]{SimpleNeRF w/o \\ Residual \\ Positional \\ Encodings}',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0051'),
        },
        {
            'name': r'\makecell[l]{SimpleNeRF w/ \\ Identical \\ Augmentations}',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0054'),
        },
    ]
    metrics_data = {
        'LPIPS': {'format': '0.04f', 'best': 'min'},
        'SSIM': {'format': '0.04f', 'best': 'max'},
        'PSNR': {'format': '0.02f', 'best': 'max'},
        'DepthMAE01': {'format': '0.04f', 'best': 'min'},
        'DepthSROCC': {'format': '0.04f', 'best': 'max'},
        'MaskedLPIPS11': {'format': '0.04f', 'best': 'min'},
        'MaskedSSIM11': {'format': '0.04f', 'best': 'max'},
        'MaskedPSNR11': {'format': '0.02f', 'best': 'max'},
        'MaskedDepthMAE11': {'format': '0.04f', 'best': 'min'},
        'MaskedDepthSROCC11': {'format': '0.04f', 'best': 'max'},
    }
    qa_data = read_qa_data(models_data, metrics_data)
    qa_data = process_qa_data(qa_data, metrics_data)
    generate_latex_table(qa_data, metrics_data)
    return


def demo2a():
    models_data = [
        {
            'name': 'InfoNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/012_InfoNeRF/runs/testing/test1011'),
        },
        {
            'name': 'DietNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/008_DietNeRF/runs/testing/test1011'),
        },
        {
            'name': 'RegNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/011_RegNeRF/runs/testing/test1012'),
        },
        {
            'name': 'DS-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/009_DS_NeRF/runs/testing/test1012'),
        },
        {
            'name': 'DDP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/010_DDP_NeRF/runs/testing/test1012'),
        },
        {
            'name': 'FreeNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/017_FreeNeRF/runs/testing/test1012'),
        },
        {
            'name': 'ViP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/018_ViP_NeRF/runs/testing/test1012'),
        },
        {
            'name': 'SimpleNeRF',
            # 'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1061'),
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1141'),
        },
    ]
    metrics_data = {
        'LPIPS': {'format': '0.02f', 'best': 'min'},
        'SSIM': {'format': '0.02f', 'best': 'max'},
        'PSNR': {'format': '0.01f', 'best': 'max'},
        'DepthMAE12': {'format': '0.02f', 'best': 'min'},
        'DepthSROCC': {'format': '0.02f', 'best': 'max'},
        'MaskedLPIPS12': {'format': '0.02f', 'best': 'min'},
        'MaskedSSIM12': {'format': '0.02f', 'best': 'max'},
        'MaskedPSNR12': {'format': '0.01f', 'best': 'max'},
        'MaskedDepthMAE32': {'format': '0.02f', 'best': 'min'},
        'MaskedDepthSROCC12': {'format': '0.02f', 'best': 'max'},
    }
    qa_data = read_qa_data(models_data, metrics_data)
    qa_data = process_qa_data(qa_data, metrics_data)
    generate_latex_table(qa_data, metrics_data)
    return


def demo2b():
    models_data = [
        {
            'name': 'InfoNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/012_InfoNeRF/runs/testing/test1021'),
        },
        {
            'name': 'DietNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/008_DietNeRF/runs/testing/test1021'),
        },
        {
            'name': 'RegNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/011_RegNeRF/runs/testing/test1022'),
        },
        {
            'name': 'DS-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/009_DS_NeRF/runs/testing/test1022'),
        },
        {
            'name': 'DDP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/010_DDP_NeRF/runs/testing/test1022'),
        },
        {
            'name': 'FreeNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/017_FreeNeRF/runs/testing/test1022'),
        },
        {
            'name': 'ViP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/018_ViP_NeRF/runs/testing/test1022'),
        },
        {
            'name': 'SimpleNeRF',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1142'),
        },
        # {
        #     'name': r'\makecell[l]{SimpleNeRF w/o \\ reliable depth}',
        #     'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1132'),
        # },
    ]
    metrics_data = {
        'LPIPS': {'format': '0.02f', 'best': 'min'},
        'SSIM': {'format': '0.02f', 'best': 'max'},
        'PSNR': {'format': '0.01f', 'best': 'max'},
        'DepthMAE12': {'format': '0.02f', 'best': 'min'},
        'DepthSROCC': {'format': '0.02f', 'best': 'max'},
        'MaskedLPIPS12': {'format': '0.02f', 'best': 'min'},
        'MaskedSSIM12': {'format': '0.02f', 'best': 'max'},
        'MaskedPSNR12': {'format': '0.01f', 'best': 'max'},
        'MaskedDepthMAE32': {'format': '0.02f', 'best': 'min'},
        'MaskedDepthSROCC12': {'format': '0.02f', 'best': 'max'},
    }
    qa_data = read_qa_data(models_data, metrics_data)
    qa_data = process_qa_data(qa_data, metrics_data)
    generate_latex_table(qa_data, metrics_data)
    return


def demo2c():
    models_data = [
        {
            'name': 'InfoNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/012_InfoNeRF/runs/testing/test1031'),
        },
        {
            'name': 'DietNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/008_DietNeRF/runs/testing/test1031'),
        },
        {
            'name': 'RegNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/011_RegNeRF/runs/testing/test1032'),
        },
        {
            'name': 'DS-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/009_DS_NeRF/runs/testing/test1032'),
        },
        {
            'name': 'DDP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/010_DDP_NeRF/runs/testing/test1032'),
        },
        {
            'name': 'FreeNeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/017_FreeNeRF/runs/testing/test1032'),
        },
        {
            'name': 'ViP-NeRF',
            'model_dirpath': Path('../../../view_synthesis/literature/018_ViP_NeRF/runs/testing/test1032'),
        },
        {
            'name': 'SimpleNeRF',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1143'),
        },
        # {
        #     'name': r'\makecell[l]{SimpleNeRF w/o \\ reliable depth}',
        #     'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1133'),
        # },
    ]
    metrics_data = {
        'LPIPS': {'format': '0.02f', 'best': 'min'},
        'SSIM': {'format': '0.02f', 'best': 'max'},
        'PSNR': {'format': '0.01f', 'best': 'max'},
        'DepthMAE12': {'format': '0.02f', 'best': 'min'},
        'DepthSROCC': {'format': '0.02f', 'best': 'max'},
        'MaskedLPIPS12': {'format': '0.02f', 'best': 'min'},
        'MaskedSSIM12': {'format': '0.02f', 'best': 'max'},
        'MaskedPSNR12': {'format': '0.01f', 'best': 'max'},
        'MaskedDepthMAE32': {'format': '0.02f', 'best': 'min'},
        'MaskedDepthSROCC12': {'format': '0.02f', 'best': 'max'},
    }
    qa_data = read_qa_data(models_data, metrics_data)
    qa_data = process_qa_data(qa_data, metrics_data)
    generate_latex_table(qa_data, metrics_data)
    return


def demo2d():
    models_data = [
        {
            'name': 'SimpleNeRF',
            # 'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1061'),
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1141'),
        },
        {
            'name': r'\makecell[l]{SimpleNeRF w/o \\ Points Augmentation}',
            # 'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1066'),
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1146'),
        },
        {
            'name': r'\makecell[l]{SimpleNeRF w/o \\ Views Augmentation}',
            # 'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1067'),
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1147'),
        },
        {
            'name': r'\makecell[l]{SimpleNeRF w/o \\ Coarse-fine \\ Consistency}',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1113'),
            # 'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1148'),
        },
        {
            'name': r'\makecell[l]{SimpleNeRF w/o \\ reliable depth}',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1150'),
        },
        {
            'name': r'\makecell[l]{SimpleNeRF w/o \\ Residual \\ Positional \\ Encodings}',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1151'),
        },
        {
            'name': r'\makecell[l]{SimpleNeRF w/ \\ Identical \\ Augmentations}',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1154'),
        },
    ]
    metrics_data = {
        'LPIPS': {'format': '0.02f', 'best': 'min'},
        'SSIM': {'format': '0.02f', 'best': 'max'},
        'PSNR': {'format': '0.01f', 'best': 'max'},
        'DepthMAE12': {'format': '0.02f', 'best': 'min'},
        'DepthSROCC': {'format': '0.02f', 'best': 'max'},
        'MaskedLPIPS12': {'format': '0.02f', 'best': 'min'},
        'MaskedSSIM12': {'format': '0.02f', 'best': 'max'},
        'MaskedPSNR12': {'format': '0.01f', 'best': 'max'},
        'MaskedDepthMAE32': {'format': '0.02f', 'best': 'min'},
        'MaskedDepthSROCC12': {'format': '0.02f', 'best': 'max'},
    }
    qa_data = read_qa_data(models_data, metrics_data)
    qa_data = process_qa_data(qa_data, metrics_data)
    generate_latex_table(qa_data, metrics_data)
    return


def main():
    # demo1a()
    # demo1b()
    # demo1c()
    # demo1d()
    # demo2a()
    # demo2b()
    # demo2c()
    # demo2d()
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
