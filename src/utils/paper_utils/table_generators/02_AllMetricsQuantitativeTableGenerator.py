# Shree KRISHNAya Namaha
# Generates latex table for comparison w.r.t. all metrics (w/ and w/o masks)
# Author: Nagabhushan S N
# Last Modified: 11/05/2023

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
        metric_qa_data_dict = {}
        for model_data in models_data:
            model_name = model_data['name']
            model_dirpath = model_data['model_dirpath']
            if not model_dirpath.exists():
                continue
            qa_filepath = model_dirpath / 'QA_Scores.json'
            with open(qa_filepath.as_posix(), 'r') as qa_file:
                qa_scores = json.load(qa_file)
            avg_qa_score = None
            for pred_type in qa_scores.keys():
                for key in qa_scores[pred_type].keys():
                    if key.startswith(metric_name):
                        avg_qa_score = qa_scores[pred_type][key]
            metric_qa_data_dict[model_name] = avg_qa_score
        qa_data_dict[metric_name] = metric_qa_data_dict
    qa_data = pandas.DataFrame(qa_data_dict)
    return qa_data


def generate_latex_table(qa_data: pandas.DataFrame, metrics_data: dict, model_length: int = 12, num_leading_spaces: int = 12):
    leading_spaces = ''.join([' '] * num_leading_spaces)

    model_names = list(qa_data.index)
    metric_names = list(qa_data.columns)
    for i, model_name in enumerate(model_names):
        if model_name == 'SimpleNeRF':
            print(leading_spaces + r'\hline')

        model_qa_data = qa_data.loc[model_name]
        num_model_trailing_spaces = model_length - len(model_name)
        model_trailing_spaces = ''.join([' '] * num_model_trailing_spaces)
        model_text = leading_spaces + model_name + model_trailing_spaces + ' & '
        for metric_name in metric_names:
            if metric_name.startswith('Masked'):
                continue

            metric_name1 = metric_name
            if re.search('\w+\d\d$', metric_name):
                metric_name1 = metric_name[:-2]

            if ('DepthMAE' in metric_name) and (('RegNeRF' in model_name) or ('FreeNeRF' in model_name)):
                model_text = model_text + r' -- & '
                continue

            # Add the masked scores
            masked_metric_name = next(filter(lambda x: x.startswith(f'Masked{metric_name1}'), metric_names))
            masked_qa_score = model_qa_data[masked_metric_name]
            if metrics_data[metric_name]['best'] == 'max':
                is_best_score = qa_data[masked_metric_name].argmax() == i
            else:
                is_best_score = qa_data[masked_metric_name].argmin() == i
            metric_format = metrics_data[masked_metric_name]['format']
            if is_best_score:
                model_text = model_text + r' \textbf{' + '{qa_score:{metric_format}}'.format(qa_score=masked_qa_score, metric_format=metric_format) + r'}'
            else:
                model_text = model_text + ' {qa_score:{metric_format}}'.format(qa_score=masked_qa_score, metric_format=metric_format) + r''

            model_qa_score = model_qa_data[metric_name]
            if metrics_data[metric_name]['best'] == 'max':
                is_best_score = qa_data[metric_name].argmax() == i
            else:
                is_best_score = qa_data[metric_name].argmin() == i
            metric_format = metrics_data[metric_name]['format']
            if is_best_score:
                model_text = model_text + r'(\textbf{' + '{qa_score:{metric_format}}'.format(qa_score=model_qa_score, metric_format=metric_format) + r'})'
            else:
                model_text = model_text + '({qa_score:{metric_format}}'.format(qa_score=model_qa_score, metric_format=metric_format) + ')'

            model_text = model_text + ' & '
        model_text = model_text[:-2]
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
        {
            'name': 'SimpleNeRF w/o points aug',
            # 'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0031'),
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0046'),
        },
        {
            'name': 'SimpleNeRF w/o views aug',
            # 'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0032'),
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0047'),
        },
        {
            'name': 'SimpleNeRF w/o coarse-fine cons',
            # 'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0033'),
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0048'),
        },
        {
            'name': 'SimpleNeRF w/o reliable depth',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0050'),
        },
        {
            'name': 'SimpleNeRF w/o residual pos enc',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test0051'),
        },
        {
            'name': 'SimpleNeRF w/ identical augs',
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
        #     'name': 'SimpleNeRF w/o reliable depth',
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
        {
            'name': 'SimpleNeRF w/o points aug',
            # 'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1066'),
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1146'),
        },
        {
            'name': 'SimpleNeRF w/o views aug',
            # 'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1067'),
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1147'),
        },
        {
            'name': 'SimpleNeRF w/o coarse-fine cons',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1113'),
            # 'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1148'),
        },
        {
            'name': 'SimpleNeRF w/o reliable depth',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1150'),
        },
        {
            'name': 'SimpleNeRF w/o residual pos enc',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1151'),
        },
        {
            'name': 'SimpleNeRF w/ identical augs',
            'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1154'),
        },
    ]
    metrics_data = {
        'LPIPS': {'format': '0.04f', 'best': 'min'},
        'SSIM': {'format': '0.04f', 'best': 'max'},
        'PSNR': {'format': '0.02f', 'best': 'max'},
        'DepthMAE12': {'format': '0.04f', 'best': 'min'},
        'DepthSROCC': {'format': '0.04f', 'best': 'max'},
        'MaskedLPIPS12': {'format': '0.04f', 'best': 'min'},
        'MaskedSSIM12': {'format': '0.04f', 'best': 'max'},
        'MaskedPSNR12': {'format': '0.02f', 'best': 'max'},
        'MaskedDepthMAE32': {'format': '0.04f', 'best': 'min'},
        'MaskedDepthSROCC12': {'format': '0.04f', 'best': 'max'},
    }
    qa_data = read_qa_data(models_data, metrics_data)
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
        #     'name': 'SimpleNeRF w/o reliable depth',
        #     'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1132'),
        # },
    ]
    metrics_data = {
        'LPIPS': {'format': '0.04f', 'best': 'min'},
        'SSIM': {'format': '0.04f', 'best': 'max'},
        'PSNR': {'format': '0.02f', 'best': 'max'},
        'DepthMAE12': {'format': '0.04f', 'best': 'min'},
        'DepthSROCC': {'format': '0.04f', 'best': 'max'},
        'MaskedLPIPS12': {'format': '0.04f', 'best': 'min'},
        'MaskedSSIM12': {'format': '0.04f', 'best': 'max'},
        'MaskedPSNR12': {'format': '0.02f', 'best': 'max'},
        'MaskedDepthMAE32': {'format': '0.04f', 'best': 'min'},
        'MaskedDepthSROCC12': {'format': '0.04f', 'best': 'max'},
    }
    qa_data = read_qa_data(models_data, metrics_data)
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
        #     'name': 'SimpleNeRF w/o reliable depth',
        #     'model_dirpath': Path('../../../view_synthesis/research/001_NeRF/runs/testing/test1133'),
        # },
    ]
    metrics_data = {
        'LPIPS': {'format': '0.04f', 'best': 'min'},
        'SSIM': {'format': '0.04f', 'best': 'max'},
        'PSNR': {'format': '0.02f', 'best': 'max'},
        'DepthMAE12': {'format': '0.04f', 'best': 'min'},
        'DepthSROCC': {'format': '0.04f', 'best': 'max'},
        'MaskedLPIPS12': {'format': '0.04f', 'best': 'min'},
        'MaskedSSIM12': {'format': '0.04f', 'best': 'max'},
        'MaskedPSNR12': {'format': '0.02f', 'best': 'max'},
        'MaskedDepthMAE32': {'format': '0.04f', 'best': 'min'},
        'MaskedDepthSROCC12': {'format': '0.04f', 'best': 'max'},
    }
    qa_data = read_qa_data(models_data, metrics_data)
    generate_latex_table(qa_data, metrics_data)
    return


def main():
    # demo1a()
    # demo1b()
    # demo1c()
    # demo2a()
    # demo2b()
    # demo2c()
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
