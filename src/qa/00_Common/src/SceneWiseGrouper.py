# Shree KRISHNAya Namaha
# Groups QA scores scene-wise
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

import datetime
import time
import traceback
from pathlib import Path

import pandas

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def get_grouped_qa_scores(qa_data: pandas.DataFrame):
    final_column_names = [x for x in qa_data.columns if x != 'pred_frame_num']
    group_column_names = list(qa_data)[:-2]
    grouped_qa_data = qa_data.groupby(by=group_column_names).mean().reset_index()[final_column_names]
    grouped_qa_data = grouped_qa_data.round({final_column_names[-1]: 4, })
    return grouped_qa_data


def group_qa_scores(testing_dirpath: Path, test_nums: list):
    for test_num in test_nums:
        qa_dirpath = testing_dirpath / f'test{test_num:04}/QA_Scores'
        for pred_dirpath in sorted(qa_dirpath.iterdir()):
            for qa_filepath in sorted(pred_dirpath.glob('*_FrameWise.csv')):
                qa_data = pandas.read_csv(qa_filepath)
                grouped_qa_data = get_grouped_qa_scores(qa_data)
                grouped_qa_filepath = qa_filepath.parent / f'{qa_filepath.stem[:-9]}SceneWise.csv'
                grouped_qa_data.to_csv(grouped_qa_filepath, index=False)
    return
