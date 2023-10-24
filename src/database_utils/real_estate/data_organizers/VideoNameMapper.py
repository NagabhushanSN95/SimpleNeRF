# Shree KRISHNAya Namaha
# Maps video name to scene num
# Author: Nagabhushan S N
# Last Modified: 22/10/2023

import datetime
import shutil
import time
import traceback
from pathlib import Path

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def copy_camera_data(unzipped_dirpath: Path, database_dirpath: Path):
    mapping = {
        0: '000c3ab189999a83',
        1: '000db54a47bd43fe',
        3: '0017ce4c6a39d122',
        4: '002ae53df0e0afe2',
        6: '0043978734eec081',
    }
    for scene_num in mapping:
        scene_dirpath = database_dirpath / f'{scene_num:05}'
        scene_dirpath.mkdir(parents=True, exist_ok=False)
        src_data_path = unzipped_dirpath / f'test/{mapping[scene_num]}.txt'
        tgt_data_path = scene_dirpath / 'CameraData.txt'
        shutil.copy(src_data_path, tgt_data_path)
    return


def demo1():
    root_dirpath = Path('../../../../data/databases/RealEstate10K/')
    unzipped_dirpath = root_dirpath / 'unzipped_data/RealEstate10K'
    database_dirpath = root_dirpath / 'data/test/database_data'

    copy_camera_data(unzipped_dirpath, database_dirpath)
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
