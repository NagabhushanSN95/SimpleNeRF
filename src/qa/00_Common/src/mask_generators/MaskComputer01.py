# Shree KRISHNAya Namaha
# Computes visibility mask. A pixel is marked as visible if PoseWarping marks it as visible and depth matches.
# Authors: Nagabhushan S N
# Last Modified: 15/06/2023
import json
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

from Warper import Warper

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class MaskComputer:
    def __init__(self, configs: dict):
        self.configs = configs
        self.warper = Warper(resolution=None)
        self.depth_error_threshold = self.configs['depth_error_threshold']
        return

    def compute_mask(self, frame_train, depth_train, depth_test, extrinsic_train, extrinsic_test, intrinsic_train, intrinsic_test):
        depth_error_threshold = self.depth_error_threshold * depth_train.max()
        warping_mask, warped_depth = self.warper.forward_warp(frame_train, None, depth_train, extrinsic_train, extrinsic_test, intrinsic_train, intrinsic_test)[1:3]
        mask = warping_mask & (numpy.abs(warped_depth - depth_test) < depth_error_threshold)
        return mask
