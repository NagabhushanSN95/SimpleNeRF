# Shree KRISHNAya Namaha
# Abstract class
# Authors: Nagabhushan S N, Adithyan K V
# Last Modified: 15/06/2023

import abc

from pathlib import Path

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class LearningRateDecayerParent:
    @abc.abstractmethod
    def get_updated_learning_rate(self, iter_num):
        pass
