# Shree KRISHNAya Namaha
# Abstract parent class
# Author: Nagabhushan S N
# Last Modified: 15/06/2023

import abc


class LossFunctionParent:
    @abc.abstractmethod
    def compute_loss(self, input_dict: dict, output_dict: dict, return_loss_maps: bool = False):
        pass
