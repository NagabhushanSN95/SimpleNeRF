# Shree KRISHNAya Namaha
# Utility functions for losses
# Authors: Nagabhushan S N, Adithyan K V
# Last Modified: 15/06/2023


def update_loss_map_dict(old_dict: dict, new_dict: dict, suffix: str):
    for key in new_dict.keys():
        old_dict[f'{key}_{suffix}'] = new_dict[key]
    return old_dict
