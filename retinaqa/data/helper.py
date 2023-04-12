import os
import glob

import numpy as np
from PIL import Image
import pandas as pd


def add_case_information(df: pd.DataFrame, simulated_ood_data):
    if simulated_ood_data:
        other_df = df['case_id'].apply(lambda c: pd.Series(get_simulated_case_details(c), index=['is_ood', 'is_retina', 'info', 'perturbation']))
    else:
        other_df = df['case_id'].apply(lambda c: pd.Series(get_real_case_details(c), index=['is_ood', 'is_retina', 'info']))
    df = pd.concat([df, other_df], axis=1)
    return df


def get_simulated_case_details(case_id):
    parts = case_id.split('_')
    is_ood = False
    perturbation = None
    if 'pert-' in parts[-1]:
        is_ood = True
        perturbation = parts[-1][len('pert-'):]
    info = parts[-2] if is_ood else parts[-1]
    assert info in ('r', 'b')
    is_retina = info == 'r'
    return is_ood, is_retina, info, perturbation


def get_real_case_details(case_id):
    info = case_id.split('_')[-1]
    assert info in ('r', 'b', 'o', 'i')
    is_ood = info == 'o'
    is_retina = info == 'r'
    return is_ood, is_retina, info


def load_all_images(dir_path, dtype=None, rescale=None, selection='*'):
    if not isinstance(selection, (list, tuple)):
        selection = [selection]

    image_dict = {}
    for sel in selection:
        for img_path in sorted(glob.glob(dir_path + f'/{sel}/*.png')):
            case_id = os.path.basename(img_path)[:-len('.png')]
            img = np.asarray(Image.open(img_path))
            if dtype:
                img = img.astype(dtype)
            if rescale:
                img *= rescale
            image_dict[case_id] = img
    return image_dict
