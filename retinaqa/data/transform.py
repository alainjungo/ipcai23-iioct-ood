import numpy as np
from PIL import Image

import pymia.data.definition as defs
KEY_IMAGES = 'ascan'
KEY_LABELS = 'target'
defs.KEY_IMAGES = KEY_IMAGES  # before transformation imports since used as default arguments
defs.KEY_LABELS = KEY_LABELS  # before transformation imports since used as default arguments
from pymia.data.transformation import (Transform, LoopEntryTransform, LambdaTransform, ComposeTransform, IntensityRescale,
                                       RandomCrop, Squeeze, UnSqueeze, check_and_return, raise_error_if_entry_not_extracted,
                                       ENTRY_NOT_EXTRACTED_ERR_MSG)
from . import perturbations as pert


class MinMaxNormalization(LoopEntryTransform):

    def __init__(self, old_range: tuple, new_range: tuple, entries=(KEY_IMAGES,)) -> None:
        super().__init__(None, entries)
        self.old_min, self.old_max = old_range
        self.new_min, self.new_max = new_range

    def transform_entry(self, np_entry, entry, loop_i=None) -> np.ndarray:
        val = (np_entry - self.old_min)/(self.old_max - self.old_min) * (self.new_max - self.new_min) + self.new_min
        assert val.min() >= self.new_min and val.max() <= self.new_max
        return val


class ToRGB(LoopEntryTransform):

    def __init__(self, entries=(KEY_IMAGES,)) -> None:
        super().__init__(None, entries)

    def transform_entry(self, np_entry, entry, loop_i=None) -> np.ndarray:
        # simulate rgb
        ret = np.stack([np_entry, np_entry, np_entry], axis=0)
        return ret


class Resize(LoopEntryTransform):

    def __init__(self, size: tuple, entries=(KEY_IMAGES,)) -> None:
        super().__init__(None, entries)
        self.size = size[::-1]  # since PIL image is translated

    def transform_entry(self, np_entry, entry, loop_i=None) -> np.ndarray:
        ret = np.array(Image.fromarray(np_entry).resize(self.size, Image.BICUBIC))
        return ret


IMAGENET_MEANS = (0.485, 0.456, 0.406)
IMAGENET_STDS = (0.229, 0.224, 0.225)


class ChannelWiseNorm(LoopEntryTransform):

    def __init__(self, norm_means=IMAGENET_MEANS, norm_stds=IMAGENET_STDS, entries=(KEY_IMAGES, )) -> None:
        super().__init__(None, entries)
        assert len(norm_means) == len(norm_stds)
        self.norm_means = norm_means
        self.norm_stds = norm_stds

    def transform_entry(self, np_entry, entry, loop_i=None) -> np.ndarray:
        assert len(np_entry) == len(self.norm_means)
        for i in range(len(self.norm_means)):
            np_entry[i] = (np_entry[i] - self.norm_means[i]) / self.norm_stds[i]
        return np_entry


class OnlinePerturbation(Transform):

    def __init__(self, perturbations: dict, entry=KEY_IMAGES, targe_entry=KEY_LABELS, condition_key=None,
                 required_condition=True) -> None:
        super().__init__()
        probabilities = [v for v in perturbations.values()]
        assert np.isclose(sum(probabilities), 1.0)
        self.perturbations_names = [k for k in perturbations.keys()]
        assert all(n in ('none', 'stripe', 'rect', 'zoom', 'shift', 'noise', 'smooth', 'contr', 'intsh') for n in self.perturbations_names)
        self.cumsum = np.cumsum([0.0] + probabilities)  # 0 as starting value
        self.entry = entry
        self.target_entry = targe_entry
        self.condition_key = condition_key
        self.required_condition = required_condition
        # assert if not known perturbation

    def __call__(self, sample: dict) -> dict:

        if self.condition_key is not None and sample[self.condition_key] == self.required_condition:
            return sample

        val = np.random.rand()

        index = np.argmax((val >= self.cumsum[:-1]) & (val < self.cumsum[1:]))  # argmax finds first and only match index
        pert_name = self.perturbations_names[index]

        np_entry = sample[self.entry]
        np_target = sample[self.target_entry] if self.target_entry is not None else None
        is_perturbed = False

        if pert_name == 'stripe':
            np_entry = pert.stripes(np_entry, min_=1, max_=2)
            is_perturbed = True
        elif pert_name == 'rect':
            np_entry = pert.rectangle(np_entry, min_height=6, max_height=10)
            is_perturbed = True
        elif pert_name == 'zoom':
            np_entry, np_target = pert.zoom(np_entry, np_target)
            is_perturbed = True
        elif pert_name == 'shift':
            # np_entry = pert.shift(np_entry, min_height=1, max_height=9, pos_at_end=True)
            np_entry, np_target = pert.shift_multiple(np_entry, np_target, min_nb=1, max_nb=1)
            is_perturbed = True
        elif pert_name == 'noise':
            np_entry = pert.noise(np_entry)
            # np_entry = pert.noise(np_entry, 15)
            is_perturbed = True
        elif pert_name == 'smooth':
            np_entry = pert.smooth(np_entry)
            # np_entry = pert.smooth(np_entry, 2)
            is_perturbed = True
        elif pert_name == 'contr':
            np_entry = pert.contrast(np_entry)
            # np_entry = pert.contrast(np_entry, (0.25, 0.5, 0.75, 1/0.75, 1/0.5, 1/0.25))
            is_perturbed = True
        elif pert_name == 'intsh':
            np_entry = pert.intensity_shift(np_entry)
            is_perturbed = True
        else:
            # 'none' -> no perturbation needed
            pass

        np_entry = np.clip(np_entry, 0., 255.)
        if np_target is not None:
            np_target = np.clip(np_target, 0., 255.)

        sample['is_perturbed'] = is_perturbed
        sample['pert_name'] = pert_name
        sample[self.entry] = np_entry
        if self.target_entry is not None:
            sample[self.target_entry] = np_target

        return sample
