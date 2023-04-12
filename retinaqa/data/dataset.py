import os
import glob

import torch.utils.data as data
import numpy as np
from PIL import Image
from torch.utils.data.dataset import T_co


class SimulatedOoD(data.Dataset):

    def __init__(self, img_dir, selection, with_ood, sequence_length=10, overlap_sequence=True,
                 transform=None, exclude_info=None) -> None:
        super().__init__()
        self.img_dir = img_dir
        if not isinstance(selection, (tuple, list)):
            selection = [selection]
        self.selection = selection
        self.sequence_length = sequence_length
        self.transform = transform
        ret = self._collect(img_dir, selection, with_ood, sequence_length, overlap_sequence, exclude_info)
        self.chunks, self.indices, self.chunk_ids, self.chunk_infos, self.is_ood_chunk = ret

    @staticmethod
    def _collect(img_dir, selection, with_ood, sequence_length, overlap_sequence, exclude_info):
        chunk_ids, chunks, indices, chunk_infos, is_ood_chunk = [], [], [], [], []
        case_idx = 0

        for item in selection:
            img_case_path = os.path.join(img_dir, item)
            img_paths = sorted(glob.glob(img_case_path + '/*.png'))

            for i, img_path in enumerate(img_paths):

                np_samples = np.asarray(Image.open(img_path)).astype(np.float32)

                chunk_id = os.path.basename(img_path)[:-len('.png')]
                parts = chunk_id.split('_')
                is_ood = 'pert-' in parts[-1]
                if not with_ood and is_ood:
                    continue

                info = parts[-2] if is_ood else parts[-1]
                if exclude_info is not None and not is_ood:
                    if info == exclude_info:
                        continue

                chunks.append(np_samples)
                chunk_ids.append(chunk_id)
                is_ood_chunk.append(is_ood)
                chunk_infos.append(info)

                cnt = len(np_samples) - sequence_length + 1
                if overlap_sequence:
                    case_indices = np.vstack([np.ones(cnt) * case_idx, np.arange(cnt)]).astype(np.int).T
                else:
                    inds = np.arange(cnt, step=sequence_length)
                    case_indices = np.vstack([np.ones(len(inds)) * case_idx, inds]).astype(np.int).T

                indices.append(case_indices)
                case_idx += 1

        return chunks, np.vstack(indices), chunk_ids, chunk_infos, is_ood_chunk


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index) -> T_co:
        return self.get_sample(index, transform=self.transform)

    def get_sample(self, index, transform=None):
        case_idx, sample_idx = self.indices[index].tolist()

        sample = {
            'sample_index': index,
            'is_retina': self.chunk_infos[case_idx] == 'r',
            'is_ood': self.is_ood_chunk[case_idx],
            'ascan': _extract_range(self.chunks[case_idx], sample_idx, sample_idx+self.sequence_length),
        }

        if transform:
            sample = transform(sample)

        return sample


class RealOoD(data.Dataset):

    def __init__(self, img_dir, selection, with_ood, sequence_length=10, overlap_sequence=False,
                 transform=None, exclude_info=None) -> None:
        super().__init__()
        self.img_dir = img_dir
        if not isinstance(selection, (tuple, list)):
            selection = [selection]
        self.selection = selection
        self.sequence_length = sequence_length
        self.transform = transform
        ret = self._collect(img_dir, selection, with_ood, sequence_length, overlap_sequence, exclude_info)
        self.chunks, self.indices, self.chunk_ids, self.chunk_infos = ret

    @staticmethod
    def _collect(img_dir, selection, with_ood, sequence_length, overlap_sequence, exclude_info):
        chunk_ids, chunks, indices, chunk_infos = [], [], [], []
        case_idx = 0

        for item in selection:
            img_case_path = os.path.join(img_dir, item)
            img_paths = sorted(glob.glob(img_case_path + '/*.png'))

            for i, img_path in enumerate(img_paths):
                np_samples = np.asarray(Image.open(img_path)).astype(np.float32)

                chunk_id = os.path.basename(img_path)[:-len('.png')]
                info = chunk_id.split('_')[-1]

                if info == 'i':
                    # ignored cases
                    continue

                if not with_ood and info == 'o':
                    continue

                if exclude_info == info:
                    continue

                chunks.append(np_samples)
                chunk_ids.append(chunk_id)
                chunk_infos.append(info)

                cnt = len(np_samples) - sequence_length + 1
                if overlap_sequence:
                    case_indices = np.vstack([np.ones(cnt) * case_idx, np.arange(cnt)]).astype(np.int).T
                else:
                    inds = np.arange(cnt, step=sequence_length)
                    case_indices = np.vstack([np.ones(len(inds)) * case_idx, inds]).astype(np.int).T

                indices.append(case_indices)
                case_idx += 1

        return chunks, np.vstack(indices), chunk_ids, chunk_infos

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index) -> T_co:
        return self.get_sample(index, transform=self.transform)

    def get_sample(self, index, transform=None):
        case_idx, sample_idx = self.indices[index].tolist()

        sample = {
            'sample_index': index,
            'is_retina': self.chunk_infos[case_idx] == 'r',
            'is_ood': self.chunk_infos[case_idx] == 'o',
            'ascan': _extract_range(self.chunks[case_idx], sample_idx, sample_idx+self.sequence_length),
        }

        if transform:
            sample = transform(sample)

        return sample


def _extract_range(arr: np.ndarray, from_, to_):
    if to_ - from_ == 1:
        return arr[from_]
    return arr[from_: to_]


class ClinicalChunks(data.Dataset):

    def __init__(self, img_dir, target_name, selection, sequence_length=1, overlap_sequence=False,
                 transform=None, only_fg=True) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.target_name = target_name
        self.selection = selection
        self.sequence_length = sequence_length
        self.transform = transform
        ret = self._collect(img_dir, target_name, selection, sequence_length, overlap_sequence, only_fg)
        self.chunks, self.targets, self.indices, self.chunk_ids = ret

    @property
    def has_targets(self):
        return len(self.targets) > 0

    @staticmethod
    def _collect(img_dir, target_name, selection, sequence_length, overlap_sequence, only_fg):
        chunks = []
        targets = []
        indices = []
        chunk_ids = []

        has_targets = target_name is not None
        img_name = os.path.basename(img_dir)

        case_idx = 0
        for item in selection:
            img_case_path = os.path.join(img_dir, item)
            img_paths = sorted(glob.glob(img_case_path + '/*.png'))

            for i, img_path in enumerate(img_paths):
                if has_targets:
                    target_path = img_path.replace('.png', f'_target.png').replace(img_name, target_name)
                    if not os.path.exists(target_path):
                        continue  # not all images have targets -> ignore the images then
                    np_targets = np.asarray(Image.open(target_path)).astype(np.float32)
                    has_any = np_targets.any()
                    if only_fg and not has_any:
                        continue
                    targets.append(np_targets)

                np_samples = np.asarray(Image.open(img_path)).astype(np.float32)
                if has_targets:
                    assert len(np_samples) == len(np_targets)
                chunk_ids.append(os.path.basename(img_path)[:-len('.png')])
                chunks.append(np_samples)

                cnt = len(np_samples) - sequence_length + 1
                if overlap_sequence:
                    case_indices = np.vstack([np.ones(cnt) * case_idx, np.arange(cnt)]).astype(np.int).T
                else:
                    inds = np.arange(cnt, step=sequence_length)
                    case_indices = np.vstack([np.ones(len(inds)) * case_idx, inds]).astype(np.int).T

                indices.append(case_indices)
                case_idx += 1

        return chunks, targets, np.vstack(indices), chunk_ids

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index) -> dict:
        return self.get_sample(index, transform=self.transform)

    def get_sample(self, index, transform=None):
        case_idx, sample_idx = self.indices[index].tolist()

        sample = {
            'sample_index': index,
            'ascan': _extract_range(self.chunks[case_idx], sample_idx, sample_idx+self.sequence_length),
        }

        if self.has_targets:
            target = _extract_range(self.targets[case_idx], sample_idx, sample_idx + self.sequence_length)
            sample['target'] = target
            sample['is_retina'] = target.any()

        if transform:
            sample = transform(sample)

        return sample




