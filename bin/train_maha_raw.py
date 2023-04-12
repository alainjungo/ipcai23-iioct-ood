import os

import numpy as np
import pandas as pd

import common.utils.idhelper as idh
import retinaqa.eval.metrics as m
import retinaqa.data.dataset as ds
import definitions as defs


def main():
    params = {
        'test_name': 'maharaw',
    }
    id_ = idh.get_unique_identifier()
    params['test_name'] = f'{id_}_{params["test_name"]}'
    out_dir = os.path.join(defs.MODELS_DIR, params['test_name'])
    os.makedirs(out_dir)

    print('Extract Maha params')
    dataset = get_dataset(defs.DATA_DIR, ['train'], with_ood=False)
    all_seqs, _, _ = get_preprocessed_seqs(dataset)

    mean, cov_inv = m.get_maha_prams(all_seqs)
    maha_params = {'mu': mean, 'covinv': cov_inv}
    maha_file = os.path.join(out_dir, 'maha.npz')
    np.savez(maha_file, **maha_params)

    print('Apply Maha params')
    dataset = get_dataset(defs.DATA_DIR, ['test'], with_ood=True)
    all_seqs, sequence_ids, case_ids = get_preprocessed_seqs(dataset)
    all_dists = m.maha_distance_2d(all_seqs, mean, cov_inv)

    seq_mean_dists = all_dists.reshape((len(case_ids), dataset.sequence_length)).mean(axis=1)

    data = {'sequence_id': sequence_ids,
            'case_id': case_ids,
            'score': seq_mean_dists
            }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(out_dir, 'maha_distances.csv'))


def get_dataset(dataset_dir, selection, with_ood):
    return ds.RealOoD(dataset_dir, selection, with_ood, sequence_length=defs.SEQL, exclude_info='b')


def get_preprocessed_seqs(dataset):
    all_seqs = np.vstack(dataset.chunks).astype(float)
    all_seqs /= 255  # from range [0,255] to [0,1]

    sequence_ids = dataset.indices[:, 1]
    case_ids = np.asarray(dataset.chunk_ids)[dataset.indices[:, 0]]
    assert (len(all_seqs) / dataset.sequence_length) == len(sequence_ids) == len(case_ids)
    return all_seqs, sequence_ids, case_ids


if __name__ == '__main__':
    main()


