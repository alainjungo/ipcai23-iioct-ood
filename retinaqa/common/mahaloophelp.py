import os

import numpy as np
import pandas as pd

import retinaqa.eval.metrics as metrics
import retinaqa.model as mdl
import retinaqa.eval.metrics as m
import common.loop as loop


class ExtractParamsInteraction(loop.TestInteraction):

    def __init__(self, maha_file) -> None:
        super().__init__()
        self.maha_file = maha_file

    def test_step(self, context: loop.TestContext, batch, loop_info):
        x = batch['ascan'].float().to(context.device)

        features = mdl.get_effnet_features(context.model, x)
        results = {f'feature_{i}': f.cpu().numpy() for i, f in enumerate(features)}

        return results

    def test_summary(self, context: loop.TestContext, results, loop_info):

        feature_keys = sorted([k for k in results if k.startswith('feature_')])
        maha_params = {}
        for k in feature_keys:
            feature_index = int(k.rsplit('_', 1)[1])
            features = results.pop(k)
            features = np.stack(features)

            mean, cov_inv = m.get_maha_prams(features)
            maha_params[f'mu_{feature_index}'] = mean
            maha_params[f'covinv_{feature_index}'] = cov_inv

        np.savez(self.maha_file, **maha_params)

        return {}


class ApplyParamsInteraction(loop.TestInteraction):

    def __init__(self, out_dir, maha_file) -> None:
        super().__init__()
        self.out_dir = out_dir
        with np.load(maha_file) as d:
            self.maha_params = {**d}

    def test_step(self, context: loop.TestContext, batch, loop_info):
        x = batch['ascan'].float().to(context.device)

        features = mdl.get_effnet_features(context.model, x)

        results = {f'feature_{i}': f.cpu().numpy() for i, f in enumerate(features)}
        return results

    def test_summary(self, context: loop.TestContext, results, loop_info):
        nb_features = len([k for k in results if k.startswith('feature_')])

        maha_dists = {}
        feature_stats = {}
        for feature_index in range(nb_features):
            mean, cov_inv = self.maha_params[f'mu_{feature_index}'], self.maha_params[f'covinv_{feature_index}']
            features = np.stack(results.pop(f'feature_{feature_index}'))

            feature_stats[f'feat_ch-mean_{feature_index}'] = features.mean(axis=-1)
            feature_stats[f'feat_ch-std_{feature_index}'] = features.std(axis=-1)

            dists = metrics.maha_distance_2d(features, mean, cov_inv)

            maha_dists[f'maha-dist_{feature_index}'] = dists

        scores = np.sum([m for m in maha_dists.values()], axis=0)

        dataset = context.test_loader.dataset
        d = {'sequence_id': dataset.indices[:, 1],
             'case_id': np.asarray(dataset.chunk_ids)[dataset.indices[:, 0]],
             'score': scores
             }
        d.update(maha_dists)
        d.update(feature_stats)

        df = pd.DataFrame(d)
        df.to_csv(os.path.join(self.out_dir, 'maha_distances.csv'))

        return {}

