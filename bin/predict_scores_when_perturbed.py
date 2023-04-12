import os
import sys

import torch
from torch.utils import data
import numpy as np
import pandas as pd
import torchvision.models as mdls

import common.utils.idhelper as idh
import common.loop as loop
import definitions as defs
import common.utils.torchhelper as th
import retinaqa.model as mdl
import retinaqa.data.transform as tfm
import retinaqa.data.dataset as ds
import retinaqa.eval.evaluator as evtor
import retinaqa.eval.metrics as metrics
import common.access.config as cfg

is_debug = sys.gettrace() is not None


def main():
    params = {
        'test_name': 'allmethods_ratio',
    }
    params['test_name'] = f'{idh.get_unique_identifier()}_{params["test_name"]}'

    out_dir = os.path.join(defs.SIMULATED_OUT_DIR, params['test_name'])
    os.makedirs(out_dir)

    config_path = os.path.join(out_dir, 'config.yaml')
    cfg.save_config(params, config_path)

    perts = ['stripe', 'rect', 'zoom', 'shift', 'noise', 'contr', 'smooth', 'intsh']
    pert_probabilities = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    for pert_prob in pert_probabilities:
        print(f'\nProbability: {pert_prob}')
        postfix = str(pert_prob).replace(".", "")

        perturbations = {'none': 1-pert_prob, **{k: pert_prob / len(perts) for k in perts}}
        perturbation = tfm.OnlinePerturbation(perturbations)

        context = MyContext(defs.DIST_DATA_DIR, perturbation)
        interaction = MyInteraction(out_dir, postfix, defs.MAHA_FILE, defs.MAHA_RAW_FILE)
        tester = loop.Tester(context, interaction, loop.ConsoleLog())
        tester.test({'est': defs.EST_CHECKPOINT, 'ood-sup': defs.SUP_CHECKPOINT, 'ood-glow': defs.GLOW_CHECKPOINT})


class MyContext(loop.TestContext):

    def __init__(self, dataset_dir, perturbation: tfm.Transform, device='cuda') -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.perturbation = perturbation
        self._device = device
        self.sequence_length = defs.SEQL
        self.selection = ['pig4']

    @property
    def device(self):
        return self._device

    def _init_model(self):
        estimation_model = mdl.GenericUnet(out_ch=1, in_ch=1, channels=[8, 16, 32, 32, 32], repetitions=1, dropout=0)
        ood_model = mdls.efficientnet_b0(weights=mdls.EfficientNet_B0_Weights.IMAGENET1K_V1)
        ood_sup_model = mdl.get_adapted_effnet(out_classes=1, weights=mdls.EfficientNet_B0_Weights.IMAGENET1K_V1)
        ood_glow_model = mdl.Glow((64, 224, 3), 512, 32, 3, 1.0, "invconv", "affine", True, 1, True, False,)
        model = torch.nn.ModuleDict({'estimator': estimation_model, 'ood-maha': ood_model, 'ood-sup': ood_sup_model, 'ood-glow': ood_glow_model})
        model.to(self.device)
        return model

    def load_checkpoint(self, checkpoint_path):
        est_path, sup_path, glow_path = checkpoint_path['est'], checkpoint_path['ood-sup'], checkpoint_path['ood-glow']

        est_checkpoint = torch.load(est_path)
        self.model['estimator'].load_state_dict(est_checkpoint.pop('model_state_dict'))

        sup_checkpoint = torch.load(sup_path)
        self.model['ood-sup'].load_state_dict(sup_checkpoint.pop('model_state_dict'))

        glow_checkpoint = torch.load(glow_path)
        self.model['ood-glow'].load_state_dict(glow_checkpoint)
        self.model['ood-glow'].set_actnorm_init()

    def _init_test_loader(self):
        transforms = [
            self.perturbation,
            tfm.MinMaxNormalization((0., 255.), (0.0, 1.0)),
            CopyTo('ood'),
            tfm.Resize(defs.RESIZE, ('ood',)),
            tfm.ToRGB(('ood',)),
            tfm.ChannelWiseNorm(entries=('ood',))
        ]

        transform = tfm.ComposeTransform(transforms)

        dataset = ds.ClinicalChunks(self.dataset_dir, defs.DIST_TARGET_NAME, self.selection,
                                    sequence_length=self.sequence_length, transform=transform)
        test_loader = data.DataLoader(dataset, batch_size=24, shuffle=False, num_workers=0 if is_debug else 4,
                                      pin_memory=True, worker_init_fn=th.seed_worker)
        return test_loader


class CopyTo(tfm.Transform):

    def __init__(self, new_entry, entry=tfm.KEY_IMAGES) -> None:
        super().__init__()
        self.new_entry = new_entry
        self.entry = entry

    def __call__(self, sample: dict) -> dict:
        sample[self.new_entry] = sample[self.entry].copy()
        return sample


class MyInteraction(loop.TestInteraction):

    def __init__(self, test_dir, postfix, maha_file, maha_img_file) -> None:
        super().__init__()
        self.test_dir = test_dir
        self.postfix = postfix
        with np.load(maha_file) as d:
            self.maha_params = {**d}
        with np.load(maha_img_file) as d:
            self.maha_img_params = {**d}

    def test_step(self, context: MyContext, batch, loop_info):
        x = batch['ascan'].float().to(context.device)
        x_ood = batch['ood'].float().to(context.device)

        single_y_hats = []
        maha_img_scores = []
        for i in range(x.size(1)):
            single_y_hats.append(context.model['estimator'](x[:, i:i+1]))
            mean, cov_inv = self.maha_img_params[f'mu'], self.maha_img_params[f'covinv']
            dist = metrics.maha_distance_2d(x[:, i:i+1].squeeze(1).cpu().numpy(), mean, cov_inv)
            maha_img_scores.append(dist)
        y_hat = torch.cat(single_y_hats, dim=1)

        maha_img_scores = np.stack(maha_img_scores, axis=1)[..., np.newaxis]

        features = mdl.get_effnet_features(context.model['ood-maha'], x_ood)
        all_dists = []
        for feature_index, feature in enumerate(features):
            mean, cov_inv = self.maha_params[f'mu_{feature_index}'], self.maha_params[f'covinv_{feature_index}']
            dists = metrics.maha_distance_2d(feature.cpu().numpy(), mean, cov_inv)
            all_dists.append(dists)
        scores = np.sum(all_dists, axis=0)
        # repeat to have one per sample
        scores = np.tile(scores[:, np.newaxis], (1, context.sequence_length))[..., np.newaxis]

        sup_y_hat = context.model['ood-sup'](x_ood)
        sup_scores = torch.sigmoid(sup_y_hat).squeeze(1).cpu().numpy()
        sup_scores = np.tile(sup_scores[:, np.newaxis], (1, context.sequence_length))[..., np.newaxis]

        glow_res = context.model['ood-glow'](x_ood)
        glow_scores = glow_res[1].cpu().numpy()
        glow_scores = np.tile(glow_scores[:, np.newaxis], (1, context.sequence_length))[..., np.newaxis]

        snr = x.mean(dim=(1, 2)) / x.std(dim=(1, 2))
        snr = snr.cpu().numpy()
        snr = np.tile(snr[:, np.newaxis], (1, context.sequence_length))[..., np.newaxis]

        est_scores = y_hat.cpu().numpy().max(axis=-1, keepdims=True)

        results = {
            'estimate': y_hat.cpu().numpy(),
            'maha-score': scores,
            'maha-img-score': maha_img_scores,
            'sup-score': sup_scores,
            'glow-score': glow_scores,
            'snr': snr,
            'est-score': est_scores,
            'target': batch['target'].numpy(),
            'pert_name': np.tile(np.asarray(batch['pert_name'])[:, np.newaxis], (1, context.sequence_length))[..., np.newaxis]
        }
        return results

    def test_summary(self, context: MyContext, results, loop_info):
        summary = {}

        estimate = np.vstack(results.pop('estimate'))
        estimate = np.expand_dims(estimate, axis=1)
        targets = np.vstack(results.pop('target'))
        targets = np.expand_dims(targets, axis=1)

        maxs = estimate.max(axis=-1)
        max_mask = maxs < 0.3
        predicted_boundaries = np.argmax(estimate, axis=-1).astype(float)
        predicted_boundaries[max_mask] = np.nan
        # predicted_boundaries_no_nan = np.nan_to_num(predicted_boundaries, nan=estimate.shape[-1] - 1)
        predicted_dist = predicted_boundaries * defs.SPACING_MM

        mask = targets.max(axis=-1) == 0
        target_boundaries = np.argmax(targets, axis=-1).astype(float)
        target_boundaries[mask] = np.nan
        target_dist = target_boundaries * defs.SPACING_MM

        label_names = ('ILM',)
        label_dict = {i: name for i, name in enumerate(label_names)}
        evaluator = evtor.DistanceEvaluator(label_dict, nan_as=(targets.shape[-1] - 1) * defs.SPACING_MM)
        evaluator.calculate_metrics(predicted_dist, target_dist)
        summary.update(evaluator.summary)

        dataset = context.test_loader.dataset
        data_inds = dataset.indices[:, 0].repeat(dataset.sequence_length)
        # cannot take the information from the dataset since depends on sequence length
        case_indices = np.asarray(dataset.chunk_ids)[data_inds]
        sequence_ids = np.tile(np.arange(len(dataset.chunks[0])), len(dataset.chunk_ids))

        maha_scores = np.vstack(results.pop('maha-score'))
        maha_img_scores = np.vstack(results.pop('maha-img-score'))
        est_scores = np.vstack(results.pop('est-score'))
        sup_scores = np.vstack(results.pop('sup-score'))
        glow_scores = np.vstack(results.pop('glow-score'))
        snr = np.vstack(results.pop('snr'))
        pert_name = np.vstack(results.pop('pert_name'))

        assert (est_scores == maxs).all()

        df = pd.DataFrame({'sequence_id': sequence_ids,
                           'case_id': case_indices,
                           'prediction': predicted_boundaries.squeeze(),
                           'target': target_boundaries.squeeze(),
                           'ae': evaluator.case_results['AE_ILM'],
                           'se': evaluator.case_results['SE_ILM'],
                           'acc': evaluator.case_results['ACC_ILM'],
                           'pert_name': pert_name.squeeze(),
                           'maha_score': maha_scores.squeeze(),
                           'estimation_score': est_scores.squeeze(),
                           'maha_img_score': maha_img_scores.squeeze(),
                           'supervised_score': sup_scores.squeeze(),
                           'glow_score': glow_scores.squeeze(),
                           'snr': snr.squeeze(),
                           })
        df.to_csv(os.path.join(self.test_dir, f'scores_and_metrics_{self.postfix}.csv'))

        return summary


if __name__ == '__main__':
    main()
