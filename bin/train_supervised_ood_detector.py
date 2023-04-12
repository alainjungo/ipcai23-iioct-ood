import sys
import random

import torch
from torch import optim
import torch.nn.functional as F
import torch.utils.tensorboard as tensorboard
from torch.utils import data
import torchvision.models as mdls
import numpy as np
import sklearn.metrics as m

from common.loop import context as ctx
import common.loop as loop
import retinaqa.common.loophelper as loop_hlp
import retinaqa.data.transform as tfm
import retinaqa.data.dataset as ds
import retinaqa.model as mdl
import definitions as defs
import common.utils.torchhelper as th

is_debug = sys.gettrace() is not None


def main():
    custom_params = {
        'train_name': 'supervised',
        'train_dir': defs.MODELS_DIR,
        'dataset_dir': defs.DATA_DIR,

        'epochs': 50,
        'seed': 42,
        'lr': 1e-5,

        'train_perturbations': ['shift', 'smooth', 'noise', 'intsh'],
        'valid_perturbations': ['shift', 'smooth', 'noise', 'intsh'],
    }
    params = loop_hlp.combine_args_default_and_params(custom_params)

    metric_objectives = {'loss': np.less, 'AUC': np.greater}

    ref = {}
    def init_classes(params, path_info, checkpoint_path):
        ref['tb'] = tensorboard.SummaryWriter(path_info['run_dir'])
        ref['checkpoint_dir'] = path_info['checkpoint_dir']
        # might have been overwritten, thus getting them here
        context = MyContext(params, ref['tb'])
        interaction = MyInteractions(path_info['valid_dir'], metric_objectives)
        callbacks = [loop.TensorBoardLog(10), loop.SaveBestMultiple(path_info['checkpoint_dir'], metric_objectives)]
        return context, interaction, callbacks

    loop_hlp.train_loop_wrap(params, init_classes, only_validate=False)
    ref['tb'].close()


class MyContext(loop.Context):

    def __init__(self, params, tb) -> None:
        super().__init__()
        self.dataset_dir = params['dataset_dir']
        self.train_pert = params['train_perturbations']
        self.valid_pert = params['valid_perturbations']
        self.sequence_length = defs.SEQL
        self.lr = params['lr']
        self.tb = tb

    @property
    def device(self):
        return 'cuda'

    def _init_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)

    def _init_model(self):
        model = mdl.get_adapted_effnet(out_classes=1, weights=mdls.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.to(self.device)
        return model

    def _init_train_loader(self):
        perturbations = {'none': 0.5, **{k: 0.5 / len(self.train_pert) for k in self.train_pert}}

        transforms = [
            tfm.OnlinePerturbation(perturbations, targe_entry=None),
            tfm.MinMaxNormalization((0., 255.), (0.0, 1.0), entries=(tfm.KEY_IMAGES,)),
            tfm.Resize(defs.RESIZE),
            tfm.ToRGB(),
            tfm.ChannelWiseNorm(),
        ]

        transform = tfm.ComposeTransform(transforms)
        # no ood required since perturbing
        dataset = ds.RealOoD(self.dataset_dir, 'dl-train', with_ood=False, transform=transform,
                             sequence_length=self.sequence_length, exclude_info='b')

        train_loader = data.DataLoader(dataset, batch_size=24, shuffle=False, num_workers=0 if is_debug else 4,
                                       pin_memory=True, worker_init_fn=th.seed_worker)
        return train_loader

    def _init_valid_loader(self):
        perturbations = {'none': 0.5, **{k: 0.5 / len(self.valid_pert) for k in self.valid_pert}}

        transforms = [
            tfm.OnlinePerturbation(perturbations, targe_entry=None),
            tfm.MinMaxNormalization((0., 255.), (0.0, 1.0), entries=(tfm.KEY_IMAGES,)),
            tfm.Resize(defs.RESIZE),
            tfm.ToRGB(),
            tfm.ChannelWiseNorm(),
        ]
        transform = tfm.ComposeTransform(transforms)

        # no ood required since perturbing
        dataset = ds.RealOoD(self.dataset_dir, 'dl-valid', with_ood=False, transform=transform,
                             sequence_length=self.sequence_length, exclude_info='b')

        # always same validation random validation perturbations
        # num_worker>=1 since otherwise seed_worker doesn't get called
        valid_loader = data.DataLoader(dataset, batch_size=24, shuffle=False, num_workers=1 if is_debug else 4,
                                       pin_memory=True, worker_init_fn=get_determinisitc_seed_worker(42))
        return valid_loader


def get_determinisitc_seed_worker(seed):
    def seed_worker(worker_id):
        # This function is to be used as 'worker_init_fn' argument to the pytorch dataloader
        # Needed because the numpy seed is shared among the workers
        worker_seed = seed % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return seed_worker


class MyInteractions(loop.Interaction):

    def __init__(self, validation_dir, metric_objectives) -> None:
        super().__init__(metric_objectives)
        self.validation_dir = validation_dir
        self.inital_xs = None

    def training_step(self, context: loop.Context, batch, loop_info):
        x, y = batch['ascan'].float().to(context.device), batch['is_perturbed'].float().to(context.device)

        y_hat = context.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat.squeeze(-1), y)

        loss.backward()
        context.optimizer.step()

        return {'loss': loss.item()}

    def validation_step(self, context: ctx.Context, batch, loop_info):
        x, y = batch['ascan'].float().to(context.device), batch['is_perturbed'].float().to(context.device)

        y_hat = context.model(x)
        y_hat = y_hat.squeeze(-1)

        probs = torch.sigmoid(y_hat)

        loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction='none')

        results = {
            'inputs': x.cpu().numpy(),
            'estimate': probs.cpu().numpy(),
            'target': y.cpu().numpy(),
            'loss': loss.cpu().numpy(),
        }

        return results

    def validation_summary(self, context: ctx.Context, results, loop_info):
        if not isinstance(context, MyContext):
            raise ValueError(f'needs being of type {MyContext.__name__}')

        loader_id = loop_info['loader_id']
        loader_str = loader_id if len(loader_id) == 0 else f'{loader_id}_'

        summary = {f'{loader_str}loss': np.mean(results.pop('loss'))}

        xs = np.stack(results.pop('inputs'))
        if self.inital_xs is None:
            self.inital_xs = xs

        equal = (xs == self.inital_xs).all()
        if not equal:
            raise ValueError('validation set has changed')

        estimate = np.stack(results.pop('estimate'))  # np.stack since estimate have a dim more
        targets = np.stack(results.pop('target'))  # np.stack since targets have a dim more

        fpr, tpr, _ = m.roc_curve(targets, estimate)
        auc = m.auc(fpr, tpr)
        ap = m.average_precision_score(targets, estimate)
        summary['AUC'] = auc
        summary['AP'] = ap

        return summary


if __name__ == '__main__':
    main()
