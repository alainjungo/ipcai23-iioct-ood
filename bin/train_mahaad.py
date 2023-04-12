import os
import sys

import torchvision.models as mdls
import torch.utils.data as data

import common.utils.idhelper as idh
import common.loop as loop
import retinaqa.data.transform as tfm
import retinaqa.data.dataset as ds
import definitions as defs
import common.utils.torchhelper as th
import retinaqa.common.mahaloophelp as maha_loop

is_debug = sys.gettrace() is not None


def main():
    params = {
        'test_name': 'mahaad',
        'sequence_length': 10,
    }
    id_ = idh.get_unique_identifier()
    params['test_name'] = f'{id_}_{params["test_name"]}'
    out_dir = os.path.join(defs.MODELS_DIR, params['test_name'])

    os.makedirs(out_dir)
    maha_file = os.path.join(out_dir, 'maha.npz')

    print('Extract params')
    context = MyContext(defs.DATA_DIR, ['train'], with_ood=False)
    extract_interaction = maha_loop.ExtractParamsInteraction(maha_file)
    tester = loop.Tester(context, extract_interaction, loop.ConsoleLog())
    tester.test('')

    print('Apply params')
    context = MyContext(defs.DATA_DIR, ['test'], with_ood=True)
    apply_interaction = maha_loop.ApplyParamsInteraction(out_dir, maha_file)
    tester = loop.Tester(context, apply_interaction, loop.ConsoleLog())
    tester.test('')


class MyContext(loop.TestContext):

    def __init__(self, dataset_dir, selection, with_ood) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.selection = selection
        self.with_ood = with_ood

    @property
    def device(self):
        return 'cuda'

    def _init_model(self):
        model = mdls.efficientnet_b0(weights=mdls.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.to(self.device)
        return model

    def load_checkpoint(self, checkpoint_path: str):
        pass  # no need to load since pretrained loaded already

    def _init_test_loader(self):
        tfms = [
            tfm.MinMaxNormalization((0., 255.), (0.0, 1.0)),
            tfm.Resize(defs.RESIZE),
            tfm.ToRGB(),
            tfm.ChannelWiseNorm()
        ]

        transform = tfm.ComposeTransform(tfms)

        dataset = ds.RealOoD(self.dataset_dir, self.selection, self.with_ood, transform=transform,
                             sequence_length=defs.SEQL, exclude_info='b')

        test_loader = data.DataLoader(dataset, batch_size=24, num_workers=0 if is_debug else 4,
                                      pin_memory=True, worker_init_fn=th.seed_worker)
        return test_loader


if __name__ == '__main__':
    main()
