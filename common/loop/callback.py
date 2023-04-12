import logging
import os
import time
import csv

import common.access.checkpoint as chk
import common.utils.callbackhelper as clbh
from . import context as ctx


class Callback:

    def on_train_start(self, context: ctx.Context, loop_info):
        pass

    def on_train_end(self, context: ctx.Context, loop_info):
        pass

    def on_epoch_start(self, context: ctx.Context, loop_info):
        pass

    def on_epoch_end(self, context: ctx.Context, summary, loop_info):
        pass

    def on_batch_start(self, context: ctx.Context, batch, loop_info):
        pass

    def on_batch_end(self, context: ctx.Context, result, loop_info):
        pass

    def on_validation_start(self, context: ctx.Context, loop_info):
        pass

    def on_validation_end(self, context: ctx.Context, summary, best, loop_info):
        pass

    def on_validation_batch_start(self, context: ctx.Context, batch, loop_info):
        pass

    def on_validation_batch_end(self, context: ctx.Context, result, loop_info):
        pass

    def on_test_start(self, context: ctx.TestContext, loop_info):
        pass

    def on_test_end(self, context: ctx.TestContext, summary, loop_info):
        pass

    def on_test_batch_start(self, context: ctx.TestContext, batch, loop_info):
        pass

    def on_test_batch_end(self, context: ctx.TestContext, result, loop_info):
        pass


class ComposeCallback(Callback):

    def __init__(self, callbacks: list) -> None:
        super().__init__()
        clbh.make_reduce_compose(self, Callback, callbacks)


class ConsoleLog(Callback):

    def __init__(self, log_every_nth=10) -> None:
        super().__init__()
        self.log_every_nth = log_every_nth
        self.train_batch_start_time = None
        self.valid_start_time = None
        self.test_start_time = None

    def on_train_start(self, context: ctx.Context, loop_info):
        logging.info('model: \n{}'.format(str(context.model)))
        params = sum(p.numel() for p in context.model.parameters() if p.requires_grad)
        logging.info('trainable parameters: {}'.format(params))
        logging.info('startup finished')

        logging.info('loop')
        self.train_batch_start_time = time.time()

    def on_batch_end(self, context: ctx.Context, result, loop_info):
        epochs, epoch = loop_info.get('epochs'),  loop_info.get('epoch')
        batches, batch_idx = loop_info.get('batches'), loop_info.get('batch_idx')

        if ((batch_idx + 1) % self.log_every_nth == 0) or (batch_idx == batches - 1):
            duration = time.time() - self.train_batch_start_time
            result_string = ' | '.join(f'{v}: {k:.2e}' for v, k in result.items())
            logging.info(f'[{epoch + 1}/{epochs}, {batch_idx + 1}/{batches}, {duration:.2f}s] {result_string}')
            # start timing here in order to take into account the data loading
            self.train_batch_start_time = time.time()

    def on_validation_start(self, context: ctx.Context, loop_info):
        logging.info('validating')
        self.valid_start_time = time.time()

    def on_validation_end(self, context: ctx.Context, summary, best, loop_info):
        epochs, epoch = loop_info.get('epochs'),  loop_info.get('epoch')

        duration = time.time() - self.valid_start_time
        result_string = ' | '.join(f'{v}: {k:.2e}' for v, k in summary.items())
        logging.info(f'[{epoch + 1}/{epochs}, {duration:.2f}s] {result_string}')

    def on_test_start(self, context: ctx.TestContext, loop_info):
        logging.info('testing')
        self.test_start_time = time.time()

    def on_test_end(self, context: ctx.TestContext, summary, loop_info):
        duration = time.time() - self.test_start_time
        result_string = ' | '.join(f'{v}: {k:.2e}' for v, k in summary.items())
        logging.info(f'[{duration:.2f}s] {result_string}')


class TensorBoardLog(Callback):

    def __init__(self, log_every_nth=1) -> None:
        super().__init__()
        self.log_every_nth = log_every_nth
        self.start_time_epoch = 0
        self.start_time_valid = 0

    def on_epoch_start(self, context: ctx.Context, loop_info):
        self.start_time_epoch = time.time()

    def on_epoch_end(self, context: ctx.Context, summary, loop_info):
        duration = time.time() - self.start_time_epoch

        tb = self._get_logger_or_raise_error(context)
        epoch = loop_info.get('epoch')

        tb.add_scalar('train/time', duration, epoch)

    def on_validation_start(self, context: ctx.Context, loop_info):
        self.start_time_valid = time.time()

    def on_batch_end(self, context: ctx.Context, result, loop_info):
        tb = self._get_logger_or_raise_error(context)

        epochs, epoch = loop_info.get('epochs'),  loop_info.get('epoch')
        batches, batch_idx = loop_info.get('batches'), loop_info.get('batch_idx')

        if ((batch_idx + 1) % self.log_every_nth == 0) or (batch_idx == batches - 1):
            step = epoch * batches + batch_idx
            for key, result in result.items():
                tb.add_scalar('train/{}'.format(key), result, step)

    def on_validation_end(self, context: ctx.Context, summary, best, loop_info):
        duration = time.time() - self.start_time_epoch

        tb = self._get_logger_or_raise_error(context)
        epoch = loop_info.get('epoch')

        for key, result, in summary.items():
            tb.add_scalar('valid/{}'.format(key), result, epoch)

        tb.add_scalar('valid/time', duration, epoch)

    @staticmethod
    def _get_logger_or_raise_error(context: ctx.Context):
        if not hasattr(context, 'tb'):
            raise ValueError('missing tensorboard logger in context')
        return context.tb


class WriteCsv(Callback):

    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.file_path = file_path
        self.header_written = False

    def on_validation_end(self, context: ctx.Context, summary, best, loop_info):
        epoch = loop_info.get('epoch')
        to_write = {'epoch': epoch, **summary}

        with open(self.file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, to_write.keys())
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            writer.writerow(to_write)


class SaveBest(Callback):

    def __init__(self, checkpoint_dir, metric, comp_fn=lambda x, y: x > y, optional=False) -> None:
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.metric = metric
        self.is_better = comp_fn
        self.optional = optional
        self.saved_best = None
        self.saved_best_path = None

    def on_validation_end(self, context: ctx.Context, summary, best, loop_info):
        if self.metric not in best:
            if self.optional:
                return
            raise ValueError(f'{self.metric} missing as entry of "best"')

        if (self.saved_best is None) or self.is_better(best[self.metric], self.saved_best):
            # first delete the existing
            if self.saved_best_path is not None and os.path.isfile(self.saved_best_path):
                os.remove(self.saved_best_path)

            epoch = loop_info.get('epoch')
            checkpoint_path = chk.get_checkpoint_path(self.checkpoint_dir, epoch, self.metric)
            context.save_checkpoint(checkpoint_path, epoch, best)
            self.saved_best = best[self.metric]
            self.saved_best_path = checkpoint_path


class SaveBestMultiple(Callback):

    def __init__(self, checkpoint_dir, metric_dict, optional=False) -> None:
        super().__init__()
        self.clbks = []
        for metric, comp_fn in metric_dict.items():
            self.clbks.append(SaveBest(checkpoint_dir, metric, comp_fn, optional=optional))

    def on_validation_end(self, context: ctx.Context, summary, best, loop_info):
        for clb in self.clbks:
            clb.on_validation_end(context, summary, best, loop_info)


class SaveNLast(Callback):

    def __init__(self, checkpoint_dir: str, n_last: int) -> None:
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.n_last = n_last
        self.saved_epochs = []

    def on_validation_end(self, context: ctx.Context, summary, best, loop_info):
        self._remove_keep_n(self.n_last - 1)

        epoch = loop_info.get('epoch')
        checkpoint_path = chk.get_checkpoint_path(self.checkpoint_dir, epoch)
        context.save_checkpoint(checkpoint_path, epoch, best)

        self.saved_epochs.append(epoch)

    def _remove_keep_n(self, n):
        while len(self.saved_epochs) > n:
            epoch_to_remove = self.saved_epochs.pop(0)
            checkpoint_path = chk.get_checkpoint_path(self.checkpoint_dir, epoch_to_remove)
            if os.path.isfile(checkpoint_path):
                os.remove(checkpoint_path)
