import sys
import argparse
import logging
import __main__

import common.loop as loop
import common.utils.threadhelper as thread
import common.access.config as config

is_debug = sys.gettrace() is not None


def train_loop_wrap(params, init_classes_fn, only_validate=False):
    try:
        run_dir, validation_dir, chk_dir, checkpoint_path = loop.init.initial_preparation(params, __main__.__file__)
        path_info = {'run_dir': run_dir, 'valid_dir': validation_dir, 'checkpoint_dir': chk_dir}

        context, interaction, callbacks = init_classes_fn(params, path_info, checkpoint_path)
        default_callbacks = [
            loop.ConsoleLog(),
            loop.SaveNLast(chk_dir, 1)
        ]
        if callbacks:
            default_callbacks.extend(callbacks)
        callback = loop.ComposeCallback(default_callbacks)

        trainer = loop.TrainLoop(context, interaction, callback, epochs=params['epochs'], seed=params['seed'],
                                 only_validate=only_validate)
        trainer.train(checkpoint_path)

    finally:
        thread.join_all()  # wait for plot to finish
        logging.exception('')  # log the exception


def combine_args_default_and_params(custom_params, default_config='{}'):
    config_arg, override_arg = parse_arguments(default_config)
    params = combine_configs(config_arg, custom_params, override_arg)
    return params


def parse_arguments(default_config):
    parser = argparse.ArgumentParser(description='Retina trajectory segmentation')

    parser.add_argument(
        '--config', type=str,
        default=default_config,
        help='Path to the configuration file or json string with config parameters'
    )
    parser.add_argument(
        '--override', type=str, help='Path to override config file or json string with config parameters.'
    )
    args = parser.parse_args()
    return args.config, args.override


def combine_configs(configuration, custom_params, override_config=None):
    params = config.get_config(configuration)

    override_params = {}
    if override_config:
        override_params = config.get_config(override_config)

    # take everything that is not in the override params from the custom ones
    config.add_config_entries(override_params, custom_params)

    # overwrite the params
    config.update_config(params, override_params)

    return params

