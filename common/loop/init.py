import os
import shutil
import logging

import common.utils.logginghelper as lh
import common.utils.idhelper as idh
import common.access.checkpoint as chk
import common.access.config as cfg


def initial_preparation(params, main_file):
    checkpoint_path = None
    # resume should be used to resume a training procedure that was interrupted or should be trained for some additional epochs
    if 'resume' in params:
        run_dir = params['resume']

        checkpoint_path = chk.get_last_checkpoint(chk.get_checkpoint_dir(run_dir))

        orig_config_path = os.path.join(run_dir, 'config.yaml')
        orig_params = cfg.read_config(orig_config_path)
        # extend resume parameters with the original parameters
        cfg.add_config_entries(params, orig_params)
    else:
        run_dir = os.path.join(params['train_dir'], f'{idh.get_unique_identifier()}_{params["train_name"]}')
        os.makedirs(run_dir)

    config_path = os.path.join(run_dir, 'config.yaml')
    _create_backup_if_exists(config_path)
    cfg.save_config(params, config_path)
    run_file_path = os.path.join(run_dir, 'run_file.py')
    _create_backup_if_exists(run_file_path)
    shutil.copyfile(main_file, run_file_path)

    lh.setup_file_logging(os.path.join(run_dir, 'log.txt'))

    logging.info(f'parameters:\n{cfg.to_str(params)}')

    validation_dir = os.path.join(run_dir, 'validation')
    os.makedirs(validation_dir, exist_ok=True)  # exist_ok for the 'resume' case

    chk_dir = chk.get_checkpoint_dir(run_dir)
    os.makedirs(chk_dir, exist_ok=True)

    return run_dir, validation_dir, chk_dir, checkpoint_path


def _create_backup_if_exists(file_path):
    if os.path.exists(file_path):
        name, ext = os.path.splitext(file_path)
        bak_path = f'{name}.bak{ext}'
        if os.path.exists(bak_path):
            os.remove(bak_path)
        shutil.copyfile(file_path, bak_path)


def initial_test_preparation(params: dict, main_file, retrieve_from_train: tuple):
    checkpoint_path = params['checkpoint']
    train_dir = os.path.dirname(os.path.dirname(checkpoint_path))

    if retrieve_from_train:
        orig_config_path = os.path.join(train_dir, 'config.yaml')
        train_params = cfg.read_config(orig_config_path)
        cfg.add_config_entries(params, train_params, selection=retrieve_from_train)

    test_dir = params.get('test_dir', None)
    if test_dir is None:
        test_dir = os.path.join(train_dir, 'test')

    os.makedirs(test_dir, exist_ok=True)

    run_dir = os.path.join(test_dir, f'{idh.get_unique_identifier()}_{params["test_name"]}')
    os.makedirs(run_dir)

    config_path = os.path.join(run_dir, 'config.yaml')
    cfg.save_config(params, config_path)
    run_file_path = os.path.join(run_dir, 'run_file.py')
    shutil.copyfile(main_file, run_file_path)

    lh.setup_file_logging(os.path.join(run_dir, 'log.txt'))

    logging.info(f'parameters:\n{cfg.to_str(params)}')

    return run_dir, checkpoint_path
