""" Script to parse output of run_model.py """

import tensorflow as tf
import argparse
import platform
import os
from img_common.parser import Parser


def load_config_procedures():
    """
    Function to read the configurations from the specified config file

    @rtype: str
    @return: directory path
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_to_parse')
    path = parser.parse_args().folder_to_parse
    return path


if __name__ == '__main__':
    devices = tf.config.experimental.list_physical_devices('GPU')
    list(map(lambda d: tf.config.experimental.set_memory_growth(d, True),
             devices))

    if not platform.system().lower() == 'linux':
        raise RuntimeError('This code currently only works linux environments')
    else:
        if 'LD_LIBRARY_PATH' not in os.environ:
            os.environ['LD_LIBRARY_PATH'] = ''
        if 'PATH' not in os.environ:
            os.environ['PATH'] = ''
        os.environ['LD_LIBRARY_PATH'] += ':./linux_binaries'
        os.environ['PATH'] += ':./linux_binaries'

    run_path = load_config_procedures()
    parser = Parser(run_path)
    parser.parse()