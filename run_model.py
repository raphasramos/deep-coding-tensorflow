""" This script runs an autoencoder model. It parses a .json file
"""

import tensorflow as tf
import argparse
from shutil import copy
import json
from pathlib import Path
from img_common.autoencoder import AutoEnc


def load_config_procedures():
    """ Function to read the configurations from the specified config file """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    config_file = Path(parser.parse_args().config_file)
    with open(config_file, 'r') as config:
        json_c = json.load(config)
    return json_c, config_file


if __name__ == '__main__':
    devices = tf.config.experimental.list_physical_devices('GPU')
    list(map(lambda d: tf.config.experimental.set_memory_growth(d, True),
             devices))
    configs, config_file = load_config_procedures()
    autoenc = AutoEnc(configs)
    copy(config_file, autoenc.out_folder / config_file.name)
    autoenc.run()
