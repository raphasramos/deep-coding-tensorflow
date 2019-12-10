""" File with the useful enums for the project """


from img_common.keras_custom import *
from enum import Enum, IntEnum


class Folders(IntEnum):
    """ Enum representing the names of the folders for output """
    RAW_DATA = 0
    PARSED_DATA = 1
    TENSORBOARD = 2
    CHECKPOINTS = 3

    def __str__(self):
        return self.name.lower()


class Optimizers(Enum):
    """ Enum representing the acceptable optimizers """
    ADAM = tf.keras.optimizers.Adam
    RMS = tf.keras.optimizers.RMSprop

    def __str__(self):
        return self.name.lower()

    @classmethod
    def _missing_(cls, value):
        return cls.__members__[value.upper()]


class Losses(Enum):
    """ Enum with the available losses """
    MSE = tf.keras.losses.MeanSquaredError()

    def __str__(self):
        return self.name.lower()

    @classmethod
    def _missing_(cls, value):
        return cls.__members__[value.upper()]


class KLayers(Enum):
    """ Enum with keras layers (the strings used in the config file) """
    CONV2D = tf.keras.layers.Conv2D
    CONV2D_LSTM = tf.keras.layers.ConvLSTM2D
    CONV2D_TRANSPOSE = tf.keras.layers.Conv2DTranspose
    CONV3D = tf.keras.layers.Conv3D
    CONV3D_TRANSPOSE = tf.keras.layers.Conv3DTranspose
    DENSE = tf.keras.layers.Dense
    SUBTRACT = tf.keras.layers.Subtract
    RESHAPE = tf.keras.layers.Reshape

    # Custom layers
    QUANTIZE = Quantize
    CONV_BIN = ConvBin
    DEPTH_TO_SPACE = DepthToSpace
    CUSTOM_CONV2D_LSTM = CustomConv2DLSTM

    def __str__(self):
        return self.name.lower()

    @classmethod
    def _missing_(cls, value):
        return cls.__members__[value.upper()]
