# zhoudoao
# 2020.5.14

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

# Dependency imports
from absl import app
from absl import flags
import matplotlib
matplotlib.use('Agg')
from matplotlib import figure  # pylint: disable=g-import-not-at-top
from matplotlib.backends import backend_agg
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tf.enable_v2_behavior()


IMAGE_SHAPE = [ ]
NUM_TRAIN_EXAMPLES = 50000
NUM_HELDOUT_EXAMPLES = 10000
NUM_CLASSES = 10


class CIFAR10Sequence(tf.keras.utils.Sequence):
    def __init__(self, data=None, batch_size=128, fake_dataset=None):
        if data:
            images, labels = data
        else:
            images, labels = MNISTSequence.__generate_fake_data(
                num_images=fake_data_size, num_classes=NUM_CLASSES)
        self.images, self.labels = MNISTSequence.__preprocessing(images, labels)
        self.batch_size = batch_size

    @staticmethod
    


