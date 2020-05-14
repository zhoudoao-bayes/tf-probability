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

IMAGE_SHAPE = [28, 28, 1]
NUM_TRAIN_EXAMPLES = 60000
NUM_HELDOUT_EXAPLES = 10000
NUM_CLASSES = 10

class MNISTSequence(tf.keras.utils.Sequence):
    def __init__(self, data=None, batch_size=128, fake_data_size=None):
        if data:
            images, labels = data
        else:
            images, labels = MNISTSequence.__generate_fake_data(
                num_images=fake_data_size, num_classes=NUM_CLASSES)
        self.images, self.labels = MNISTSequence.__preprocessing(images, labels)
        self.batch_size = batch_size
    
    @staticmethod
    def __generate_fake_data(num_images, num_classes):
        """Generates fake data in the shape of the MNIST dataset for unittest.

            Args:
            num_images: Integer, the number of fake images to be generated.
            num_classes: Integer, the number of classes to be generate.
            Returns:
            images: Numpy `array` representing the fake image data. The
                    shape of the array will be (num_images, 28, 28).
            labels: Numpy `array` of integers, where each entry will be
                    assigned a unique integer.
        """
        images = np.random.randint(low=0, high=256, 
            size=(num_images, IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
        labels = np.random.randint(low=0, high=num_classes,
            size=num_images)
        return images, labels 

    
    @staticmethod
    def __preprocessing(images, labels):
        """Preprocesses image and labels data.

        Args:
        images: Numpy `array` representing the image data.
        labels: Numpy `array` representing the labels data (range 0-9).

        Returns:
        images: Numpy `array` representing the image data, normalized
                and expanded for convolutional network input.
        labels: Numpy `array` representing the labels data (range 0-9),
                as one-hot (categorical) values.
        """
        images = 2 * (images / 255. ) - 1. 
        images = images[..., tf.newaxis]

        labels = tf.keras.utils.to_categorical(labels)
        return images, labels
    

    def __len__(self):
        return int(tf.math.ceil(len(self.images) / self.batch_size))

    
    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size: (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx+1) * self.batch_size]
        return batch_x, batch_y

