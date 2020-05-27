# zhoudoao@foxmail.com
# 2020.5.26
""" Preprocess MNIST datasets.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow.compat.v2 as tf

tf.enable_v2_behavior()


class MNISTSequence(tf.keras.utils.Sequence):
    """Produces a sequence of MNIST digits with labels."""

    def __init__(self, 
                 data=None, 
                 batch_size=128, 
                 fake_data_size=None,
                 num_classes=10,
                 image_shape=[28, 28, 1]):
        """Initializes the sequence.

        Args:
        data: Tuple of numpy `array` instances, the first representing images and
                the second labels.
        batch_size: Integer, number of elements in each training batch.
        fake_data_size: Optional integer number of fake datapoints to generate.
        """
        if data:
            images, labels = data
        else:
            images, labels = MNISTSequence.__generate_fake_data(
                num_images=fake_data_size, num_classes=num_classes, image_shape=image_shape)
        self.batch_size = batch_size
        self.images, self.labels = MNISTSequence.__preprocessing(
            images, labels)
        
        # self.num_classes = num_classes

    @staticmethod
    def __generate_fake_data(num_images, num_classes, image_shape):
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
                                   size=(num_images, image_shape[0],
                                         image_shape[1]))
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
        images = 2 * (images / 255.) - 1.
        images = images[..., tf.newaxis]

        labels = tf.keras.utils.to_categorical(labels)
        return images, labels

    def __len__(self):
        return int(tf.math.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y
