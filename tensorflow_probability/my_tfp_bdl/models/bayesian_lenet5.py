# zhoudoao@foxmail.com
# 2020.5.12
""" Bayesian LeNet-5 and LeNet-5 model.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tf.enable_v2_behavior()
tfd = tfp.distributions 


def bayesian_lenet5(num_classes,
                    kl_divergence_function):
    model = tf.keras.models.Sequential([
        tfp.layers.Convolution2DFlipout(
            6, kernel_size=5, padding='SAME',
            kernel_divergence_fn=kl_divergence_function,
            activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(
            pool_size=[2, 2], strides=[2, 2],
            padding='SAME'),
        tfp.layers.Convolution2DFlipout(
            16, kernel_size=5, padding='SAME',
            kernel_divergence_fn=kl_divergence_function,
            activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(
            pool_size=[2, 2], strides=[2, 2],
            padding='SAME'),
        tfp.layers.Convolution2DFlipout(
            120, kernel_size=5, padding='SAME',
            kernel_divergence_fn=kl_divergence_function,
            activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        tfp.layers.DenseFlipout(
            84, kernel_divergence_fn=kl_divergence_function,
            activation=tf.nn.relu),
        tfp.layers.DenseFlipout(
            num_classes, kernel_divergence_fn=kl_divergence_function,
            activation=tf.nn.softmax)
    ])

    return model


def lenet5(num_classes, activation=tf.nn.relu):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(6, kernel_size=5, padding='SAME',
            activation=activation),
        tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
            strides=[2, 2], padding='SAME'),
        tf.keras.layers.Conv2D(16, kernel_size=5, padding='SAME',
            activation=activation),
        tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2],
            padding='SAME'),
        tf.keras.layers.Conv2D(120, kernel_size=5, padding='SAME',
           activation=activation),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(84, activation=activation),
        tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
    ])

    return model





