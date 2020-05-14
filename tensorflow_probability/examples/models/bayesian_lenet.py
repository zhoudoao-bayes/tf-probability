# zhoudoao
# 2020.5.14
"""Build a Bayesian Lenet5 model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions 

def bayesian_lenet(
    input_shape,
    num_classes=10,
    kernel_posterior_scale_mean=-9.0,
    kernel_posterior_scale_stddev=0.1,
    kernel_posterior_scale_mean=0.2
):

    def _untransformed_scale_constraint(t):
        return tf.clip_by_value(t, -1000,
            tf.math.log(kernel_posterior_scale_constraint))


    kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                            tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))

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
            NUM_CLASSES, kernel_divergence_fn=kl_divergence_function,
            activation=tf.nn.softmax)
    ])

    return model

    

    


