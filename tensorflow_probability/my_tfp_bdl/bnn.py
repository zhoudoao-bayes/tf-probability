# zhoudoao@foxmail.com
# 2020.5.26
""" Trains a Bayesian LeNet-5 to classify MNIST.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings
import pdb

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

import sys
sys.path.append('..')

from models.bayesian_lenet5 import bayesian_lenet5, lenet5
from datasets.mnist import MNISTSequence
from utils.plot_helpers import plot_weight_posteriors, plot_heldout_prediction

tf.enable_v2_behavior()
tfd = tfp.distributions

warnings.simplefilter(action='ignore')

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# pdb.set_trace()

# Parameters for MNIST
IMAGE_SHAPE = [28, 28, 1]
NUM_TRAIN_EXAMPLES = 60000
NUM_VALIDATION_EXAMPLES = 10000
NUM_CLASSES = 10

# Python parameters
flags.DEFINE_float('learning_rate',
                   default=0.001,
                   help='Initial learning rate.')
flags.DEFINE_integer('num_epochs',
                     default=10,
                     help='Number of training steps to run.')
flags.DEFINE_integer('batch_size',
                     default=128,
                     help='Batch size.')
flags.DEFINE_string('data_dir',
                    default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                                         'bayesian_neural_network/data'),
                    help='Directory where data is stored (if using real data).')
flags.DEFINE_string(
    'model_dir',
    default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                         'bayesian_neural_network/'),
    help="Directory to put the model's fit.")
flags.DEFINE_integer('viz_steps',
                     default=400,
                     help='Frequency at which save visualizations.')
flags.DEFINE_integer('num_monte_carlo',
                     default=50,
                     help='Network draws to compute predictive probabilities.')
flags.DEFINE_bool('fake_data',
                  default=False,
                  help='If true, uses fake data. Defaults to real data.')

FLAGS = flags.FLAGS


def create_bayesian_lenet5():
    kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) 
        / tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32)) 
    bayesian_lenet5_model = bayesian_lenet5(NUM_CLASSES, kl_divergence_function)
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    bayesian_lenet5_model.compile(optimizer, loss='categorical_crossentropy',
        metrics=['accuracy'], experimental_run_tf_function=False)
    
    return bayesian_lenet5_model


def main(argv):
    del argv
    if tf.io.gfile.exists(FLAGS.model_dir):
        tf.io.gfile.rmtree(FLAGS.model_dir)
    tf.io.gfile.makedirs(FLAGS.model_dir)

    if FLAGS.fake_data:
        train_seq = MNISTSequence(batch_size=FLAGS.batch_size,
                                  fake_data_size=NUM_TRAIN_EXAMPLES)
        validation_seq = MNISTSequence(batch_size=FLAGS.batch_size,
                                       fake_data_size=NUM_VALIDATION_EXAMPLES)
    else:
        train_set, validation_set = tf.keras.datasets.mnist.load_data(
            os.path.join(FLAGS.data_dir, 'mnist.npz'))
        train_seq = MNISTSequence(data=train_set, batch_size=FLAGS.batch_size)
        validation_seq = MNISTSequence(data=validation_set, batch_size=FLAGS.batch_size)
        validation = MNISTSequence(data=validation_set, batch_size=NUM_VALIDATION_EXAMPLES)
    
    bayesian_lenet5_model = create_bayesian_lenet5()

    bayesian_lenet5_model.build(input_shape=[None, 28, 28, 1])

    # Training and validation process
    train_accuracy, train_loss = [], []
    validation_accuracy, validation_loss = [], []
    print(' ... Training bayesian lenet-5 ...')
    for epoch in range(FLAGS.num_epochs):

        # Validation 
        for validation_x, validation_y in validation:
            validation_batch_loss, validation_batch_accuracy = bayesian_lenet5_model.evaluate(
                x=validation_x, y=validation_y, batch_size=NUM_VALIDATION_EXAMPLES)
            
            validation_loss.append(validation_batch_accuracy)
            validation_accuracy.append(validation_batch_loss)
        
        # Training
        for step, (batch_x, batch_y) in enumerate(train_seq):
            train_batch_loss, train_batch_accuracy = bayesian_lenet5_model.train_on_batch(
                batch_x, batch_y)   
            train_accuracy.append(train_batch_accuracy)
            train_loss.append(train_batch_loss)

            if step % 10 == 0:
                print('Epoch: {}, Batch index: {}, Loss: {:.3f}, Accuracy: {:.3f}'.format(
                  epoch, step, tf.reduce_mean(train_loss), tf.reduce_mean(train_accuracy)))
            
            if (step+1) % FLAGS.viz_steps == 0:
                # Compute log prob of heldout set by averaging draws from the model:
                # p(heldout | train) = int_model p(heldout|model) p(model|train)
                #                   ~= 1/n * sum_{i=1}^n p(heldout | model_i)
                # where model_i is a draw from the posterior p(model|train).
                print(' ... Running monte carlo inference')
                probs = tf.stack([bayesian_lenet5_model.predict(validation_seq, verbose=1)
                                    for _ in range(FLAGS.num_monte_carlo)], axis=0)
                mean_probs = tf.reduce_mean(probs, axis=0)
                heldout_log_prob = tf.reduce_mean(tf.math.log(mean_probs))
                print(' ... Held-out nats: {:.3f}'.format(heldout_log_prob))

                if HAS_SEABORN:
                    names = [layer.name for layer in bayesian_lenet5_model.layers
                        if 'flipout' in layer.name]
                    qm_vals = [layer.kernel_posterior.mean()
                                for layer in bayesian_lenet5_model.layers
                                if 'flipout' in layer.name]
                    qs_vals = [layer.kernel_posterior.stddev()
                                for layer in bayesian_lenet5_model.layers
                                if 'flipout' in layer.name]
                    plot_weight_posteriors(names, qm_vals, qs_vals,
                                            fname=os.path.join(
                                                FLAGS.model_dir,
                                                'epoch{}_step{:05d}_weights.png'.format(
                                                    epoch, step)))
                    plot_heldout_prediction(validation_seq.images, probs,
                                            fname=os.path.join(
                                                FLAGS.model_dir,
                                                'epoch{}_step{}_pred.png'.format(
                                                    epoch, step)),
                                            title='mean heldout logprob {:.2f}'
                                            .format(heldout_log_prob))
    folder = '../results/bnn_results/' 
    prefix = 'lr_' + str(FLAGS.learning_rate)
    save_list_as_npz(train_accuracy, folder+prefix+'train_accuracy.npy')
    save_list_as_npz(train_loss, folder+prefix+'train_loss.npy')
    save_list_as_npz(test_accuracy, folder+prefix+'test_accuracy.npy')
    save_list_as_npz(test_loss, folder+prefix+'test_loss.npy')


if __name__ == '__main__':
    app.run(main)
