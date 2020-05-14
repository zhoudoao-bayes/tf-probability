# zhoudoao@foxmail.com
# 2020.5.14
"""Trains a distilled cnn for a bnn.

The arhitecutres are lenet and bayesian lenet
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl import app
from absl import flags
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import figure  # pylint: disable=g-import-not-at-top
from matplotlib.backends import backend_agg
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

import sys
sys.path.append('..')

from models.bayesian_lenet import bayesian_lenet, lenet
from datasets.mnist import MNISTSequence

tf.enable_v2_behavior()

warnings.simplefilter(action='ignore')

tfd = tfp.distribution

try:
  import seaborn as sns  # pylint: disable=g-import-not-at-top
  HAS_SEABORN = True
except ImportError:
  HAS_SEABORN = False


if DATASETS == 'MNIST':
    IMAGE_SHAPE = [28, 28, 1]
    NUM_TRAIN_EXAMPLES = 60000
    NUM_HELDOUT_EXAMPLES = 10000
    NUM_CLASSES = 10
elif DATASETS = "CIFAR10":
    pass 
elif DATASETS = "CIFAR100":
    pass 


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


def plot_weight_posteriors(names, qm_vals, qs_vals, fname):
  """Save a PNG plot with histograms of weight means and stddevs.

  Args:
    names: A Python `iterable` of `str` variable names.
      qm_vals: A Python `iterable`, the same length as `names`,
      whose elements are Numpy `array`s, of any shape, containing
      posterior means of weight varibles.
    qs_vals: A Python `iterable`, the same length as `names`,
      whose elements are Numpy `array`s, of any shape, containing
      posterior standard deviations of weight varibles.
    fname: Python `str` filename to save the plot to.
  """
  fig = figure.Figure(figsize=(6, 3))
  canvas = backend_agg.FigureCanvasAgg(fig)

  ax = fig.add_subplot(1, 2, 1)
  for n, qm in zip(names, qm_vals):
    sns.distplot(tf.reshape(qm, shape=[-1]), ax=ax, label=n)
  ax.set_title('weight means')
  ax.set_xlim([-1.5, 1.5])
  ax.legend()

  ax = fig.add_subplot(1, 2, 2)
  for n, qs in zip(names, qs_vals):
    sns.distplot(tf.reshape(qs, shape=[-1]), ax=ax)
  ax.set_title('weight stddevs')
  ax.set_xlim([0, 1.])

  fig.tight_layout()
  canvas.print_figure(fname, format='png')
  print('saved {}'.format(fname))


def plot_heldout_prediction(input_vals, probs,
                            fname, n=10, title=''):
  """Save a PNG plot visualizing posterior uncertainty on heldout data.

  Args:
    input_vals: A `float`-like Numpy `array` of shape
      `[num_heldout] + IMAGE_SHAPE`, containing heldout input images.
    probs: A `float`-like Numpy array of shape `[num_monte_carlo,
      num_heldout, num_classes]` containing Monte Carlo samples of
      class probabilities for each heldout sample.
    fname: Python `str` filename to save the plot to.
    n: Python `int` number of datapoints to vizualize.
    title: Python `str` title for the plot.
  """
  fig = figure.Figure(figsize=(9, 3*n))
  canvas = backend_agg.FigureCanvasAgg(fig)
  for i in range(n):
    ax = fig.add_subplot(n, 3, 3*i + 1)
    ax.imshow(input_vals[i, :].reshape(IMAGE_SHAPE[:-1]), interpolation='None')

    ax = fig.add_subplot(n, 3, 3*i + 2)
    for prob_sample in probs:
      sns.barplot(np.arange(10), prob_sample[i, :], alpha=0.1, ax=ax)
      ax.set_ylim([0, 1])
    ax.set_title('posterior samples')

    ax = fig.add_subplot(n, 3, 3*i + 3)
    sns.barplot(np.arange(10), tf.reduce_mean(probs[:, i, :], axis=0), ax=ax)
    ax.set_ylim([0, 1])
    ax.set_title('predictive probs')
  fig.suptitle(title)
  fig.tight_layout()

  canvas.print_figure(fname, format='png')
  print('saved {}'.format(fname))


def create_bayesian_lenet_model():
    kl_divergence_function = (lambda q, p, _: tfd.kl_divergence_function(q, p) /
        tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))
    
    model = bayesian_lenet()
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)

    model.compile(optimizer, loss='categorical_crossentropy',
       metrics=['accuracy'], experimental_run_tf_function=False)
    
    return model


def create_lenet_model():
    model = lenet()
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    model.compile(optimizer, loss='categorical_crossentropy',
        metrics=['accuracy'], experimental_run_tf_function=False)
    
    return model


def main(argv):
    del argv
    if tf.io.gfile.exists(FLAGS.model_dir):
        tf.compat.v1.logging.warnings('Warning: deleting old log directory at {}'.format(FLAGS.model_dir))
        tf.io.gfile.rmtree(FLAGS.model_dir)
    tf.io.gfile.makedirs(FLAGS.model_dir)

    if FLAGS.fake_data:
        train_seq = MNISTSequence(batch_size=FLAGS.batch_size,
                                fake_data_size=NUM_TRAIN_EXAMPLES)
        heldout_seq = MNISTSequence(batch_size=FLAGS.batch_size,
                                    fake_data_size=NUM_HELDOUT_EXAMPLES)
    else:
        train_set, heldout_set = tf.keras.datasets.mnist.load_data()
        train_seq = MNISTSequence(data=train_set, batch_size=FLAGS.batch_size)
        heldout_seq = MNISTSequence(data=heldout_set, batch_size=FLAGS.batch_size)

    bayesian_lenet_model = create_bayesian_lenet_model()
    bayesian_lenet_model.build(input_shape=[None, 28, 28, 1])

    lenet_model = create_lenet_model()
    lenet_model.build(input_shape=[None, 28, 28, 1])

    print('... Training convolutional neutral network')

    for epoch in range(FLAGS.num_epochs):
        epoch_accuracy_bayesian, epoch_loss_bayesian = [], []
        epoch_accuracy_lenet, epoch_loss_lenet = [], []

        # Training data for bayesian lenet
        for step, (batch_x, batch_y) in enumerate(train_seq):
            start_time = time.time()
            batch_loss_bayesian, batch_accuracy_bayesian = bayesian_lenet_model.train_on_batch(batch_x, batch_y)
            epoch_accuracy_bayesian.append(batch_accuracy)
            epoch_loss_bayesian.append(batch_loss)
            end_time_bayesian = time.time()
            cost_time_bayesian = end_time_bayesian - start_time
            
            # Training data for lenet
            batch_x_perturbation = batch_x + noise
            batch_y_pred = model.predict(batch_x_perturbation)

            batch_loss_lenet, batch_accuracy_lenet = 
                lenet_model.train_on_batch(batch_x_perturbation, batch_y_pred)
            epoch_accuracy_lenet.append(batch_accuracy_lenet)
            epoch_loss_lenet.append(batch_loss_bayesian)
            end_time_lenet = time.time()
            cost_time_lenet = end_time_lenet - end_time_bayesian

            if step % 100 == 0:
                print('Epoch {}, Batch index: {}, '
                    'Bayesian Loss: {:.3f}, Accuracy: {:.3f}, Cost time:'.format(
                      epoch, step, tf.reduce_mean(epoch_loss_bayesian),
                      tf.reduce_mean(epoch_accuracy_bayesian),
                      cost_time_bayesian
                ))
                print('Distilled Loss: {:.3f}, Accuracy: {:.3f}, Cost time:'.format(
                    tf.reduce_mean(epoch_accuracy_lenet),
                    tf.reduce_mean(epoch_loss_bayesian),
                    cost_time_lenet
                ))
            
            if (step+1) % FLAGS.viz_steps == 0:
                print('... Running monte carlo inference')
                probs_bayesian = tf.stack([model.predict(heldout_seq, verbose=1)
                    for _ in range(FLAGS.num_monte_carlo)], axis=0)
                mean_probs_bayesian = tf.reduce_mean(probs_bayesian, axis=0)
                heldout_log_prob_bayesian = tf.reduce_mean(tf.math.log(mean_probs_bayesian))
                print('... Heldout nats: {:.3f}'.format(heldout_log_prob_bayesian))
    
    a = 1


if __name__ == '__main__':
  app.run(main)
                

              
              

            
            







        



