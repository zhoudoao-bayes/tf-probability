# zhoudoao@foxmail.com
# 2020.5.12
"""Trains a Bayesian cnn to classify MNIST/CIFAR10/CIFAR100 datasets.

The architecture is bayesian vgg and bayesian resnet.
"""


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

import sys
sys.path.append('..')

from models.bayesian_lenet import bayesian_lenet
from datasets.mnist import MNISTSequence

tf.enable_v2_behavior()

# warnings.simplefilter(action='ignore')

import pdb; pdb.set_trace()

tfd = tfp.distributions

try:
  import seaborn as sns  # pylint: disable=g-import-not-at-top
  HAS_SEABORN = True
except ImportError:
  HAS_SEABORN = False

DATASETS = 'MNIST'

if DATASETS == 'MNIST':
    IMAGE_SHAPE = [28, 28, 1]
    NUM_TRAIN_EXAMPLES = 60000
    NUM_HELDOUT_EXAMPLES = 10000
    NUM_CLASSES = 10
elif DATASETS == "CIFAR10":
    pass 
elif DATASETS == "CIFAR100":
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
                                         'tf2_bayes_cnn/data'),
                    help='Directory where data is stored (if using real data).')
flags.DEFINE_string(
    'model_dir',
    default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                         'tf2_bayes_cnn/'),
    help="Directory to put the model's fit.")
flags.DEFINE_integer('viz_steps',
                     default=40,
                     help='Frequency at which save visualizations.')
flags.DEFINE_integer('num_monte_carlo',
                     default=5,
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


def create_model() :
    # kl_divergence_function = (lambda q, p, _: tfd.kl_divergence_function(q, p) /
    #     tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32)) 
    
    model = bayesian_lenet(NUM_CLASSES, NUM_TRAIN_EXAMPLES)

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)

    model.compile(optimizer, loss='categorical_crossentropy',
        metrics=['accuracy'], experimental_run_tf_function=False)
    
    return model 



def main(argv):
  del argv
  if tf.io.gfile.exists(FLAGS.model_dir):
    # tf.compat.v1.logging.warnings('Warning: deleting old log directory at {}'.format(FLAGS.model_dir))
    tf.io.gfile.rmtree(FLAGS.model_dir)
  tf.io.gfile.makedirs(FLAGS.model_dir)

  if FLAGS.fake_data:
    train_seq = MNISTSequence(batch_size=FLAGS.batch_size,
                              fake_data_size=NUM_TRAIN_EXAMPLES)
    heldout_seq = MNISTSequence(batch_size=FLAGS.batch_size,
                                fake_data_size=NUM_HELDOUT_EXAMPLES)
  else:
    print(os.path.join(FLAGS.data_dir, 'mnist.npz'))
    train_set, heldout_set = tf.keras.datasets.mnist.load_data(os.path.join(FLAGS.data_dir,
        'mnist.npz'))
    train_seq = MNISTSequence(data=train_set, batch_size=FLAGS.batch_size)
    heldout_seq = MNISTSequence(data=heldout_set, batch_size=FLAGS.batch_size)
  

  model = create_model()

  model.build(input_shape=[None, 28, 28, 1])

  print('... Training convolutional neutral network')

  import pdb; pdb.set_trace()

  for epoch in range(FLAGS.num_epochs):
    epoch_accuracy, epoch_loss = [], []
    for step, (batch_x, batch_y) in enumerate(train_seq):
      batch_loss, batch_accuracy = model.train_on_batch(batch_x, batch_y)
      epoch_accuracy.append(batch_accuracy)
      epoch_loss.append(batch_loss)

      if step % 100 == 0:
        print('Epoch: {}, Batch index: {}, '
              'Loss: {:.3f}, Accuracy: {:.3f}'.format(
                  epoch, step,
                  tf.reduce_mean(epoch_loss),
                  tf.reduce_mean(epoch_accuracy)))
      
      if (step+1) % FLAGS.viz_steps == 0:
        # Compute log prob of heldout set by averaging draws from the model:
        # p(heldout | train) = int_model p(heldout|model) p(model|train)
        #                   ~= 1/n * sum_{i=1}^n p(heldout | model_i)
        # where model_i is a draw from the posterior p(model|train).
        print(' ... Running monte carlo inference')
        probs = tf.stack([model.predict(heldout_seq, verbose=1)
                          for _ in range(FLAGS.num_monte_carlo)], axis=0)
        mean_probs = tf.reduce_mean(probs, axis=0)
        heldout_log_prob = tf.reduce_mean(tf.math.log(mean_probs))
        print(' ... Held-out nats: {:.3f}'.format(heldout_log_prob))

        if HAS_SEABORN:
          names = [layer.name for layer in model.layers
                   if 'flipout' in layer.name]
          qm_vals = [layer.kernel_posterior.mean()
                     for layer in model.layers
                     if 'flipout' in layer.name]
          qs_vals = [layer.kernel_posterior.stddev()
                     for layer in model.layers
                     if 'flipout' in layer.name]
          plot_weight_posteriors(names, qm_vals, qs_vals,
                                 fname=os.path.join(
                                     FLAGS.model_dir,
                                     'epoch{}_step{:05d}_weights.png'.format(
                                         epoch, step)))
          plot_heldout_prediction(heldout_seq.images, probs,
                                  fname=os.path.join(
                                      FLAGS.model_dir,
                                      'epoch{}_step{}_pred.png'.format(
                                          epoch, step)),
                                  title='mean heldout logprob {:.2f}'
                                  .format(heldout_log_prob))

if __name__ == '__main__':
  app.run(main)
      












