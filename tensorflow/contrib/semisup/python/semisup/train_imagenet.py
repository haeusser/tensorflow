"""
Copyright 2016 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Association-based semi-supervised training module.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import numpy as np
import os
import tensorflow as tf
import sys


import tensorflow.contrib.semisup as semisup
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import inception_v3

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

sys.path.insert(0, '/usr/wiss/haeusser/libs/tfmodels/inception')
from inception.imagenet_data import ImagenetData
from inception import image_processing

FLAGS = flags.FLAGS


flags.DEFINE_string('dataset', 'svhn', 'Which dataset to work on.')

flags.DEFINE_string('architecture', 'svhn', 'Which dataset to work on.')

flags.DEFINE_integer(
    'sup_per_class', 100,
    'Number of labeled samples used per class in total. -1 = all')

flags.DEFINE_integer('unsup_samples', -1,
                     'Number of unlabeled samples used in total. -1 = all.')

flags.DEFINE_integer('sup_seed', -1,
                     'Integer random seed used for labeled set selection.')

flags.DEFINE_integer('sup_batch_size', 740,
                     'Number of labeled samples per batch.')

flags.DEFINE_integer('unsup_batch_size', 740,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')

flags.DEFINE_float('minimum_learning_rate', 1e-6,
                   'Lower bound for learning rate.')

flags.DEFINE_float('decay_factor', 0.33, 'Learning rate decay factor.')

flags.DEFINE_float('decay_steps', 60000,
                   'Learning rate decay interval in steps.')

flags.DEFINE_float('visit_weight', 0.0, 'Weight for visit loss.')

flags.DEFINE_float('visit_weight_sigmoid', False,
                   'Increase visit weight with a sigmoid envelope.')

flags.DEFINE_float('logit_weight', 1.0, 'Weight for logit loss.')

flags.DEFINE_integer('max_steps', 100000, 'Number of training steps.')

flags.DEFINE_bool('augmentation', False,
                  'Apply data augmentation during training.')

flags.DEFINE_integer('new_size', 128,
                     'If > 0, resize image to this width/height.')

flags.DEFINE_integer('virtual_embeddings', 0,
                     'How many virtual embeddings to add.')

flags.DEFINE_string('logdir', '/tmp/semisup', 'Training log path.')

flags.DEFINE_integer('save_summaries_secs', 150,
                     'How often should summaries be saved (in seconds).')

flags.DEFINE_integer('save_interval_secs', 300,
                     'How often should checkpoints be saved (in seconds).')

flags.DEFINE_integer('log_every_n_steps', 100,
                     'Logging interval for slim training loop.')

flags.DEFINE_float(
    'batch_norm_decay', 0.99,
    'Batch norm decay factor (only used for STL-10 at the moment.')

flags.DEFINE_integer('remove_classes', 0,
                     'Remove this number of classes from the labeled set, '
                     'starting with highest label number.')

flags.DEFINE_string('master', '',
                    'BNS name of the TensorFlow master to use.')

flags.DEFINE_integer(
    'ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')

# TODO(haeusser) convert to argparse as gflags will be discontinued
#flags.DEFINE_multi_float('custom_lr_vals', None,
#                         'For custom lr schedule: lr values.')

#flags.DEFINE_multi_int('custom_lr_steps', None,
#                       'For custom lr schedule: step values.')

FLAGS.custom_lr_vals = None
FLAGS.custom_lr_steps = None


def piecewise_constant(x, boundaries, values, name=None):
  """This is tf.train.piecewise_constant.

  Due to some bug, it is inaccessible.
  Remove this when the issue is resolved.

  Piecewise constant from boundaries and interval values.

  Example: use a learning rate that's 1.0 for the first 100000 steps, 0.5
    for steps 100001 to 110000, and 0.1 for any additional steps.

  ```python
  global_step = tf.Variable(0, trainable=False)
  boundaries = [100000, 110000]
  values = [1.0, 0.5, 0.1]
  learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

  # Later, whenever we perform an optimization step, we increment global_step.
  ```

  Args:
    x: A 0-D scalar `Tensor`. Must be one of the following types: `float32`,
      `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`.
    boundaries: A list of `Tensor`s or `int`s or `float`s with strictly
      increasing entries, and with all elements having the same type as `x`.
    values: A list of `Tensor`s or float`s or `int`s that specifies the values
      for the intervals defined by `boundaries`. It should have one more element
      than `boundaries`, and all elements should have the same type.
    name: A string. Optional name of the operation. Defaults to
      'PiecewiseConstant'.

  Returns:
    A 0-D Tensor. Its value is `values[0]` when `x <= boundaries[0]`,
    `values[1]` when `x > boundaries[0]` and `x <= boundaries[1]`, ...,
    and values[-1] when `x > boundaries[-1]`.

  Raises:
    ValueError: if types of `x` and `buondaries` do not match, or types of all
        `values` do not match.
  """
  with tf.name_scope(name, 'PiecewiseConstant',
                     [x, boundaries, values, name]) as name:
    x = tf.convert_to_tensor(x)
    # Avoid explicit conversion to x's dtype. This could result in faulty
    # comparisons, for example if floats are converted to integers.
    boundaries = [tf.convert_to_tensor(b) for b in boundaries]
    for b in boundaries:
      if b.dtype != x.dtype:
        raise ValueError('Boundaries (%s) must have the same dtype as x (%s).' %
                         (b.dtype, x.dtype))
    values = [tf.convert_to_tensor(v) for v in values]
    for v in values[1:]:
      if v.dtype != values[0].dtype:
        raise ValueError(
            'Values must have elements all with the same dtype (%s vs %s).' %
            (values[0].dtype, v.dtype))

    pred_fn_pairs = {}
    pred_fn_pairs[x <= boundaries[0]] = lambda: values[0]
    pred_fn_pairs[x > boundaries[-1]] = lambda: values[-1]
    for low, high, v in zip(boundaries[:-1], boundaries[1:], values[1:-1]):
      # Need to bind v here; can do this with lambda v=v: ...
      pred = (x > low) & (x <= high)
      pred_fn_pairs[pred] = lambda v=v: v

    # The default isn't needed here because our conditions are mutually
    # exclusive and exhaustive, but tf.case requires it.
    default = lambda: values[0]
    return tf.case(pred_fn_pairs, default, exclusive=True)


def inception_model(inputs,
                    emb_size=128,
                    is_training=True):
    _, end_points = inception_v3.inception_v3(inputs, is_training=is_training, reuse=True)
    net = end_points['Mixed_7c']
    net = slim.flatten(net, scope='flatten')
    with slim.arg_scope([slim.fully_connected], normalizer_fn=None):
        emb = slim.fully_connected(net, emb_size, scope='fc')
    return emb


def main(_):
  # Load dataset
  tf.app.flags.FLAGS.data_dir ='/work/haeusser/data/imagenet/shards'
  dataset = ImagenetData(subset='train')
  assert dataset.data_files()

  num_labels = dataset.num_classes() + 1
  image_shape = [FLAGS.image_size, FLAGS.image_size, 3]
  visit_weight = FLAGS.visit_weight
  logit_weight = FLAGS.logit_weight




  graph = tf.Graph()
  with graph.as_default():
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks, merge_devices=True)):

      images, labels = image_processing.batch_inputs(
          dataset, 32, train=True,
          num_preprocess_threads=16,
          num_readers=FLAGS.num_readers)


      t_sup_images, t_sup_labels = tf.train.batch(
                                                  [images, labels],
                                                  batch_size=FLAGS.sup_batch_size,
                                                  enqueue_many=True,
                                                  num_threads=16,
                                                  capacity=1000 + 3 * FLAGS.sup_batch_size,
                                                  )

      t_unsup_images, _ = tf.train.batch(
                                                  [images, labels],
                                                  batch_size=FLAGS.sup_batch_size,
                                                  enqueue_many=True,
                                                  num_threads=16,
                                                  capacity=1000 + 3 * FLAGS.sup_batch_size,
                                                  )


      # Apply augmentation
      if FLAGS.augmentation:
        if hasattr(tools, 'augmentation_params'):
          augmentation_function = partial(
              apply_augmentation, params=tools.augmentation_params)
        else:
          augmentation_function = apply_affine_augmentation
      else:
        augmentation_function = None


      # Set up semisup model.
      model = semisup.SemisupModel(inception_model, num_labels, image_shape)

      # Compute embeddings and logits.
      t_sup_emb = model.image_to_embedding(t_sup_images)
      t_unsup_emb = model.image_to_embedding(t_unsup_images)

      # Add virtual embeddings.
      if FLAGS.virtual_embeddings:
        t_sup_emb = tf.concat(0, [
            t_sup_emb, semisup.create_virt_emb(FLAGS.virtual_embeddings, 128)
        ])

        if not FLAGS.remove_classes:
          # need to add additional labels for virtual embeddings
          t_sup_labels = tf.concat(0, [
              t_sup_labels,
              (num_labels + tf.range(1, FLAGS.virtual_embeddings + 1, tf.int64))
              * tf.ones([FLAGS.virtual_embeddings], tf.int64)
          ])

      t_sup_logit = model.embedding_to_logit(t_sup_emb)

      # Add losses.
      if FLAGS.visit_weight_sigmoid:
        visit_weight = logistic_growth(model.step, FLAGS.visit_weight,
                                       FLAGS.max_steps)
      else:
        visit_weight = FLAGS.visit_weight
      tf.summary.scalar('VisitLossWeight', visit_weight)

      if FLAGS.unsup_samples != 0:
        model.add_semisup_loss(
            t_sup_emb, t_unsup_emb, t_sup_labels, visit_weight=visit_weight)
      model.add_logit_loss(t_sup_logit, t_sup_labels, weight=logit_weight)

      # Set up learning rate schedule if necessary.
      if FLAGS.custom_lr_vals is not None and FLAGS.custom_lr_steps is not None:
        boundaries = [
            tf.convert_to_tensor(x, tf.int64) for x in FLAGS.custom_lr_steps
        ]

        t_learning_rate = piecewise_constant(model.step, boundaries,
                                             FLAGS.custom_lr_vals)
      else:
        t_learning_rate = tf.maximum(
            tf.train.exponential_decay(
                FLAGS.learning_rate,
                model.step,
                FLAGS.decay_steps,
                FLAGS.decay_factor,
                staircase=True),
            FLAGS.minimum_learning_rate)

      # Create training operation and start the actual training loop.
      train_op = model.create_train_op(t_learning_rate)

      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True

      slim.learning.train(
          train_op,
          logdir=FLAGS.logdir,
          save_summaries_secs=FLAGS.save_summaries_secs,
          save_interval_secs=FLAGS.save_interval_secs,
          master=FLAGS.master,
          is_chief=(FLAGS.task == 0),
          startup_delay_steps=(FLAGS.task * 20),
          log_every_n_steps=FLAGS.log_every_n_steps,
          session_config=config)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run()
