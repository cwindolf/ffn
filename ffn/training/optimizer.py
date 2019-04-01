# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities to configure TF optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from absl import flags


FLAGS = flags.FLAGS

flags.DEFINE_string('optimizer', 'sgd',
                    'Which optimizer to use. Valid values are: '
                    'momentum, sgd, adagrad, adam, rmsprop.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('momentum', 0.9, 'Momentum.')
flags.DEFINE_float('learning_rate_decay_factor', 0.94,
                   'Learning rate decay factor.')
flags.DEFINE_float('num_epochs_per_decay', 2.0,
                   'Number of epochs after which learning rate decays.')
flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')
flags.DEFINE_float('adam_beta1', 0.9, 'Gradient decay term for Adam.')
flags.DEFINE_float('adam_beta2', 0.999, 'Gradient^2 decay term for Adam.')
flags.DEFINE_float('epsilon', 1e-8, 'Epsilon term for RMSProp and Adam.')
flags.DEFINE_bool('synchronous', False, 'Wrap opt in SyncReplicasOptimizer')
flags.DEFINE_integer('sync_n_replicas', 0,
                     'If --synchronous, put the total number of workers here.')
flags.DEFINE_enum('scale_lr', 'noscale', ['noscale', 'sqrt', 'linear'],
                  'If --synchronous, scale the learning rate by this fn of --sync_n_replicas')


def optimizer_from_flags():
  lr = FLAGS.learning_rate

  if FLAGS.synchronous:
    if FLAGS.scale_lr == 'sqrt':
      lr *= math.sqrt(FLAGS.sync_n_replicas)
    elif FLAGS.scale_lr == 'linear':
      lr *= FLAGS.sync_n_replicas

  if FLAGS.optimizer == 'momentum':
    opt = tf.train.MomentumOptimizer(lr, FLAGS.momentum)
  elif FLAGS.optimizer == 'sgd':
    opt = tf.train.GradientDescentOptimizer(lr)
  elif FLAGS.optimizer == 'adagrad':
    opt = tf.train.AdagradOptimizer(lr)
  elif FLAGS.optimizer == 'adam':
    opt = tf.train.AdamOptimizer(learning_rate=lr,
                                  beta1=FLAGS.adam_beta1,
                                  beta2=FLAGS.adam_beta2,
                                  epsilon=FLAGS.epsilon)
  elif FLAGS.optimizer == 'rmsprop':
    opt = tf.train.RMSPropOptimizer(lr, FLAGS.rmsprop_decay,
                                     momentum=FLAGS.momentum,
                                     epsilon=FLAGS.epsilon)
  else:
    raise ValueError('Unknown optimizer: %s' % FLAGS.optimizer)

  if FLAGS.synchronous:
    opt = tf.train.SyncReplicasOptimizer(opt, FLAGS.sync_n_replicas, FLAGS.sync_n_replicas)

  return opt
