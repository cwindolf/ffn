# Copyright 2017-2018 Google Inc.
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
"""Driver script for FFN training.

The FFN is first run on single seed point. The mask prediction for that seed
point is then used to train subsequent steps of the FFN by moving the field
of view in a way dependent on the initial predictions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
from io import BytesIO
from functools import partial
import itertools
import json
import logging
import os
import random
import time

import h5py
import numpy as np

import PIL
import PIL.Image

import six

from scipy.special import expit
from scipy.special import logit
from sklearn.metrics import adjusted_rand_score
import tensorflow as tf

from absl import app
from absl import flags
from tensorflow import gfile

from ffn.inference import movement
from ffn.training import mask
from ffn.training.import_util import import_symbol
from ffn.training import inputs
from ffn.training import augmentation
# Necessary so that optimizer and training flags are defined.
# pylint: disable=unused-import
from ffn.training import optimizer
from ffn.training import training_flags
# pylint: enable=unused-import


# Data parallel training options.
# See also some of the training infra options above.
flags.DEFINE_integer('task', 0, 'Task id of the replica running the training.')
# flags.DEFINE_string('master', '', 'Network address of the master.')
flags.DEFINE_integer('ps_tasks', 0, 'Number of tasks in the ps job.')
flags.DEFINE_list('ps_hosts', '',
                  'Parameter servers. Comma-separated list of '
                  '<hostname>:<port> pairs.')
flags.DEFINE_list('worker_hosts', '',
                  'Worker servers. Comma-separated list of '
                  '<hostname>:<port> pairs.')
flags.DEFINE_enum('job_name', 'worker', ['ps', 'worker'],
                  'Am I a parameter server or a worker?')

FLAGS = flags.FLAGS


class EvalTracker(object):
  """Tracks eval results over multiple training steps."""

  def __init__(self, eval_shape, prefix=None):
    self.eval_labels = tf.placeholder(
        tf.float32, [1] + eval_shape + [1], name='eval_labels')
    self.eval_preds = tf.placeholder(
        tf.float32, [1] + eval_shape + [1], name='eval_preds')
    self.eval_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.eval_preds, labels=self.eval_labels))
    self.reset()
    self.eval_threshold = logit(0.9)
    self.sess = None
    self._eval_shape = eval_shape
    self.prefix = ''
    if prefix:
      self.prefix = prefix.rstrip('/') + '/'

  def reset(self):
    """Resets status of the tracker."""
    self.loss = 0
    self.num_patches = 0
    self.tp = 0
    self.tn = 0
    self.fn = 0
    self.fp = 0
    self.total_voxels = 0
    self.masked_voxels = 0
    self.images_xy = deque(maxlen=16)
    self.images_xz = deque(maxlen=16)
    self.images_yz = deque(maxlen=16)
    self.adj_rand_score = 0

  def slice_image(self, labels, predicted, weights, slice_axis):
    """Builds a tf.Summary showing a slice of an object mask.

    The object mask slice is shown side by side with the corresponding
    ground truth mask.

    Args:
      labels: ndarray of ground truth data, shape [1, z, y, x, 1]
      predicted: ndarray of predicted data, shape [1, z, y, x, 1]
      weights: ndarray of loss weights, shape [1, z, y, x, 1]
      slice_axis: axis in the middle of which to place the cutting plane
          for which the summary image will be generated, valid values are
          2 ('x'), 1 ('y'), and 0 ('z').

    Returns:
      tf.Summary.Value object with the image.
    """
    zyx = list(labels.shape[1:-1])
    selector = [0, slice(None), slice(None), slice(None), 0]
    selector[slice_axis + 1] = zyx[slice_axis] // 2
    selector = tuple(selector)

    del zyx[slice_axis]
    h, w = zyx

    buf = BytesIO()
    labels = (labels[selector] * 255).astype(np.uint8)
    predicted = (predicted[selector] * 255).astype(np.uint8)
    weights = (weights[selector] * 255).astype(np.uint8)

    im = PIL.Image.fromarray(np.concatenate([labels, predicted,
                                             weights], axis=1), 'L')
    im.save(buf, 'PNG')

    axis_names = 'zyx'
    axis_names = axis_names.replace(axis_names[slice_axis], '')

    return tf.Summary.Value(
        tag=f'{self.prefix}final_{axis_names[::-1]}',
        image=tf.Summary.Image(
            height=h, width=w * 3, colorspace=1,  # greyscale
            encoded_image_string=buf.getvalue()))

  def add_patch(self, labels, predicted, weights,
                coord=None, volname=None, patches=None):
    """Evaluates single-object segmentation quality."""

    predicted = mask.crop_and_pad(predicted, (0, 0, 0), self._eval_shape)
    weights = mask.crop_and_pad(weights, (0, 0, 0), self._eval_shape)
    labels = mask.crop_and_pad(labels, (0, 0, 0), self._eval_shape)
    loss, = self.sess.run([self.eval_loss], {self.eval_labels: labels,
                                             self.eval_preds: predicted})
    self.loss += loss
    self.total_voxels += labels.size
    self.masked_voxels += np.sum(weights == 0.0)

    pred_mask = predicted >= self.eval_threshold
    true_mask = labels > 0.5
    pred_bg = np.logical_not(pred_mask)
    true_bg = np.logical_not(true_mask)

    self.adj_rand_score += adjusted_rand_score(true_mask.ravel(), pred_mask.ravel())

    self.tp += np.sum(pred_mask & true_mask)
    self.fp += np.sum(pred_mask & true_bg)
    self.fn += np.sum(pred_bg & true_mask)
    self.tn += np.sum(pred_bg & true_bg)
    self.num_patches += 1

    predicted = expit(predicted)
    self.images_xy.append(self.slice_image(labels, predicted, weights, 0))
    self.images_xz.append(self.slice_image(labels, predicted, weights, 1))
    self.images_yz.append(self.slice_image(labels, predicted, weights, 2))

  def get_summaries(self):
    """Gathers tensorflow summaries into single list."""

    if not self.total_voxels:
      return []

    precision = self.tp / max(self.tp + self.fp, 1)
    recall = self.tp / max(self.tp + self.fn, 1)

    for images in self.images_xy, self.images_xz, self.images_yz:
      for i, summary in enumerate(images):
        summary.tag += '/%d' % i

    summaries = (
        list(self.images_xy) + list(self.images_xz) + list(self.images_yz) + [
            tf.Summary.Value(tag=f'{self.prefix}masked_voxel_fraction',
                             simple_value=(self.masked_voxels /
                                           self.total_voxels)),
            tf.Summary.Value(tag=f'{self.prefix}eval/patch_loss',
                             simple_value=self.loss / self.num_patches),
            tf.Summary.Value(tag=f'{self.prefix}eval/patches',
                             simple_value=self.num_patches),
            tf.Summary.Value(tag=f'{self.prefix}eval/accuracy',
                             simple_value=(self.tp + self.tn) / (
                                 self.tp + self.tn + self.fp + self.fn)),
            tf.Summary.Value(tag=f'{self.prefix}eval/precision',
                             simple_value=precision),
            tf.Summary.Value(tag=f'{self.prefix}eval/recall',
                             simple_value=recall),
            tf.Summary.Value(tag=f'{self.prefix}eval/specificity',
                             simple_value=self.tn / max(self.tn + self.fp, 1)),
            tf.Summary.Value(tag=f'{self.prefix}eval/f1',
                             simple_value=(2.0 * precision * recall /
                                           (precision + recall))),
            tf.Summary.Value(tag=f'{self.prefix}eval/adjusted_rand_score',
                             simple_value=self.adj_rand_score / self.num_patches),
        ])

    return summaries


def run_training_step(sess, model, fetch_summary, feed_dict):
  """Runs one training step for a single FFN FOV."""
  ops_to_run = [model.train_op, model.global_step, model.logits]

  if fetch_summary is not None:
    ops_to_run.append(fetch_summary)

  results = sess.run(ops_to_run, feed_dict)
  step, prediction = results[1:3]

  if fetch_summary is not None:
    summ = results[-1]
  else:
    summ = None

  return prediction, step, summ


def run_validation_step(sess, model, feed_dict):
  """Runs one validation step for a single FFN FOV."""
  prediction = sess.run(model.logits, feed_dict)
  return prediction


def fov_moves():
  # Add one more move to get a better fill of the evaluation area.
  if FLAGS.fov_policy == 'max_pred_moves':
    return FLAGS.fov_moves + 1
  return FLAGS.fov_moves


def train_labels_size(model):
  return (np.array(model.pred_mask_size) +
          np.array(model.deltas) * 2 * fov_moves())


def train_eval_size(model):
  return (np.array(model.pred_mask_size) +
          np.array(model.deltas) * 2 * FLAGS.fov_moves)


def train_image_size(model):
  return (np.array(model.input_image_size) +
          np.array(model.deltas) * 2 * fov_moves())


def train_canvas_size(model):
  return (np.array(model.input_seed_size) +
          np.array(model.deltas) * 2 * fov_moves())


def _get_offset_and_scale_map():
  if not FLAGS.image_offset_scale_map:
    return {}

  ret = {}
  for vol_def in FLAGS.image_offset_scale_map:
    vol_name, offset, scale = vol_def.split(':')
    ret[vol_name] = float(offset), float(scale)

  return ret


def _get_reflectable_axes():
  return [int(x) + 1 for x in FLAGS.reflectable_axes]


def _get_permutable_axes():
  return [int(x) + 1 for x in FLAGS.permutable_axes]


def define_data_input(model, queue_batch=None, val=False):
  """Adds TF ops to load input data."""
  # This method handles the creation of data loading ops for
  # training data by default, or validation data if val=True.
  label_volumes = FLAGS.label_volumes
  data_volumes = FLAGS.data_volumes
  train_coords = FLAGS.train_coords
  if val:
    label_volumes = FLAGS.validation_label_volumes
    data_volumes = FLAGS.validation_data_volumes
    train_coords = FLAGS.validation_coords

  label_volume_map = {}
  for vol in label_volumes.split(','):
    print(vol)
    volname, path, dataset = vol.split(':')
    label_volume_map[volname] = h5py.File(path, 'r')[dataset]

  image_volume_map = {}
  for vol in data_volumes.split(','):
    volname, path, dataset = vol.split(':')
    image_volume_map[volname] = h5py.File(path, 'r')[dataset]

  if queue_batch is None:
    queue_batch = FLAGS.batch_size

  # Fetch sizes of images and labels
  label_size = train_labels_size(model)
  image_size = train_image_size(model)

  label_radii = (label_size // 2).tolist()
  label_size = label_size.tolist()
  image_radii = (image_size // 2).tolist()
  image_size = image_size.tolist()

  # Fetch a single coordinate and volume name from a queue reading the
  # coordinate files or from saved hard/important examples
  coord, volname = inputs.load_patch_coordinates(train_coords)

  # Load object labels (segmentation).
  labels = inputs.load_from_numpylike(
      coord, volname, label_size, label_volume_map)

  label_shape = [1] + label_size[::-1] + [1]
  labels = tf.reshape(labels, label_shape)

  loss_weights = tf.constant(np.ones(label_shape, dtype=np.float32))

  # Load image data.
  patch = inputs.load_from_numpylike(
      coord, volname, image_size, image_volume_map)
  data_shape = [1] + image_size[::-1] + [1]
  patch = tf.reshape(patch, shape=data_shape)

  if ((FLAGS.image_stddev is None or FLAGS.image_mean is None) and
      not FLAGS.image_offset_scale_map):
    raise ValueError('--image_mean, --image_stddev or --image_offset_scale_map '
                     'need to be defined')

  # Convert segmentation into a soft object mask.
  lom = tf.logical_and(
      labels > 0,
      tf.equal(labels, labels[0,
                              label_radii[2],
                              label_radii[1],
                              label_radii[0],
                              0]))
  labels = inputs.soften_labels(lom)

  # Apply basic augmentations.
  transform_axes = augmentation.PermuteAndReflect(
      rank=5, permutable_axes=_get_permutable_axes(),
      reflectable_axes=_get_reflectable_axes())
  labels = transform_axes(labels)
  patch = transform_axes(patch)
  loss_weights = transform_axes(loss_weights)

  # Normalize image data.
  patch = inputs.offset_and_scale_patches(
      patch, volname[0],
      offset_scale_map=_get_offset_and_scale_map(),
      default_offset=FLAGS.image_mean,
      default_scale=FLAGS.image_stddev)

  # Create a batch of examples. Note that any TF operation before this line
  # will be hidden behind a queue, so expensive/slow ops can take advantage
  # of multithreading.
  patches, labels, loss_weights = tf.train.shuffle_batch(
      [patch, labels, loss_weights], queue_batch,
      num_threads=max(1, FLAGS.batch_size // 2),
      capacity=32 * FLAGS.batch_size,
      min_after_dequeue=4 * FLAGS.batch_size,
      enqueue_many=True)

  return patches, labels, loss_weights, coord, volname


def prepare_ffn(model):
  """Creates the TF graph for an FFN."""
  shape = [FLAGS.batch_size] + list(model.pred_mask_size[::-1]) + [1]

  model.labels = tf.placeholder(tf.float32, shape, name='labels')
  model.loss_weights = tf.placeholder(tf.float32, shape, name='loss_weights')
  model.define_tf_graph()


def fixed_offsets(model, seed, fov_shifts=None):
  """Generates offsets based on a fixed list."""
  for off in itertools.chain([(0, 0, 0)], fov_shifts):
    if model.dim == 3:
      is_valid_move = seed[:,
                           seed.shape[1] // 2 + off[2],
                           seed.shape[2] // 2 + off[1],
                           seed.shape[3] // 2 + off[0],
                           0] >= logit(FLAGS.threshold)
    else:
      is_valid_move = seed[:,
                           seed.shape[1] // 2 + off[1],
                           seed.shape[2] // 2 + off[0],
                           0] >= logit(FLAGS.threshold)

    if not is_valid_move:
      continue

    yield off


def max_pred_offsets(model, seed):
  """Generates offsets with the policy used for inference."""
  # Always start at the center.
  queue = deque([(0, 0, 0)])
  done = set()

  train_image_radius = train_image_size(model) // 2
  input_image_radius = np.array(model.input_image_size) // 2

  while queue:
    offset = queue.popleft()

    # Drop any offsets that would take us beyond the image fragment we
    # loaded for training.
    if np.any(np.abs(np.array(offset)) + input_image_radius >
              train_image_radius):
      continue

    # Ignore locations that were visited previously.
    quantized_offset = (
        offset[0] // max(model.deltas[0], 1),
        offset[1] // max(model.deltas[1], 1),
        offset[2] // max(model.deltas[2], 1))

    if quantized_offset in done:
      continue

    done.add(quantized_offset)

    yield offset

    # Look for new offsets within the updated seed.
    curr_seed = mask.crop_and_pad(seed, offset, model.pred_mask_size[::-1])
    todos = sorted(
        movement.get_scored_move_offsets(
            model.deltas[::-1],
            curr_seed[0, ..., 0],
            threshold=logit(FLAGS.threshold)), reverse=True)
    queue.extend((x[2] + offset[0],
                  x[1] + offset[1],
                  x[0] + offset[2]) for _, x in todos)


def get_example(load_example, eval_tracker, model, get_offsets):
  """Generates individual training examples.

  Args:
    load_example: callable returning a tuple of image and label ndarrays
                  as well as the seed coordinate and volume name of the example
    eval_tracker: EvalTracker object
    model: FFNModel object
    get_offsets: iterable of (x, y, z) offsets to investigate within the
        training patch

  Yields:
    tuple of:
      seed array, shape [1, z, y, x, 1]
      image array, shape [1, z, y, x, 1]
      label array, shape [1, z, y, x, 1]
  """
  seed_shape = train_canvas_size(model).tolist()[::-1]

  while True:
    full_patches, full_labels, loss_weights, coord, volname = load_example()
    # Always start with a clean seed.
    seed = logit(mask.make_seed(seed_shape, 1, pad=FLAGS.seed_pad))

    for off in get_offsets(model, seed):
      predicted = mask.crop_and_pad(seed, off, model.input_seed_size[::-1])
      patches = mask.crop_and_pad(full_patches, off, model.input_image_size[::-1])
      labels = mask.crop_and_pad(full_labels, off, model.pred_mask_size[::-1])
      weights = mask.crop_and_pad(loss_weights, off, model.pred_mask_size[::-1])

      # Necessary, since the caller is going to update the array and these
      # changes need to be visible in the following iterations.
      assert predicted.base is seed
      yield predicted, patches, labels, weights

    eval_tracker.add_patch(
        full_labels, seed, loss_weights, coord, volname, full_patches)


def get_batch(load_example, eval_tracker, model, batch_size, get_offsets):
  """Generates batches of training examples.

  Args:
    load_example: callable returning a tuple of image and label ndarrays
                  as well as the seed coordinate and volume name of the example
    eval_tracker: EvalTracker object
    model: FFNModel object
    batch_size: desidred batch size
    get_offsets: iterable of (x, y, z) offsets to investigate within the
        training patch

  Yields:
    tuple of:
      seed array, shape [b, z, y, x, 1]
      image array, shape [b, z, y, x, 1]
      label array, shape [b, z, y, x, 1]

    where 'b' is the batch_size.
  """
  def _batch(iterable):
    for batch_vals in iterable:
      # `batch_vals` is sequence of `batch_size` tuples returned by the
      # `get_example` generator, to which we apply the following transformation:
      #   [(a0, b0), (a1, b1), .. (an, bn)] -> [(a0, a1, .., an),
      #                                         (b0, b1, .., bn)]
      # (where n is the batch size) to get a sequence, each element of which
      # represents a batch of values of a given type (e.g., seed, image, etc.)
      yield zip(*batch_vals)

  # Create a separate generator for every element in the batch. This generator
  # will automatically advance to a different training example once the allowed
  # moves for the current location are exhausted.
  for seeds, patches, labels, weights in _batch(six.moves.zip(
      *[get_example(load_example, eval_tracker, model, get_offsets) for _
        in range(batch_size)])):

    batched_seeds = np.concatenate(seeds)

    yield (batched_seeds, np.concatenate(patches), np.concatenate(labels),
           np.concatenate(weights))

    # batched_seed is updated in place with new predictions by the code
    # calling get_batch. Here we distribute these updated predictions back
    # to the buffer of every generator.
    for i in range(batch_size):
      seeds[i][:] = batched_seeds[i, ...]


def eval_batch(load_example, eval_tracker, model, batch_size, get_offsets):
  seed_shape = train_canvas_size(model).tolist()[::-1]
  for _ in range(batch_size):
    full_patches, full_labels, loss_weights, coord, volname = load_example()
    seed = logit(mask.make_seed(seed_shape, 1, pad=FLAGS.seed_pad))
    eval_tracker.add_patch(
        full_labels, seed, loss_weights, coord, volname, full_patches)


def save_flags():
  gfile.MakeDirs(FLAGS.train_dir)
  with gfile.Open(os.path.join(FLAGS.train_dir,
                               'flags.%d' % time.time()), 'w') as f:
    for mod, flag_list in FLAGS.flags_by_module_dict().items():
      if (mod.startswith('google3.research.neuromancer.tensorflow') or
          mod.startswith('/')):
        for flag in flag_list:
          f.write('%s\n' % flag.serialize())


# for done queue reference, see:
# - https://github.com/tensorflow/tensorflow/issues/4713
# - https://gist.github.com/yaroslavvb/ea1b1bae0a75c4aae593df7eca72d9ca
def create_done_queue(i):
  with tf.device("/job:ps/task:%d" % i):
    return tf.FIFOQueue(len(FLAGS.worker_hosts),
      tf.int32, shared_name="done_queue"+str(i))


def create_done_queues():
  return [create_done_queue(i) for i in range(FLAGS.ps_tasks)]


def ps_train_ffn(model_cls, cluster_spec=None, **model_kwargs):
  # Distributed or local training server
  if cluster_spec:
    server = tf.train.Server(cluster_spec,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task)
  else:
    server = tf.train.Server.create_local_server()

  # Parameter servers wait for instructions
  logging.info('Starting parameter server.')
  queue = create_done_queue(FLAGS.task)
  # Lots of times ps don't need a session -- this one does to
  # listen to its done queue
  sess = tf.Session(server.target,
    config=tf.ConfigProto(
      log_device_placement=False,
      allow_soft_placement=True))

  # Wait for quit queue signals from workers
  for i in range(len(FLAGS.worker_hosts)):
    sess.run(queue.dequeue())
    logging.info('PS' + str(FLAGS.task) + ' got quit sig number ' + str(i))

  # Quit
  logging.info('PS' + str(FLAGS.task) + ' exiting.')
  time.sleep(1.0)
  time.sleep(1.0)
  # For some reason the sess.close is hanging, this hard kill is the only
  # way I can find to exit.
  os._exit(0)


def worker_train_ffn(model_cls, cluster_spec=None, **model_kwargs):
  # First worker does some extra work
  is_chief = FLAGS.task == 0

  # Check if we have validation data
  do_val = bool(FLAGS.validation_data_volumes)

  # Distributed or local training server
  if cluster_spec:
    server = tf.train.Server(cluster_spec,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task)
  else:
    server = tf.train.Server.create_local_server()

  # Workers get to work.
  # `job_name=='worker'` by default, so this path is hit when not distributed
  logging.info('Starting worker.')

  with tf.Graph().as_default():
    # This is how we tell the parameter servers we're finished --
    # We'll enqueue something onto a queue for each parameter server
    # Only runs if distributed.
    if cluster_spec:
      done_ops = []
      with tf.device('/job:worker/task:%d' % FLAGS.task):
        done_msg = f'Worker {FLAGS.task} signaling parameter servers'
        done_ops = ([q.enqueue(1) for q in create_done_queues()]
                   + [tf.Print(tf.constant(0), [], message=done_msg)])

    with tf.device(tf.train.replica_device_setter(
        ps_tasks=FLAGS.ps_tasks,
        cluster=cluster_spec,
        merge_devices=True)):
      # The constructor might define TF ops/placeholders, so it is important
      # that the FFN is instantiated within the current context.
      model = model_cls(**model_kwargs)
      eval_shape_zyx = train_eval_size(model).tolist()[::-1]

      eval_tracker = EvalTracker(eval_shape_zyx)
      load_data_ops = define_data_input(model, queue_batch=1)
      prepare_ffn(model)

      if is_chief:
        # Chief writes out the parameters that started the run
        save_flags()

        # Chief does validation
        if do_val:
          val_eval_tracker = EvalTracker(eval_shape_zyx, prefix='validation')
          val_load_data_ops = define_data_input(model, queue_batch=1, val=True)

      merge_summaries_op = tf.summary.merge_all()
      summary_writer = None
      saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.25)
      scaffold = tf.train.Scaffold(saver=saver)

      # In some situations, we will add hooks.
      hooks = []

      if cluster_spec:
        # If distributed, make sure the done queue ops run.
        hooks.append(tf.train.FinalOpsHook(done_ops))

      # Support for synchronous optimizers
      if FLAGS.synchronous:
        print(f'Running with a synchronous optimizer')
        hooks.append(model.opt.make_session_run_hook(is_chief, num_tokens=0))

      # Distributed communication
      device_filters = None
      if cluster_spec:
        device_filters = ['/job:ps', '/job:worker/task:%d' % FLAGS.task]

      print(f'Running with device_filters {device_filters}')

      with tf.train.MonitoredTrainingSession(
          master=server.target,
          is_chief=is_chief,
          save_summaries_steps=None,
          save_checkpoint_secs=300,
          config=tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True,
            device_filters=device_filters),
          checkpoint_dir=FLAGS.train_dir,
          scaffold=scaffold,
          hooks=hooks) as sess:

        eval_tracker.sess = sess
        if is_chief and do_val:
          val_eval_tracker.sess = sess
        step = int(sess.run(model.global_step))

        if FLAGS.task > 0:
          # To avoid early instabilities when using multiple replicas, we use
          # a launch schedule where new replicas are brought online gradually.
          # logging.info('Delaying replica start.')
          while (not FLAGS.synchronous) and step < FLAGS.replica_step_delay * FLAGS.task:
            time.sleep(5.0)
            step = int(sess.run(model.global_step))
          logging.info(f'Worker task {FLAGS.task} coming online')
        else:
          summary_writer = tf.summary.FileWriterCache.get(FLAGS.train_dir)
          summary_writer.add_session_log(
              tf.summary.SessionLog(status=tf.summary.SessionLog.START), step)

        fov_shifts = list(model.shifts)  # x, y, z
        if FLAGS.shuffle_moves:
          random.shuffle(fov_shifts)

        policy_map = {
            'fixed': partial(fixed_offsets, fov_shifts=fov_shifts),
            'max_pred_moves': max_pred_offsets
        }
        batch_it = get_batch(lambda: sess.run(load_data_ops),
                             eval_tracker, model, FLAGS.batch_size,
                             policy_map[FLAGS.fov_policy])

        if is_chief and do_val:
          val_batch_it = get_batch(lambda: sess.run(val_load_data_ops),
                                   val_eval_tracker, model, FLAGS.batch_size,
                                   policy_map[FLAGS.fov_policy])

        t_last = time.time()

        while not sess.should_stop() and step < FLAGS.max_steps:
          # Run summaries periodically.
          t_curr = time.time()
          if t_curr - t_last > FLAGS.summary_rate_secs and is_chief:
            logging.info('Saving summaries at step ' + str(step))
            summ_op = merge_summaries_op
            t_last = t_curr
          else:
            summ_op = None

          if is_chief and do_val:
            # Run validation step
            # Might end up reducing the freq at which this runs.
            vs, vp, vl, vw = next(val_batch_it)
            updated_vs = run_validation_step(sess, model,
                feed_dict={
                    model.loss_weights: vw,
                    model.labels: vl,
                    model.input_patches: vp,
                    model.input_seed: vs,
                })
            mask.update_at(vs, (0, 0, 0), updated_vs)

          seed, patches, labels, weights = next(batch_it)

          updated_seed, step, summ = run_training_step(
              sess, model, summ_op,
              feed_dict={
                  model.loss_weights: weights,
                  model.labels: labels,
                  model.input_patches: patches,
                  model.input_seed: seed,
              })

          # Save prediction results in the original seed array so that
          # they can be used in subsequent steps.
          mask.update_at(seed, (0, 0, 0), updated_seed)

          # Record summaries.
          if summ is not None:
            logging.info('Saving summaries.')
            summ = tf.Summary.FromString(summ)

            # Compute a loss over the whole training patch (i.e. more than a
            # single-step field of view of the network). This quantifies the
            # quality of the final object mask.
            summ.value.extend(eval_tracker.get_summaries())
            if is_chief and do_val:
              summ.value.extend(val_eval_tracker.get_summaries())
            eval_tracker.reset()

            assert summary_writer is not None
            summary_writer.add_summary(summ, step)

      if summary_writer is not None:
        summary_writer.flush()


def get_cluster_spec():
  '''
  Convert the `{--ps_hosts,--worker_hosts}` flags into a definition of the
  distributed cluster we're running on.

  Returns:
    None                  if these flags aren't present, which signals local
                          training.
    tf.train.ClusterSpec  describing the cluster otherwise
  '''
  logging.info('%s %d Building cluster from ps_hosts %s, worker_hosts %s'
               % (FLAGS.job_name, FLAGS.task, FLAGS.ps_hosts, FLAGS.worker_hosts))
  if not (FLAGS.ps_hosts or FLAGS.worker_hosts or FLAGS.ps_tasks):
    return None
  elif FLAGS.ps_hosts and FLAGS.worker_hosts and FLAGS.ps_tasks > 0:
    cluster_spec = tf.train.ClusterSpec({
        'ps': FLAGS.ps_hosts,
        'worker': FLAGS.worker_hosts,
      })
    return cluster_spec
  else:
    raise InvalidArgumentError(
      'Set either all or none of --ps_hosts, --worker_hosts, --ps_tasks')


def main(argv=()):
  del argv  # Unused.

  model_class = import_symbol(FLAGS.model_name)
  # Multiply the task number by a value large enough that tasks starting at a
  # similar time cannot end up with the same seed.
  seed = int(time.time() * 1000 + FLAGS.task * 3600 * 24)
  logging.info('Random seed: %r', seed)
  random.seed(seed)

  if FLAGS.job_name == 'ps':
    train_ffn = ps_train_ffn
  elif FLAGS.job_name == 'worker':
    train_ffn = worker_train_ffn

  train_ffn(model_class,
            cluster_spec=get_cluster_spec(),
            batch_size=FLAGS.batch_size,
            **json.loads(FLAGS.model_args))


if __name__ == '__main__':
  flags.mark_flag_as_required('train_coords')
  flags.mark_flag_as_required('data_volumes')
  flags.mark_flag_as_required('label_volumes')
  flags.mark_flag_as_required('model_name')
  flags.mark_flag_as_required('model_args')
  app.run(main)
