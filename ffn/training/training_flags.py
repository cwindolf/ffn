'''
This file allows the training configuration to be shared between scripts.
Doesn't worry about the cluster flags since those are different across
scripts.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags


# Options related to training data.
flags.DEFINE_string('train_coords', None,
                    'Glob for the TFRecord of training coordinates.')
flags.DEFINE_string('data_volumes', None,
                    'Comma-separated list of <volume_name>:<volume_path>:'
                    '<dataset>, where volume_name need to match the '
                    '"label_volume_name" field in the input example, '
                    'volume_path points to HDF5 volumes containing uint8 '
                    'image data, and `dataset` is the name of the dataset '
                    'from which data will be read.')
flags.DEFINE_string('label_volumes', None,
                    'Comma-separated list of <volume_name>:<volume_path>:'
                    '<dataset>, where volume_name need to match the '
                    '"label_volume_name" field in the input example, '
                    'volume_path points to HDF5 volumes containing int64 '
                    'label data, and `dataset` is the name of the dataset '
                    'from which data will be read.')
flags.DEFINE_string('model_name', None,
                    'Name of the model to train. Format: '
                    '[<packages>.]<module_name>.<model_class>, if packages is '
                    'missing "ffn.training.models" is used as default.')
flags.DEFINE_string('model_args', None,
                    'JSON string with arguments to be passed to the model '
                    'constructor.')

# Training infra options.
flags.DEFINE_string('train_dir', '/tmp',
                    'Path where checkpoints and other data will be saved.')
flags.DEFINE_integer('batch_size', 4, 'Number of images in a batch.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to train for.')
flags.DEFINE_integer('replica_step_delay', 300,
                     'Require the model to reach step number '
                     '<replica_step_delay> * '
                     '<replica_id> before starting training on a given '
                     'replica.')
flags.DEFINE_integer('summary_rate_secs', 120,
                     'How often to save summaries (in seconds).')



# FFN training options.
flags.DEFINE_float('seed_pad', 0.05,
                   'Value to use for the unknown area of the seed.')
flags.DEFINE_float('threshold', 0.9,
                   'Value to be reached or exceeded at the new center of the '
                   'field of view in order for the network to inspect it.')
flags.DEFINE_enum('fov_policy', 'fixed', ['fixed', 'max_pred_moves'],
                  'Policy to determine where to move the field of the '
                  'network. "fixed" tries predefined offsets specified by '
                  '"model.shifts". "max_pred_moves" moves to the voxel with '
                  'maximum mask activation within a plane perpendicular to '
                  'one of the 6 Cartesian directions, offset by +/- '
                  'model.deltas from the current FOV position.')



# TODO(mjanusz): Implement fov_moves > 1 for the 'fixed' policy.
flags.DEFINE_integer('fov_moves', 1,
                     'Number of FOV moves by "model.delta" voxels to execute '
                     'in every dimension. Currently only works with the '
                     '"max_pred_moves" policy.')
flags.DEFINE_boolean('shuffle_moves', True,
                     'Whether to randomize the order of the moves used by the '
                     'network with the "fixed" policy.')

flags.DEFINE_float('image_mean', None,
                   'Mean image intensity to use for input normalization.')
flags.DEFINE_float('image_stddev', None,
                   'Image intensity standard deviation to use for input '
                   'normalization.')
flags.DEFINE_list('image_offset_scale_map', None,
                  'Optional per-volume specification of mean and stddev. '
                  'Every entry in the list is a colon-separated tuple of: '
                  'volume_label, offset, scale.')

flags.DEFINE_list('permutable_axes', ['1', '2'],
                  'List of integers equal to a subset of [0, 1, 2] specifying '
                  'which of the [z, y, x] axes, respectively, may be permuted '
                  'in order to augment the training data.')

flags.DEFINE_list('reflectable_axes', ['0', '1', '2'],
                  'List of integers equal to a subset of [0, 1, 2] specifying '
                  'which of the [z, y, x] axes, respectively, may be reflected '
                  'in order to augment the training data.')