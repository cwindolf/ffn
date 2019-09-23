'''
Builds an srun command to run a distributed training job.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import subprocess

from absl import flags
from absl import app

# Flags needed by train.py live in these files
# pylint: disable=unused-import
from ffn.training import optimizer
from ffn.training import training_flags
# pylint: enable=unused-import

# Args for configuring the cluster.
flags.DEFINE_integer('nworkers', 1,
                     'Number of gpus to allocate for this computation',
                     lower_bound=1)
flags.DEFINE_string('port_start', '2220',
                    'Ports will be allocated starting here.')

FLAGS = flags.FLAGS


def main(argv):
    # Munge args ------------------------------------------------------

    # We want to pass the optimizer flags and the training flags to
    # `slurm_node.py`. So let's serialize those.
    module_dict = FLAGS.flags_by_module_dict()
    train_flags = [f.serialize()
                   for f in module_dict['ffn.training.training_flags']
                   if f.present]
    optimizer_flags = [f.serialize()
                       for f in module_dict['ffn.training.optimizer']
                       if f.present]

    # Write out the flags

    with open(os.path.join(FLAGS.train_dir, 'flagfile.txt'), 'w') as flagfile:
        flagfile.write(FLAGS.flags_into_string())

    # Run the job -----------------------------------------------------
    res = subprocess.run(
        ['srun',
         '--job-name', 'ffntrain',
         '--output', os.path.join(FLAGS.train_dir, 'ffn_%N_%j_%t.out'),
         '--error', os.path.join(FLAGS.train_dir, 'ffn_%N_%j_%t.err'),

         # Job spec
         '--ntasks', str(FLAGS.nworkers),
         '-p', 'gpu',
         '--gpus-per-task', 'v100-32gb:1',
         '--cpus-per-task', '9',

         # Each task runs...
         'python',
         'slurm_train_task.py',
         '--port_start', FLAGS.port_start,
         ]
        + train_flags
        + optimizer_flags)

    print('srun ran with return code', res.returncode)
    print('bye!')


if __name__ == '__main__':
    # These flags are required in train.py.
    # Bail now rather than allocating nodes and then bailing.
    flags.mark_flag_as_required('train_coords')
    flags.mark_flag_as_required('data_volumes')
    flags.mark_flag_as_required('label_volumes')
    flags.mark_flag_as_required('model_name')
    flags.mark_flag_as_required('model_args')

    app.run(main)
