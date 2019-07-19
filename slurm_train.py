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
flags.DEFINE_integer('num_nodes', 1,
                     'Number of nodes to allocate for this computation',
                     lower_bound=1)
flags.DEFINE_integer('num_ps', 1,
                     'Total number of parameter servers',
                     lower_bound=1)
flags.DEFINE_string('ps_port', '2220',
                    'Port for parameter servers')
flags.DEFINE_string('worker_port_min', '2221',
                    'Port for first worker on each node')
flags.DEFINE_string('exclude', 'workergpu[00-02]',
                    'Slurm hostname list of machines to avoid')
flags.DEFINE_string('slurm_log_dir', 'logs/',
                    'Have slurm save program std{err,out} in this dir')
flags.DEFINE_string('gres', 'gpu:4', 'gpu:<num_gpus>')

FLAGS = flags.FLAGS


def main(argv):
    # *********************************************************************** #
    # Munge args

    # We want to pass the optimizer flags and the training flags to
    # `slurm_node.py`. So let's serialize those.
    module_dict = FLAGS.flags_by_module_dict()
    train_flags = [f.serialize()
                   for f in module_dict['ffn.training.training_flags']
                   if f.present]
    optimizer_flags = [f.serialize()
                       for f in module_dict['ffn.training.optimizer']
                       if f.present]

    # *********************************************************************** #
    # Write out the flags

    with open(os.path.join(FLAGS.train_dir, 'flagfile.txt'), 'w') as flagfile:
        flagfile.write(FLAGS.flags_into_string())

    # *********************************************************************** #
    # Run the job

    print('Running job with log dir', os.path.abspath(FLAGS.slurm_log_dir))

    res = subprocess.run(
        ['srun',
         # srun args
         '--job-name', 'ffntrain',
         '--ntasks-per-node=1',
         '--nodes', str(FLAGS.num_nodes),
         '--output', os.path.join(FLAGS.slurm_log_dir, 'ffn_%N_%j.out'),
         '--error', os.path.join(FLAGS.slurm_log_dir, 'ffn_%N_%j.err'),
         '-p', 'gpu',
         f'--gres={FLAGS.gres}',
         # '--exclude', FLAGS.exclude,
         '--constraint=v100',
         # '-c40',
         # '--ntasks-per-node=1',
         '--exclusive',

         # trainer script and its args
         'python', 'slurm_node.py',
         '--num_ps', str(FLAGS.num_ps),
         '--ps_port', FLAGS.ps_port,
         '--worker_port_min', FLAGS.worker_port_min,
         '--node_log_dir', os.path.join(FLAGS.slurm_log_dir)]
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
