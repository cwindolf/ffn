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
flags.DEFINE_integer('num_ps', 0,
                     'How many nodes should run a parameter server? '
                     'If unset, every node will run one.')
flags.DEFINE_string('ps_port', '2224',
                    'Port for parameter servers')
flags.DEFINE_string('worker_a_port', '2222',
                    'Port for first worker on each node')
flags.DEFINE_string('worker_b_port', '2223',
                    'Port for second worker')
flags.DEFINE_string('exclude', 'workergpu[00-02]',
                    'Slurm hostname list of machines to avoid')
flags.DEFINE_string('slurm_log_dir', 'logs/',
                    'Have slurm save program std{err,out} in this dir')

FLAGS = flags.FLAGS


def main(argv):
    # *********************************************************************** #
    # Munge args

    # Default number of parameter servers
    if FLAGS.num_ps <= 0 or FLAGS.num_ps > FLAGS.num_nodes:
        num_ps = str(FLAGS.num_nodes)
    else:
        num_ps = str(FLAGS.num_ps)

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
    # Run the job

    # Check env for srun
    try:
        subprocess.run(['srun', '--version'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        import sys
        sys.exit('srun: command not found.\n'
                 'Try a `module load slurm`')

    print('Running job with log dir', os.path.abspath(FLAGS.slurm_log_dir))

    res = subprocess.run(['srun',
            # srun args
            '--nodes', str(FLAGS.num_nodes),
            '--output', os.path.join(FLAGS.slurm_log_dir, 'ffn_%N_%j.out'),
            '--error', os.path.join(FLAGS.slurm_log_dir, 'ffn_%N_%j.err'),
            '-p', 'gpu',
            '--gres=gpu:2',
            '--exclude', FLAGS.exclude,
            '--exclusive',

            # trainer script and its args
            'python', 'slurm_node.py',
            '--ps_tasks', num_ps,
            '--ps_port', FLAGS.ps_port,
            '--worker_a_port', FLAGS.worker_a_port,
            '--worker_b_port', FLAGS.worker_b_port]
            + train_flags + optimizer_flags)

    print('srun ran with return code', res.returncode)
    print('bye!')


if __name__ == '__main__':
    # These flags are required in train.py, so let's bail now rather than
    # allocating a node and then bailing.
    flags.mark_flag_as_required('train_coords')
    flags.mark_flag_as_required('data_volumes')
    flags.mark_flag_as_required('label_volumes')
    flags.mark_flag_as_required('model_name')
    flags.mark_flag_as_required('model_args')
    app.run(main)
