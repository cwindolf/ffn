'''
Starts a parameter server and a worker process on this node.
Meant to be run by slurm_train.py, not manually.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import time

from absl import flags
from absl import app

# Flags needed by train.py live in these files
# pylint: disable=unused-import
from ffn.training import optimizer
from ffn.training import training_flags
# pylint: enable=unused-import


flags.DEFINE_integer('ps_tasks', 1,
                     'How many parameter servers?')
flags.DEFINE_string('ps_port', '2224',
                    'Port for parameter servers')
flags.DEFINE_string('worker_a_port', '2222',
                    'Port for workers')
flags.DEFINE_string('worker_b_port', '2223',
                    'Second port for workers')

FLAGS = flags.FLAGS


def build_cluster_args():
    '''
    Parse slurm environment variables to figure out what other nodes exist
    and build args for `train.py`
    '''
    # `$ scontrol show hostnames` spits out hosts, one per line.
    hostnames_res = subprocess.run(['scontrol', 'show', 'hostnames'],
        stdout=subprocess.PIPE)
    assert hostnames_res.returncode == 0
    hostnames = hostnames_res.stdout.decode().split()

    # `$SLURMD_NODENAME` is the name of the host we're running on
    me = os.environ['SLURMD_NODENAME']

    # Figure out which nodes will be running parameter servers
    num_nodes = len(hostnames)
    assert num_nodes >= FLAGS.ps_tasks
    node_idx = hostnames.index(me)
    ps_hostnames = hostnames[:FLAGS.ps_tasks]

    # The args themselves
    task = str(node_idx)
    b_task = str(node_idx + num_nodes)
    ps_hosts = ','.join(host + ':' + FLAGS.ps_port for host in ps_hostnames)
    worker_hosts = ','.join(host + ':' + FLAGS.worker_a_port for host in hostnames)
    worker_hosts += ','
    worker_hosts += ','.join(host + ':' + FLAGS.worker_b_port for host in hostnames)

    return task, b_task, ps_hosts, worker_hosts, node_idx, num_nodes


def launch_procs(task, b_task, worker_hosts, ps_hosts, run_ps):
    '''
    Launch two workers (one for each gpu -- edit this if different machines
    become available), and a parameter server if `run_ps`.

    task, b_task, worker_hosts, ps_hosts
        all strings. will be used as command line args.

    run_ps      bool
    '''
    # If any training arguments are set, we want to send them to `train.py`
    module_dict = FLAGS.flags_by_module_dict()
    train_flags = [f.serialize() 
                   for f in module_dict['ffn.training.training_flags']
                   if f.present]
    optimizer_flags = [f.serialize() 
                       for f in module_dict['ffn.training.optimizer']
                       if f.present]

    # Worker A
    worker_a_env = os.environ.copy()
    worker_a_env['CUDA_VISIBLE_DEVICES'] = '0'

    worker_a_proc = subprocess.Popen(['python', 'train.py',
            # Cluster config
            '--job_name', 'worker', # !
            '--task', task,
            '--ps_tasks', str(FLAGS.ps_tasks),
            '--ps_hosts', ps_hosts,
            '--worker_hosts', worker_hosts]
            + train_flags + optimizer_flags,
        env=worker_a_env)


    # Worker B
    worker_b_env = os.environ.copy()
    worker_b_env['CUDA_VISIBLE_DEVICES'] = '1'

    worker_b_proc = subprocess.Popen(['python', 'train.py',
            # Cluster config
            '--job_name', 'worker', # !
            '--task', b_task,
            '--ps_tasks', str(FLAGS.ps_tasks),
            '--ps_hosts', ps_hosts,
            '--worker_hosts', worker_hosts]
            + train_flags + optimizer_flags,
        env=worker_b_env)


    # Parameter server
    if run_ps:
        ps_env = os.environ.copy()
        ps_env['CUDA_VISIBLE_DEVICES'] = ''

        ps_proc = subprocess.Popen(['python', 'train.py',
                # Cluster config
                '--job_name', 'ps', # !
                '--task', task,
                '--ps_tasks', str(FLAGS.ps_tasks),
                '--ps_hosts', ps_hosts,
                '--worker_hosts', worker_hosts]
                + train_flags + optimizer_flags,
            env=ps_env)

    return [worker_a_proc, worker_b_proc] + ([ps_proc] if run_ps else [])


def main(_):
    # See what nodes we are running on
    (task, b_task, ps_hosts,
     worker_hosts, node_idx, num_nodes) = build_cluster_args()
    run_ps = node_idx < FLAGS.ps_tasks

    # Launch training processes
    print('Node', node_idx, 'of', num_nodes, 'launching workers'
      + ('and a ps' if run_ps else ''))
    procs = launch_procs(task, b_task, worker_hosts, ps_hosts, run_ps)

    # Wait for join and log GPU usage
    while None in [proc.poll() for proc in procs]:
        subprocess.run(['nvidia-smi'])
        time.sleep(10.0)

    # Done now.
    for proc in procs:
        print(proc, proc.communicate())

    print('Node', node_idx, 'finished.')


if __name__ == '__main__':
    app.run(main)
   