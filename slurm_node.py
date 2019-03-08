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
flags.DEFINE_integer('ps_port', 2220,
                    'Port for parameter servers')
flags.DEFINE_integer('worker_port_min', 2221,
                    'Low port for workers. Workers task i allocates port '
                    'worker_port_min+i. On a machine with 4 gpus, ports '
                    'min,min+1,min+2,min+3 will be allocated for workers.')
flags.DEFINE_string('node_log_dir', '', 'This script logs here.')

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

    # Figure out how many GPUs each host has. Actually just how many I have,
    # assume homogeneous for now.
    gpus_res = subprocess.run(['nvidia-smi', '-L'],
        stdout=subprocess.PIPE)
    assert gpus_res.returncode == 0
    # Subtract 1 for trailing newline
    n_gpus = len(gpus_res.stdout.decode().split('\n')) - 1

    # Figure out which nodes will be running parameter servers
    num_nodes = len(hostnames)
    assert num_nodes >= FLAGS.ps_tasks
    node_idx = hostnames.index(me)
    ps_hostnames = hostnames[:FLAGS.ps_tasks]

    # The args themselves
    ps_task = str(node_idx)
    ps_hosts = ','.join(host + ':' + str(FLAGS.ps_port) for host in ps_hostnames)

    # A worker per gpu per host.
    # A worker needs to know its hostname:port, the index of its gpu, and its task number.
    worker_hosts = []
    worker_tasks = []
    worker_gpu_inds = []
    cur_task = 0

    for h in hostnames:
        for i in range(n_gpus):
            host_and_port = h + ':' + str(FLAGS.worker_port_min + i)
            worker_hosts.append(host_and_port)

            if h == me:
                worker_gpu_inds.append(i)
                worker_tasks.append(cur_task)

            cur_task += 1

    worker_hosts = ','.join(worker_hosts)

    return ps_task, worker_tasks, worker_gpu_inds, ps_hosts, worker_hosts, node_idx, num_nodes


def launch_procs(ps_task, worker_tasks, worker_gpu_inds, ps_hosts, worker_hosts, run_ps):
    '''
    Launch one worker for each GPU, and a parameter server if `run_ps`.
    '''
    # If any training arguments are set, we want to send them to `train.py`
    module_dict = FLAGS.flags_by_module_dict()
    train_flags = [f.serialize()
                   for f in module_dict['ffn.training.training_flags']
                   if f.present]
    optimizer_flags = [f.serialize()
                       for f in module_dict['ffn.training.optimizer']
                       if f.present]

    worker_procs = []
    for worker_task, gpu_idx in zip(worker_tasks, worker_gpu_inds):
        worker_env = os.environ.copy()
        worker_env['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)

        worker_proc = subprocess.Popen(['python', 'train.py',
                # Cluster config
                '--job_name', 'worker', # !
                '--task', str(worker_task),
                '--ps_tasks', str(FLAGS.ps_tasks),
                '--ps_hosts', ps_hosts,
                '--worker_hosts', worker_hosts]
                + train_flags + optimizer_flags,
            env=worker_env)

        worker_procs.append(worker_proc)

    # Parameter server
    if run_ps:
        ps_env = os.environ.copy()
        ps_env['CUDA_VISIBLE_DEVICES'] = ''

        ps_proc = subprocess.Popen(['python', 'train.py',
                # Cluster config
                '--job_name', 'ps', # !
                '--task', ps_task,
                '--ps_tasks', str(FLAGS.ps_tasks),
                '--ps_hosts', ps_hosts,
                '--worker_hosts', worker_hosts]
                + train_flags + optimizer_flags,
            env=ps_env)

    return worker_procs + ([ps_proc] if run_ps else [])


def main(_):
    # See what nodes we are running on
    (ps_task, worker_tasks, worker_gpu_inds, ps_hosts,
        worker_hosts, node_idx, num_nodes) = build_cluster_args()
    run_ps = node_idx < FLAGS.ps_tasks

    # Launch training processes
    procs = launch_procs(ps_task, worker_tasks, worker_gpu_inds, ps_hosts, worker_hosts, run_ps)

    # Wait for join and log GPU usage
    while None in [proc.poll() for proc in procs]:
        subprocess.run(['nvidia-smi'])
        time.sleep(60.0)

    # Done now.
    for proc in procs:
        print(proc, proc.communicate())


if __name__ == '__main__':
    app.run(main)
