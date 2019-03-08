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
flags.DEFINE_integer('num_extra_ps', 0,
                     'Run this many additional parameter servers on '
                     'dedicated nodes.')
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

    # Figure out how many GPUs this host has
    gpus_res = subprocess.run(['nvidia-smi', '-L'],
        stdout=subprocess.PIPE)
    print(gpus_res.stdout.decode())
    # Subtract 1 for trailing newline
    n_gpus = len(gpus_res.stdout.decode().split('\n')) - 1
    gpu_inds = list(range(n_gpus))

    # Extra parameter servers
    extra_ps_hostnames = []
    worker_hostnames = hostnames
    if FLAGS.num_extra_ps:
        extra_ps_hostnames = hostnames[-FLAGS.num_extra_ps:]
        worker_hostnames = hostnames[:FLAGS.num_extra_ps]
    am_extra_ps = me in extra_ps_hostnames

    # Compute the hostnames for workers and ps
    # The result should be comma separated lists host1:port1,host2:port2,...
    # Also, compute the ps and worker task indices that will run on this node
    ps_hosts = []
    ps_tasks = []
    cur_ps_task = 0
    for host in hostnames[:FLAGS.ps_tasks]:
        if host in extra_ps_hostnames:
            continue
        ps_hosts.append(f'{host}:{FLAGS.ps_port}')
        if host == me:
            ps_tasks.append(cur_ps_task)
        cur_ps_task += 1
    for host in extra_ps_hostnames:
        # ps take the role of workers, use worker ports.
        for i in gpu_inds:
            ps_hosts.append(f'{host}:{FLAGS.worker_port_min + i}')
            if host == me:
                ps_tasks.append(cur_ps_task)
            cur_ps_task += 1
    ps_hosts = ','.join(ps_hosts)

    worker_hosts = []
    worker_tasks = []
    cur_worker_task = 0
    for host in worker_hostnames:
        for i in gpu_inds:
            worker_hosts.append(f'{host}:{FLAGS.worker_port_min + i}')
            if host == me:
                worker_tasks.append(cur_worker_task)
            cur_worker_task += 1
    worker_hosts = ','.join(worker_hosts)

    return ps_tasks, worker_tasks, gpu_inds, ps_hosts, worker_hosts


def launch_procs(ps_tasks, worker_tasks, gpu_inds, ps_hosts, worker_hosts):
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
    for worker_task, gpu_idx in zip(worker_tasks, gpu_inds):
        worker_env = os.environ.copy()
        worker_env['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
        print('Worker adding gpu', gpu_idx)

        worker_proc = subprocess.Popen(['python', 'train.py',
                # Cluster config
                '--job_name', 'worker', # !
                '--task', str(worker_task),
                '--ps_tasks', str(len(ps_tasks)),
                '--ps_hosts', ps_hosts,
                '--worker_hosts', worker_hosts]
                + train_flags + optimizer_flags,
            env=worker_env)

        worker_procs.append(worker_proc)

    # Parameter server
    ps_procs = []
    for i, ps_task in enumerate(ps_tasks):
        # Is there a gpu for this ps?
        ps_gpu = ''
        if i < len(gpu_inds) - len(worker_tasks):
            ps_gpu = str(gpu_inds[len(worker_tasks) + i])

        ps_env = os.environ.copy()
        ps_env['CUDA_VISIBLE_DEVICES'] = ps_gpu
        print('PS adding gpu', ps_gpu)

        ps_proc = subprocess.Popen(['python', 'train.py',
                # Cluster config
                '--job_name', 'ps', # !
                '--task', str(ps_task),
                '--ps_tasks', str(len(ps_tasks)),
                '--ps_hosts', ps_hosts,
                '--worker_hosts', worker_hosts]
                + train_flags + optimizer_flags,
            env=ps_env)

        ps_procs.append(ps_proc)

    return worker_procs + ps_procs


def main(_):
    # See what nodes we are running on
    (ps_tasks, worker_tasks, gpu_inds, ps_hosts,
        worker_hosts) = build_cluster_args()

    # Launch training processes
    procs = launch_procs(ps_tasks, worker_tasks, gpu_inds, ps_hosts, worker_hosts)

    # Wait for join and log GPU usage
    while None in [proc.poll() for proc in procs]:
        subprocess.run(['nvidia-smi'])
        time.sleep(60.0)

    # Done now, join procs
    for proc in procs:
        print(proc, proc.communicate())


if __name__ == '__main__':
    app.run(main)
