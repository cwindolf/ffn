'''
Run a train task.

This will launch either a worker, or a worker and a ps if rank is 0.

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

flags.DEFINE_integer('port_start', 2220,
                     'Port for all tasks')

FLAGS = flags.FLAGS


def build_cluster_args():
    '''
    Parse slurm environment to build list of server:port for all
    workers and all parameter servers.

    At the same time, figure out how this task fits into the grand
    scheme.
    '''
    assert int(os.environ['SLURM_STEPID']) == 0

    # Each instantiation of this script will get a unique rank,
    # and a per-machine unique local rank
    my_rank = int(os.environ['SLURM_PROCID'])

    # How many total workers?
    n_ranks = int(os.environ['SLURM_NTASKS'])

    # `$ scontrol show hostnames` spits out hosts, one per line.
    # This is easier than parsing `$SLURM_NODELIST`.
    hostnames_res = subprocess.run(['scontrol', 'show', 'hostnames'],
                                   stdout=subprocess.PIPE)
    assert hostnames_res.returncode == 0
    hostnames = hostnames_res.stdout.decode().split()

    # Figure out the rank->host mapping
    rank_to_host = []
    local_ranks = []
    host_idx = 0
    local_rank = 0
    # example: SLURM_TASKS_PER_NODE=2(x2),1
    # result here should be [host0, host0, host1, host1, host2]
    for expr in os.environ['SLURM_TASKS_PER_NODE'].split(','):
        if '(x' in expr:
            beg, end = expr.split('(x')
            n_tasks_on_host = int(beg)
            n_repeats = int(end.rstrip(')'))
            for i in range(n_repeats):
                for j in range(n_tasks_on_host):
                    rank_to_host.append(hostnames[host_idx])
                    local_ranks.append(local_rank)
                    local_rank += 1
                host_idx += 1
                local_rank = 0
        else:
            n_tasks_on_host = int(expr)
            for i in range(n_tasks_on_host):
                rank_to_host.append(hostnames[host_idx])
                local_ranks.append(local_rank)
                local_rank += 1
            host_idx += 1
            local_rank = 0

    # Just making sure.
    assert len(rank_to_host) == len(local_ranks) == n_ranks

    # Each host runs a worker
    worker_port_start = FLAGS.port_start + 1
    worker_addrs = [f'{h}:{worker_port_start + lr}'
                    for h, lr in zip(rank_to_host, local_ranks)]

    # Figure out which machines will run parameter servers
    run_ps = my_rank == 0
    ps_addrs = [f'{rank_to_host[0]}:{FLAGS.port_start}']

    return {
        'worker_task': str(my_rank),
        'run_ps': run_ps,
        'worker_addrs': ','.join(worker_addrs),
        'ps_addrs': ','.join(ps_addrs),
    }


def launch_procs(worker_task=None, run_ps=None,
                 worker_addrs=None, ps_addrs=None):
    # Get some flags to pass to trainer
    module_dict = FLAGS.flags_by_module_dict()
    extra_flags = [f.serialize()
                   for f in (module_dict['ffn.training.training_flags']
                             + module_dict['ffn.training.optimizer'])
                   if f.present]

    procs = []

    # Launch worker
    print(f'Launching worker {worker_task}')
    procs.append(subprocess.Popen(
        ['python',
         'train.py',
         '--job_name', 'worker',  # !
         '--task', worker_task,
         '--ps_tasks', '1',
         '--ps_hosts', ps_addrs,
         '--worker_hosts', worker_addrs,
         ]
        + extra_flags))

    # Launch ps if nec.
    if run_ps:
        ps_env = os.environ.copy()
        ps_env['CUDA_VISIBLE_DEVICES'] = ''
        print('Launching PS')
        procs.append(subprocess.Popen(
            ['python',
             'train.py',
             '--job_name', 'ps',  # !
             '--task', '0',
             '--ps_tasks', '1',
             '--ps_hosts', ps_addrs,
             '--worker_hosts', worker_addrs,
             ]
            + extra_flags,
            env=ps_env))

    return procs


def main(_):
    procs = launch_procs(**build_cluster_args())

    # Wait for join and log GPU usage
    while None in [proc.poll() for proc in procs]:
        subprocess.run(['nvidia-smi'])
        time.sleep(360.0)

    # Done now.
    for proc in procs:
        print(proc, proc.communicate())


if __name__ == '__main__':
    app.run(main)
