'''
Starts a parameter server and a worker process on this node.
'''
import argparse
import os
import subprocess
import time


# *************************************************************************** #
if __name__ == '__main__':
    # *********************************************************************** #
    # Args

    ap = argparse.ArgumentParser(__doc__)

    ap.add_argument('--train_dir', type=str, default='/tmp')
    ap.add_argument('--train_coords', type=str, required=True,
                    help='Training coordinates from `build_coordinates.py`')
    ap.add_argument('--data_volumes', type=str, required=True,
                    help='hdf5 triplet for raw data')
    ap.add_argument('--label_volumes', type=str, required=True,
                    help='hdf5 triplet for segment data')
    ap.add_argument('--model_name', type=str,
                    default='convstack_3d.ConvStack3DFFNModel')
    ap.add_argument('--model_args', type=str,
                    default="{\"depth\": 12, \"fov_size\": [33, 33, 33], \"deltas\": [8, 8, 8]}")
    ap.add_argument('--image_mean', type=str, default='128')
    ap.add_argument('--image_stddev', type=str, default='33')
    ap.add_argument('--ps_tasks', type=int, required=True,
                    help='How many parameter servers?')
    ap.add_argument('--ps_port', type=str, default='2223',
                    help='Port for parameter servers')
    ap.add_argument('--no_gpu_for_ps', action='store_true')
    ap.add_argument('--worker_port', type=str, default='2222',
                    help='Port for workers')
    ap.add_argument('--max_steps', type=str, default='10000')
    ap.add_argument('--batch_size', type=str, default='4')

    args = ap.parse_args()


    # *********************************************************************** #
    # Parse slurm environment variables to figure out what other nodes exist
    # and build args for `train.py`

    # `$ scontrol show hostnames` spits out hosts, one per line.
    hostnames_res = subprocess.run(['scontrol', 'show', 'hostnames'],
        stdout=subprocess.PIPE)
    assert hostnames_res.returncode == 0
    hostnames = hostnames_res.stdout.decode().split()

    # `$SLURMD_NODENAME` is the name of the host we're running on
    me = os.environ['SLURMD_NODENAME']

    # Figure out which nodes will be running parameter servers
    num_nodes = len(hostnames)
    assert num_nodes >= args.ps_tasks
    node_idx = hostnames.index(me)
    worker_tasks = num_nodes + num_nodes - args.ps_tasks
    ps_hostnames = hostnames[:args.ps_tasks]

    # Will there be a ps on this node, or just a worker?
    run_ps = node_idx < args.ps_tasks

    # The args themselves
    task = str(node_idx)
    ps_hosts = ','.join(host + ':' + args.ps_port for host in ps_hostnames)
    worker_hosts = ','.join(host + ':' + args.worker_port for host in hostnames)


    # *********************************************************************** #
    # Launch processes

    print('Node', node_idx, 'of', num_nodes, 'launching worker'
          + ('and a ps' if run_ps else ''))

    # Worker
    worker_env = os.environ.copy()
    if run_ps and not args.no_gpu_for_ps:
        worker_env['CUDA_VISIBLE_DEVICES'] = '0'
    # else worker takes all gpus

    worker_proc = subprocess.Popen(['python', 'train.py',
            # The usual training args
            '--train_dir', args.train_dir,
            '--train_coords', args.train_coords,
            '--train_dir', args.train_dir,
            '--data_volumes', args.data_volumes,
            '--label_volumes', args.label_volumes,
            '--model_name', args.model_name,
            '--model_args', args.model_args,
            '--image_mean', args.image_mean,
            '--image_stddev', args.image_stddev,
            '--max_steps', args.max_steps,
            '--batch_size', args.batch_size,

            # Cluster config
            '--job_name', 'worker', # !
            '--task', task,
            '--ps_tasks', str(args.ps_tasks),
            '--ps_hosts', ps_hosts,
            '--worker_hosts', worker_hosts],
        env=worker_env)


    # Parameter server
    if run_ps:
        ps_env = os.environ.copy()
        if args.no_gpu_for_ps:
            ps_env['CUDA_VISIBLE_DEVICES'] = ''
        else:
            ps_env['CUDA_VISIBLE_DEVICES'] = '1'

        ps_proc = subprocess.Popen(['python', 'train.py',
                # The usual training args
                '--train_dir', args.train_dir,
                '--train_coords', args.train_coords,
                '--data_volumes', args.data_volumes,
                '--label_volumes', args.label_volumes,
                '--model_name', args.model_name,
                '--model_args', args.model_args,
                '--image_mean', args.image_mean,
                '--image_stddev', args.image_stddev,
                '--max_steps', args.max_steps,
                '--batch_size', args.batch_size,

                # Cluster config
                '--job_name', 'ps', # !
                '--task', task,
                '--ps_tasks', str(args.ps_tasks),
                '--ps_hosts', ps_hosts,
                '--worker_hosts', worker_hosts],
            env=ps_env)


    # *********************************************************************** #
    # Wait for join

    # Res of communicate should be (None, None)
    # communicate waits until the process finishes so this is one way to hang
    # around til then.
    print('Joining worker:', worker_proc.communicate())
    print(worker_proc)
    if run_ps:
        print('Joining ps:', ps_proc.communicate())
        print(ps_proc)
