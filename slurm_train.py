'''
Builds an srun command to run a distributed training job.
'''
import argparse
import os.path
import subprocess

# *************************************************************************** #
if __name__ == '__main__':
    # *********************************************************************** #
    # Args

    ap = argparse.ArgumentParser(__doc__)

    ap.add_argument('num_nodes', type=int,
                    help='Number of nodes to allocate for this computation')
    ap.add_argument('--num_ps', type=int, default=0,
                    help='How many nodes should run a parameter server? '
                         'If unset, every node will run one.')
    ap.add_argument('--ps_port', type=str, default='2223',
                    help='Port for parameter servers')
    ap.add_argument('--worker_port', type=str, default='2222',
                    help='Port for workers')
    ap.add_argument('--exclude', type=str, default='workergpu[00-02]',
                    help='Cluster machines to avoid.')
    ap.add_argument('--log_dir', type=str, default='logs/',
                    help='Directory for log files')

    tp = ap.add_argument_group('Training config')
    tp.add_argument('--train_dir', type=str, default='/tmp')
    tp.add_argument('--train_coords', type=str, required=True,
                    help='Training coordinates from `build_coordinates.py`')
    tp.add_argument('--data_volumes', type=str, required=True,
                    help='hdf5 triplet for raw data')
    tp.add_argument('--label_volumes', type=str, required=True,
                    help='hdf5 triplet for segment data')
    tp.add_argument('--model_name', type=str,
                    default='convstack_3d.ConvStack3DFFNModel')
    tp.add_argument('--model_args', type=str,
                    default="{\"depth\": 12, \"fov_size\": [33, 33, 33], \"deltas\": [8, 8, 8]}")
    tp.add_argument('--image_mean', type=str, default='128')
    tp.add_argument('--image_stddev', type=str, default='33')
    tp.add_argument('--max_steps', type=str, default='10000')
    tp.add_argument('--batch_size', type=str, default='4')

    args = ap.parse_args()

    # Default number of parameter servers
    if args.num_ps <= 0 or args.num_ps > args.num_nodes:
        num_ps = str(args.num_nodes)
    else:
        num_ps = str(args.num_ps)


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

    res = subprocess.run(['srun',
            # srun args
            '--nodes', str(args.num_nodes),
            '--output', os.path.join(args.log_dir, 'ffn_%N_%j.out'),
            '--error', os.path.join(args.log_dir, 'ffn_%N_%j.err'),
            '-p', 'gpu',
            '--gres=gpu:2',
            '--exclude', args.exclude,
            '--exclusive',

            # trainer script and its args
            'python', './slurm/slurm_node.py',
            '--ps_tasks', num_ps,
            '--train_dir', args.train_dir,
            '--train_coords', args.train_coords,
            '--data_volumes', args.data_volumes,
            '--label_volumes', args.label_volumes,
            '--model_name', args.model_name,
            '--model_args', args.model_args,
            '--image_mean', args.image_mean,
            '--image_stddev', args.image_stddev,
            '--ps_port', args.ps_port,
            '--worker_port', args.worker_port,
            '--batch_size', args.batch_size,
            '--max_steps', args.max_steps])

    print('bye')
    print('srun ran with return code', res.returncode)
