import os
import time
import subprocess
from absl import flags
from absl import app
from ffn.training import optimizer  # noqa: W0611
from secgan.training import secgan_flags  # noqa: W0611
import logging


# ------------------------------- flags -------------------------------

flags.DEFINE_string('run_name', None, 'Name of run, folderst to create, etc.')
flags.DEFINE_string(
    'base_folder',
    '/mnt/ceph/data/neuro/wasp_em/ffndat/secgan/',
    'Stuff will live in base_folder/run_name.',
)

# This script manages both a slurm launcher and the nodes launched.
# User does not need to set this. Don't even worry about it.
flags.DEFINE_enum('role', 'launcher', ['launcher', 'node'], 'Leave me alone.')

# Args for configuring the cluster.
flags.DEFINE_integer(
    'num_nodes',
    1,
    'Number of nodes to allocate for this computation',
    lower_bound=1,
)
flags.DEFINE_integer(
    'num_ps', 1, 'Total number of parameter servers', lower_bound=1
)
flags.DEFINE_integer('ps_port', 2220, 'Port for parameter servers')
flags.DEFINE_integer('worker_port_min', 2221, 'Worker ports start here')
flags.DEFINE_string('gres', 'gpu:v100-32gb:4', '')

FLAGS = flags.FLAGS


# ----------------------------- launcher ------------------------------


def launcher(train_flags, optimizer_flags):
    subprocess.run(
        [
            'srun',
            # srun args
            '--job-name',
            FLAGS.run_name,
            '--ntasks-per-node=1',
            '--nodes',
            str(FLAGS.num_nodes),
            '--output',
            os.path.join(
                FLAGS.base_folder,
                FLAGS.run_name,
                f'{FLAGS.run_name}_%N_%j.out',
            ),
            '--error',
            os.path.join(
                FLAGS.base_folder,
                FLAGS.run_name,
                f'{FLAGS.run_name}_%N_%j.err',
            ),
            '-p',
            'gpu',
            f'--gres={FLAGS.gres}',
            '--exclusive',
            # Launch nodes
            'python',
            'slurm_train_secgan.py',
            '--run_name',
            FLAGS.run_name,
            '--role',
            'node',
            '--num_ps',
            str(FLAGS.num_ps),
            '--ps_port',
            str(FLAGS.ps_port),
            '--worker_port_min',
            str(FLAGS.worker_port_min),
        ]
        + train_flags
        + optimizer_flags
    )


# ------------------------------- node --------------------------------


def build_cluster_args():
    '''
    Parse slurm environment variables to figure out what other nodes exist
    and build args for `train_secgan.py`
    '''
    # `$ scontrol show hostnames` spits out hosts, one per line.
    hostnames_res = subprocess.run(
        ['scontrol', 'show', 'hostnames'], stdout=subprocess.PIPE
    )
    assert hostnames_res.returncode == 0
    hostnames = hostnames_res.stdout.decode().split()

    # `$SLURMD_NODENAME` is the name of the host we're running on
    me = os.environ['SLURMD_NODENAME']

    # Figure out how many GPUs each host has. Actually just how many I have,
    # assume homogeneous for now.
    gpus_res = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE)
    assert gpus_res.returncode == 0
    # Subtract 1 for trailing newline
    n_gpus = len(gpus_res.stdout.decode().split('\n')) - 1
    logging.info(f'{me} found n_gpus={n_gpus}')

    # Figure out which nodes will be running parameter servers
    num_nodes = len(hostnames)
    node_idx = hostnames.index(me)

    # The args themselves
    ps_task = str(node_idx)
    ps_hostnames = hostnames[0 : FLAGS.num_ps]
    logging.info(f'parameter server hosts: {ps_hostnames}')
    run_ps = me in ps_hostnames
    ps_hosts = ','.join(f'{host}:{FLAGS.ps_port}' for host in ps_hostnames)

    # A worker per gpu per host.
    # A worker needs to know its hostname:port, the index of its gpu,
    # and its task number.
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

    return (
        ps_task,
        run_ps,
        worker_tasks,
        worker_gpu_inds,
        ps_hosts,
        worker_hosts,
        node_idx,
        num_nodes,
    )


def launch_procs(
    train_flags,
    optimizer_flags,
    ps_task,
    run_ps,
    worker_tasks,
    worker_gpu_inds,
    ps_hosts,
    worker_hosts,
):
    '''
    Launch one worker for each GPU, and a parameter server if `run_ps`.
    Returns `processes`, all of the popen objects.
    '''
    processes = []
    start_synchronous_chief = False
    common_args = (
        [
            'python',
            'train_secgan.py',
            '--ps_tasks',
            str(FLAGS.num_ps),
            '--ps_hosts',
            ps_hosts,
            '--worker_hosts',
            worker_hosts,
            '--train_dir',
            os.path.join(FLAGS.base_folder, FLAGS.run_name),
        ]
        + train_flags
        + optimizer_flags
    )

    # Launch workers
    for worker_task, gpu_idx in zip(worker_tasks, worker_gpu_inds):
        if FLAGS.synchronous and worker_task == 0:
            # Delay the chief worker
            start_synchronous_chief = True
            chief_gpu = gpu_idx
            continue

        worker_env = os.environ.copy()
        worker_env['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
        worker_proc = subprocess.Popen(
            common_args + ['--job_name', 'worker', '--task', str(worker_task)],
            env=worker_env,
        )
        processes.append(worker_proc)

    # Launch parameter server
    if run_ps:
        ps_env = os.environ.copy()
        ps_env['CUDA_VISIBLE_DEVICES'] = ''
        ps_proc = subprocess.Popen(
            common_args + ['--job_name', 'ps', '--task', ps_task], env=ps_env
        )
        processes.append(ps_proc)

    # Launch synchronous chief
    if start_synchronous_chief:
        # Give other workers time to build their graphs
        time.sleep(60.0)
        worker_env = os.environ.copy()
        worker_env['CUDA_VISIBLE_DEVICES'] = str(chief_gpu)
        worker_proc = subprocess.Popen(
            common_args + ['--job_name', 'worker', '--task', '0'],
            env=worker_env,
        )
        processes.append(worker_proc)

    return processes


def node(train_flags, optimizer_flags):
    # See what nodes we are running on and what processes this node
    # should run
    (
        ps_task,
        run_ps,
        worker_tasks,
        worker_gpu_inds,
        ps_hosts,
        worker_hosts,
        node_idx,
        num_nodes,
    ) = build_cluster_args()

    # Launch training processes
    procs = launch_procs(
        train_flags,
        optimizer_flags,
        ps_task,
        run_ps,
        worker_tasks,
        worker_gpu_inds,
        ps_hosts,
        worker_hosts,
    )

    # Wait for join and log GPU usage
    while None in [proc.poll() for proc in procs]:
        subprocess.run(['nvidia-smi'])
        time.sleep(600.0)

    # Done now.
    for proc in procs:
        logging.info(str(proc))
        logging.info(str(proc.communicate()))


# ------------------------------- main --------------------------------


def main(argv):
    # Extract optimizer flags and the training flags
    module_dict = FLAGS.flags_by_module_dict()
    train_flags = [
        f.serialize()
        for f in module_dict['training.secgan_flags']
        if f.present
    ]
    optimizer_flags = [
        f.serialize()
        for f in module_dict['ffn.training.optimizer']
        if f.present
    ]

    if FLAGS.train_dir is not None:
        raise ValueError(
            'Please set --base_folder and --run_name '
            'instead of --train_dir.'
        )

    # Make sure train folder exists
    os.makedirs(os.path.join(FLAGS.base_folder, FLAGS.run_name), exist_ok=True)

    # Launch or run node
    if FLAGS.role == 'launcher':
        launcher(train_flags, optimizer_flags)
    elif FLAGS.role == 'node':
        node(train_flags, optimizer_flags)


if __name__ == '__main__':
    app.run(main)
