"""
This script contains utilities to simplify writing scripts
that batch FFN tasks over multiple Slurm nodes.

Basically, these just call `srun` on the runner scripts in
the root of this repo.
"""
from multiprocessing.pool import ThreadPool
from pathlib import Path
import subprocess
import sys
import time

from tensorflow import gfile
from google.protobuf import text_format

from ffn.inference import inference_pb2
from ffn.utils import bounding_box_pb2
from ffn.inference import storage


# -- constants

# the APIs of the functions below will allow users to override these
# defaults, which assume that the user wants to just grab a GPU and 1/4
# of the GPU node's compute for any given task.
default_slurm_kwargs = {
    "-n": "1",
    "-N": "1",
    "-p": "gpu",
    "--gres": "gpu:1",
    '--constraint': 'v100',
    "-c": "8",
}


# -- helpers

def _readfile(fn):
    assert Path(fn).is_file()
    with open(fn, 'r') as f:
        text = ' '.join(f.read().split())
    return text


# -- slurm library

def srun_inference(infreq, bbox, local=False, slurm_kwargs=None):
    """Run an inference job on a Slurm node.

    Arguments
    ---------
    infreq : str
        An InferenceRequest proto in its string representation.
    bbox : str
        A BoundingBox proto in its string representation.
    local : bool
        Run locally rather than on a Slurm node.
    slurm_kwargs : dict of command line arguments for srun

    Returns
    -------
    seg_dir : str
        The `segmentation_output_dir` field of the InferenceRequest.
    """

    # -- bail out early
    # Will bail under the same circumstances that ffn Runner does.
    # Copied from `Runner.run` method.
    # This is nice because you can glob infreqs in bash and not worry
    # about doing the same job twice, and because we won't launch
    # a whole srun just for the thing to bail right away.
    bbox_pb = bounding_box_pb2.BoundingBox()
    text_format.Parse(bbox, bbox_pb)
    infreq_pb = inference_pb2.InferenceRequest()
    text_format.Parse(infreq, infreq_pb)
    seg_dir = infreq_pb.segmentation_output_dir
    corner = (bbox_pb.start.z, bbox_pb.start.y, bbox_pb.start.x)
    seg_path = storage.segmentation_path(seg_dir, corner)
    if gfile.Exists(seg_path):
        print(f"{seg_path} already exists.")
        return seg_dir

    print("Starting worker for ", seg_dir)

    # -- process srun arguments
    sargs = default_slurm_kwargs.copy()
    if slurm_kwargs:
        sargs.update((k, v) for k, v in slurm_kwargs.items() if v)
    if "--job-name" not in sargs and "-J" not in sargs:
        sargs["--job-name"] = f"inf{bbox_pb.size.x}"

    # -- build command line
    wrapper = []
    if not local:
        wrapper = ["srun"]
        for k, v in sargs.items():
            wrapper += [k, v]
    inference_cmd = [
        "python",
        "run_inference.py",
        "--inference_request",
        infreq,
        "--bounding_box",
        bbox,
    ]

    # -- run and make sure process exit
    print("Running:", repr(wrapper + inference_cmd))
    process = subprocess.run(
        wrapper + inference_cmd,
        stdin=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
    )
    print(process.poll(), process)

    # read output until we see that process is done
    # they hang sometimes and can need multiple sigterms
    # because of srun.
    for bline in iter(process.stdout.readline, ''):
        line = bline.decode(sys.stdout.encoding).strip()
        if line:
            print(line, flush=True)
            if 'terminating' in line:
                print('>>> Process seems to be done.')
                print('>>> Waiting for a bit and then killing.')
                time.sleep(60.0)
                process.kill()
                time.sleep(30.0)
                if process.poll() is None:
                    process.kill()
                    process.kill()
                time.sleep(5.0)
                print('Killed? poll:', process.poll())
    print(f'Done with {seg_path}...')

    return seg_dir


def srun_multiple_inferences(
    infreqs, bboxes, n_workers=-1, local=False, slurm_kwargs=None
):
    """Run multiple inferences at once on a Slurm cluster

    Arguments
    ---------
    infreqs : list of str
        Either a list of InferenceRequest protos in their string
        representation, or a list of filenames containing
        InferenceRequests.
    bboxes : list of str
        Similar, but for BoundingBox protos.
        Length of bboxes should be either len(infreqs) or 1.
        In the latter case, that bbox is used in all inferences.
    n_workers : int
        Number of inferences to run at a time. If negative, run
        as many workers as there are inference requests.
    local : bool
        Run locally rather than via Slurm. If set, implies
        n_workers=1 even if n_workers is set.
    slurm_kwargs : dict of command line args for srun

    Returns
    -------
    seg_dirs : list of str
        The segmentation_output_dir fields of the supplied infreqs.
    """
    n_infreqs = len(infreqs)

    # -- load up protos if they are files
    if infreqs[0].endswith(".pbtxt"):
        infreqs = [_readfile(fn) for fn in infreqs]
    if bboxes[0].endswith(".pbtxt"):
        bboxes = [_readfile(fn) for fn in bboxes]

    # broadcast bboxes
    if len(bboxes) == 1:
        bboxes = [bboxes[0] for _ in range(n_infreqs)]

    # -- build and run jobs
    job_args = zip(
        infreqs,
        bboxes,
        [local for _ in range(n_infreqs)],
        [slurm_kwargs for _ in range(n_infreqs)],
    )

    if n_workers < 1:
        n_workers = n_infreqs
    if local:
        n_workers = 1

    res_dirs = []
    with ThreadPool(n_workers) as pool:
        for res_dir in pool.starmap(srun_inference, job_args):
            res_dirs.append(res_dir)

    return res_dirs
