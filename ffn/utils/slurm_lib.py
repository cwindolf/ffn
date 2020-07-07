"""
This script contains utilities to simplify writing scripts
that batch FFN tasks over multiple Slurm nodes.

Basically, these just call `srun` on the runner scripts in
the root of this repo.
"""
from pathlib import Path

from tensorflow import gfile
from google.protobuf import text_format

from ffn.inference import inference_pb2
from ffn.utils import bounding_box_pb2
from ffn.inference import storage
import ffn

# the APIs of the functions below will allow users to override these
# defaults, which assume that the user wants to just grab a GPU and 1/4
# of the GPU node's compute for any given task.
default_slurm_kwargs = {
    "-n": "1",
    "-N": "1",
    "-p": "gpu",
    "--gres": "gpu:v100-32gb:1",
    "-c": "8",
}

# the functions below batch out to scripts in the ffn root. we find
# that by inspecting the ffn module.
# __file__ will be .../ffn/ffn/__init__.py__, we want .../ffn/.
ffnpath = Path(ffn.__file__).parent.parent.resolve()


def srun_inference(infreq, bbox, **slurm_kwargs):
    """Run an inference job on a Slurm node.

    Arguments
    ---------
    infreq : str
        An InferenceRequest proto in its string representation.
    bbox : str
        A BoundingBox proto in its string representation.

    Returns
    -------
    seg_dir : str
        The `segmentation_output_dir` field of the InferenceRequest.
    """
    sargs = default_slurm_kwargs.copy()
    sargs.update((k, str(v)) for k, v in slurm_kwargs.items())
    if "--job-name" not in sargs and "-J" not in sargs:
        sargs["--job-name"]

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
        print(f'{seg_path} already exists.')
        return seg_dir
