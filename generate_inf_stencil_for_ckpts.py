"""Helper script for generating InferenceRequests and storage folders

You have several checkpoints, and maybe you want to make both a
PolicyPeaks and a PolicyInvertOrigins inference request for both.
To use this script, make basic inference requests (here, one for
peaks and one for inverse), and pass those in along with the
checkpoints to this script. This script will write the checkpoint
paths into the inference requests in their `model_checkpoint_path`
field, as well as determine paths for each (here, peaks and inverse)
inference and make those folders.
"""
import argparse
import glob
import os
from os.path import join
from google.protobuf import text_format
from ffn.inference import inference_pb2

# -- args
ap = argparse.ArgumentParser()

ap.add_argument("infreqs", nargs="+")
ap.add_argument("--ckpts", nargs="+")
ap.add_argument("--ckpt-dir")
ap.add_argument("--output-dir")

args = ap.parse_args()

# -- load the inference requests
infreqs = []
for infreq_fn in args.infreqs:
    # load proto
    infreq = inference_pb2.InferenceRequest()
    with open(infreq_fn) as infreq_f:
        text_format.Parse(infreq_f.read(), infreq)

    # validate that this is a PolicyPeaks or PolicyInvertOrigins
    # this script could be extended to other types, but these are
    # what I use.
    assert infreq.seed_policy in ("PolicyPeaks", "PolicyInvertOrigins")

    # ok, save it
    infreqs.append(infreq)

# -- figure out checkpoint paths
ckpt_paths = [
    d.split(".")[0]
    for d in glob.glob(join(args.ckpt_dir, "model.ckpt-*.meta"))
    if any(c in d for c in args.ckpts)
]
assert len(ckpt_paths) == len(args.ckpts)

# make sure order matches
ckpt_paths = sorted(ckpt_paths, key=lambda x: int(x.split("model.ckpt-")[1]))
ckpts = sorted(args.ckpts, key=int)

# -- write InferenceRequests and make folders
invert_policy = "{\"segmentation_dir\": \"{peaks_dir}\"}"
for ckpt, ckpt_path in zip(ckpts, ckpt_paths):
    # paths
    peaks_dir = join(args.output_dir, f"{ckpt}_peaks")
    invert_dir = join(args.output_dir, f"{ckpt}_invert")
    peaks_infreq_fn = join(args.output_dir, f"{ckpt}_peaks.pbtxt")
    invert_infreq_fn = join(args.output_dir, f"{ckpt}_invert.pbtxt")

    # make them
    os.makedirs(peaks_dir, exist_ok=True)
    os.makedirs(invert_dir, exist_ok=True)

    for infreq in infreqs:
        # write model path
        infreq.model_checkpoint_path = ckpt_path

        # write policy-specific stuff, and write to disk
        if infreq.seed_policy == "PolicyPeaks":
            infreq.segmentation_output_dir = peaks_dir

            with open(peaks_infreq_fn, "w") as out:
                out.write(text_format.MessageToString(infreq))

        elif infreq.seed_policy == "PolicyInvertOrigins":
            infreq.segmentation_output_dir = invert_dir
            infreq.seed_policy_args = invert_policy.format(peaks_dir=peaks_dir)

            with open(invert_infreq_fn, "w") as out:
                out.write(text_format.MessageToString(infreq))

        else:
            assert False
