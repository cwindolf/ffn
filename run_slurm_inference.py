"""Slurm cluster inference

One inference per GPU, no intra-GPU parallelism.

For a given collection of inference requests, this runs each as its own
Slurm job independent from the others. There is no BatchExecutor here,
for that behavior see run_batch_inference.py -- read rest of this doc
to understand the difference.

This script allows you to specify the inferences in two ways. For one,
you can manually pass paths to multiple inference requests to run to
the flag --infreqs.

Inference requests can also be created dynamically by overwriting fields
in a base inference request passed to --infreqs by pointing to some
paths to checkpoints with --ckpts. You can also pass in a --ckpt_dir
containing many checkpoints, and select checkpoints from there by
passing a YAML file to --yaml. The yaml file should have format like:

```yaml
ckpts:
  name_of_first_ckpt: 11111111
  name_of_second_ckpt: 22222222
  name_of_ckpt_that_we_dont_want_to_use: 000000001

cull:
  - name_of_ckpt_that_we_dont_want_to_use
```

In all cases, you can supply either one path to a BoundingBox proto
in --bboxes, in which case that bounding box will be used for all
inferences, or you can pass multiple paths to --bboxes, in which
case they will correspond to each inference request passed to
--infreqs in the order provided.

The use case of this script is to batch out inference over multiple
checkpoints. Since multiple models do not fit in the same GPU, we
cannot run multiple inferences in parallel inside each GPU.

If, on the other hand, you are in a situation where the
same checkpoint is being used for multiple inferences, the script
run_batch_inference.py is able to parallelize over those inferences
inside each GPU, and will probably be faster, or at least more
efficient.
"""
import re
import os
from os.path import abspath, join
import glob

from absl import flags
from absl import app
from google.protobuf import text_format
import yaml

from ffn.inference import inference_pb2
from ffn.utils import slurm_lib


flags.DEFINE_spaceseplist(
    "infreqs",
    [],
    'Try --infreqs="$(echo <glob>)". ' "(space separated list of filenames)",
)
flags.DEFINE_spaceseplist(
    "bboxes",
    [],
    "One bbox means same bbox for all infreqs. Or, supply one bbox "
    "for each infreq.",
)
flags.DEFINE_string(
    "out_dir",
    "",
    "Where to write output to when ckpt_dir and n_infs are supplied",
)

# yaml interface
flags.DEFINE_string(
    "ckpt_dir",
    "",
    "Optionally, just supply a single infreq and a single bbox, "
    "and --n_infs inferences will be run using evenly spaced "
    "checkpoints from this directory.",
)
flags.DEFINE_spaceseplist(
    "ckpts",
    [],
    "Manually specify ffn ckpts. This overrides ckpt_dir and whatever "
    "was in the infreq originally.",
)

flags.DEFINE_string(
    "yaml",
    "",
    "Manually specify ffn ckpts and names in a yaml file. This "
    "overrides ckpt_dir and whatever was in the infreq originally,"
    "as well as --ckpts",
)

flags.DEFINE_string(
    "forward_dir",
    "",
    "Interpret infreqs as inverse, and fill in the forward paths from "
    "dirs in here.",
)

flags.DEFINE_enum(
    "where",
    "local",
    ["local", "cluster"],
    "What machines should run the job?",
)
flags.DEFINE_string(
    "t", "", "-t for slurm jobs.",
)

FLAGS = flags.FLAGS


def main(_):
    # validate args
    if FLAGS.yaml or FLAGS.ckpts:
        assert len(FLAGS.infreqs) == 1
    assert len(FLAGS.infreqs) > 0
    assert len(FLAGS.bboxes) in (1, len(FLAGS.infreqs))

    # apply --yaml or --ckpts if present
    if FLAGS.yaml:
        with open(FLAGS.yaml, "r") as ymlf:
            yml = yaml.load(ymlf)
        name2ckpt = yml["ckpts"]
        if "cull" in yml:
            cull = yml["cull"]
            name2ckpt = {n: c for n, c in name2ckpt.items() if n not in cull}
        names = list(name2ckpt.keys())
        ckpt_inds = list(name2ckpt.values())
        print(ckpt_inds)
        ckpt_paths = glob.glob(join(FLAGS.ckpt_dir, "model.ckpt-*.meta"))
        ckpt_paths = [p.strip(".meta") for p in ckpt_paths]
        ckpts = []
        for c in ckpt_inds:
            for p in ckpt_paths:
                if str(c) in p:
                    ckpts.append(p)
        # could fail if not all ckpt_inds have same length,
        # then you would need to pad with zeros, TODO.
        assert len(ckpt_inds) == len(ckpts)
        ckpts.sort(key=lambda c: int(re.findall(r"\d+", c)[-1]))
    elif FLAGS.ckpts:
        ckpts = FLAGS.ckpts
        ckpt_inds = [re.findall(r"\d+", c)[-1] for c in ckpts]
        names = None
    else:
        # in this case we just use the --infreqs themselves
        ckpts = None
        ckpt_inds = None
        names = None

    # if --yaml or --ckpts, let's build new infreqs
    if ckpts:
        # Parse prototype infreq
        inf = inference_pb2.InferenceRequest()
        text_format.Parse(open(FLAGS.infreqs[0]).read(), inf)

        # Make inference requests for each of these
        infreqs = []
        for i, c in enumerate(ckpts):
            name = names[i] if names else ""
            # patch checkpoint path
            inf.model_checkpoint_path = c
            # patch output path
            nbatch = re.findall(r"\d+", c)[-1]
            subfolder = f"{nbatch}_res"
            inf.segmentation_output_dir = join(
                abspath(FLAGS.out_dir), subfolder
            )
            # if this is an inverse inference, patch seed
            # policy arguments with the forward inference
            if FLAGS.forward_dir:
                forward_dir = [
                    d
                    for d in os.listdir(FLAGS.forward_dir)
                    if str(ckpt_inds[i]) in d
                ]
                if len(forward_dir) == 0:
                    continue
                if len(forward_dir) > 1:
                    raise ValueError(
                        "Found", len(forward_dir), "candidates for"
                        "forward directory for ckpt", name, ckpt_inds[i],
                        "which were", forward_dir
                    )
                forward_dir = join(FLAGS.forward_dir, forward_dir[0])
                # filter on existence of forward segmentation
                if glob.glob(join(forward_dir, "*/*/seg*.npz")):
                    print(
                        "Found forward", forward_dir, "for", ckpt_inds[i],
                    )
                else:
                    print("No seg for", forward_dir, "skipping", c)
                    continue
                inf.seed_policy_args = (
                    f'{{"segmentation_dir": "{forward_dir}"}}'  # noqa
                )
            os.makedirs(inf.segmentation_output_dir, exist_ok=True)
            infreq_text = text_format.MessageToString(inf)
            # write infreq to the output folder for posterity
            with open(
                join(inf.segmentation_output_dir, "infreq.pbtxt"), "w"
            ) as infreq_f:
                infreq_f.write(infreq_text)
            infreqs.append(infreq_text)

    # -- run
    res_dirs = slurm_lib.srun_multiple_inferences(
        infreqs,
        FLAGS.bboxes,
        local=FLAGS.where == "local",
        slurm_kwargs={"-t", FLAGS.t},
    )
    print("Ok, all done. Segs have been saved to:")
    print("\n".join(res_dirs))


if __name__ == "__main__":
    app.run(main)
