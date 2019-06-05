'''
For a given collection of inference requests, this runs each as its own
slurm job.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import subprocess
import re
import os.path
import glob

import numpy as np

from tensorflow import gfile

from absl import flags
from absl import app

from google.protobuf import text_format
from ffn.inference import inference_pb2
from ffn.utils import bounding_box_pb2
from ffn.inference import storage


flags.DEFINE_spaceseplist(
    'infreqs', [],
    'Try --infreqs="$(echo <glob>)". '
    '(space separated list of filenames)')
flags.DEFINE_spaceseplist(
    'bboxes', [],
    'One bbox means same bbox for all infreqs. Or, supply one bbox '
    'for each infreq.')

flags.DEFINE_string(
    'ckpt_dir', "",
    'Optionally, just supply a single infreq and a single bbox, '
    'and --n_infs inferences will be run using evenly spaced '
    'checkpoints from this directory.')
flags.DEFINE_integer(
    'n_infs', 0,
    'Use with ckpt_dir and out_dir. If --n_infs=1, the latest ckpt is used.')
flags.DEFINE_string(
    'out_dir', '',
    'Where to write output to when ckpt_dir and n_infs are supplied')


flags.DEFINE_integer(
    'batch_size', 1,
    'Batch size for inference')
flags.DEFINE_integer(
    'n_concurrent', 1,
    'Number of threas in executor?')

FLAGS = flags.FLAGS


def read(fn):
    with open(fn, 'r') as f:
        res = ' '.join(f.read().split())
    return res


def worker(i_and_b):
    infreq, bbox = i_and_b

    # Bail out early under the same circumstances that ffn Runner
    # does. Copied from `Runner.run` method.
    # This is nice because you can glob infreqs in bash and not worry
    # about doing the same job twice.
    bbox_pb = bounding_box_pb2.BoundingBox()
    text_format.Parse(bbox, bbox_pb)
    infreq_pb = inference_pb2.InferenceRequest()
    text_format.Parse(infreq, infreq_pb)
    seg_dir = infreq_pb.segmentation_output_dir
    corner = (bbox_pb.start.z, bbox_pb.start.y, bbox_pb.start.x)
    seg_path = storage.segmentation_path(seg_dir, corner)
    if gfile.Exists(seg_path):
        print(f'{seg_path} already exists.')
        return 2

    print(f'Launching job for {seg_path}')

    res = subprocess.run(
        ['srun',
         '--job-name', f'inf{bbox_pb.size.x}',
         '-n', '1',
         '-p', 'gpu',
         '--gres', 'gpu:1',
         '--constraint', 'v100',
         '-c', '12',
         # '--output', out,
         # '--error', err,
         #
         'python',
         'run_inference.py',
         '--inference_request', infreq,
         '--bounding_box', bbox])

    print(f'    Done with {seg_path}...')

    return res.returncode, seg_dir


def main(_):
    infreqs = list(map(read, FLAGS.infreqs))
    bboxes = list(map(read, FLAGS.bboxes))

    # Two behaviors:
    # (a) n_infs > 0 and there is 1 infreq and 1 bbbox and a ckpt_dir
    # (b) just running the supplied infreqs and bboxes.

    # Make infreqs for (a) and then let (b) code deal with bbox assignment
    if (FLAGS.n_infs > 0 and FLAGS.ckpt_dir and FLAGS.out_dir
        and len(infreqs) == len(bboxes) == 1):
        # Get all checkpoints in ckpt_dir
        ckpts = glob.glob(os.path.join(FLAGS.ckpt_dir, 'model.ckpt-*.meta'))
        ckpts = [p.strip('.meta') for p in ckpts]
        ckpts.sort(key=lambda c: int(re.findall(r'\d+', c)[-1]))

        # Extract n_infs of them
        if FLAGS.n_infs == 1:
            # Use the latest if only doing 1.
            ckpts = [ckpts[-1]]
        else:
            choice_inds = np.linspace(0, len(ckpts) - 1,
                                      num=FLAGS.n_infs, dtype=int)
            ckpts = np.asarray(ckpts)[choice_inds]

        # Parse prototype infreq
        inf = inference_pb2.InferenceRequest()
        text_format.Parse(infreqs[0], inf)

        # Make inference requests for each of these
        infreqs = []
        for c in ckpts:
            # Patch checkpoint
            inf.model_checkpoint_path = c
            # Patch output path
            nbatch = re.findall(r'\d+', c)[-1]
            subfolder = f'{nbatch}_res'
            inf.segmentation_output_dir = os.path.join(
                os.path.abspath(FLAGS.out_dir), subfolder)
            # Fix up batchsz
            inf.batch_size = FLAGS.batch_size
            inf.concurrent_requests = FLAGS.n_concurrent
            # Done
            infreqs.append(text_format.MessageToString(inf))

    assert infreqs
    assert len(bboxes) in (1, len(infreqs))

    # Apply the same bbox to all infreqs if there is only one
    if len(bboxes) == 1:
        bboxes = bboxes * len(infreqs)

    job_args = zip(infreqs, bboxes)
    segdirs = []

    with multiprocessing.Pool(len(infreqs)) as p:
        for res, segdir in p.imap_unordered(worker, job_args):
            print(f'    {res}', flush=True)
            segdirs.append(segdir)

    print('Ok, all done. '
          'You might like to know that the segs have been saved to:')
    print('\n'.join(segdirs))


if __name__ == '__main__':
    app.run(main)
