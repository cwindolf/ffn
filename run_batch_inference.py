"""Runs FFN inference by batching over all subvolumes in a bounding box.

Inference is performed with lots of threads.

This script supports multiple inference requests at once, as long
as they share a model (and they are forced to share a global bbox
and subvolume structure). Each subvolume will have each inference
request run on it in the order they were provided. Really the only
thing allowed is different seed policies.

These are supported by launching a runner per inference request,
but making sure that the runners all share an executor and session
and model etc.
"""
import os
import time
import logging
import itertools
import numpy as np
from multiprocessing.pool import ThreadPool

from google.protobuf import text_format
from absl import app
from absl import flags
from tensorflow import gfile

from ffn.utils import bounding_box_pb2
from ffn.utils import bounding_box
from ffn.inference import inference_pb2
from ffn.inference import inference

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'inference_requests',
    None,
    'One or more InferenceRequest pbtxt filenames, space separated.'
)
flags.DEFINE_string(
    'bounding_box',
    None,
    'BoundingBox pbtxt file.',
)
flags.DEFINE_integer('subvolume_size', -1, '')
flags.DEFINE_integer('subvolume_overlap', 48, '')
flags.DEFINE_integer('rank', -1, 'My worker id.')
flags.DEFINE_integer('nworkers', -1, 'Number of workers.')
flags.DEFINE_boolean(
    'advise',
    False,
    'Don\'t run anything, just give advice.'
)


def _thread_main(runners_and_bbox):
    thread_start = time.time()
    runners, bbox = runners_and_bbox
    logging.info(
        f'Thread calling run at (zyx)={bbox.start}, '
        f'(dzyx)={bbox.size}.'
    )
    for irun, runner in enumerate(runners):
        logging.info('Thread starting run %d/%d', irun + 1, len(runners))
        runner.run(bbox.start, bbox.size)
        logging.info('Thread finished run %d/%d', irun + 1, len(runners))
    return time.time() - thread_start


def infer():
    requests = []
    for infreq_filename in FLAGS.inference_requests.split():
        request = inference_pb2.InferenceRequest()
        with open(infreq_filename) as infreq_f:
            text_format.Parse(infreq_f.read(), request)
        if not gfile.Exists(request.segmentation_output_dir):
            gfile.MakeDirs(request.segmentation_output_dir)
        requests.append(request)

    # Some asserts to make sure things don't go haywire
    batch_size = requests[0].batch_size
    assert all(r.batch_size == batch_size for r in requests)
    concurrent_requests = requests[0].concurrent_requests
    assert all(r.concurrent_requests == concurrent_requests for r in requests)
    model_name = requests[0].model_name
    assert all(r.model_name == model_name for r in requests)
    model_args = requests[0].model_args
    assert all(r.model_args == model_args for r in requests)
    model_checkpoint_path = requests[0].model_checkpoint_path
    assert all(
        r.model_checkpoint_path == model_checkpoint_path for r in requests
    )

    # Global bounding box ---------------------------------------------
    outer_bbox_pb = bounding_box_pb2.BoundingBox()
    if FLAGS.bounding_box:
        with open(FLAGS.bounding_box) as bbox_f:
            text_format.Parse(bbox_f.read(), outer_bbox_pb)
    # Cast to fancy bounding box with subvolume calculations
    outer_bbox = bounding_box.BoundingBox(outer_bbox_pb)

    # Subvolumes ------------------------------------------------------
    if FLAGS.subvolume_size < 0:
        svsize = outer_bbox.size
    else:
        svsize = [FLAGS.subvolume_size] * 3

    # NB: Right now we're truncating the volume when
    # subvols don't fit. See svcalc's kwargs to fix that up.
    svcalc = bounding_box.OrderlyOverlappingCalculator(
        outer_bbox, svsize, [FLAGS.subvolume_overlap] * 3
    )
    nsb = svcalc.num_sub_boxes()
    subvols = svcalc.generate_sub_boxes()

    # If we are one of many workers, take the ith subvol
    # only when (i mod nworkers) == rank
    # TODO: This method suffers from load balance problems.
    if FLAGS.nworkers > 0 and FLAGS.rank >= 0:
        assert FLAGS.rank < FLAGS.nworkers
        subvols = itertools.islice(subvols, FLAGS.rank, nsb, FLAGS.nworkers)
        # Compute correct number of subvolumes for this worker
        # nsb // nworkers can give the wrong answer.
        nsb = len(range(*slice(FLAGS.rank, nsb, FLAGS.nworkers).indices(nsb)))

    # Figure out how many threads
    if concurrent_requests < 0:
        concurrent_requests = nsb
    elif concurrent_requests < request.batch_size:
        raise ValueError(
            'Please let concurrent_requests < 0 or '
            'concurrent_requests >= batch_size.'
        )
    # Update infreq so runner doesn't freak out if something changed.
    for request in requests:
        request.concurrent_requests = concurrent_requests

    # Initialize inference runner -------------------------------------
    runner0 = inference.Runner()
    runner0.start(
        requests[0],
        executor_expected_clients=len(requests) * nsb,
    )
    runners = [runner0]
    for request in requests:
        runner = inference.Runner()
        runner.start(
            request,
            session=runner0.session,
            model=runner0.model,
            executor=runner0.executor,
        )
        runners.append(runner)

    # Main loop -------------------------------------------------------
    # Log some details
    if FLAGS.nworkers > 0 and FLAGS.rank >= 0:
        logging.info(
            f'Hi. This is rank {FLAGS.rank} out of '
            f'{FLAGS.nworkers} workers.'
        )
    logging.info(f'Launching worker threads.')
    logging.info(
        f'{nsb} subvolumes to get through with '
        f'{concurrent_requests} workers sharing '
        f'{batch_size} slots, and {len(requests)} '
        f'infreqs to run per subvol.'
    )

    # Start threads
    start_time = time.time()
    job_args = zip(itertools.repeat(runners), subvols)
    dts = []
    with ThreadPool(concurrent_requests) as pool:
        for dt in pool.imap_unordered(_thread_main, job_args):
            dts.append(dt)
            logging.info(
                f'gathered run in {dt:0.3f}s. '
                f'nrun: {len(dts)}. expected total: {nsb}'
            )
    logging.info(f'total nrun: {len(dts)}. expected: {nsb}')
    end_time = time.time()

    # Log time info to see how bad the load balance is.
    time_hrs = (end_time - start_time) / 60 / 60
    min_per_seg = (end_time - start_time) / 60 / len(dts)
    logging.info(f'Took {time_hrs:0.3f} hours.')
    logging.info(f'That\'s {min_per_seg:0.3f} mins per subvolume.')

    logging.info('Timing from inside the threads. All in minutes.')
    dts = np.array(dts) / 60
    logging.info(
        f'min={dts.min():0.3f}, mean={dts.mean():0.3f}, '
        f'max={dts.max():0.3f}. std={dts.std()}.'
    )

    # Save counters for fun -------------------------------------------
    for runner, request in zip(runners, requests):
        counter_path = os.path.join(
            request.segmentation_output_dir, 'counters.txt'
        )
        if not gfile.Exists(counter_path):
            runner.counters.dump(counter_path)

    # Clean up --------------------------------------------------------
    logging.info('Done. Stopping executor.')
    del runner


def advise():
    outer_bbox_pb = bounding_box_pb2.BoundingBox()
    if FLAGS.bounding_box:
        with open(FLAGS.bounding_box) as bbox_f:
            text_format.Parse(bbox_f.read(), outer_bbox_pb)
    outer_bbox = bounding_box.BoundingBox(outer_bbox_pb)
    if FLAGS.subvolume_size < 0:
        svsize = outer_bbox.size
    else:
        svsize = [FLAGS.subvolume_size] * 3
    svcalc = bounding_box.OrderlyOverlappingCalculator(
        outer_bbox, svsize, [FLAGS.subvolume_overlap] * 3
    )
    nsb = svcalc.num_sub_boxes()
    print(f'num subvols={nsb}.')


def main(unused_argv):
    if FLAGS.advise:
        advise()
    else:
        infer()


if __name__ == '__main__':
    app.run(main)
