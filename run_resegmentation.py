import numpy as np
import os.path
import logging

from google.protobuf import text_format
from ffn.inference import inference_pb2
from ffn.inference import inference
from ffn.inference import resegmentation
from ffn.inference import resegmentation_analysis

import joblib

from absl import app
from absl import flags


flags.DEFINE_string(
    "resegmentation_request", "", "Path to ResegmentationRequest proto",
)

flags.DEFINE_boolean(
    "analyze_results",
    False,
    "Run this script once without this flag to run the required "
    "resegmentation inferences, then run with this flag set to "
    "analyze the results of those resegmentations.",
)
flags.DEFINE_string(
    "affinities_npy",
    "",
    "If --analyze_results, we will save affinities to this "
    ".npy path in the neuclease merge table format.",
)

VOXEL_SZ = [1, 1, 1]

# mpi-style script level parallelism
flags.DEFINE_integer('rank', -1, 'My worker id.')
flags.DEFINE_integer('nworkers', -1, 'Number of workers.')

FLAGS = flags.FLAGS


# What's this? See:
# github.com/janelia-flyem/neuclease/blob/master/neuclease/merge_table.py
MERGE_TABLE_DTYPE = [('id_a', '<u8'),
                     ('id_b', '<u8'),
                     ('xa', '<u4'),
                     ('ya', '<u4'),
                     ('za', '<u4'),
                     ('xb', '<u4'),
                     ('yb', '<u4'),
                     ('zb', '<u4'),
                     ('score', '<f4')]


def analyze_results():
    """Produce affinities from completed ResegmentationRequest."""
    # Configure logger
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("affinities")

    assert not os.path.exists(FLAGS.affinities_npy)

    # Get the ResegmentationRequest
    resegmentation_request = inference_pb2.ResegmentationRequest()
    with open(FLAGS.resegmentation_request, "r") as reseg_req_f:
        text_format.Parse(reseg_req_f.read(), resegmentation_request)
    npoints = len(resegmentation_request.points)

    results = []
    for i in range(npoints):
        # Get output path for this point
        target_path = resegmentation.get_target_path(resegmentation_request, i)
        logger.info(
            "Processing point %d/%d, with path %s", i, npoints, target_path
        )

        # Analyze...
        pair_resegmentation_result = resegmentation_analysis.evaluate_pair_resegmentation( # noqa
            target_path,
        )

        results.append(pair_resegmentation_result)

    # Build merge table for neuclease
    merge_table = [
        (
            res.id_a,
            res.id_b,
            res.eval.from_a.origin.x,
            res.eval.from_a.origin.y,
            res.eval.from_a.origin.z,
            res.eval.from_b.origin.x,
            res.eval.from_b.origin.y,
            res.eval.from_b.origin.z,
            # XXX Seems like IOU is the score to use?
            res.eval.iou,
        )
        for res in results
    ]
    merge_table = np.array(merge_table, dtype=MERGE_TABLE_DTYPE)
    np.save(FLAGS.affinities_npy, merge_table)



def do_resegmentation():
    """Run inferences specified in a ResegmentationRequest."""
    # Configure logger
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("reseg")

    # Get the ResegmentationRequest
    resegmentation_request = inference_pb2.ResegmentationRequest()
    if FLAGS.resegmentation_request.endswith('txt'):
        logger.info("Loading resegmentation request from text format...")
        with open(FLAGS.resegmentation_request, "r") as reseg_req_f:
            text_format.Parse(reseg_req_f.read(), resegmentation_request)
    else:
        logger.info("Loading resegmentation request from binary format...")
        with open(FLAGS.resegmentation_request, "rb") as reseg_req_f:
            resegmentation_request.ParseFromString(reseg_req_f.read())
    logger.info("Done")

    # Figure out this rank's role (might be the only rank.)
    nworkers = FLAGS.nworkers if FLAGS.nworkers > 0 else 1
    rank = FLAGS.rank if FLAGS.nworkers > 0 else 0
    num_points = len(resegmentation_request.points)
    my_points = list(range(rank, num_points, nworkers))
    nthreads = resegmentation_request.inference.concurrent_requests

    logger.info("%d points to process on %d ranks", num_points, nworkers)
    logger.info(
        "rank %d processing %d points on %d threads",
        rank,
        len(my_points),
        nthreads,
    )

    # Build run scaffold objects
    runner = inference.Runner()
    runner.start(
        resegmentation_request.inference,
        executor_expected_clients=len(my_points),
    )

    # Launch threads
    logger.info("Starting resegmentation")
    with joblib.Parallel(
        nthreads, prefer='threads', verbose=100
    ) as par:
        for _ in par(
            joblib.delayed(resegmentation.process_point)(
                resegmentation_request, runner, i, VOXEL_SZ
            )
            for i in my_points
        ):
            pass
    logger.info("All done.")


def main(unused_argv):
    if FLAGS.analyze_results:
        analyze_results()
    else:
        do_resegmentation()


if __name__ == '__main__':
    app.run(main)
