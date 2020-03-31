"""
I recommend running this command inside GNU parallel like:

 $ parallel --lb \
   "srun $(SRUNFLAGS) python run_resegmentation.py \
       run \
       --resegmentation_request XXX \
       --rank {} --nworkers 4" \
   ::: 0 1 2 3

for easy rank-based parallelism.
"""
import argparse
import numpy as np
import os.path
import joblib
import h5py
import logging

from google.protobuf import text_format
from ffn.inference import inference_pb2
from ffn.inference import inference
from ffn.inference import resegmentation
from ffn.inference import resegmentation_analysis
from ffn.utils import geom_utils


# This is used by resegmentation during EDT to specify
# anisotropy of the metric. Our voxels are isotropic so
# not worried about setting this to physical units.
VOXEL_SZ = [1, 1, 1]


# What's this? See:
# github.com/janelia-flyem/neuclease/blob/master/neuclease/merge_table.py
MERGE_TABLE_DTYPE = [
    ("id_a", "<u8"),
    ("id_b", "<u8"),
    ("xa", "<u4"),
    ("ya", "<u4"),
    ("za", "<u4"),
    ("xb", "<u4"),
    ("yb", "<u4"),
    ("zb", "<u4"),
    ("score", "<f4"),
]


def get_resegmentation_request(resegmentation_request_path):
    resegmentation_request = inference_pb2.ResegmentationRequest()
    if resegmentation_request_path.endswith("txt"):
        logging.info("Loading resegmentation request from text format...")
        with open(resegmentation_request_path, "r") as reseg_req_f:
            text_format.Parse(reseg_req_f.read(), resegmentation_request)
    else:
        logging.info("Loading resegmentation request from binary format...")
        with open(resegmentation_request_path, "rb") as reseg_req_f:
            resegmentation_request.ParseFromString(reseg_req_f.read())
    return resegmentation_request


def analyze_results(reseg_req_path, affinities_npy, bigmem, nthreads):
    """Produce affinities from completed ResegmentationRequest."""
    # Configure logger
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("affinities")

    assert bool(affinities_npy)
    assert not os.path.exists(affinities_npy)
    assert affinities_npy.endswith(".npy")

    # Get the ResegmentationRequest
    resegmentation_request = get_resegmentation_request(reseg_req_path)
    npoints = len(resegmentation_request.points)

    # Load up seg volume
    (
        path,
        dataset,
    ) = resegmentation_request.inference.init_segmentation.hdf5.split(":")
    init_segmentation = h5py.File(path, "r")[dataset]
    if bigmem:
        logger.info("Loading segmentation into memory, since you asked.")
        init_segmentation = init_segmentation[:]

    # Other params
    reseg_radius = geom_utils.ToNumpy3Vector(resegmentation_request.radius)[
        ::-1
    ]
    analysis_radius = geom_utils.ToNumpy3Vector(
        resegmentation_request.analysis_radius
    )[::-1]

    def analyze_point(i):
        # Get output path for this point
        target_path = resegmentation.get_target_path(
            resegmentation_request, i, return_early=False
        )
        logger.info(
            "Processing point %d/%d, with path %s", i, npoints, target_path
        )

        # Analyze...
        pair_resegmentation_result = resegmentation_analysis.evaluate_pair_resegmentation(
            target_path,
            init_segmentation,
            reseg_radius,
            analysis_radius,
            VOXEL_SZ,
        )

        return pair_resegmentation_result

    results = []
    nthreads = -1 if nthreads == 0 else nthreads
    with joblib.Parallel(nthreads, require="sharedmem", verbose=100) as par:
        for result in par(
            joblib.delayed(analyze_point)(i) for i in range(npoints)
        ):
            results.append(result)

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
    np.save(affinities_npy, merge_table)


def run_inferences(reseg_req_path, nthreads, rank, nworkers):
    """Run inferences specified in a ResegmentationRequest."""
    # Configure logger
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(f"[reseg rank{rank}]")

    # Get the ResegmentationRequest
    resegmentation_request = get_resegmentation_request(reseg_req_path)

    # Figure out this rank's role (might be the only rank.)
    num_points_total = len(resegmentation_request.points)
    my_points = list(range(rank, num_points_total, nworkers))
    if nthreads == 0:
        nthreads = resegmentation_request.inference.concurrent_requests
    else:
        logger.info(
            "Overriding number of threads set in proto. Was %d, now %d.",
            resegmentation_request.inference.concurrent_requests,
            nthreads,
        )
    logger.info("%d points to process on %d ranks", num_points_total, nworkers)
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
    with joblib.Parallel(nthreads, prefer="threads", verbose=100) as par:
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


if __name__ == "__main__":
    # ----------------------------- args ------------------------------
    ap = argparse.ArgumentParser()

    # We'll use a subparser interface to break up the steps.
    sp = ap.add_subparsers(dest="task")

    # Step 1: Resegmentation request runner ---------------------------
    run_p = sp.add_parser(
        "run",
        help="Step 1: Run the inferences specified in the "
        "ResegmentationRequest.",
    )
    run_p.add_argument(
        "--resegmentation_request",
        help="Path to (binary or text) ResegmentationRequest proto.",
    )
    run_p.add_argument(
        "--nthreads",
        type=int,
        default=0,
        help="Number of threads per rank (overrides concurrent_requests field "
        "in the proto if != 0).",
    )
    mpi_g = run_p.add_argument_group(
        title="Inter-node Parallelism",
        description="MPI style rank-based multi node parallelism. Work is "
        "split naively across the ranks.",
    )
    mpi_g.add_argument("--rank", type=int, default=0)
    mpi_g.add_argument("--nworkers", type=int, default=1)

    # Step 2: Analysis / merge table builder --------------------------
    ana_p = sp.add_parser(
        "analyze",
        help="Step 2: Analyze results from run step and spit out merge table.",
    )
    ana_p.add_argument(
        "--resegmentation_request",
        help="Path to (binary or text) ResegmentationRequest proto.",
    )
    ana_p.add_argument(
        "--affinities_npy",
        help="Place to save the affinities (aka merge table for neuclease).",
    )
    ana_p.add_argument(
        "--nthreads", help="Number of analysis threads.",
    )
    ana_p.add_argument("--bigmem", type=bool, help="Load whole seg into mem.")

    args = ap.parse_args()

    # ------------------------- run the task --------------------------
    if args.task == "run":
        run_inferences(
            args.resegmentation_request,
            args.nthreads,
            args.rank,
            args.nworkers,
        )
    elif args.task == "analyze":
        analyze_results(
            args.resegmentation_request,
            args.affinities_npy,
            args.bigmem,
            args.nthreads,
        )
