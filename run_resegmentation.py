"""run_resegmentation.py

Try
 $ python run_resegmentation.py --help

This script handles all of the business related to affinities. This
involves the following 4 steps, each of which has a corresponding
subcommand.

[Step 0]
      Figure out which pairs of supervoxels should be candidates for
      merges and the point at which a targeted inference should be run
      to determine their affinity. Writes a ResegmentationRequest proto
      for later steps to process. Run
            $ python run_resegmentation.py build_req --help
      to see how.
[Step 1]
    Run resegmentation to determine affinities. This writes a ton of
    .npz files with targeted inference results which Step 2 will
    process.
        $ python run_resegmentation.py run --help
    will tell you more. But this is the most expensive step! It can be
    an absolutely huge number of inferences! You gotta parallelize it.
    So, this step supports multi-node parallelism in addition to intra-
    node parallelism. To use the multi-node, I recommend running inside
    GNU parallel:
         $ parallel --lb \
           "srun $(SRUNFLAGS) python run_resegmentation.py \
               run \
               --resegmentation_request XXX \
               --rank {} --nworkers 4" \
           ::: seq 0 3
    Except maybe with more workers...haha. Hope you have a lot of GPUs.
[Step 2]
    Analyze the results of those resegmentation inferences, and write a
    table of affinities, aka a "merge table" in neuclease jargon.
        $ python run_resegmentation.py analyze --help
[Step 3]
    Given an affinity threshold in (0, 1), merge all neurites with
    affinity greater than the threshold. Saves the resulting supervoxel
    to body mapping to disk. Can optionally preserve some previously
    existing mappings.
        $ python run_resegmentation.py automerge --help
"""
import argparse
import json
import logging
import os
import time

import h5py
import joblib
import networkx as nx
import numpy as np
import pandas as pd
from google.protobuf import text_format

import neuclease.merge_table
import neuclease.dvid

from ffn.inference import inference_pb2
from ffn.utils import bounding_box_pb2
from ffn.utils import geom_utils
from ffn.utils import pair_detector
from ffn.inference import inference
from ffn.inference import resegmentation
from ffn.inference import resegmentation_analysis

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

# ------------------------------ library ------------------------------


def shist(arr, bins="auto", width=72):
    """A basic string histogram for logging

    Modified from @tammoippen's crappyhist.py

    Arguments
    ---------
    arr : np.array to get histogram of
    bins : passed through to np.histogram
    width : width of generated string

    Returns: histogram string. print it out!
    """
    # Compute and normalize histogram
    hist, edges = np.histogram(arr, bins=bins)
    hist = hist / hist.max()

    # Text padding details for this particular data
    # Any need for a negative size?
    hasneg = (edges < 0).any()
    # How much space will printing bin edges take up?
    intlen = hasneg + len(str(int(np.ceil(np.abs(edges).max()))))
    if np.issubdtype(arr.dtype, np.integer):
        declen = 0
    else:
        declen = 3
    # the 3 is for ' | '. the declen > 0 is for the decimal point.
    totlen = intlen + (declen > 0) + declen + 3
    # How much space is left over for the histogram?
    histlen = width - totlen

    # Loop to build output string
    messages = [
        f"min: {arr.min():3g}, "
        f"median: {np.median(arr):3g}, "
        f"max: {arr.max():3g}"
    ]
    for freq, edge in zip(hist, edges):
        # No divider to indicate frequency of 0
        cchar = " " if freq == 0 else "|"
        barlen = int(freq * histlen)
        messages.append(f'{edge:{intlen}.{declen}f} {cchar} {"#" * barlen}')

    return "\n".join(messages)


class timer:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        print(f"{self.message} Took {time.time() - self.start:3g} s.")


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


# --------------- step 0: build resegmentation request ----------------


def build_reseg_req(
    resegmentation_request_path,
    inference_request_path,
    bounding_box_path,
    init_segmentation_h5,
    output_directory,
    subdir_digits,
    max_retry_iters,
    segment_recovery_fraction,
    max_distance,
    radius,
    nthreads,
    bigmem,
):
    logger = logging.getLogger("buildrr")
    # Make sure good ext -- we use ext to decide behavior.
    assert resegmentation_request_path.split(".")[-1] in ("pb", "pbtxt")

    # Load up inference request
    inference_request = inference_pb2.InferenceRequest()
    if inference_request_path:
        with open(inference_request_path) as inference_request_f:
            text_format.Parse(inference_request_f.read(), inference_request)

    # Load up bounding box
    # We compute pair resegmentation points for all neurons
    # that intersect with the bounding box, but provide no
    # guarantees about what will happen just outside the boundary.
    # There might be some points there or there might not.
    bbox = bounding_box_pb2.BoundingBox()
    with open(bounding_box_path) as bbox_f:
        text_format.Parse(bbox_f.read(), bbox)

    # Reseg points ----------------------------------------------------
    # Build a data structure that will help us quickly check
    # whether two segments cannot possibly overlap.
    pd = pair_detector.PairDetector(
        init_segmentation_h5,
        bbox,
        nthreads,
        max_distance=max_distance,
        bigmem=bigmem,
    )
    logger.info(
        "Points were in bounding box: %s-%s", pd.min_approach, pd.max_approach,
    )

    # List points
    resegmentation_points = []
    for id_a, id_b, point in pd.pairs_and_points():
        # Build ResegmentationPoint proto
        rp = inference_pb2.ResegmentationPoint()
        rp.id_a = id_a
        rp.id_b = id_b
        # PairDetector gets things XYZ ordered.
        rp.point.x, rp.point.y, rp.point.z = point
        # OK bai
        resegmentation_points.append(rp)

    # Build the ResegmentationRequest ---------------------------------
    logger.info("Building the ResegmentationRequest...")

    # Some fields we set with the constructor...
    resegmentation_request = inference_pb2.ResegmentationRequest(
        inference=inference_request,
        points=resegmentation_points,
        output_directory=output_directory,
        subdir_digits=subdir_digits,
        max_retry_iters=max_retry_iters,
        segment_recovery_fraction=segment_recovery_fraction,
    )
    # Some (nested) fields are easier to just set.
    # Patch the inference request to point to the initial segmentation
    resegmentation_request.inference.init_segmentation.hdf5 = (
        init_segmentation_h5
    )
    # Resegmentation and analysis radius
    resegmentation_request.radius.x = radius
    resegmentation_request.radius.y = radius
    resegmentation_request.radius.z = radius
    # Following suggestion in a comment in ResegmentationRequest proto,
    # compute analysis radius by subtracting FFN's FOV radius from
    # the resegmentation radius.
    model_args = json.loads(inference_request.model_args)
    ffn_fov_radius = model_args.get("fov_size", [24, 24, 24])
    resegmentation_request.analysis_radius.x = radius - ffn_fov_radius[0]
    resegmentation_request.analysis_radius.y = radius - ffn_fov_radius[1]
    resegmentation_request.analysis_radius.z = radius - ffn_fov_radius[2]
    # Write request to output file
    if resegmentation_request_path.endswith(".pbtxt"):
        # Text output for humans
        with open(resegmentation_request_path, "w") as out:
            out.write(text_format.MessageToString(resegmentation_request))
    elif resegmentation_request_path.endswith(".pb"):
        # Binary output for robots
        with open(resegmentation_request_path, "wb") as out:
            out.write(resegmentation_request.SerializeToString())
    else:
        assert False
    logger.info(f"OK, I wrote {resegmentation_request_path}. Bye...")


# --------------- step 1: run resegmentation inferences ---------------


def run_inferences(reseg_req_path, nthreads, rank, nworkers):
    """Run inferences specified in a ResegmentationRequest."""
    # Configure logger
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
            pass  # no-op to consume generator.
    logger.info("All done.")


# ----- step 2: analyze results from step 1 and make merge table ------


def analyze_results(reseg_req_path, affinities_npy, bigmem, nthreads):
    """Produce affinities from completed ResegmentationRequest."""
    # Crash now rather than after doing all the work
    assert bool(affinities_npy)
    assert not os.path.exists(affinities_npy)
    assert affinities_npy.endswith(".npy")

    # Configure logger
    logger = logging.getLogger("affinities")

    # Get the ResegmentationRequest
    resegmentation_request = get_resegmentation_request(reseg_req_path)
    npoints = len(resegmentation_request.points)

    # Load up seg volume
    init_seg_path = resegmentation_request.inference.init_segmentation.hdf5
    path, dataset = init_seg_path.split(":")
    init_segmentation = h5py.File(path, "r")[dataset]
    if bigmem:
        logger.info("Loading segmentation into memory, since you asked.")
        init_segmentation = init_segmentation[:]

    # Other params
    reseg_radius = geom_utils.ToNumpy3Vector(resegmentation_request.radius)
    analysis_radius = geom_utils.ToNumpy3Vector(
        resegmentation_request.analysis_radius
    )
    # These are XYZ in the proto, but we need ZYX.
    reseg_radius = reseg_radius[::-1]
    analysis_radius = analysis_radius[::-1]

    def analyze_point(i):
        # Get output path for this point
        target_path = resegmentation.get_target_path(
            resegmentation_request, i, return_early=False
        )
        logger.info(
            "Processing point %d/%d, with path %s", i, npoints, target_path
        )

        # Analyze...
        pair_resegmentation_result = resegmentation_analysis.evaluate_pair_resegmentation(  # noqa
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
    logger.info("Done. Building merge table to save to %s", affinities_npy)
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


# ---------------------- step 3: automatic merge ----------------------


def automerge(
    affinities_npy, mappings_output, threshold, preserve_mappings
):
    """
    Many thanks to Stuart Berg @stuarteberg for the code. To get this
    working please clone https://github.com/janelia-flyem/neuclease/
    and pip install .
    You might also need to `conda install -c flyem-forge dvidutils`
    Of course the whole concept of automerging is kind of useless
    without neuclease, so you probably already have it.
    """
    assert 0 < threshold < 1
    assert mappings_output.endswith(".pkl")
    assert preserve_mappings is None or preserve_mappings.endswith(".pkl")

    # Figure out supervoxels that should not be touched ---------------
    preserve_sv = np.array([])
    if preserve_mappings is not None:
        original_mappings = pd.Series.from_pickle(preserve_mappings)
        preserve_sv = np.ascontiguousarray(original_mappings.index)

    # Decide on automerge ---------------------------------------------
    # neuclease will normalize the merge table a little for us
    merge_table = neuclease.merge_table.load_merge_table(affinities_npy)
    open_to_merge = merge_table[
        np.logical_not(
            np.isin(merge_table["id_a"], preserve_sv)
            | np.isin(merge_table["id_b"], preserve_sv)
        )
    ]
    thresholded = open_to_merge[open_to_merge["score"] > threshold]

    # Get all supervoxel IDs present in merge table
    all_svids = np.union1d(
        merge_table["id_a"].values, merge_table["id_b"].values
    )
    nsvs = all_svids.size
    del all_svids
    merge_svids = np.union1d(
        thresholded["id_a"].values, thresholded["id_b"].values
    )
    merge_svids.sort()
    assert merge_svids[0] > 0  # We should not be getting background here.
    nmergedsvs = merge_svids.size
    print(
        f"Attempting to merge {nmergedsvs} svs out of {nsvs} total, "
        f"where {open_to_merge.index.size} are not being perserved."
    )

    # Cluster using connected components (networkx)
    with timer("Clustered."):
        edges = [(row.id_a, row.id_b) for row in thresholded.itertuples()]
        merges = [
            sorted(cc) for cc in nx.connected_components(nx.Graph(edges))
        ]
        csizes = np.array([len(merge) for merge in merges])

    # Log stats
    print(
        f"Agglomeration merged {nmergedsvs} out of {nsvs} original "
        f"supervoxels into {len(merges)} neurites. That means the new "
        f"neurites have a mean of {nmergedsvs / len(merges):3g} svs, "
        f"median of {np.median(csizes):3g} and stddev of "
        f"{np.std(csizes):3g} svs. Smallest and largest contain "
        f"{min(csizes)}, {max(csizes)} supervoxels."
    )
    print("Histogram of cluster sizes:")
    print(shist(csizes, bins=16))
    assert all(sv in merge_svids for merge in merges for sv in merge)

    # Upload new mappings ---------------------------------------------
    with timer("Made mapping series."):
        # Make pd.Series with the new labels
        svids = [svid for merge in merges for svid in merge]
        bodies = [merge[0] for merge in merges for _ in merge]
        mappings = pd.Series(data=bodies, index=svids)

        # concatenate preserve_mappings
        if preserve_mappings is not None:
            mappings = pd.concat(original_mappings, mappings)

    with timer("Saved mappings to disk."):
        mappings.to_pkl(mappings_output)


# ------------------------------- main --------------------------------
if __name__ == "__main__":
    # ----------------------------- args ------------------------------
    ap = argparse.ArgumentParser()

    # We'll use a subparser interface to break up the steps.
    subparsers = ap.add_subparsers(dest="task")

    # Step 0: Detect pair candidates and build ResegmentationRequest --
    brr_p = subparsers.add_parser(
        "build_req",
        help="Step 0: Detect possible pairs and make a Resegmentation"
        "Request proto to pass to later steps.",
    )
    brr_p.add_argument(
        "--resegmentation_request",
        help="Path to write ResegmentationRequest proto. If ends with .pb, "
        "a binary proto is written. If ends with .pbtxt, text proto. Binary "
        "is much faster to load later, text is just there for debugging.",
    )
    brr_p.add_argument(
        "--inference_request",
        help="Path to the original InferenceRequest proto. The Resegmentation"
        "Request proto that this script generates will be based on it.",
    )
    brr_p.add_argument(
        "--bounding_box",
        help="Path to a BoundingBox proto. We will look for reseg pairs in "
        "this bbox.",
    )
    brr_p.add_argument(
        "--init_segmentation",
        help="<hdf5 path>:<dset> pointing to the initial consensus "
        "segmentation (result of run_consensus.py after run_inference.py).",
    )
    brr_p.add_argument(
        "--nthreads",
        type=int,
        default=-1,
        help="Number of threads to use in pair detector.",
    )
    brr_p.add_argument(
        "--bigmem",
        action="store_true",
        help="Load init segmentation into memory when detecting pairs.",
    )
    brr_rr_g = brr_p.add_argument_group(
        title="ResegmentationRequest proto fields",
        description="These flags will set fields in the output proto. Check "
        "out inference.proto to see their documentation, and feel free to add "
        "more of them here.",
    )
    brr_rr_g.add_argument("--output_directory")
    brr_rr_g.add_argument("--subdir_digits", type=int, default=2)
    brr_rr_g.add_argument("--max_retry_iters", type=int, default=1)
    brr_rr_g.add_argument(
        "--segment_recovery_fraction", type=float, default=0.5
    )
    brr_rr_g.add_argument("--max_distance", type=float, default=2.0)
    brr_rr_g.add_argument("--radius", type=int, default=48)

    # Step 1: Resegmentation request runner ---------------------------
    run_p = subparsers.add_parser(
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
    run_par_g = run_p.add_argument_group(
        title="Inter-node Parallelism",
        description="MPI style rank-based multi node parallelism. Work is "
        "split naively across the ranks.",
    )
    run_par_g.add_argument("--rank", type=int, default=0)
    run_par_g.add_argument("--nworkers", type=int, default=1)

    # Step 2: Analysis / merge table builder --------------------------
    ana_p = subparsers.add_parser(
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
    ana_p.add_argument(
        "--bigmem", action="store_true", help="Load whole seg into mem."
    )

    # Step 3: Post automerge to DVID ----------------------------------
    automerge_p = subparsers.add_parser(
        "automerge",
        help="Use the merge table from the last step to decide on an "
        "automated merge, and write the resulting mapping to disk.",
    )
    automerge_p.add_argument(
        "--affinities_npy", help="Merge table from step 2."
    )
    automerge_p.add_argument(
        "--preserve_mappings",
        default=None,
        help="Optional path to pd.Series in .pkl format. This would "
        "contain supervoxel to bodyID mappings to leave unchanged "
        "during the automerge. The Series should have supervoxel ID "
        "as its index and body ID as values.",
    )
    automerge_p.add_argument(
        "--mappings_output",
        requred=True,
        help="Path to .pkl where a pd.Series containing the supervoxel "
        "ID to body ID mapping will be written. If --preserve_mappings "
        "is set, the output here will contain that input.",
    )
    automerge_p.add_argument(
        "--threshold", type=float, help="Automatic merge threshold."
    )

    args = ap.parse_args()

    # ------------------------- run the task --------------------------
    if "DEBUG" in os.environ:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Switch on task to dispatch to its runner function.
    if args.task == "build_req":
        build_reseg_req(
            args.resegmentation_request,
            args.inference_requset,
            args.bounding_box,
            args.init_segmentation,
            args.output_directory,
            args.subdir_digits,
            args.max_retry_iters,
            args.segment_recovery_fraction,
            args.max_distance,
            args.radius,
            args.nthreads,
            args.bigmem,
        )
    elif args.task == "run":
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
    elif args.task == "automerge":
        automerge(
            args.affinities_npy,
            args.mappings_output,
            args.threshold,
            args.preserve_mappings,
        )
