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
    affinity greater than the threshold. Then, post that merge (in a
    very destructive fashion!) to the labels instance in a DVID repo.
    Don't use a repo that has been traced.
        $ python run_resegmentation.py post_automerge --help
"""
import argparse
import collections
import datetime
import json
import logging
import os
import time

import h5py
import joblib
import numpy as np
import pandas as pd
from google.protobuf import text_format
from sklearn.cluster import AgglomerativeClustering

from ffn.inference import inference_pb2
from ffn.utils import bounding_box_pb2
from ffn.utils import geom_utils
from ffn.utils import pair_detector

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
    from ffn.inference import inference
    from ffn.inference import resegmentation

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
    from ffn.inference import resegmentation
    from ffn.inference import resegmentation_analysis

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


# ------------------ step 3: post an automatic merge ------------------


def post_automerge(
    affinities_npy, threshold, dvid_host, repo_uuid, indices_batch_sz=512
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

    # I didn't want to make neuclease a hard dependency for the rest
    # of the script.
    import neuclease.merge_table
    import neuclease.dvid

    # Hit host with a simple api call to make sure it works
    with timer("Fetched mutid."):
        repo_info = neuclease.dvid.fetch_repo_info(dvid_host, repo_uuid)
        last_mutid = repo_info["MutationID"]
    print("It was", last_mutid)

    # Decide on automerge ---------------------------------------------
    # neuclease will normalize the merge table a little for us
    merge_table = neuclease.merge_table.load_merge_table(affinities_npy)
    thresholded = merge_table[merge_table['score'] > threshold]

    # Get all supervoxel IDs present in merge table
    all_svids = np.union1d(merge_table["id_a"].values, merge_table["id_b"].values)
    nsvs = all_svids.size
    del all_svids
    merge_svids = np.union1d(thresholded["id_a"].values, thresholded["id_b"].values)
    np.sort(merge_svids)
    assert merge_svids[0] > 0  # We should not be getting background here.
    nmergedsvs = merge_svids.size
    print(f"Attempting to merge {nmergedsvs} svs out of {nsvs} total.")
    # Make a reverse index
    svid2zid = dict(zip(merge_svids, np.arange(nmergedsvs)))

    with timer("Clustered."):
        # Make merge table into wide nsvs x nsvs affinity matrix
        with timer("Merge table long -> wide."):
            affinities = np.zeros((nmergedsvs, nmergedsvs), dtype=np.float)
            for row in thresholded.itertuples():
                i, j = svid2zid[row.id_a], svid2zid[row.id_b]
                affinities[i, j] = affinities[j, i] = row.score

        # Do clustering
        distances = 1 - affinities
        del affinities
        agg = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage="single",
            distance_threshold=1 - threshold,
        )
        agg.fit(distances)

    # Log stats
    print(
        f"Agglomeration merged {agg.n_leaves_} out of {nsvs} original "
        f"supervoxels into {agg.n_clusters_} neurites."
    )

    # These are in the same order as svids above. So, we got the merges
    # now, OK? We just need to find out like, who was actually part of
    # a merge, i.e. what svids are part of cluster labels with more
    # than one member. And then tell DVID.
    with timer("Processed clusters into merges."):
        # Collect svids for each label
        label_svids = collections.defaultdict(set)
        for label, svid in zip(agg.labels_, merge_svids):
            label_svids[label].add(svid)

        # See what svids are part of clusters with more than one label
        svids_in_merges = set()
        merges = []
        for label, svids in label_svids.items():
            if len(svids) > 1:
                svids_in_merges |= svids
                merges.append(list(sorted(svids)))
        svids_in_merges = np.array(list(sorted(svids_in_merges)))
        assert (svids_in_merges == merge_svids).all()
    del merge_svids

    # Update label index ----------------------------------------------
    # Get the old index for svids who are gonna change
    # Do this by hitting GET .../indices with batches of svids
    with timer("Downloaded label indices."):
        plis = {}
        for i in range(0, len(svids_in_merges), indices_batch_sz):
            svid_batch = svids_in_merges[i : i + indices_batch_sz]
            for pli in neuclease.dvid.fetch_labelindices(
                dvid_host, repo_uuid, "labels", svid_batch, format="pandas"
            ):
                plis[pli.label] = pli

    # Update these indices according to merges and post to DVID
    the_time = datetime.datetime.now().isoformat()
    li_proto_batch = []
    batch_i = 0
    nbatch = len(merges) // indices_batch_sz
    with timer("Posted all merged label indices."):
        for merge in merges:
            merge_pli_blocks = pd.concat(
                [plis[sv].blocks for sv in merge], ignore_index=True
            )
            # Create a neuclease PandasLabelIndex containing our automerge
            new_pli = neuclease.dvid.PandasLabelIndex(
                merge_pli_blocks,
                merge[0],  # New label is gonna be the smallest svid in merge
                last_mutid + 1,
                the_time,
                os.environ.get("USER", "automerge_unknown_user"),
            )
            # Converts pandas -> protobuf for posting
            new_li_proto = neuclease.dvid.create_labelindex(new_pli)
            li_proto_batch.append(new_li_proto)
            if len(li_proto_batch) >= indices_batch_sz:
                # Hit POST .../indices once for each merge batch
                with timer(f"Posted batch {batch_i} / {nbatch}."):
                    neuclease.dvid.post_labelindices(
                        dvid_host,
                        repo_uuid,
                        "labels",
                        new_li_proto,
                        li_proto_batch,
                    )
                li_proto_batch = []
        if li_proto_batch:
            with timer("Posted final batch."):
                neuclease.dvid.post_labelindices(
                    dvid_host,
                    repo_uuid,
                    "labels",
                    new_li_proto,
                    li_proto_batch,
                )


    # Upload new mappings ---------------------------------------------
    with timer("Made mapping series."):
        # Make pd.Series with the new labels
        print(svids_in_merges[:10])
        svids = [svid for merge in svids_in_merges for svid in merge]
        bodies = [merge[0] for merge in svids_in_merges for _ in merge]
        mappings = pd.Series(data=bodies, index=svids)

    with timer("Posted mappings to DVID."):
        # This hits POST .../mappings
        neuclease.dvid.post_mappings(
            dvid_host, repo_uuid, "labels", mappings, last_mutid + 2
        )


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
    post_p = subparsers.add_parser(
        "post_automerge",
        help="Use the merge table from the last step to decide on an "
        "automated merge, and post that to DVID. Please only use this on "
        "brand new repos, since it is destructive.",
    )
    post_p.add_argument("--affinities_npy", help="Merge table from step 2.")
    post_p.add_argument(
        "--threshold", type=float, help="Automatic merge threshold."
    )
    post_p.add_argument(
        "--indices_batch_size",
        type=int,
        default=512,
        help="Number of labelindices to post at a time.",
    )

    post_dvid_g = post_p.add_argument_group(
        title="DVID",
        description="Not to harp on this, but like, don't use a repo that has "
        "been traced, this will destroy the work.",
    )
    post_dvid_g.add_argument(
        "--repo",
        help="UUID of DVID repo whose labels and mappings we will overwrite.",
    )
    post_dvid_g.add_argument(
        "--dvid",
        default="",
        help="<host>:<port>. If environment variable DVIDHOST is set, we will "
        "use that for the host, and 8000 for the port unless DVIDPORT is also "
        "set.",
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
    elif args.task == "post_automerge":
        # Munge DVID host
        dvid_host = args.dvid
        if not dvid_host and ("DVIDHOST" in os.environ):
            port = os.environ.get("DVIDPORT", 8000)
            dvid_host = f"{os.environ['DVIDHOST']}:{port}"
        elif not dvid_host:
            raise ValueError("Please pass --dvid or set DVIDHOST.")

        post_automerge(
            args.affinities_npy,
            args.threshold,
            dvid_host,
            args.repo,
            indices_batch_sz=args.indices_batch_size,
        )
