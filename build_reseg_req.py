"""Resegmentation request builder

You want to make affinities for segment agglomeration. The way to do
this is "Pair Resegmentation". This builds a protobuf that tells FFN
to do that.

Given an initial InferenceRequest proto which has already been run, and
the final output segmentation (passed via this script's flag
`--init_segmentation`), build a ResegmentationRequest proto for pair
resegmentation.

You're reading this script, and you're saying "why are we doing this?"
It might help to take a look at `inference.proto`, specifically the
ResegmentationRequest message. We're building a ResegmentationRequest,
so we are working through the fields there. The tough one is the
repeated ResegmentationPoint field -- we want to do pair resegmentation,
so how do we choose these points?

QUOTH bioRxiv FFN paper:

To​​ determine​​ whether​​ a ​​pair ​​of ​​segments​ ​in ​​spatial​​
proximity​ ​are​ ​part ​​of the ​​same ​​neurite, ​​we extracted a small
subvolume (about 1 cubic micron in size) around the point of their
closest approach. We then placed seeds in parts of the two objects
inside the subvolume, at locations maximally distant from object
boundaries, and performed two independent FFN inference runs, one for
each of these seeds, while keeping the remaining objects fixed. [...
then some logic about merging which is not relevant here]

So what points should this script output: the "point of their closest
approach", or the seeds in the subvolume extracted around that point?
To understand this, look at `resegmentation.py`, which will be our
method for running resegmentation. Specifically the function
`process_point`, which has most of the logic. We can see that this
function extracts a volume around the ResegmentationPoint, does some
fancy EDT, and uses the result of that to get some points
(the line with the call to `get_starting_location`), which it then
passes to its `canvas.segment_at(...)`.
>>> So, we conclude that the ResegmentationPoint is the point of
    closest approach. The solemn duty of this script is to determine
    these points, and send them off to `resegmentation.py`.
"""
import json
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

from google.protobuf import text_format
from absl import app
from absl import flags
import multiprocessing.pool

import ppx.data_util as dx

# Faster than skfmm and scipy.
import edt

from scipy.spatial import distance
from skimage import segmentation

from ffn.utils import bounding_box_pb2
from ffn.utils import bounding_box
from ffn.inference import inference_pb2


flags.DEFINE_string(
    "output_file",
    None,
    "Write the ResegmentationRequest proto here. "
    "If this is a .pbtxt, will write proto in human-readable "
    "text format. Else binary.",
)

flags.DEFINE_string(
    "inference_request",
    "",
    "The path to an InferenceRequest proto that will configure "
    "the inference runner used to run this ResegmentationRequest. "
    "Optional -- if you don't supply it, you'll just need to edit "
    "the output proto.",
)
flags.DEFINE_string(
    "bounding_box",
    None,
    "Path to a BoundingBox proto. The `ResegmentationPoint`s will "
    "be chosen within this box.",
)
flags.DEFINE_string(
    "init_segmentation",
    None,
    "HDF5 datspec pointing to the initial segmentation of this "
    "volume. Will be written to the InferenceRequest stored in the "
    "output ResegmentationRequest proto, and will also be used to "
    "figure out the `ResegmentationPoint`s.",
)


flags.DEFINE_string(
    "method",
    "pdist",
    "edt or pdist. Should be equivalent (???), pdist is faster (???)",
)

flags.DEFINE_integer(
    "radius",
    96,
    "The radius around the ResegmentationPoint to look at during"
    "resegmentation",
)
flags.DEFINE_string(
    "output_directory", None, "The output directory for resegmentation."
)
flags.DEFINE_integer("subdir_digits", 0, "See ResegmentationRequest proto.")
flags.DEFINE_integer("max_retry_iters", 1, "See ResegmentationRequest proto.")
flags.DEFINE_float(
    "segment_recovery_fraction",
    0.5,
    "Important one. See ResegmentationRequest proto.",
)

flags.DEFINE_float(
    "max_distance",
    2.0,
    "Segments that are farther apart than this number will be excluded"
    "from consideration as possible merge candidates.",
)

flags.DEFINE_integer(
    "nworkers", None, "Number of processes. All cores by default."
)
flags.DEFINE_boolean(
    "bigmem",
    False,
    "Load the segmentation into memory. Fine for small segmentations, "
    "but for those it shouldn't even matter. For big segs, make sure "
    "you have a lot of memory.",
)


FLAGS = flags.FLAGS


class PairDetector:
    """Spatial hashing for segmentation overlap + closest approach

    At some point, we'll have to loop over all pairs. We don't want to
    run an expensive overlap test for each volume. Using a spatial
    hash helps us spend very little time in the O(n^2) part of the
    algorithm: although there are n^2 possible pairs of segments,
    in an oversegmented volume we should probably expect O(n) actual
    pairs. The spatial hash helps quickly rule out segments that are
    not close to each other, by looking at which segments lie in
    overlapping subvolumes.

    It also uses this info and some Euclidean distance transforms
    to compute the points of closest approach between pairs that
    get close enough (according to FLAGS.max_distance)
    """

    BUCKET_SIZE = 64

    def __init__(self, init_segmentation_spec, bbox, method="edt"):
        """Constructor finds the requested pairs.

        Warning, it looks at the FLAGS.
        """
        # Chunk into subvolumes
        # (these will be buckets in the spatial hash)
        overlap = [int(np.ceil(2 * FLAGS.max_distance))] * 3
        svcalc = bounding_box.OrderlyOverlappingCalculator(
            bbox,
            [PairDetector.BUCKET_SIZE] * 3,
            overlap,
            include_small_sub_boxes=True,
        )

        # Store all of the (sorted) pairs that have been
        # seen together as keys in a dictionary. The values in this
        # dict will describe the closest approach between these two
        # segments so far, with a tuple: (approach_point, approach_dist).
        # approach_dist is the distance of this approach point to the
        # two segments in the pair (they will be equidistant from the
        # point).
        pair2approach = {}

        # Might as well store the set of segids, while we're at it.
        segids = set()

        if FLAGS.bigmem:
            # Load the whole segmentation into a global, so that worker
            # processes inherit the memory. If not set, worker processes
            # just make their own HDF5 handle, and they don't load the
            # thing into memory at all.
            logging.info("Loading segmentation into memory, since you asked.")
            global init_segmentation
            init_segmentation = dx.loadspec(init_segmentation_spec)

        # Pool maps over subvolumes, main thread loop reduces
        # results into `segids` and `pair2approach`.
        logging.info("Detecting pairs.")
        with multiprocessing.pool.Pool(
            FLAGS.nworkers,
            initializer=PairDetector.proc_initializer,
            initargs=(init_segmentation_spec, svcalc, method),
        ) as pool, tqdm(
            total=svcalc.num_sub_boxes(),
            smoothing=0.0,
            mininterval=1.0,
            ncols=60,
        ) as t:
            t.set_description(f"0 / 0")
            for niter, (segids_at_sv, pair2approach_at_sv) in enumerate(
                pool.imap_unordered(
                    PairDetector.process_subvolume,
                    range(svcalc.num_sub_boxes()),
                )
            ):
                # Update segid set with segids in this subvolume
                segids.update(segids_at_sv)

                # Incorporate new closest approaches
                for pair, (point, dist) in pair2approach_at_sv.items():
                    if pair in pair2approach:
                        _, prev_dist = pair2approach[pair]
                        if dist > prev_dist:
                            continue
                    pair2approach[pair] = point, dist

                # Update progress
                t.update()
                if not niter % 100:
                    t.set_description(
                        f"{len(pair2approach)} / {len(segids)}^2"
                    )

        n_pairs = len(pair2approach)
        n_possible = len(segids) * (len(segids) - 1) / 2
        logging.info(
            f"Spatial hash finished. Found {n_pairs} pairs between "
            f"the {len(segids)} bodies, out of {n_possible} possible "
            f"pairs. That's a rate of {n_pairs / len(segids)} "
            "partners per body."
        )

        # Store things for my friends
        self.pair2approach = pair2approach
        self.svcalc = svcalc
        self.segids = segids

    def pairs_and_points(self):
        """This is basically the reason this class exists."""
        for (id_a, id_b), (point, dist) in self.pair2approach.items():
            yield id_a, id_b, point

    @staticmethod
    def proc_initializer(init_segmentation_spec, svcalc, method):
        """Manage variables that don't change across subvolumes."""
        if FLAGS.bigmem:
            global init_segmentation
        else:
            init_segmentation = dx.loadspec(
                init_segmentation_spec, readonly_mmap=True
            )
        PairDetector.process_subvolume.init_segmentation = init_segmentation
        PairDetector.process_subvolume.svcalc = svcalc
        PairDetector.process_subvolume.method = method

    @staticmethod
    def process_subvolume(svid):
        """Determines the approach points within a subvolume.

        Helper to add parallelism during __init__ since otherwise this
        would take like a day to run.
        """
        method = PairDetector.process_subvolume.method

        # Get all segids for this subvolume index
        svcalc = PairDetector.process_subvolume.svcalc
        subvolume = svcalc.index_to_sub_box(svid)
        init_segmentation = PairDetector.process_subvolume.init_segmentation
        seg_at_sv = init_segmentation[subvolume.to_slice()]
        segids_at_sv = np.setdiff1d(np.unique(seg_at_sv), [0])

        # Compute EDTs to help later compute the approach
        # points
        if method == "edt":
            edts = [edt.edt3d(seg_at_sv != segid) for segid in segids_at_sv]
        elif method == "pdist":
            # Get coords of points on body boundaries
            shells = segmentation.find_boundaries(seg_at_sv, mode="inner")
            pointclouds = [
                np.array(
                    np.logical_and(shells, seg_at_sv == segid).nonzero(),
                    dtype=np.float32,
                )
                for segid in segids_at_sv
            ]
        else:
            assert False

        # Get pair approaches for this subvolume
        pair2approach_at_sv = {}
        for j, segid in enumerate(segids_at_sv):

            if method == "edt":
                segid_edt = edts[j]
            elif method == "pdist":
                pointcloud = pointclouds[j]
            else:
                assert False

            for k in range(j + 1, len(segids_at_sv)):
                other = segids_at_sv[k]

                # Find closest approach of other and segid in the
                # subvolume. We define the closest approach to be
                # the point whose sum distance to both bodies is
                # smallest out of all the equidistant points.

                # Simple method with pairwise distances
                if method == "pdist":
                    other_pointcloud = pointclouds[k]

                    # Compute pairwise distances between points
                    dists = distance.cdist(pointcloud.T, other_pointcloud.T)
                    amin = np.argmin(dists)
                    ji, ki = np.unravel_index(np.atleast_1d(amin), dists.shape)
                    ji, ki = ji[0], ki[0]
                    approach_offset = np.round(
                        0.5 * (pointcloud[:, ji] + other_pointcloud[:, ki])
                    ).astype(int)
                    approach_dist = dists[ji, ki]

                # More "sophisticated" method with Euc distance transform
                elif method == "edt":
                    other_edt = edts[k]

                    # Constrained minimum
                    # We consider a voxel to be more or less equidistant
                    # from the two bodies when the difference between its
                    # distances from the two bodies less than root 3,
                    # since that's the max distance between neighboring
                    # voxels in 3D. If we had no tolerance here, we would
                    # lose a lot of candidates due to grid effects.
                    equis = np.where(
                        np.abs(segid_edt - other_edt) < np.sqrt(3.0) + 1e-5,
                        segid_edt + other_edt,
                        np.inf,
                    )

                    approach_offset = np.unravel_index(
                        np.argmin(equis), equis.shape
                    )
                    approach_dist = (
                        segid_edt[approach_offset] + other_edt[approach_offset]
                    )

                else:
                    assert False

                # Check close enough
                if approach_dist < FLAGS.max_distance:
                    # Key into dictionary
                    pair = int(segid), int(other)

                    # Convert local offset to global point
                    approach_point = subvolume.start + approach_offset

                    # OK, store this approach
                    pair2approach_at_sv[pair] = approach_point, approach_dist

        return segids_at_sv, pair2approach_at_sv


# ------------------------------- main --------------------------------


def main(unused_argv):
    # Load up inference request
    inference_request = inference_pb2.InferenceRequest()
    if FLAGS.inference_request:
        with open(FLAGS.inference_request) as inference_request_f:
            text_format.Parse(inference_request_f.read(), inference_request)

    # Load up bounding box
    # We compute pair resegmentation points for all neurons
    # that intersect with the bounding box, but provide no
    # guarantees about what will happen just outside the boundary.
    # There might be some points there or there might not.
    bbox = bounding_box_pb2.BoundingBox()
    with open(FLAGS.bounding_box) as bbox_f:
        text_format.Parse(bbox_f.read(), bbox)
    bbox = bounding_box.BoundingBox(bbox)

    # Build a data structure that will help us quickly check
    # whether two segments cannot possibly overlap.
    pair_detector = PairDetector(
        FLAGS.init_segmentation, bbox, method=FLAGS.method
    )

    # Get the resegmentation points
    resegmentation_points = []
    for id_a, id_b, point in pair_detector.pairs_and_points():
        # Build ResegmentationPoint proto
        rp = inference_pb2.ResegmentationPoint()
        rp.id_a = id_a
        rp.id_b = id_b
        # TODO: um. does this want xyz or zyx order???
        # `point` is an index into the array, aka zyx...
        rp.point.z, rp.point.y, rp.point.x = point

        # OK bai
        resegmentation_points.append(rp)

    # Build the ResegmentationRequest
    logging.info("Building the ResegmentationRequest...")
    resegmentation_request = inference_pb2.ResegmentationRequest(
        inference=inference_request,
        points=resegmentation_points,
        output_directory=FLAGS.output_directory,
        subdir_digits=FLAGS.subdir_digits,
        max_retry_iters=FLAGS.max_retry_iters,
        segment_recovery_fraction=FLAGS.segment_recovery_fraction,
    )

    # Some fields are easier to just set.

    # Patch the inference request to point to the initial segmentation
    resegmentation_request.inference.init_segmentation.hdf5 = FLAGS.init_segmentation  # noqa

    # Resegmentation and analysis radius
    resegmentation_request.radius.x = FLAGS.radius
    resegmentation_request.radius.y = FLAGS.radius
    resegmentation_request.radius.z = FLAGS.radius

    # Following suggestion in a comment in ResegmentationRequest proto,
    # compute analysis radius by subtracting FFN's FOV radius from
    # the resegmentation radius.
    model_args = json.loads(inference_request.model_args)
    ffn_fov_radius = model_args.get("fov_size", [24, 24, 24])
    resegmentation_request.analysis_radius.x = FLAGS.radius - ffn_fov_radius[0]
    resegmentation_request.analysis_radius.y = FLAGS.radius - ffn_fov_radius[1]
    resegmentation_request.analysis_radius.z = FLAGS.radius - ffn_fov_radius[2]

    # Write request to output file
    if FLAGS.output_file.endswith('txt'):
        with open(FLAGS.output_file, "w") as out:
            out.write(text_format.MessageToString(resegmentation_request))
    else:
        with open(FLAGS.output_file, "wb") as out:
            out.write(resegmentation_request.SerializeToString())

    logging.info(f"OK, I wrote {FLAGS.output_file}. Bye...")


if __name__ == "__main__":
    app.run(main)
