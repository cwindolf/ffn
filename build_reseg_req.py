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

from ffn.utils import bounding_box_pb2
from ffn.utils import bounding_box
from ffn.utils import vector_pb2
from ffn.inference import inference_pb2


flags.DEFINE_string(
    "output_file",
    None,
    "Write the ResegmentationRequest as a text proto here.",
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
    None,
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
    MAX_DIST = np.sqrt(3 * 64 ** 2)

    def __init__(self, init_segmentation_spec, bbox):

        # Get the subvolumes
        overlap = [int(np.ceil(2 * FLAGS.max_distance))] * 3
        svcalc = bounding_box.OrderlyOverlappingCalculator(
            bbox,
            [PairDetector.BUCKET_SIZE] * 3,
            overlap,
            include_small_sub_boxes=True,
        )

        # Find all the segids
        segids = set()

        # Store all of the (sorted) pairs that have been
        # seen together as keys in a dictionary. The values in this
        # dict will describe the closest approach between these two
        # segments so far, with a tuple: (approach_point, approach_dist).
        # approach_dist is the distance of this approach point to the
        # two segments in the pair (they will be equidistant from the
        # point).
        # Note that the distances are really squared distances...
        pair2approach = {}

        if FLAGS.bigmem:
            # Load the whole segmentation into memory, so that worker
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
            initargs=(init_segmentation_spec, svcalc),
        ) as pool, tqdm(
            total=svcalc.num_sub_boxes(),
            smoothing=0.0,
            mininterval=1.0,
            ncols=100,
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

        logging.info(
            f"Spatial hash finished. Found {len(pair2approach)} pairs "
            f"out of {len(segids) ** 2} possible pairs."
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
    def proc_initializer(init_segmentation_spec, svcalc):
        """Manage variables that don't change across subvolumes."""
        if FLAGS.bigmem:
            global init_segmentation
        else:
            init_segmentation = dx.loadspec(
                init_segmentation_spec, readonly_mmap=True
            )
        PairDetector.process_subvolume.init_segmentation = init_segmentation
        PairDetector.process_subvolume.svcalc = svcalc

    @staticmethod
    def process_subvolume(svid):
        """Determines the approach points within a subvolume.

        Helper to add parallelism during __init__ since otherwise this
        would take like a day to run.
        """
        # Get all segids for this subvolume index
        svcalc = PairDetector.process_subvolume.svcalc
        subvolume = svcalc.index_to_sub_box(svid)
        init_segmentation = PairDetector.process_subvolume.init_segmentation
        seg_at_sv = init_segmentation[subvolume.to_slice()]
        segids_at_sv = np.setdiff1d(np.unique(seg_at_sv), [0])

        # Compute EDTs to help later compute the approach
        # points
        edts = [edt.edt3d(seg_at_sv != segid) for segid in segids_at_sv]

        # Get pair approaches for this subvolume
        pair2approach_at_sv = {}
        for j, segid in enumerate(segids_at_sv):
            segid_edt = edts[j]
            for k in range(j + 1, len(segids_at_sv)):
                other = segids_at_sv[k]
                pair = int(segid), int(other)
                # Find closest approach of other and segid
                # in the subvolume. We define closest point
                # the equidistant point with the smallest
                # distance.
                other_edt = edts[k]

                # We take a Lagrangian approach to finding closest
                # equidistant point
                L = np.abs(segid_edt - other_edt).astype(np.float64)
                L *= 2.0 * PairDetector.MAX_DIST
                L += segid_edt
                L += other_edt

                approach_offset = np.unravel_index(np.argmin(L), L.shape)
                approach_dist = segid_edt[approach_offset]
                approach_dist_ = other_edt[approach_offset]

                # Check close enough
                if approach_dist + approach_dist_ < FLAGS.max_distance:
                    if abs(approach_dist - approach_dist_) > np.sqrt(3.0):
                        logging.critical("Bad equidistance violation...")
                        logging.critical(
                            f"Pair was {pair} with dists "
                            f"{approach_dist}, {other_edt[approach_offset]}"
                        )

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
    pair_detector = PairDetector(FLAGS.init_segmentation, bbox)

    # Get the resegmentation points
    resegmentation_points = []
    for id_a, id_b, point in pair_detector.pairs_and_points():
        # TODO: um. does this want xyz or zyx order???
        # `point` is an index into the array, aka zyx...
        p = vector_pb2.Vector3j()
        p.z, p.y, p.x = point

        # Build ResegmentationPoint proto
        rp = inference_pb2.ResegmentationPoint()
        rp.id_a = id_a
        rp.id_b = id_b
        rp.point = p

        # OK bai
        resegmentation_points.append(rp)

    # Build the ResegmentationRequest
    resegmentation_request = inference_pb2.ResegmentationRequest()

    # Write the fields that are specified directly in our flags
    resegmentation_request.output_directory = FLAGS.output_directory
    resegmentation_request.subdir_digits = FLAGS.subdir_digits
    resegmentation_request.max_retry_iters = FLAGS.max_retry_iters
    resegmentation_request.segment_recovery_fraction = (
        FLAGS.segment_recovery_fraction
    )

    # Add init_segmentation field to infreq and save into RR
    inference_request.init_segmentation = FLAGS.init_segmentation
    resegmentation_request.inference = inference_request

    # Add points
    resegmentation_request.points = resegmentation_points

    # Resegmentation and analysis radius
    radius = vector_pb2.Vector3j()
    radius.x = FLAGS.radius
    radius.y = FLAGS.radius
    radius.z = FLAGS.radius
    resegmentation_request.radius = radius

    # Following suggestion in a comment in ResegmentationRequest proto,
    # compute analysis radius by subtracting FFN's FOV radius from
    # the resegmentation radius.
    model_args = json.loads(inference_request.model_args)
    ffn_fov_radius = model_args.get("fov_size", [24, 24, 24])
    analysis_radius = vector_pb2.Vector3j()
    analysis_radius.x = FLAGS.radius - ffn_fov_radius[0]
    analysis_radius.y = FLAGS.radius - ffn_fov_radius[1]
    analysis_radius.z = FLAGS.radius - ffn_fov_radius[2]
    resegmentation_request.analysis_radius = analysis_radius

    # Write request to output file
    with open(FLAGS.output_file, "w") as out:
        out.write(text_format.MessageToString(resegmentation_request))

    logging.info(f"OK, I wrote {FLAGS.output_file}. Bye...")


if __name__ == "__main__":
    app.run(main)
