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
import os
import json
import logging

if "DEBUG" in os.environ:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

from google.protobuf import text_format
from absl import app
from absl import flags

from ffn.utils import bounding_box_pb2
from ffn.utils import bounding_box
from ffn.utils import pair_detector
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

flags.DEFINE_integer(
    "radius",
    96,
    "The radius around the ResegmentationPoint to look at during"
    "resegmentation",
)
flags.DEFINE_string(
    "output_directory", None, "The output directory for resegmentation."
)
flags.DEFINE_integer("subdir_digits", 2, "See ResegmentationRequest proto.")
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

# ------------------------------- main --------------------------------


def main(unused_argv):
    # Params ----------------------------------------------------------
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

    # Reseg points ----------------------------------------------------
    # Build a data structure that will help us quickly check
    # whether two segments cannot possibly overlap.
    pd = pair_detector.PairDetector(
        FLAGS.init_segmentation, bbox, FLAGS.nworkers, max_distance=FLAGS.max_distance, bigmem=FLAGS.bigmem,
    )
    logging.info(
        "Points were in bounding box: %s-%s",
        pd.min_approach,
        pd.max_approach,
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
    logging.info("Building the ResegmentationRequest...")

    # Some fields we set with the constructor...
    resegmentation_request = inference_pb2.ResegmentationRequest(
        inference=inference_request,
        points=resegmentation_points,
        output_directory=FLAGS.output_directory,
        subdir_digits=FLAGS.subdir_digits,
        max_retry_iters=FLAGS.max_retry_iters,
        segment_recovery_fraction=FLAGS.segment_recovery_fraction,
    )

    # Some (nested) fields are easier to just set.
    # Patch the inference request to point to the initial segmentation
    resegmentation_request.inference.init_segmentation.hdf5 = (
        FLAGS.init_segmentation
    )
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
    if FLAGS.output_file.endswith("txt"):
        # Text output for humans
        with open(FLAGS.output_file, "w") as out:
            out.write(text_format.MessageToString(resegmentation_request))
    else:
        # Binary output for robots
        with open(FLAGS.output_file, "wb") as out:
            out.write(resegmentation_request.SerializeToString())
    logging.info(f"OK, I wrote {FLAGS.output_file}. Bye...")


if __name__ == "__main__":
    app.run(main)
