"""
QUOTH bioRxiv FFN paper:

To determine whether a pair of segments in spatial proximity are part
of the same neurite, we extracted a small subvolume (about 1 cubic
micron in size) around the point of their closest approach. We then
placed seeds in parts of the two objects inside the subvolume, at
locations maximally distant from object boundaries, and performed two
independent FFN inference runs, one for each of these seeds, while
keeping the remaining objects fixed. [... then some logic about merging
which is not relevant here]

So what points should the pair detector output: the "point of their closest
approach", or the seeds in the subvolume extracted around that point?
To understand this, look at `resegmentation.py`, which will be our
method for running resegmentation. Specifically the function
`process_point`, which has most of the logic. We can see that this
function extracts a volume around the ResegmentationPoint, does some
fancy EDT, and uses the result of that to get some points
(the line with the call to `get_starting_location`), which it then
passes to its `canvas.segment_at(...)`.

>> So, we conclude that the ResegmentationPoint is the point of
   closest approach. The solemn duty of this class is to determine
   these points.
"""

import logging
import multiprocessing.pool
import numpy as np
from tqdm import tqdm

from scipy.spatial import distance
from skimage import segmentation

import ppx.data_util as dx
from ffn.utils import bounding_box


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
    Points are returned in XYZ order.
    It also uses this info and some Euclidean distance transforms
    to compute the points of closest approach between pairs that
    get close enough (according to FLAGS.max_distance)
    """

    BUCKET_SIZE = 64

    def __init__(
        self,
        init_segmentation_spec,
        bbox,
        nthreads,
        max_distance=2.0,
        bigmem=False,
    ):
        """Constructor finds the requested pairs.
        Warning, it looks at the FLAGS.
        """
        logging.info(f"Spatial hashing {init_segmentation_spec} in {bbox}")
        # Chunk into subvolumes
        # (these will be buckets in the spatial hash)
        overlap = [int(np.ceil(2 * max_distance))] * 3
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

        # Also store bbox for approach points
        min_approach = np.full(3, np.inf)
        max_approach = np.full(3, -np.inf)

        if bigmem:
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
            nthreads,
            initializer=PairDetector.proc_initializer,
            initargs=(init_segmentation_spec, svcalc, bigmem),
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
                    min_approach = np.minimum(min_approach, point)
                    max_approach = np.maximum(max_approach, point)

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
        self.min_approach = tuple(min_approach)
        self.max_approach = tuple(max_approach)

    def pairs_and_points(self):
        """This is basically the reason this class exists."""
        for (id_a, id_b), (point, dist) in self.pair2approach.items():
            yield id_a, id_b, point

    @staticmethod
    def proc_initializer(init_segmentation_spec, svcalc, bigmem):
        """Manage variables that don't change across subvolumes."""
        if bigmem:
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
        # Load up things stored here by the initializer
        svcalc = PairDetector.process_subvolume.svcalc
        init_segmentation = PairDetector.process_subvolume.init_segmentation

        # Get all segids for this subvolume index
        subvolume = svcalc.index_to_sub_box(svid)
        seg_at_sv = init_segmentation[subvolume.to_slice()]
        segids_at_sv = np.setdiff1d(np.unique(seg_at_sv), [0])

        # Get coords of points on body boundaries
        shells = segmentation.find_boundaries(seg_at_sv, mode="inner")
        pointclouds = [
            np.array(
                np.logical_and(shells, seg_at_sv == segid).nonzero(),
                dtype=np.float32,
            )
            for segid in segids_at_sv
        ]

        # Get pair approaches for this subvolume
        pair2approach_at_sv = {}
        for j, segid in enumerate(segids_at_sv):
            pointcloud = pointclouds[j]
            for k in range(j + 1, len(segids_at_sv)):
                other = segids_at_sv[k]
                other_pointcloud = pointclouds[k]

                # Find closest approach of other and segid in the
                # subvolume. We define the closest approach to be
                # the point whose sum distance to both bodies is
                # smallest out of all the equidistant points.

                # Compute pairwise distances between points
                dists = distance.cdist(pointcloud.T, other_pointcloud.T)

                # Find indices of closest points
                # Ties are broken arbitrarily for now (by argmin)
                amin = np.argmin(dists)
                ji, ki = np.unravel_index(np.atleast_1d(amin), dists.shape)
                ji, ki = ji[0], ki[0]
                approach_offset = np.round(
                    0.5 * (pointcloud[:, ji] + other_pointcloud[:, ki])
                ).astype(int)

                # Check close enough. If so, store this approach.
                approach_dist = dists[ji, ki]
                if approach_dist < FLAGS.max_distance:
                    # Key into dictionary
                    pair = int(segid), int(other)
                    # Convert local offset to global point, XYZ
                    approach_point = subvolume.start + approach_offset[::-1]
                    # OK, store this approach
                    pair2approach_at_sv[pair] = approach_point, approach_dist

        return segids_at_sv, pair2approach_at_sv
