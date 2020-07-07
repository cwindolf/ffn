r"""
Please observe that the split oversegmentation operation forms
an abelian group on the set of segmentations. So, it doesn't
matter if we first apply the split consensus to merge subvols
into large vols or if we apply on subvols first. In fact, this
split consensus can be considered as the meet operation on a
certain poset described below.

In other words, we want to "reduce" a bunch of segs using the
split consensus and foldl == foldr.

It's easy to parallelize the bit that works over subvolumes,
so we'll fan out over that and then funnel those results into
an output volume, which I guess we'll just keep as a memmaped
hdf5 file.

Join and meet for segmentations
-------------------------------

Recall the notions of join and meet:
https://en.wikipedia.org/wiki/Join_and_meet
Note that we can define a partial order on segmentations as
follows. Let $a$ and $b$ be two segmentations of the same
region, i.e., $a,b : Z_{pqr} -> N$ are two
assignments of natural numbers to every site on the finite
3D lattice $Z_{pqr} := Z_p \times Z_q \times Z_r$. (There's
nothing important about the lattice being 3D, in fact let's
forget about it and equivalently consider $Z_{pqr}$ to be
the enumeration of its sites $Z_{pqr}=\{0,...,pqr\}$.

OK, now we can define a partial order, using the symbol $\leq$.
This relation can be read "is splittier than" in a sense.
Intuitively, we say that $a \leq b$ when every segment inside
$a$ is part of only one segment in $b$, so that either $a$ is
just less dense than $b$ (in the specific sense that every
segmentation in $a$ is smaller than the one in $b$ or just
absent), or it's more split than $b$. More precisely, every
segment in $a$ can be obtained by erasing voxels in a single
segment from $b$ (more than one segment in $a$ can arise from
a segment in $b$ in this way). Equivalently and more formally,
let a segment in a segmentation $x$ be a level set
$S_k = \{z \in Z_{pqr} : x(z) = k\}$ of $x$ where $k \neq 0$.
Then, $a \leq b$ when for all segments $S_k$ of $a$, there
exists some $j\in N$ such that $S_k \subseteq S_j$ and $S_j$
is a segment in $b$.

Some examples: Let $a$ be some segmentation, and say that
there exist voxels labeled 1 and 2 in $a$. Let $f: N -> N$
be the function such that $f(n)=n$ for all $n$ except $n=2$,
and $f(2)=1$. Then the composition $b(z) = f(a(z))$ is the
segmentation equal to $a$ but with 1 and 2 merged. Clearly
the segment $b == 2$ is empty, and $b==1$ is the union of
$a==1$ and $a==2$. For all $n\notin\{1,2\}$, $b==n$ is the
same as $a==n$. Thust $a \leq b$, so that merges move you
up the partial order.

Similarly, erasing voxels and splitting subvoxels are
operations that move you down the partial order.

Now that we have a handle on this partial order, let's
consider its meet and join operations.

For meet: let $c,d$ be segmentations, and don't assume
that $c\leq d$ or $d\leq c$. Form a new segmentation
$e$ (the join) as follows.  Let $J: N\times N -> N$ be
the canonical bijection of the pairs of natural numbers
to the natural numbers, and make sure $J(0,0)=0$. Then
let $e(z)=J(c(z),d(z))$. I claim that $e\leq c$ and
$e\leq d$, and what's more that there is no $e\leq e'$
such that $e'\leq c$, $e'\leq d$, and $e\neq e'$.

This meet is defined in a function below. It's a
commutative version of the split seg by intersection
function in ffn.inference.segmentation.

For join: basically the idea is, for any segment ids that
overlap in $c$ and $d$, they need to get merged. No choice.
I'll let the reader formalize this.

Notice that the suprema of this poset lattice, the
kind of Zorn's lemma maximal elements, are the segmentations
that assign all sites to a single id $k\neq 0$. This suggests
that we should really "mod out" the permutations of seg IDs,
i.e., consider the equivalence classes of the eq. rel. $a==b$
if there exists a permutation $\sigma$ s.t. $\sigma(a(z))=b(z)$
for all $z$.

Interestingly, the minimal element is unique: the 0 segmentation.
"""
# TODO No merging on boundaries.
# TODO Strip margin?
import numpy as np

# We're trying to squeeze into uint32. Let's be careful about it.
np.seterr(over="raise")

import os
import h5py
import ppx.data_util as dx
import multiprocessing
from itertools import repeat
import logging
import joblib
from joblib import delayed

logging.basicConfig(level=logging.INFO)

from absl import app
from absl import flags

from ffn.inference import storage
from ffn.inference import segmentation
from ffn.utils import bounding_box


FLAGS = flags.FLAGS


flags.DEFINE_string(
    "segmentation_dirs",
    None,
    "Space separated segmentation spots. "
    "The first is given a little more authority.",
)
flags.DEFINE_string("outspec", None, "hdf5 spec to write to.")
flags.DEFINE_integer(
    "min_ffn_size", 0, "Minimum segment size to consider ffn output viable"
)
# flags.DEFINE_integer(
#    "min_split_size",
#    0,
#    "Minimum segment size to consider split consensus output viable",
# )


def merge_from_min_id(out, seg, mask, min_new_id, scratch=None):
    assert out.ndim == 3
    assert seg.ndim == 1
    assert out.size == seg.size
    assert min_new_id > 0
    # Write the segmentation into scratch
    if scratch is not None:
        assert scratch.ndim == 1
        assert scratch.size == seg.size
        scratch.fill(0)
    else:
        scratch = np.zeros(out.size, dtype=np.uint32)
    scratch[mask] = seg[mask]
    # Split CCs
    segmentation.clean_up(scratch.reshape(out.shape))
    # Linear ID space starting at min_new_id
    relabeled, id_map = segmentation.make_labels_contiguous(scratch)
    max_new_id = min_new_id + max(new_id for _, new_id in id_map)
    assert max_new_id < np.iinfo(np.uint32).max
    # Write new IDs and check invariant
    scratch[mask] = min_new_id + relabeled[mask]
    assert min_new_id == max_new_id or scratch.max() == max_new_id
    # Write output and update ID invariant
    out.flat[mask] = scratch[mask]
    return max_new_id


def seg_meet(a, b, min_new_id=1):
    """Merge a and b into a single segmentation by "split consensus"

    This implements the "meet" operation described in detail in this
    file's docstring.

    Parameters
    ----------
    a, b : np.array, integer typed, same shape
    min_new_id : int
        The output's ID space will start here

    Returns: uint32 np.array with same shape as `a`
    """
    if a.shape != b.shape:
        raise ValueError("Segmentations had different shapes.")
    orig_shape = a.shape
    assert min_new_id < np.iinfo(np.uint32).max
    # Boolean foreground / background masks
    a = a.ravel()
    b = b.ravel()
    a_fg = a != 0
    b_fg = b != 0
    b_and_a = np.logical_and(b_fg, a_fg)
    b_not_a = np.logical_and(b_fg, np.logical_not(a_fg))
    a_not_b = np.logical_and(a_fg, np.logical_not(b_fg))
    # Sparsify
    b_and_a = b_and_a.nonzero()
    b_not_a = b_not_a.nonzero()
    a_not_b = a_not_b.nonzero()
    # Output storage
    out = np.zeros(orig_shape, dtype=np.uint32)
    scratch = np.empty(a.shape, dtype=np.uint32)
    # Perform the merges
    new_max_id = merge_from_min_id(
        out, a, a_not_b, min_new_id, scratch=scratch
    )
    new_max_id = merge_from_min_id(
        out, b, b_not_a, new_max_id + 1, scratch=scratch
    )
    new_max_id = merge_from_min_id(
        out, a, b_and_a, new_max_id + 1, scratch=scratch
    )
    assert new_max_id < np.iinfo(np.uint32).max

    logging.info(f"seg_meet max id {new_max_id}")

    return out.reshape(orig_shape)


def split_merge_sv(subvolume, segmentation_dirs, min_ffn_size):
    segdirs = segmentation_dirs.split()
    # First seg determines bg and all that
    seg, _ = storage.load_segmentation(
        segdirs[0], subvolume.start, min_size=min_ffn_size
    )
    seg = seg.astype(np.uint64)

    for segdir in segdirs[1:]:
        seg_b, _ = storage.load_segmentation(
            segdir, subvolume.start, min_size=min_ffn_size
        )
        seg_b = seg_b.astype(np.uint64)
        assert seg.shape == seg_b.shape
        # segmentation.split_segmentation_by_intersection(seg, seg_b, 0)
        seg = seg_meet(seg, seg_b)

    seg = segmentation.make_labels_contiguous(seg)[0]
    seg = seg.astype(np.uint64)

    return tuple(reversed(subvolume.to_slice())), seg


def merge_into_main(a, b, old_max_id):
    """Merge a new segmentation `b` into the large vol

    Merge keeps `a` fixed, and inserts segments from `b` into the
    background, remapping them to a new contiguous id space.

    To preserve the contiguity of IDs, this also handles the
    cleaning step (split CCs and cull vols smaller than min_size),
    so that those IDs aren't removed along the way.

    Since contiguity is a priority, this also ensures that things
    fit into 32 bit uints.

    Arguments
    ---------
    a : 3d uint32 arr
        The main volume segmentation for this subvolume
    b : 3d uint32 arr
        The new segmentation for that subvolume
    old_max_id : int
        The maximum ID in the main volume

    # Old kwarg that could be nice to have again.
    # I decided that model SVs should be discarded
    # rather than post-split ones, since these
    # post-split could just need a merge to be
    # happy.
    min_size : int
        Min # of voxels for new bodies after split

    Returns
    -------
    merged : 3d uint32 arr
        The result of the merge, to be assigned into the main vol
    new_max_id : int
        The new maximum ID after this merge
    old_max_id : int
        Maximum id of `a` before merging
    b_not_a : indices
        the indices where b had a seg and a didn't
    """
    if a.shape != b.shape:
        raise ValueError
    orig_shape = a.shape

    # XXX Flat views because why again?
    a = a.ravel()
    b = b.ravel()

    # Figure out min ID for new segments
    a_max = a.max()
    old_max_id = max(a_max, old_max_id)
    # assert a_max <= old_max_id
    min_new_id = old_max_id + 1
    assert min_new_id < np.iinfo(np.uint32).max

    # Boolean foreground / background masks
    a_fg = a != 0
    b_fg = b != 0
    b_not_a = np.logical_and(b_fg, np.logical_not(a_fg))
    a_fg = a_fg.nonzero()
    b_not_a = b_not_a.nonzero()

    # Write A
    out = np.zeros(orig_shape, dtype=np.uint32)
    out.flat[a_fg] = a[a_fg]

    # Clean / process B \ A and write
    new_max_id = merge_from_min_id(out, b, b_not_a, min_new_id)

    return out, new_max_id, old_max_id, b_not_a


def _thread_main(subvolume__params):
    subvolume, params = subvolume__params
    segmentation_dirs = params["segmentation_dirs"]
    min_ffn_size = params["min_ffn_size"]

    # Split consensus in subvoume
    logging.info("Calling split consensus")
    slicer, result = split_merge_sv(subvolume, segmentation_dirs, min_ffn_size)

    # Read current state SWMR style.
    data = _thread_main.data
    data.id.refresh()
    data.refresh()
    cur_seg_out = data[slicer]

    # Merge subvol with its friend in the h5 array
    if not (cur_seg_out > 0).any():
        logging.info("Blank main, no merge necessary.")
    else:
        logging.info("Main not blank, gotta merge.")

    logging.info("Merging with main")
    merge, sv_max_id, sv_old_max_id, new_mask = merge_into_main(
        cur_seg_out, result, old_max_id=0,
    )

    logging.info("Returning.")
    return merge, sv_max_id, sv_old_max_id, new_mask, slicer


def _thread_init(outf, dset):
    # Need to set libver to use swmr
    logging.info("Initializing thread")
    seg_outf = h5py.File(outf, "r", libver="latest", swmr=True)
    _thread_main.data = seg_outf[dset]


def get_subvolumes(segdir):
    bboxes = []
    for corner in sorted(storage.get_existing_corners(segdir)):
        seg, _ = storage.load_segmentation(
            segdir, corner, split_cc=False, mmap_mode="r"
        )
        # XXX Reverse size, right?
        size = seg.shape
        bboxes.append(bounding_box.BoundingBox(start=corner, size=size))
    return bboxes


def main(_):
    # Check args ------------------------------------------------------
    # Outspec needs to be simple h5
    outf, dset, slice_expr, in_ax = dx.parse_spec(FLAGS.outspec)
    assert not slice_expr
    assert not in_ax
    assert outf.endswith(".h5")
    assert not os.path.exists(outf)
    assert dset

    # Need some segdirs
    segdirs = FLAGS.segmentation_dirs.split()
    assert len(segdirs) > 0
    assert all(os.path.isdir(sd) for sd in segdirs)

    # Subvolumes ------------------------------------------------------
    ncpu = multiprocessing.cpu_count()
    with multiprocessing.pool.ThreadPool(ncpu) as pool:
        allsvs_ = list(pool.map(get_subvolumes, segdirs))
    allsvs = allsvs_[0]

    # Check all the same
    for svs in allsvs_[1:]:
        assert all(a == b for a, b in zip(allsvs, svs))
    del allsvs_

    # See what we got...
    nsb = len(allsvs)

    # Log info
    logging.info(f"Found num subvols {nsb}")
    print("The boxes:\n\t", "\n\t".join(str(s) for s in allsvs))
    print("The slices:\n\t", "\n\t".join(str(s.to_slice()) for s in allsvs))
    outer_bbox = bounding_box.containing(*allsvs)
    print("The global bounding box:", outer_bbox)

    # Get "tiers" of subvolumes to help with parallelism
    # The idea is that within each tier, the subvolumes don't overlap.
    # So, an entire tier can be run (in parallel) and merged into the
    # main volume (in parallel), and once a previous tier has been
    # merged, the next can be run (in parallel) to merge into the previous
    # results.
    # How do we choose the tiers?
    # It's a "Chessboard" idea. In 2D, a chessboard would require 4
    # tiers for overlapping tiles (as long as the overlap is less than
    # half the tile size.) (The black and white chessboard has
    # overlap=0). The tiers are 0, x+, y+, and xy+.
    # In 3D, we need 8 tiers: 0, x+, y+, z+, xy+, xz+, yz+, xyz+.
    # How can we get these from the subvolume calculator?
    # Honestly, it's easier to brute force it than to derive some
    # algorithm using intuition from 3D chessboard.
    tiers = []
    unused_indices = list(range(len(allsvs)))
    while unused_indices:
        new_tier = [allsvs[unused_indices.pop(0)]]
        new_unused_indices = []
        for ind in unused_indices:
            sv = allsvs[ind]
            intersections = bounding_box.intersections([sv], new_tier)
            if intersections:
                # This box intersects someone in the tier
                new_unused_indices.append(ind)
            else:
                new_tier.append(sv)
        unused_indices = new_unused_indices
        tiers.append(new_tier)

    assert len(tiers) == min(nsb, 8)
    assert sum(len(tier) for tier in tiers) == nsb
    logging.info(f"Tier lengths: {[len(tier) for tier in tiers]}")

    # Run -------------------------------------------------------------

    # Put flags in a dict because FLAGS cannot be pickled
    thread_params = {
        "segmentation_dirs": FLAGS.segmentation_dirs,
        "min_ffn_size": FLAGS.min_ffn_size,
    }

    # This is the only writeable reference to our output file.
    # Only the main thread writes. Other threads will read.
    # h5 needs libver='latest' to use single writer/multiple reader.
    with h5py.File(outf, "w", libver="latest", swmr=True) as seg_outf:
        # Note the fillvalue=0, so we can depend on the seg to
        # be initialized to background.
        seg_out = seg_outf.create_dataset(
            dset,
            outer_bbox.size,
            dtype=np.uint32,
            fillvalue=0,
            chunks=(64, 64, 64),
        )

        # Store max id in h5 file
        seg_outf.create_dataset("max_id", (1,), dtype=np.uint32, data=[0])

        # Set swmr so that threads can read their area of the seg.
        seg_outf.swmr_mode = True

        # Make an appropriately sized process pool
        nthreads = min(max(map(len, tiers)), ncpu)

        # XXX The SWMR thing only seems to work with threads. It
        #     doesn't make a difference for performance since this
        #     is numpy heavy code, so whatever, but seriously, I don't
        #     know why it's like this?
        with multiprocessing.pool.ThreadPool(
            nthreads, initializer=_thread_init, initargs=(outf, dset)
        ) as pool:
            # Loop over tiers outside imap to ensure that
            # each tier is run independently of the others.
            n_done = 0
            for tier_num, tier in enumerate(tiers):
                logging.info(f"Running tier {tier_num}")
                for (
                    merge,
                    sv_max_id,
                    sv_old_max_id,
                    new_mask,
                    slicer,
                ) in pool.imap_unordered(
                    _thread_main, zip(tier, repeat(thread_params))
                ):
                    logging.info("Gathered result.")

                    # On the main thread, deal with the logic of
                    # merging the ID spaces -- cannot be parallelized.

                    # Get a flat view because new_mask wants flat
                    sv_flat = merge.ravel()

                    # IDs in this region start at sv_old_max_id + 1
                    # and go up to sv_max_id.
                    # We want them to start at max_id, so add
                    # max_id - sv_old_max_id.
                    max_id = seg_outf["max_id"][0]
                    id_diff = max_id - sv_old_max_id
                    assert id_diff >= 0
                    assert sv_max_id + id_diff < np.iinfo(np.uint32).max

                    if id_diff > 0:
                        # Add the diff into the view
                        sv_flat[new_mask] += id_diff

                    # Update the global max id
                    assert sv_flat.max() == merge.max()
                    new_max_id = max(max_id, merge.max())
                    seg_outf["max_id"][0] = new_max_id
                    seg_outf["max_id"].flush()
                    assert (
                        new_max_id - max_id == seg_outf["max_id"][0] - max_id
                    )

                    # Write to the hdf5
                    seg_out[slicer] = merge
                    seg_out.flush()

                    n_done += 1
                    logging.info(f"{n_done} done out of {nsb}")
                    logging.info(f"Old max id: {max_id}. New: {new_max_id}")


if __name__ == "__main__":
    app.run(main)
