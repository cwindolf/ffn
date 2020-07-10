"""Subvolume utilities

These utilities help deal with segmentations possibly containing
many overlapping subvolumes. You can answer questions like, is
this segmentation done? What subvolumes are there in this segmentation?
Did these several segmentations use the same subvolume structure?

Importantly you can also ask, what are the maximal groups of
subvolumes such that inside each group none of the subvolumes
overlap? That can help parallelize code that could be embarassingly
parallel when subvolumes don't overlap.
"""
import glob
import logging
import multiprocessing.pool
from os.path import join, exists

from ffn.inference import storage
from ffn.utils import bounding_box


# -- subvolume structure helpers

def get_subvolumes(segdir):
    """Get all of the subvolumes in an FFN segmentation directory.

    This only selects finished subvolumes. So, to check if the seg
    was completely done, you'll have to call `check_finished` first.
    """
    bboxes = []
    for corner in sorted(storage.get_existing_corners(segdir)):
        # load mmap style to avoid using memory
        seg, _ = storage.load_segmentation(
            segdir, corner, split_cc=False, mmap_mode="r"
        )
        # XXX Reverse size, right?
        size = seg.shape
        bboxes.append(bounding_box.BoundingBox(start=corner, size=size))
    return bboxes


def check_same_subvolumes(segdirs, return_bool=False):
    """Check that multiple segmentations used the same subvolumes

    Returns the subvolumes too. So, you can call this instead
    of `get_subvolumes` to add an error check.

    By default, raises when the subvolumes are different. But
    passing `return_bool` will return just the boolean indicating
    same subvolume structure rather than returning the subvolumes
    and raising if not.
    """
    with multiprocessing.pool.ThreadPool() as pool:
        allsvs_ = list(pool.map(get_subvolumes, segdirs))
    allsvs = allsvs_[0]

    # Check all the same
    for svs in allsvs_[1:]:
        same = all(a == b for a, b in zip(allsvs, svs))

        if not same and return_bool:
            return False

        elif not same:
            raise ValueError("Subvolume structure did not match.")

    if return_bool:
        return True

    return allsvs


def check_finished(segdir):
    """Check whether a segmentation with subvolumes is finished

    Returns a boolean indicating whether all subvolumes in the
    segmentation have a segmentation .npz file, and that there
    aren't any with just a .cpoint, and that the segmentation
    at least has some subvolume.
    """
    if not glob.glob(join(segdir, "*/*")):
        return False

    for subdir in glob.glob(join(segdir, "*/*")):
        if not glob.glob(join(subdir, "seg-*_*_*.npz")):
            return False

        for cpoint in glob.glob(join(subdir, "seg-*_*_*.cpoint")):
            assert cpoint.endswith(".cpoint")
            npz = cpoint[:-len(".cpoint")] + ".npz"
            if not exists(npz):
                return False

    return True


# -- subvolume-parallelism helpers

def subvolume_tiers(subvolumes):
    """Find maximal groups of non-overlapping subvolumes

    The idea is that some operations (such as meet) can be run over
    multiple subvolumes in parallel when it's guaranteed that the
    subvolumes don't overlap at all. So, let's find the largest
    possible groups of subvolumes that don't overlap -- these will
    be called "tiers".

    How do we choose the tiers?

    It's a "Chessboard" idea. In 2D, a chessboard would require 4 tiers
    for overlapping tiles (as long as the overlap is less than half the
    tile size.) (The black and white chessboard has overlap=0). The
    tiers are 0, x+, y+, and xy+. In 3D, we need 8 tiers: 0, x+, y+, z+,
    xy+, xz+, yz+, xyz+.

    That's nice, but honestly, it's easier to just compute these
    greedily by brute force it than to derive some algorithm using
    intuition from 3D chessboard. Still, we'll get the same result.

    Arguments
    ---------
    subvolumes : list of BoundingBox

    Returns
    -------
    tiers : list of list of BoundingBox
    """
    tiers = []
    unused_indices = list(range(len(subvolumes)))
    while unused_indices:
        new_tier = [subvolumes[unused_indices.pop(0)]]
        new_unused_indices = []
        for ind in unused_indices:
            sv = subvolumes[ind]
            intersections = bounding_box.intersections([sv], new_tier)
            if intersections:
                # this box intersects another in the tier
                new_unused_indices.append(ind)
            else:
                new_tier.append(sv)
        unused_indices = new_unused_indices
        tiers.append(new_tier)

    assert len(tiers) == min(len(subvolumes), 8)
    assert sum(len(tier) for tier in tiers) == len(subvolumes)
    logging.info(f"Tier lengths: {[len(tier) for tier in tiers]}")

    return tiers
