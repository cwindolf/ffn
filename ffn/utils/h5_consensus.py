"""
Utilities for computing the meet consensus of many overlapping
segmented subvolumes, using SWMR HDF5 as a support system for
multithreaded computation on disk.

The HDF5 code here is a little brittle: for instance, we use
a ThreadPool below, and I've found that it doesn't work with
processes.
"""
import logging
import multiprocessing
import os.path

import h5py
import numpy as np

from ffn.inference import storage
from ffn.utils import bounding_box
from ffn.utils import meet_consensus


# -- subvolume-parallelism helpers

def get_subvolumes(segdir):
    """Get all of the subvolumes in an FFN segmentation directory.
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


def check_same_subvolumes(segdirs):
    """Check that multiple segmentations used the same subvolumes

    Returns the subvolumes too. So, you can call this instead
    of `get_subvolumes` to add an error check.
    """
    with multiprocessing.pool.ThreadPool() as pool:
        allsvs_ = list(pool.map(get_subvolumes, segdirs))
    allsvs = allsvs_[0]

    # Check all the same
    for svs in allsvs_[1:]:
        assert all(a == b for a, b in zip(allsvs, svs))
    del allsvs_

    return allsvs


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


# -- threading helpers for main HDF5 consensus routine


def _h5_consensus_thread_main(subvolume):
    """Main function for consensus worker threads."""
    # compute the meet across segmentation dirs for this subvolume
    logging.info(f"Calling split consensus for {subvolume.start}")
    subvol_slice, meet_result = meet_consensus.load_and_compute_meet(
        subvolume,
        _h5_consensus_thread_main.segmentation_dirs,
        _h5_consensus_thread_main.min_ffn_size,
    )

    # refresh SWMR state
    consensus_data = _h5_consensus_thread_main.consensus_data
    consensus_data.refresh()

    # merge `meet_result` with its friend in the h5 array
    # this does not actually assign to the h5 -- we are just a reader
    # here, we cannot write. that's why we return the result and the
    # subvolume slice below, so the main (writer) thread can do the write
    # the IDs are there so that the main thread can deal with the contiguous
    # ID space on its own, it's the only place where enough info is present
    logging.info(f"Merging with main at {subvolume.start}")
    merge, sv_max_id, sv_old_max_id, new_mask = meet_consensus.paste_new_seg(
        consensus_data[subvol_slice], meet_result
    )

    return merge, sv_max_id, sv_old_max_id, new_mask, subvol_slice


def _h5_consensus_thread_initializer(
    out_fn, dset, segmentation_dirs, min_ffn_size
):
    """Initializer for h5 consensus threads.

    Purpose of the initializer is to store data that threads will need.
    In particular, we store some parameters (the segmentation directories
    and the min segment size at load time), and we also store a read-only
    reference to the output dataset in the HDF5 file.

    It's kind of a hack, but I've seen all over the place this being done
    by assigning them as properties of the thread's main function.
    """
    # store parameters
    _h5_consensus_thread_main.segmentation_dirs = segmentation_dirs
    _h5_consensus_thread_main.min_ffn_size = min_ffn_size

    # store read-only data reference for use in SWMR scheme
    seg_outf = h5py.File(out_fn, "r", libver="latest", swmr=True)
    _h5_consensus_thread_main.consensus_data = seg_outf[dset]


# -- HDF5 consensus main thread

def hdf5_meet_consensus(
    out_fn, segmentation_dirs, dset="seg", min_ffn_size=0, chunksize=64
):
    """Merge overlapping subvolumes into a single segmentation

    Can combine multiple segmentations with the same overlapping
    subvolume structure by meet consensus.

    Arguments
    ---------
    out_fn : str
        The HDF5 filename where we'll write the results
    segmentation_dirs : list of str
        One or more directories containing FFN segmentations
        If it's just one directory, we compute consensus over the
        subvolumes in that inference. If it's more than one directory,
        first these are reduced into a single set of segmentations of
        overlapping subvolumes with meet consensus, and then the
        consensus over subvolumes is computed.
    dset : str, optional
        The dataset to write in the HDF5 file
    min_ffn_size : int, optional
        The minimum size neurite to allow when loading subvolumes
        from disk.
    chunksize : int, optional
        Size of chunks in the hdf5 file. 64 should be fine.
    """
    # since we're trying to fit into uint32, we need to be super careful.
    np.seterr(over="raise")

    # -- arg checks
    assert len(segmentation_dirs) > 0, "Need at least one segmentation."
    assert all(
        os.path.isdir(segdir) for segdir in segmentation_dirs
    ), "All segmentation directories should exist..."
    assert not os.path.exists(out_fn), f"{out_fn} already exists."

    # -- get subvolume structure
    subvolumes = check_same_subvolumes(segmentation_dirs)
    outer_bbox = bounding_box.containing(*subvolumes)
    tiers = subvolume_tiers(subvolumes)

    # -- main portion
    # get appropriate size for thread pool
    nthreads = min(max(map(len, tiers)), multiprocessing.cpu_count())

    # count number done so far
    n_done = 0

    # this is the only writeable reference to our output file
    # h5 needs libver latest to use single writer/multiple reader
    # also enter ThreadPool context here to avoid super deep nesting
    with h5py.File(
        out_fn, "w", libver="latest", swmr=True
    ) as seg_outf, multiprocessing.pool.ThreadPool(
        nthreads,
        initializer=_h5_consensus_thread_initializer,
        initargs=(out_fn, dset, segmentation_dirs, min_ffn_size),
    ) as pool:
        # -- set up data
        # note the fillvalue=0 to initialize seg to all background
        seg_out = seg_outf.create_dataset(
            dset,
            outer_bbox.size,
            dtype=np.uint32,
            fillvalue=0,
            chunks=(chunksize, chunksize, chunksize),
        )
        # store max id in h5 file
        seg_outf.create_dataset("max_id", (1,), dtype=np.uint32, data=[0])
        # set swmr so that threads can read in parallel
        seg_outf.swmr_mode = True

        # -- main loop
        # serial over tiers
        for tier_num, tier in enumerate(tiers, start=1):
            logging.info(f"Running tier {tier_num} / {len(tiers)}")

            # parallel within tiers
            for (
                merge, sv_max_id, sv_old_max_id, new_mask, subvol_slice
            ) in pool.imap_unordered(_h5_consensus_thread_main, tier):
                # -- contiguous ID space logic
                # done on the main thread since it's hard to parallelize
                max_id = seg_outf["max_id"][0]
                id_diff = max_id - sv_old_max_id
                assert id_diff >= 0
                assert sv_max_id + id_diff < np.iinfo(np.uint32).max
                # add the diff into the view
                if id_diff > 0:
                    merge[new_mask] += id_diff
                # update the global max id
                new_max_id = max(max_id, merge.max())
                seg_outf["max_id"][0] = new_max_id
                seg_outf["max_id"].flush()
                # check that it wrote OK
                assert (
                    new_max_id - max_id == seg_outf["max_id"][0] - max_id
                )

                # -- write subvolume to HDF5
                seg_out[subvol_slice] = merge
                seg_out.flush()

                # -- inform the public
                n_done += 1
                logging.info(
                    f"[sv {n_done} / {len(subvolumes)}] old max id: "
                    f"{max_id}, new max id: {new_max_id}."
                )

    logging.info(f"Done with consensus. Wrote dataset {dset} in {out_fn}")