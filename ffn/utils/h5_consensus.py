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

from ffn.utils import bounding_box
from ffn.utils import meet_consensus
from ffn.utils import subvols


# -- threading helpers for main HDF5 consensus routine


def _h5_consensus_thread_main(subvolume):
    """Main function for consensus worker threads."""
    # compute the meet across segmentation dirs for this subvolume
    logging.info(f"Calling split consensus for {subvolume.start}")
    subvol_slice, meet_result = meet_consensus.load_and_compute_meet(
        _h5_consensus_thread_main.segmentation_dirs,
        subvolume=subvolume,
        min_ffn_size=_h5_consensus_thread_main.min_ffn_size,
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
    subvolumes = subvols.check_same_subvolumes(segmentation_dirs)
    outer_bbox = bounding_box.containing(*subvolumes)
    tiers = subvols.subvolume_tiers(subvolumes)

    # -- main portion
    # get appropriate size for thread pool
    nthreads = min(max(map(len, tiers)), multiprocessing.cpu_count())

    # count number done so far
    n_done = 0

    # this is the only writeable reference to our output file
    # h5 needs libver latest to use single writer/multiple reader
    with h5py.File(
        out_fn, "w", libver="latest", swmr=True
    ) as seg_outf:
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
        with multiprocessing.pool.ThreadPool(
            nthreads,
            initializer=_h5_consensus_thread_initializer,
            initargs=(out_fn, dset, segmentation_dirs, min_ffn_size),
        ) as pool:
            # serial over tiers
            for tier_num, tier in enumerate(tiers, start=1):
                logging.info(f"Running tier {tier_num} / {len(tiers)}")

                # parallel within tiers
                for (
                    merge, sv_max_id, sv_old_max_id, new_mask, subvol_slice
                ) in pool.imap_unordered(_h5_consensus_thread_main, tier):
                    logging.info(f"Merging in {subvol_slice}")
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
