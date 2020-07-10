"""Combine possibly multiple inferences overlapping subvolumes

See docs for ffn/utils/h5_consensus.py and the function
`h5_meet_consensus` in that file for more details.

To run lots of consensuses, I think it's easiest to mimic
the call to `h5_consensus.hdf5_meet_consensus` below in a
loop in an IPython session.
"""
import argparse
import logging
import os
from os.path import exists, isdir, join

from ffn.utils import h5_consensus
from ffn.utils import subvols


logging.basicConfig(level=logging.INFO)


# -- args

ap = argparse.ArgumentParser()

ap.add_argument(
    "output",
    help="Path to hdf5 to write to. Must not exist. "
    "If --transpose is supplied, this should be a directory rather than "
    "a .h5, and possibly several HDF5 files will be created there.",
)
ap.add_argument("--dset", default="seg", help="Dataset to write in HDF5.")
ap.add_argument(
    "--segmentation-dirs",
    metavar="SEGDIR",
    type=str,
    nargs="+",
    help="At least 1 FFN segmentation directory, each possibly "
    "containing an inference split over many overlapping subvolumes."
    "Consensus will be carried out across these directories, "
    "reducing all of them down to a single segmentation. In other "
    "words, this script takes `M` inferences of the same region, "
    "and computes a single consensus segmentation of that region."
    "However, if you supply the --transpose flag, this flag will be "
    "interpreted differently. See that flag for details.",
)
ap.add_argument(
    "--min-ffn-size",
    type=int,
    default=0,
    help="Minimum size segment to allow when loading inferences.",
)
ap.add_argument(
    "--transpose",
    action="store_true",
    help="This flag 'transposes' the 'axes' of the consensus. If set, "
    "and if `M` directories are passed to --segmentation_dirs, each "
    "of those directories should contain `N` FFN segmentation result "
    "directories. In this setting, `output` should be a directory, "
    "and up to `N` .h5 consensus segmentations will be written there "
    "by looking for subdirectories of the --segmentation_dirs with "
    "matching names and existing compatible segmentations, and writing "
    "a .h5 to `output` with a name based on their name.",
)

args = ap.parse_args()


# -- run it

if not args.transpose:
    h5_consensus.hdf5_meet_consensus(
        args.out_fn,
        args.segmentation_dirs,
        dset=args.dset,
        min_ffn_size=args.min_ffn_size,
    )
else:
    # check output path does not exist or is an empty directory
    # make directory if necessary
    if exists(args.output):
        assert isdir(args.output) and not os.listdir(args.output)
    else:
        os.makedirs(args.output)

    # -- find matching subdirs with existing compatible segmentations
    # list all subdirs
    subdirs = []
    for segdir in args.segmentation_dirs:
        subdirs.append([d for d in os.listdir(segdir) if isdir(d)])
    logging.info(f"Found candidate segmentations: {subdirs}")

    # those that exist across all
    matching = [d for d in subdirs[0] if all(d in sd for sd in subdirs)]

    # check that those subdirs are finished
    finished = [
        d for d in matching
        if all(
            subvols.check_finished(join(segdir, d))
            for segdir in args.segmentation_dirs
        )
    ]

    # check that they had the same subvolume structure
    valid = [
        d for d in finished
        if subvols.check_same_subvolumes(
            [join(segdir, d) for segdir in args.segmentation_dirs],
            return_bool=True,
        )
    ]
    logging.info(f"{len(valid)}/{len(matching)} candidates were all good.")

    # build output paths
    h5_paths = [d + "_cons.h5" for d in valid]
    logging.info(f"Will write {h5_paths} in {args.output}")
    h5_paths = [join(args.output, h5p) for h5p in h5_paths]

    # -- loop consensus
    for h5p, d in zip(h5_paths, valid):
        if exists(h5p):
            logging.info(f"{h5p} exists. Skipping {d}.")

        segdirs = [join(segdir, d) for segdir in args.segmentation_dirs]
        logging.info(f"{h5p} <- consensus({segdirs})")
        h5_consensus.hdf5_meet_consensus(
            h5p,
            segdirs,
            dset=args.dset,
            min_ffn_size=args.min_ffn_size,
        )
