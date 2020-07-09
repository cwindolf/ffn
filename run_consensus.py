"""Combine possibly multiple inferences overlapping subvolumes

See docs for ffn/utils/h5_consensus.py and the function
`h5_meet_consensus` in that file for more details.

To run lots of consensuses, I think it's easiest to mimic
the call to `h5_consensus.hdf5_meet_consensus` below in a
loop in an IPython session.
"""
import argparse
from ffn.utils import h5_consensus

# -- args
ap = argparse.ArgumentParser()
ap.add_argument("out_fn", help="Path to hdf5 to write to. Must not exist.")
ap.add_argument("--dset", default="seg", help="Dataset to write in HDF5.")
ap.add_argument(
    "--segmentation-dirs",
    metavar="SEGDIR",
    type=str,
    nargs="+",
    help="At least 1 FFN segmentation directory, each possibly "
    "containing an inference split over many overlapping subvolumes.",
)
ap.add_argument(
    "--min-ffn-size",
    type=int,
    default=0,
    help="Minimum size segment to allow when loading inferences.",
)
args = ap.parse_args()

# -- run it
h5_consensus.hdf5_meet_consensus(
    args.out_fn,
    args.segmentation_dirs,
    dset=args.dset,
    min_ffn_size=args.min_ffn_size,
)
