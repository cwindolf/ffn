import argparse
import glob
import ipdb
from multiprocessing.pool import ThreadPool
import os

import tqdm

import ffn.utils.datspec as dx
# pip install -e . in neuroglancer/python/
import neuroglancer as ng


# Shader for normalizing [-1, 1] data to [0, 1], which NG prefers
centered_shader = '''void main() {
  emitGrayscale(toNormalized((getDataValue() + 1.0) / 2.0));
}
'''


def extract_ints(string, matches=-1):
    '''Extract ints from strings

    Given a string containing some ints, return the ints at the
    positions indicated by matches.

    I often use this to sort lists of files, like maybe a folder
    has dir/1.png -- dir/144.png which don't sort lexicographically,
    but you can do `sorted(glob.glob("dir/*.png"), key=extract_ints)`
    to deal with that.

    If matches is a single index, return just that match.
    (e.g. if there are 5 integers in the string and matches=-1,
    the last one is returned. if matches=0, the first is returned.)
    If it's a sequence, return a list of matches at the indicated
    positions. (e.g. if matches=(0,-1), you will get the first
    and last integers in the string.)
    '''
    import re

    ints = re.findall(r'\d+', string)

    if matches is None:
        return [int(i) for i in ints]

    try:
        return [int(ints[i]) for i in matches]
    except TypeError:
        # `matches` was not iterable. Assuming it was just one
        # index into `ints`.
        return int(ints[matches])


def _loadspec_thread(spec):
    return dx.loadspec(spec), spec


def get_ckptdir_ckpt(seg_npz):
    corner_dir = seg_npz.split("/seg")[0]
    ckpt_val_dir = os.path.abspath(os.path.join(corner_dir, "..", ".."))
    return extract_ints(ckpt_val_dir)


def main():
    # args ------------------------------------------------------------
    ap = argparse.ArgumentParser()

    ap.add_argument(
        'specs', nargs='*',
        help='Some datspec volumes to display. If only one is '
        'supplied and it\'s a directory, treat it like a dir '
        'built by runwatcher.py and load all segs in there.'
    )
    ap.add_argument('--raw', required=True, help='Image stack.')
    # TODO: Allow multiple raws.
    #       Use case: Raw second specimen data together
    #       with SECGAN-processed version of that data.
    ap.add_argument(
        '--host', '--ip',  # Who can remember, let's do both.
        help='A flag where one can write 0.0.0.0 if they want.',
    )
    ap.add_argument('--names', nargs='+')

    args = ap.parse_args()

    if args.host:
        ng.set_server_bind_address(args.host)

    specs = args.specs
    if len(specs) == 1 and os.path.isdir(specs[0]):
        specs = sorted(
            glob.glob(os.path.join(specs[0], '*/*/*/seg*.npz')),
            key=get_ckptdir_ckpt,
        )
        names = [str(get_ckptdir_ckpt(s)) for s in specs]
        specs = [f'{s}:segmentation' for s in specs]
    elif not args.names:
        names = [
            os.path.basename(s.split(':')[0]).rstrip('.h5') for s in specs
        ]
    else:
        names = args.names
    assert len(specs) > 0

    # open ng server and add data -------------------------------------
    viewer = ng.Viewer()
    dimensions = ng.CoordinateSpace(
        names=['z', 'y', 'x'], units='nm', scales=[8, 8, 8]
    )

    # Set global coords
    with viewer.txn() as v:
        v.dimensions = dimensions

    # Load and normalize raw data for display. ng is happier with [0,1]
    print("Loading raw data...")
    raw = dx.loadspec(args.raw)
    if raw.dtype.kind == 'f':
        if 'gppx' not in args.raw:
            raw -= raw.min()
            raw /= raw.max()

    with viewer.txn() as v:
        v.layers['raw'] = ng.ImageLayer(
            source=ng.LocalVolume(
                data=raw, dimensions=dimensions
            ),
            shader=centered_shader,
        )
        v.layers['raw'].visible = False

    # Use case for this script is loading up a large number of
    # segmentations. So let's do that on many threads.
    print(f"Loading {len(specs)} segmentations...")
    with ThreadPool(min(len(specs), os.cpu_count())) as p:
        # Need imap not unordered to make sure things show up
        # in order in ng's top bar.
        for i, (volume, spec) in tqdm.tqdm(
            enumerate(p.imap(_loadspec_thread, specs)), total=len(specs)
        ):
            with viewer.txn() as v:
                # Tag segmentation layers with the checkpoint.
                # Of course this should be behind a flag if another use
                # case other than checkpoint selection shows up.
                name = names[i]

                v.layers[name] = ng.SegmentationLayer(
                    source=ng.LocalVolume(data=volume, dimensions=dimensions)
                )
                # Start with segs hidden so that you don't
                # have to hide them all yourself.
                v.layers[name].visible = False
                # default is 0.5 which is a little much.
                v.layers[name].selected_alpha = 0.35

    # Get viewer url
    print(str(viewer))

    # Need to keep the process alive. *~*
    ipdb.set_trace()
    # Hi. Feel free to interact with `viewer`.


# ---------------------------------------------------------------------
if __name__ == '__main__':
    main()

# see:
# https://github.com/google/neuroglancer/blob/master/python/README.md
# make sure you run all the steps and bundle client and whatever
# especially if you see any 404 warnings
