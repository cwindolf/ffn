import glob
import os.path
import imageio
import numpy as np
import h5py


def loadspec(datspec, **kwargs):
    """loadspec: load data from a couple of formats.

    Parameters
    ----------
    datspec : str
        Should be <path>[:<dset>], a so-called datspec that indicates
        what data to load.
            if <path> is          then      <dset> should be:
                a .npz file,       ->           name of array in .npz
                a .npy file, or    ->           not present
                a directory.       ->           an extension, like .png
        This function will grab the array from the .npz, load the .npy,
        or load all .png files (sort order) from the directory in those
        cases.

    Returns
    -------
    A numpy array.
    """
    # -- split path and dset if present
    dset = None
    if ":" in datspec:
        assert sum(c == ":" for c in datspec) == 1
        path, dset = datspec.split(":")
    else:
        path = datspec

    # -- expand
    path = os.path.abspath(os.path.expanduser(path))

    # -- now, load up the data
    if path.endswith(".npy"):
        assert dset is None, "You passed a dset when loading .npy?"
        return np.load(path, **kwargs)

    elif path.endswith(".npz"):
        if dset is None:
            # try to guess it
            with np.load(path) as npz:
                dsets = list(npz.keys())
            assert len(dsets) == 1, f"You pick the dset, ok? {dsets}"
            dset = dsets[0]

        return np.load(path, **kwargs)[dset]

    elif path.endswith(".h5"):
        if dset is None:
            # try to guess it
            with h5py.File(path, "r") as h5:
                dsets = set(h5.keys())
            assert len(dsets) == 1, f"You pick the dset, ok? {dsets}"
            dset = dsets[0]

        with h5py.File(path, "r") as h5:
            return h5[dset]

    elif os.path.isdir(path):
        pattern = os.path.join(path, f"*.{dset}")
        files = list(sorted(glob.glob(pattern)))
        assert files, f"No files matched glob {pattern}."
        return np.array([imageio.imread(f) for f in files])

    else:
        assert 0, f"Invalid datspec {datspec} -> path='{path}', dset='{dset}'."


def writeh5spec(datspec, data, h5_attrs=None, overwrite_behavior='prompt'):
    """Given a datspec path:dset, write the dataset `dset` in h5 file `path`

    Will also write hdf5 attributes if supplied, and will try to be nice
    when the data already exists in the file.

    Arguments
    ---------
    datspec : str
        String like <path to hdf5 file>:<group in hdf5 file>
    data : numpy array or similar
        Data to write to the dataset in the hdf5 file
    h5_attrs : dict, optional
        Attributes to tag the dataset with in the hdf5 file.
    overwrite_behavior: one of "always", "quit", "raise", "prompt"
        This controls what happens when data exists. By default,
        prompt user to decide what to do.
    """
    assert sum(c == ":" for c in datspec) == 1
    path, dset = datspec.split(":")
    assert path.endswith(".h5")

    with h5py.File(path, "a") as out_f:
        has_dset = dset in out_f

        if not has_dset:
            # OK, no overwriting, just go a head and write.
            out_f.create_dataset(dset, data=data)
            return

        msg = f'Dataset "{dset}" already exists in {path}.'

        # Handle overwrite
        if overwrite_behavior == 'always':
            pass
        elif overwrite_behavior == 'quit':
            return
        elif overwrite_behavior == 'raise':
            raise ValueError(msg)
        elif overwrite_behavior == 'prompt':
            print(msg)
            while True:
                res = input('Overwrite? y/n\n')
                res = res.lower()
                if res == 'n':
                    return
                elif res == 'y':
                    break
                else:
                    print(f'Did not understand {res}')

        # Perform overwrite
        del out_f[dset]
        out_f.create_dataset(dset, data=data)

        if h5_attrs is not None:
            for k, v in h5_attrs.items():
                out_f[dset].attrs[k] = v
