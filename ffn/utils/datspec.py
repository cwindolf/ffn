import glob
import os.path
import imageio
import numpy as np
import h5py

def parse_slice_expr(slice_expr):
    '''Parse a slice expression into an actual slice

    Examples:
     - Given the string "1:30,:20,10:", this returns
       the tuple:
            (slice(1, 30, None),
             slice(None, 20, None),
             slice(10, None, None))
     - On the input "1,:" it returns
            (1, slice(None, None, None))

    If the first tuple above were stored in a variable `slicer`, you
    could use it to slice a numpy array `x`:
        x.shape == (200, 200, 200)
    implies
        x[slicer].shape == (29, 20, 190)

    This can also deal with the brackets and handle well-formed
    expressions with whitespace in them.

    Arguments
    ---------
    slice_expr : str
        A comma-separated bunch of colon-separated integers/
        empty strings. There should be at most 2 colons in each bunch.

    Returns: a tuple of `slice` objects.
    '''
    try:
        return tuple(
            slice(
                *(int(n.strip()) if n.strip() else None for n in s.split(':'))
            )
            if ':' in s
            else int(s.strip())
            for s in slice_expr.strip('[]').split(',')
        )
    except ValueError:
        raise ValueError(f'Could not parse slice expression: {slice_expr}')


def parse_spec(spec):
    '''Parse a datspec, as described in loadspec doc

    In more detail, a datspec is a string in the form:
        <resource>[:<dataset>][<slice>][@<in_ax>]

    <resource> should be a file name.

    <dataset> indicates which dataset in the file should be loaded --
    this is a concept that array storage formats like .npz, .mat, .h5
    support, and datspecs are meant to be used to describe those
    resources. On the other hand, .mrc and .npy don't use this
    parameter one, so it's optional.

    The [<slice>] part means an optional slice notation. This should
    just look like a typical python indexing expression:
    "[10,14:20,::-1]"

    The [@in_ax] part means an optional suffix like `@xyz` describing
    the axis order of the data represented by this datspec on disk.
    Used in conjunction with the `out_ax` kwarg of loadspec to transpose
    the resource at load time.

    Arguments
    ---------
    spec : str
        A datspec as described above.

    Returns
    -------
    parsed_spec : 4-tuple of string
        The (file path, group in file, slice in group, axis order)
        The last three can be None but the first is always there.
    '''
    # get resource part -----------------------------------------------
    srcf = spec + ''
    if '[' in srcf:
        srcf = srcf.split('[')[0]
    if ':' in srcf:
        srcf = srcf.split(':')[0]

    # normalize data path
    srcf = os.path.abspath(os.path.expanduser(srcf))

    # get dataset part -- optional ------------------------------------
    dset = None
    if ':' in spec:
        possible_dset = spec + ''
        if '@' in possible_dset:
            possible_dset = possible_dset.split('@')[0]
        if '[' in possible_dset:
            possible_dset = possible_dset.split('[')[0]
        if ':' in possible_dset:
            dset = possible_dset.split(':')[1]

    # get slice part -- optional --------------------------------------
    slice_expr = None
    if '[' in spec:
        after_bracket = spec.split('[')[1]
        assert ']' in after_bracket
        slice_expr = after_bracket.split(']')[0]

    parsed_spec = srcf, dset, slice_expr
    return parsed_spec


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
    # -- parse datspec
    path, dset, slice_expr = parse_spec(datspec)

    # -- now, load up the data
    if path.endswith(".npy"):
        assert dset is None, "You passed a dset when loading .npy?"
        data = np.load(path, **kwargs)
        if slice_expr is not None:
            return data[slice_expr]
        else:
            return data

    elif path.endswith(".npz"):
        if dset is None:
            # try to guess it
            with np.load(path) as npz:
                dsets = list(npz.keys())
            assert len(dsets) == 1, f"You pick the dset, ok? {dsets}"
            dset = dsets[0]

        data = np.load(path, **kwargs)[dset]
        if slice_expr is not None:
            return data[slice_expr]
        else:
            return data

    elif path.endswith(".h5"):
        if dset is None:
            # try to guess it
            with h5py.File(path, "r") as h5:
                dsets = set(h5.keys())
            assert len(dsets) == 1, f"You pick the dset, ok? {dsets}"
            dset = dsets[0]

        with h5py.File(path, "r") as h5:
            data = h5[dset]
            if slice_expr is not None:
                return data[slice_expr]
            else:
                return data

    elif os.path.isdir(path):
        pattern = os.path.join(path, f"*.{dset}")
        files = list(sorted(glob.glob(pattern)))
        assert files, f"No files matched glob {pattern}."
        data = np.array([imageio.imread(f) for f in files])
        if slice_expr is not None:
            return data[slice_expr]
        else:
            return data

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
