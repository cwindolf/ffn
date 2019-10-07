import numpy as np
from skimage.util import view_as_windows
import preprocessing.data_util as dx


def random_fovs(
    volume_spec,
    batch_size,
    fov_size,
    image_mean=128.0,
    image_stddev=33.0,
    permutable_axes=(0, 1, 2),
    reflectable_axes=(0, 1, 2),
):
    '''A generator looping through batches of  random fovs in volume

    This will center and scale the dataset by image_mean/stddev,
    and apply basic augmentations.

    Arguments
    ---------
    volume_spec : datspec pointing to 3d array
    fov_size : array-like with three ints
        Size of fovs to extract
    permutable_axes : tuple of pairs from (0, 1, 2)
    reflectable_axes : sub-tuple of (0, 1, 2)
        These define augmentations, which are run live batchwise.
    '''
    # Load / preprocess array
    volume = dx.loadspec(volume_spec)
    volume = (volume.astype(np.float32) - image_mean) / image_stddev

    # Stride with fov_size to get patches
    all_fovs = view_as_windows(volume, fov_size)
    n_fovs = all_fovs.shape[0]

    # Add 1 to augmentation axes to avoid batch dim
    pax = bool(permutable_axes)
    rax = bool(reflectable_axes)
    if pax:
        permutable_axes = np.asarray(permutable_axes) + 1
    if rax:
        reflectable_axes = np.asarray(reflectable_axes) + 1

    # Loop to yield batches
    while True:
        # Load
        batch = (
            all_fovs[
                np.random.randint(n_fovs, size=batch_size),
                np.random.randint(n_fovs, size=batch_size),
                np.random.randint(n_fovs, size=batch_size),
            ]
            + 0.0
        )

        # Augment
        if pax:
            batch = batch.swapaxes(*np.random.choice(permutable_axes, size=2))
        if rax:
            batch = np.flip(batch, axis=np.random.choice(reflectable_axes))

        yield batch[..., None]
