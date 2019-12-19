import numpy as np
from scipy.special import logit
from skimage.util import view_as_windows
import preprocessing.data_util as dx
import logging


def fixed_seed_batch(
    batch_size, fov_size, seed_pad, seed_init, with_init=True, seed_logit=True
):
    '''A batch of "prior" seeds'''
    if isinstance(fov_size, int):
        fov_size = (fov_size, fov_size, fov_size)
    fixed_seed = np.full(fov_size, seed_pad, dtype=np.float32)
    if with_init:
        fov_center = tuple(list(np.array(fov_size) // 2))
        fixed_seed[fov_center] = seed_init
    # logit or not ???
    if seed_logit:
        fixed_seed_batch = np.array([logit(fixed_seed)] * batch_size)[
            ..., None
        ]
    else:
        fixed_seed_batch = np.array([fixed_seed] * batch_size)[..., None]
    return fixed_seed_batch


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
    fov_size : array-like with three ints or one int
        Size of fovs to extract
    permutable_axes : tuple of pairs from (0, 1, 2)
    reflectable_axes : sub-tuple of (0, 1, 2)
        These define augmentations, which are run live batchwise.
    '''
    if isinstance(fov_size, int):
        fov_size = (fov_size, fov_size, fov_size)

    # Load / preprocess array
    volume = dx.loadspec(volume_spec)
    volume = volume.astype(np.float32)
    if image_mean is not None:
        volume = (volume - image_mean) / image_stddev
    else:
        # Map to [-1, 1]
        logging.info(f'No image_mean passed when loading {volume_spec}')
        logging.info(f'Using secgan preprocessing.')
        volume /= 127.5
        volume -= 1.0

    # Stride with fov_size to get patches
    all_fovs = view_as_windows(volume, fov_size)
    fovs_per_side = all_fovs.shape[0]

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
                np.random.randint(fovs_per_side, size=batch_size),
                np.random.randint(fovs_per_side, size=batch_size),
                np.random.randint(fovs_per_side, size=batch_size),
            ]
            + 0.0
        )

        # Augment
        if pax:
            batch = batch.swapaxes(*np.random.choice(permutable_axes, size=2))
        if rax:
            batch = np.flip(batch, axis=np.random.choice(reflectable_axes))

        yield batch[..., None]


def multi_random_fovs(
    volume_specs,
    batch_size,
    fov_size,
    permutable_axes=(0, 1, 2),
    reflectable_axes=(0, 1, 2),
):
    '''A generator looping through batches of  random fovs in volume

    This will center and scale the dataset by image_mean/stddev,
    and apply basic augmentations.

    Arguments
    ---------
    volume_spec : datspec pointing to 3d array
    fov_size : array-like with three ints or one int
        Size of fovs to extract
    permutable_axes : tuple of pairs from (0, 1, 2)
    reflectable_axes : sub-tuple of (0, 1, 2)
        These define augmentations, which are run live batchwise.
    '''
    if isinstance(fov_size, int):
        fov_size = (fov_size, fov_size, fov_size)

    # Load / preprocess array
    logging.info(f'multi_fovs using secgan preprocessing.')
    volumes = []
    for spec in volume_specs:
        v = dx.loadspec(spec).astype(np.float32)
        v /= 127.5
        v -= 1.0
        volumes.append(v)

    # Stride with fov_size to get patches
    all_fovs = []
    fovs_per_side = []
    n_fovs = []
    for v in volumes:
        v_fovs = view_as_windows(v, fov_size)
        fovs_per_side.append(v_fovs.shape[0])
        all_fovs.append(v_fovs)
        n_fovs.append(v_fovs.shape[0] * v_fovs.shape[1] * v_fovs.shape[2])
    tot_fovs = sum(n_fovs)

    # Add 1 to augmentation axes to avoid batch dim
    pax = bool(permutable_axes)
    rax = bool(reflectable_axes)
    if pax:
        permutable_axes = np.asarray(permutable_axes) + 1
    if rax:
        reflectable_axes = np.asarray(reflectable_axes) + 1

    # Loop to yield batches
    while True:
        # Weighted volume choice
        loc = np.random.randint(tot_fovs, dtype=int)
        for i, n in enumerate(n_fovs):
            loc = loc - n
            if loc <= 0:
                break

        # Load
        batch = (
            all_fovs[i][
                np.random.randint(fovs_per_side[i], size=batch_size),
                np.random.randint(fovs_per_side[i], size=batch_size),
                np.random.randint(fovs_per_side[i], size=batch_size),
            ]
            + 0.0
        )

        # Augment
        if pax:
            batch = batch.swapaxes(*np.random.choice(permutable_axes, size=2))
        if rax:
            batch = np.flip(batch, axis=np.random.choice(reflectable_axes))

        yield batch[..., None]


def batch_by_fovs(volume, fov_size, batch_size):
    '''Basically, extract volume patches onto batch dimension
    '''
    # Support channel dimension
    volume = volume.squeeze()
    if volume.ndim == 4:
        fov_size = (*fov_size, volume.shape[3])

    # Now yield the batches
    ilim = volume.shape[0] - fov_size[0]
    jlim = volume.shape[1] - fov_size[1]
    klim = volume.shape[2] - fov_size[2] - batch_size
    logging.info(f'batch_by_fovs: ({ilim},{jlim},{klim}).')
    for i in range(0, ilim):
        logging.info(f'{i}/{ilim}')
        for j in range(0, jlim):
            for k in range(0, klim, batch_size):
                islice = slice(i, i + fov_size[0])
                jslice = slice(j, j + fov_size[1])
                subvolume = volume[
                    islice, jslice, k : k + fov_size[2] + batch_size - 1
                ]
                subvolume_patches = view_as_windows(subvolume, fov_size)
                # logging.info(f'subvolume: {subvolume.shape}')
                # logging.info(f'patches: {subvolume_patches.shape}')
                # logging.info(f'for fov, batch = {fov_size}, {batch_size}')
                # assert subvolume_patches.shape[0] == 1
                # assert subvolume_patches.shape[1] == 1
                # assert subvolume_patches.shape[2] == batch_size
                if volume.ndim == 4:
                    assert subvolume_patches.shape[3] == 1
                    subvolume_patches = subvolume_patches[:, :, :, 0]
                slices = [
                    (islice, jslice, slice(k_, k_ + fov_size[2]))
                    for k_ in range(k, k + batch_size)
                ]
                batch = subvolume_patches[0, 0]
                if batch.ndim == 4:
                    batch = batch[..., None]
                yield slices, batch
