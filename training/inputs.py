import numpy as np
from skimage.util import view_as_windows
import preprocessing.data_util as dx


def random_fovs(
    volume_spec, batch_size, fov_size, image_mean=128.0, image_stddev=33.0
):
    '''A generator looping through batches of  random fovs in volume

    This will center and scale the dataset by image_mean/stddev.

    Arguments
    ---------
    volume_spec : datspec pointing to 3d array
    fov_size : array-like with three ints
        Size of fovs to extract
    '''
    # Load / preprocess array
    volume = dx.loadspec(volume_spec)
    volume = (volume.astype(np.float32) - image_mean) / image_stddev

    # Stride with fov_size to get patches
    all_fovs = view_as_windows(volume, fov_size)
    print('FOVs shape:', all_fovs.shape)
    n_fovs = all_fovs.shape[0]

    # Loop
    while True:
        yield all_fovs[np.random.randint(n_fovs, size=batch_size)] + 0.0
