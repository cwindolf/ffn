from collections import namedtuple
import numpy as np
from ffn.inference import segmentation


UnsupMetrics = namedtuple(
    "UnsupMetrics",
    "density min_size median_size max_size n_islands",
)


def unsupervised_metrics(seg, margin=24):
    """These are grouped so that computations can be shared."""
    ccs = segmentation.split_disconnected_components(seg)

    # density
    foreground_mask = ccs != 0
    density = foreground_mask.mean()

    # get ids and sizes
    ids, sizes = np.unique(ccs[foreground_mask], return_counts=True)

    # size statistics
    np.sort(sizes)
    median_size = np.median(sizes)
    min_size = sizes.min()
    max_size = sizes.max()

    # ids that don't get near the boundary
    island_ids = np.unique(
        ccs[margin:-margin, margin:-margin, margin:-margin]
    )
    island_ids = np.setdiff1d(island_ids, [0])
    n_islands = island_ids.size

    return UnsupMetrics(
        density, min_size, median_size, max_size, n_islands
    )
