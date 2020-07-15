from collections import namedtuple
import numpy as np
from ffn.inference import segmentation


UnsupMetrics = namedtuple(
    "UnsupMetrics",
    "density nsegs min_size mean_size median_size max_size "
    "n_islands split_nsegs split_min_size split_mean_size "
    "split_median_size split_max_size split_n_islands "
    "decrumbed_nsegs decrumbed_density decrumbed_min_size "
    "decrumbed_mean_size decrumbed_median_size decrumbed_max_size "
    "decrumbed_n_islands",
)


def unsupervised_metrics(seg, margin=24):
    """These are grouped so that computations can be shared."""
    # density
    foreground_mask = seg != 0
    density = foreground_mask.mean()

    # size statistics
    ids, sizes = np.unique(seg[foreground_mask], return_counts=True)
    nsegs = ids.size
    median_size = np.median(sizes)
    mean_size = sizes.mean()
    min_size = sizes.min()
    max_size = sizes.max()

    # ids that don't get near the boundary
    island_ids = np.unique(
        seg[margin:-margin, margin:-margin, margin:-margin]
    )
    island_ids = np.setdiff1d(island_ids, [0])
    n_islands = island_ids.size

    # split cc size statistics
    ccs = segmentation.split_disconnected_components(seg)
    # get ids and sizes
    split_ids, split_sizes = np.unique(
        ccs[foreground_mask], return_counts=True
    )
    split_nsegs = split_ids.size
    split_median_size = np.median(split_sizes)
    split_mean_size = split_sizes.mean()
    split_min_size = split_sizes.min()
    split_max_size = split_sizes.max()

    # ids that don't get near the boundary
    split_island_ids = np.unique(
        ccs[margin:-margin, margin:-margin, margin:-margin]
    )
    split_island_ids = np.setdiff1d(split_island_ids, [0])
    split_n_islands = split_island_ids.size

    # decrumbed statistics
    decrumbed = segmentation.clear_dust(seg)
    decrumbed_foreground = decrumbed != 0
    decrumbed_density = decrumbed_foreground.mean()
    decrumbed_ids, decrumbed_sizes = np.unique(
        decrumbed[decrumbed_foreground], return_counts=True
    )
    decrumbed_nsegs = decrumbed_ids.size
    decrumbed_median_size = np.median(decrumbed_sizes)
    decrumbed_mean_size = decrumbed_sizes.mean()
    decrumbed_min_size = decrumbed_sizes.min()
    decrumbed_max_size = decrumbed_sizes.max()

    # ids that don't get near the boundary
    decrumbed_island_ids = np.unique(
        ccs[margin:-margin, margin:-margin, margin:-margin]
    )
    decrumbed_island_ids = np.setdiff1d(decrumbed_island_ids, [0])
    decrumbed_n_islands = decrumbed_island_ids.size

    return UnsupMetrics(
        density,
        nsegs,
        min_size, mean_size, median_size, max_size,
        n_islands,
        split_nsegs,
        split_min_size, split_mean_size, split_median_size, split_max_size,
        split_n_islands,
        decrumbed_nsegs, decrumbed_density,
        decrumbed_min_size, decrumbed_mean_size, decrumbed_median_size,
        decrumbed_max_size, decrumbed_n_islands,
    )
