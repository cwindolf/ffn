r"""Meet consensus for overlapping volumes

Please observe that the split oversegmentation operation forms
an abelian group on the set of segmentations. So, it doesn't
matter if we first apply the split consensus to merge subvols
into large vols or if we apply on subvols first. In fact, this
split consensus can be considered as the meet operation on a
certain poset described below.

In other words, we want to "reduce" a bunch of segs using the
split consensus and foldl == foldr.

It's easy to parallelize the bit that works over subvolumes,
so we'll fan out over that and then funnel those results into
an output volume, which I guess we'll just keep as a memmaped
hdf5 file.

Join and meet for segmentations
-------------------------------

Recall the notions of join and meet:
https://en.wikipedia.org/wiki/Join_and_meet
Note that we can define a partial order on segmentations as
follows. Let $a$ and $b$ be two segmentations of the same
region, i.e., $a,b : Z_{pqr} -> N$ are two
assignments of natural numbers to every site on the finite
3D lattice $Z_{pqr} := Z_p \times Z_q \times Z_r$. (There's
nothing important about the lattice being 3D, in fact let's
forget about it and equivalently consider $Z_{pqr}$ to be
the enumeration of its sites $Z_{pqr}=\{0,...,pqr\}$.

OK, now we can define a partial order, using the symbol $\leq$.
This relation can be read "is splittier than" in a sense.
Intuitively, we say that $a \leq b$ when every segment inside
$a$ is part of only one segment in $b$, so that either $a$ is
just less dense than $b$ (in the specific sense that every
segmentation in $a$ is smaller than the one in $b$ or just
absent), or it's more split than $b$. More precisely, every
segment in $a$ can be obtained by erasing voxels in a single
segment from $b$ (more than one segment in $a$ can arise from
a segment in $b$ in this way). Equivalently and more formally,
let a segment in a segmentation $x$ be a level set
$S_k = \{z \in Z_{pqr} : x(z) = k\}$ of $x$ where $k \neq 0$.
Then, $a \leq b$ when for all segments $S_k$ of $a$, there
exists some $j\in N$ such that $S_k \subseteq S_j$ and $S_j$
is a segment in $b$.

Some examples: Let $a$ be some segmentation, and say that
there exist voxels labeled 1 and 2 in $a$. Let $f: N -> N$
be the function such that $f(n)=n$ for all $n$ except $n=2$,
and $f(2)=1$. Then the composition $b(z) = f(a(z))$ is the
segmentation equal to $a$ but with 1 and 2 merged. Clearly
the segment $b == 2$ is empty, and $b==1$ is the union of
$a==1$ and $a==2$. For all $n\notin\{1,2\}$, $b==n$ is the
same as $a==n$. Thust $a \leq b$, so that merges move you
up the partial order.

Similarly, erasing voxels and splitting subvoxels are
operations that move you down the partial order.

Now that we have a handle on this partial order, let's
consider its meet and join operations.

For meet: let $c,d$ be segmentations, and don't assume
that $c\leq d$ or $d\leq c$. Form a new segmentation
$e$ (the join) as follows.  Let $J: N\times N -> N$ be
the canonical bijection of the pairs of natural numbers
to the natural numbers, and make sure $J(0,0)=0$. Then
let $e(z)=J(c(z),d(z))$. I claim that $e\leq c$ and
$e\leq d$, and what's more that there is no $e\leq e'$
such that $e'\leq c$, $e'\leq d$, and $e\neq e'$.

This meet is defined in a function below. It's a
commutative version of the split seg by intersection
function in ffn.inference.segmentation.

For join: basically the idea is, for any segment ids that
overlap in $c$ and $d$, they need to get merged. No choice.
I'll let the reader formalize this.

Notice that the suprema of this poset lattice, the
kind of Zorn's lemma maximal elements, are the segmentations
that assign all sites to a single id $k\neq 0$. This suggests
that we should really "mod out" the permutations of seg IDs,
i.e., consider the equivalence classes of the eq. rel. $a==b$
if there exists a permutation $\sigma$ s.t. $\sigma(a(z))=b(z)$
for all $z$.

Interestingly, the minimal element is unique: the 0 segmentation.
"""
import logging

import numpy as np

from ffn.inference import segmentation
from ffn.inference import storage


# -- basic library for computing meet of two segmentations

def merge_from_min_id(out, seg, mask, min_new_id, scratch=None):
    """Write seg[mask] into out[mask] using a new contiguous ID
    space starting from min_new_id.
    """
    assert out.shape == seg.shape
    assert min_new_id > 0
    # Write the segmentation into scratch
    if scratch is not None:
        assert scratch.shape == seg.shape
        scratch.fill(0)
    else:
        scratch = np.zeros(out.shape, dtype=np.uint32)
    scratch[mask] = seg[mask]
    # Split CCs
    segmentation.clean_up(scratch)
    # Linear ID space starting at min_new_id
    relabeled, id_map = segmentation.make_labels_contiguous(scratch)
    max_new_id = min_new_id + max(new_id for _, new_id in id_map)
    assert max_new_id < np.iinfo(np.uint32).max
    # Write new IDs and check invariant
    scratch[mask] = min_new_id + relabeled[mask]
    assert min_new_id == max_new_id or scratch.max() == max_new_id
    # Write output and update ID invariant
    out[mask] = scratch[mask]
    return max_new_id


def seg_meet(a, b, min_new_id=1):
    """Merge a and b into a single segmentation by "split consensus"

    This implements the "meet" operation described in detail in this
    file's docstring.

    Parameters
    ----------
    a, b : np.array, integer typed, same shape
    min_new_id : int
        The output's ID space will start here

    Returns: uint32 np.array with same shape as `a`
    """
    if a.shape != b.shape:
        raise ValueError("Segmentations had different shapes.")
    assert min_new_id < np.iinfo(np.uint32).max
    # Boolean foreground / background masks
    a_fg = a != 0
    b_fg = b != 0
    b_and_a = np.logical_and(b_fg, a_fg)
    b_not_a = np.logical_and(b_fg, np.logical_not(a_fg))
    a_not_b = np.logical_and(a_fg, np.logical_not(b_fg))
    # Sparsify
    b_and_a = b_and_a.nonzero()
    b_not_a = b_not_a.nonzero()
    a_not_b = a_not_b.nonzero()
    # Output storage
    out = np.zeros(a.shape, dtype=np.uint32)
    scratch = np.empty(a.shape, dtype=np.uint32)
    # Perform the merges
    new_max_id = merge_from_min_id(
        out, a, a_not_b, min_new_id, scratch=scratch
    )
    new_max_id = merge_from_min_id(
        out, b, b_not_a, new_max_id + 1, scratch=scratch
    )
    new_max_id = merge_from_min_id(
        out, a, b_and_a, new_max_id + 1, scratch=scratch
    )
    assert new_max_id < np.iinfo(np.uint32).max

    logging.info(f"seg_meet max id {new_max_id}")

    return out


# -- asymmetric merge, not in-place

def paste_new_seg(a, b, old_max_id=0):
    """Paste `b` into the background of `a`

    Merge keeps `a` fixed, and inserts segments from `b` into the
    background, remapping them to a new contiguous id space. This
    is done in a copy, input arrays are not modified.

    To preserve the contiguity of IDs, this also handles the
    cleaning step (split CCs and cull vols smaller than min_size),
    so that those IDs aren't removed along the way.

    Since contiguity is a priority, this also ensures that things
    fit into 32 bit uints.

    Arguments
    ---------
    a : 3d uint32 arr
        The main volume segmentation for this subvolume
    b : 3d uint32 arr
        The new segmentation for that subvolume
    old_max_id : int
        The maximum ID in the main volume

    Returns
    -------
    merged : 3d uint32 arr
        The result of the merge
    new_max_id : int
        The new maximum ID after this merge
    old_max_id : int
        Maximum id of `a` before merging
    b_not_a : indices
        The indices where `b` had segmentation and `a` had 0,
        which is the same as the indices that were part of the
        merge.
    """
    if a.shape != b.shape:
        raise ValueError("a and b had different shape.")

    # figure out min ID for new segments
    old_max_id = max(a.max(), old_max_id)
    min_new_id = old_max_id + 1
    assert min_new_id < np.iinfo(np.uint32).max

    # foreground and background indices
    a_fg = a != 0
    b_not_a = np.logical_and(b != 0, np.logical_not(a_fg)).nonzero()
    a_fg = a_fg.nonzero()

    # compute and write the pasted result
    merged = np.zeros(a.shape, dtype=np.uint32)
    merged[a_fg] = a[a_fg].astype(np.uint32)
    new_max_id = merge_from_min_id(merged, b, b_not_a, min_new_id)

    return merged, new_max_id, old_max_id, b_not_a


# -- utility for computing meet consensus on FFN output

def load_and_compute_meet(segmentation_dirs, subvolume=None, min_ffn_size=0):
    """Iteratively compute meet on multiple segmentations

    Loads segmentation at `subvolume` for each directory in
    `segmentation_dirs`, and iteratively computes the meet.

    Arguments
    ---------
    segmentation_dirs : list of str
        Directories containing segmentations. The result of this
        function will be the meet of the segmented `subvolume` from
        each directory in `segmentation_dirs`
    subvolume : optional BoundingBox
        Will load the subvolume whose min corner is subvolume.start.
        If None, loads the subvolume at (0, 0, 0)

    Returns
    -------
    subvol_slice : tuple of slice
        The representation of this subvolume as a slice. Makes it
        easy to assign the result of this function into a global
        space.
    seg : np.array
        The meet described above.
    """
    # location of segs to load
    load_corner = (0, 0, 0)
    if subvolume is not None:
        load_corner = subvolume.start

    # load first seg
    seg, _ = storage.load_segmentation(
        segmentation_dirs[0], load_corner, min_size=min_ffn_size
    )
    seg = seg.astype(np.uint64)

    # reduce with meet over the rest of the seg
    # meet is commutative, so the order doesn't matter
    for segdir in segmentation_dirs[1:]:
        seg_b, _ = storage.load_segmentation(
            segdir, load_corner, min_size=min_ffn_size
        )
        assert seg.shape == seg_b.shape
        seg = seg_meet(seg, seg_b.astype(np.uint64))

    subvol_slice = tuple(reversed(subvolume.to_slice()))
    return subvol_slice, seg
