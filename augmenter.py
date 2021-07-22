import numpy as np
import torch
from tqdm import tqdm

from utils import get_everything_from_adios2


def augment_along_fieldlines(
    Zif,
    rz,
    surf_idx,
    train_indices,
    subdivisions: int = 2,
):
    """
    Zif is original data,
    rz is coordinates of original data,
    surf_idx is an array of the fieldlines
    train_indices is an array of indices

    Between any two neighbors on a fieldline, add in subdivisions-1 many images.

    Returns the tuple (Zif, rz) with the augmented data.
    """
    if subdivisions > 2:
        raise NotImplementedError
    # Might be faster to preallocate the array.
    imgs = []
    coords = []
    # This speeds up the lookups in the loop
    train_indices = set(train_indices)
    for fieldline in tqdm(surf_idx):
        fieldline = np.trim_zeros(fieldline)
        for (first, second) in zip(fieldline, fieldline[1:]):
            if first in train_indices and second in train_indices:
                avg = np.mean(Zif[[first, second]], axis=0)
                coord = np.mean(rz[[first, second]], axis=0)
                assert coord.shape == (2,)
                imgs.append(avg)
                coords.append(coord)

    imgs = np.stack(imgs)
    coords = np.stack(coords)
    imgs = np.concatenate((Zif, imgs))
    coords = np.concatenate((rz, coords))

    return imgs, coords


if __name__ == "__main__":
    """
    This just tests the above code.
    """
    (
        Z0,
        Zif,
        zmu,
        zsig,
        zmin,
        zmax,
        zlb,
        nnode,
        ncells,
        rz,
        conn,
        psi,
        nextnode,
        epsilon,
        node_vol,
        node_vol_nearest,
        psi_surf,
        surf_idx,
        surf_len,
        theta,
    ) = get_everything_from_adios2()

    assert surf_idx.shape == (101, 338)
    assert nnode == 16395
    H = 39
    W = 39
    assert Zif.shape == (nnode, H, W)

    test_split = 0.8
    all_indices = torch.randperm(nnode)
    train_indices = all_indices[: int(test_split * nnode)].tolist()
    test_indices = all_indices[int(test_split * nnode) :].tolist()

    imgs, coords = augment_along_fieldlines(Zif, rz, surf_idx, train_indices)
