import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

from utils import get_everything_from_adios2
from XGC import XGC


def augmented_dataset_along_fieldlines(
    dataset,
    fieldlines,
    subdivisions: int = 2,
):
    """
    surf_idx is an array of the fieldlines

    Between any two neighbors on a fieldline, add in subdivisions-1 many images.

    Returns the tuple (Zif, rz) with the augmented data.
    """
    if subdivisions > 2:
        raise NotImplementedError
    # Might be faster to preallocate the array.
    imgs = []
    coords = []
    nodeids = dataset[:][-1].tolist()

    reindex = {nodeid: nodeids.index(nodeid) for nodeid in nodeids}

    # Make this a set to speed up lookups in the loop
    set_nodeids = set(nodeids)

    for fieldline in tqdm(fieldlines):
        fieldline = np.trim_zeros(fieldline)
        for (first, second) in zip(fieldline, fieldline[1:]):
            if first in set_nodeids and second in set_nodeids:
                first = reindex[first]
                second = reindex[second]
                stacked_img = torch.stack((dataset[first][0], dataset[second][0]))
                stacked_coord = torch.stack((dataset[first][1], dataset[second][1]))
                img = torch.mean(stacked_img, axis=0)
                coord = torch.mean(stacked_coord, axis=0)
                assert img.shape == (39, 39)
                assert coord.shape == (2,)
                imgs.append(img)
                coords.append(coord)

    return TensorDataset(
        torch.stack(imgs),  # imgs
        torch.stack(coords),  # coords
        torch.tensor([-1] * len(imgs)),  # set nodeid of synthetic data to -1
    )


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

    dataset = XGC()
    print("Augmenting on whole dataset")
    augmented_dataset = augmented_dataset_along_fieldlines(
        dataset=dataset,
        fieldlines=surf_idx,
    )

    if len(dataset) == nnode:
        # If we use the full XGC dataset, there are no broken pairs, and we can calculate exactly how many images are being added.
        assert len(augmented_dataset) == sum(np.count_nonzero(surf_idx, axis=1) - 1)

    from torch.utils.data import Subset

    subset = Subset(dataset, indices=range(20, 40))
    print("Augmenting on small subset")
    augmented_dataset = augmented_dataset_along_fieldlines(
        dataset=subset,
        fieldlines=surf_idx,
    )
