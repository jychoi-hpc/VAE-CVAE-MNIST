import numpy as np
import torch
from torch.utils.data import TensorDataset

from augmenter import augment_along_fieldlines
from utils import get_everything_from_adios2


def XGC(
    extend_angles: bool = False,
    coordinate: str = "cartesian",
    extra_channels: bool = False,
    nodes_for_augmentation=[],
):
    """
    Returns TensorDataset, for input into DataLoader

    nodes_for_augmentation is a list of indices.
    If nodes_for_augmentation is nonempty, then perform augmentation for only nodes in nodes_for_augmentation.
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

    if nodes_for_augmentation:
        Zif, rz = augment_along_fieldlines(Zif, rz, surf_idx, nodes_for_augmentation)

    N, H, W = Zif.shape

    if coordinate == "cartesian":
        pass
    elif coordinate == "polar":
        shifted = rz - rz[0]
        radius = np.linalg.norm(shifted, axis=1)
        angles = np.arctan2(shifted[:, 1], shifted[:, 0])
        rz = np.column_stack((radius, angles))
        assert rz.shape == (N, 2)

        if extend_angles:
            # raise NotImplementedError
            shift_up = np.column_stack((radius, angles + 2 * np.pi))
            shift_dn = np.column_stack((radius, angles - 2 * np.pi))
            rz = np.row_stack((rz, shift_up, shift_dn))
            Zif = np.row_stack((Zif, Zif, Zif))
            assert rz.shape == (3 * N, 2)
    else:
        raise ValueError

    if extra_channels:
        raise NotImplementedError
        # (16k, 1, 39, 39)
        images = np.expand_dims(Zif, 1)
        assert images.shape == (N, 1, H, W)
        # (16k, 2, 39, 39)
        ones = np.ones((N, 2, H, W))
        # (16k, 2, 39, 39) by (16k, 2, 1, 1) = (16k, 2, 39, 39)
        coords_in_extra_channel = ones * np.expand_dims(Zif, axis=(2, 3))
        assert coords_in_extra_channel.shape == (N, 2, 39, 39)
        # (16k, 3, 39, 39)
        images = np.concatenate([images, coords_in_extra_channel], axis=1)
        assert images.shape == (N, 1 + 2, H, W)
    else:
        images = torch.tensor(Zif)

    # Pytorch seems to expect a float32 default datatype.
    dataset = TensorDataset(images, torch.tensor(rz, dtype=torch.float32))

    return dataset


if __name__ == "__main__":
    """This just tests the above function"""

    import itertools

    for options in itertools.product(
        [False, True],  # extend angles
        ["cartesian", "polar"],  # coordinate
        [False],  # extra_channels
        [[], [0, 1, 2, 3, 4, 5]],  # nodes_for_augmentation
    ):
        print(*options)
        XGC(*options)
