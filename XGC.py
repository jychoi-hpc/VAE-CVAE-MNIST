import numpy as np
import torch
from torch.utils.data import TensorDataset

from utils import get_everything_from_adios2


def XGC(
    extend_angles: bool = False,
    coordinate: str = "cartesian",
    extra_channels: bool = False,
):
    """
    Returns TensorDataset, for input into DataLoader
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
    if coordinate == "cartesian":
        coord = rz
    elif coordinate == "polar":
        raise NotImplementedError
        shifted = rz - rz[0]
        radius = np.linalg.norm(shifted, axis=1)
        angles = np.arctan2(shifted[:, 1], shifted[:, 0])
        coord = np.column_stack((radius, angles))

        if extend_angles:
            shift_up = np.column_stack((radius, angles + 2 * np.pi))
            shift_dn = np.column_stack((radius, angles - 2 * np.pi))
            coord = np.row_stack((coord, shift_up, shift_dn))
    else:
        raise ValueError

    if extra_channels:
        raise NotImplementedError
        # (16k, 1, 39, 39)
        images = np.expand_dims(Zif, 1)
        # (16k, 2, 39, 39)
        ones = np.ones((Zif.shape[0], coord.shape[1], Zif.shape[1], Zif.shape[2]))
        # (16k, 2, 39, 39) by (16k, 2, 1, 1) = (16k, 2, 39, 39)
        coords_in_extra_channel = ones * np.expand_dims(coord, axis=(2, 3))
        # (16k, 3, 39, 39)
        images = np.concatenate([images, coords_in_extra_channel], axis=1)
    else:
        images = torch.tensor(Zif)

    # Pytorch seems to expect a float32 default datatype.
    dataset = TensorDataset(images, torch.tensor(coord, dtype=torch.float32))

    return dataset


if __name__ == "__main__":
    """This just tests the above function"""

    import itertools

    for extend_angles, coordinate, extra_channels in itertools.product(
        [False],
        ["cartesian"],
        [False],
    ):
        print(extend_angles, coordinate, extra_channels)
        XGC(extend_angles, coordinate, extra_channels)
