import adios2 as ad2
import numpy as np

from XGC import read_f0

# psi_surf: psi value of each surface
# surf_len: # of nodes of each surface
# surf_idx: list of node index of each surface
with ad2.open("d3d_coarse_v2_colab/xgc.mesh.bp", "r") as f:
    nnodes = int(f.read("n_n"))
    ncells = int(f.read("n_t"))
    rz = f.read("rz")
    conn = f.read("nd_connect_list")
    psi = f.read("psi")
    nextnode = f.read("nextnode")
    epsilon = f.read("epsilon")
    node_vol = f.read("node_vol")
    node_vol_nearest = f.read("node_vol_nearest")
    psi_surf = f.read("psi_surf")
    surf_idx = f.read("surf_idx")
    surf_len = f.read("surf_len")
    theta = f.read("theta")

Z0, Zif, zmu, zsig, zmin, zmax, zlb = read_f0(420, expdir="d3d_coarse_v2_colab", iphi=0)

assert surf_idx.shape == (101, 338)
assert nnodes == 16395
assert Zif.shape == (16395, 39, 39)

# expect n_expected many additional images.
# The inside counts the number of nonzeros nodes per row, and has shape (101,)
# Then subtract one, and sum up to get the total number of additional pairs.
n_expected = sum(np.count_nonzero(surf_idx, axis=1) - 1)
assert n_expected == 15898

# Might be faster to preallocate the array.
# images = np.zeros([n_expected, 39, 39])
imgs = []
coords = []

for i, fieldline in enumerate(surf_idx):
    fieldline = np.trim_zeros(fieldline)
    for j, (first, second) in enumerate(zip(fieldline, fieldline[1:])):
        avg = np.mean(Zif[[first, second]], axis=0)
        coord = np.mean(rz[[first, second]], axis=0)
        assert avg.shape == (39, 39)
        assert coord.shape == (2,)
        # idx = np.cumsum(np.count_nonzero(surf_idx, axis=1)-1)[i]+j
        imgs.append(avg)
        coords.append(coord)

imgs = np.stack(imgs)
assert imgs.shape == (n_expected, 39, 39)
coords = np.stack(coords)
assert coords.shape == (n_expected, 2)
