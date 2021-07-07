import logging
import os
import random

import adios2 as ad2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from tqdm import tqdm


def adios2_get_shape(f, varname):
    nstep = int(f.available_variables()[varname]["AvailableStepsCount"])
    shape = f.available_variables()[varname]["Shape"]
    lshape = None
    if shape == "":
        ## Accessing Adios1 file
        ## Read data and figure out
        v = f.read(varname)
        lshape = v.shape
    else:
        lshape = tuple([int(x.strip(",")) for x in shape.strip().split()])
    return (nstep, lshape)


def read_f0(
    istep,
    expdir=None,
    iphi=None,
    inode=0,
    nnodes=None,
    average=False,
    randomread=0.0,
    nchunk=16,
    fieldline=False,
):
    """
    Read XGC f0 data
    """

    fname = os.path.join(expdir, "restart_dir/xgc.f0.%05d.bp" % istep)
    if randomread > 0.0:
        ## prefetch to get metadata
        with ad2.open(fname, "r") as f:
            nstep, nsize = adios2_get_shape(f, "i_f")
            ndim = len(nsize)
            nphi = nsize[0]
            _nnodes = nsize[2] if nnodes is None else nnodes
            nmu = nsize[1]
            nvp = nsize[3]
        assert _nnodes % nchunk == 0
        _lnodes = list(range(inode, inode + _nnodes, nchunk))
        lnodes = random.sample(_lnodes, k=int(len(_lnodes) * randomread))
        lnodes = np.sort(lnodes)

        lf = list()
        li = list()
        for i in tqdm(lnodes):
            li.append(np.array(range(i, i + nchunk), dtype=np.int32))
            with ad2.open(fname, "r") as f:
                nphi = nsize[0] if iphi is None else 1
                iphi = 0 if iphi is None else iphi
                start = (iphi, 0, i, 0)
                count = (nphi, nmu, nchunk, nvp)
                _f = f.read("i_f", start=start, count=count).astype("float64")
                lf.append(_f)
        i_f = np.concatenate(lf, axis=2)
        lb = np.concatenate(li)
    elif fieldline is True:
        import networkx as nx

        fname2 = os.path.join(expdir, "xgc.mesh.bp")
        with ad2.open(fname2, "r") as f:
            _nnodes = int(f.read("n_n"))
            nextnode = f.read("nextnode")

        G = nx.Graph()
        for i in range(_nnodes):
            G.add_node(i)
        for i in range(_nnodes):
            G.add_edge(i, nextnode[i])
            G.add_edge(nextnode[i], i)
        cc = [x for x in list(nx.connected_components(G)) if len(x) >= 16]

        li = list()
        for k, components in enumerate(cc):
            DG = nx.DiGraph()
            for i in components:
                DG.add_node(i)
            for i in components:
                DG.add_edge(i, nextnode[i])

            cycle = list(nx.find_cycle(DG))
            DG.remove_edge(*cycle[-1])

            path = nx.dag_longest_path(DG)
            # print (k, len(components), path[0])
            for i in path[: len(path) - len(path) % 16]:
                li.append(i)

        with ad2.open(fname, "r") as f:
            nstep, nsize = adios2_get_shape(f, "i_f")
            ndim = len(nsize)
            nphi = nsize[0] if iphi is None else 1
            iphi = 0 if iphi is None else iphi
            _nnodes = nsize[2]
            nmu = nsize[1]
            nvp = nsize[3]
            start = (iphi, 0, 0, 0)
            count = (nphi, nmu, _nnodes, nvp)
            logging.info(f"Reading: {start} {count}")
            i_f = f.read("i_f", start=start, count=count).astype("float64")

        _nnodes = len(li) - inode if nnodes is None else nnodes
        lb = np.array(li[inode : inode + _nnodes], dtype=np.int32)
        logging.info(f"Fieldline: {len(lb)}")
        logging.info(f"{lb}")
        i_f = i_f[:, :, lb, :]
    else:
        with ad2.open(fname, "r") as f:
            nstep, nsize = adios2_get_shape(f, "i_f")
            ndim = len(nsize)
            nphi = nsize[0] if iphi is None else 1
            iphi = 0 if iphi is None else iphi
            _nnodes = nsize[2] - inode if nnodes is None else nnodes
            nmu = nsize[1]
            nvp = nsize[3]
            start = (iphi, 0, inode, 0)
            count = (nphi, nmu, _nnodes, nvp)
            logging.info(f"Reading: {start} {count}")
            i_f = f.read("i_f", start=start, count=count).astype("float64")
            # e_f = f.read('e_f')
        li = list(range(inode, inode + _nnodes))
        lb = np.array(li, dtype=np.int32)

    # if i_f.shape[3] == 31:
    #     i_f = np.append(i_f, i_f[...,30:31], axis=3)
    #     # e_f = np.append(e_f, e_f[...,30:31], axis=3)
    # if i_f.shape[3] == 39:
    #     i_f = np.append(i_f, i_f[...,38:39], axis=3)
    #     i_f = np.append(i_f, i_f[:,38:39,:,:], axis=1)

    Z0 = np.moveaxis(i_f, 1, 2)

    if average:
        Z0 = np.mean(Z0, axis=0)
        zlb = lb
    else:
        Z0 = Z0.reshape((-1, Z0.shape[2], Z0.shape[3]))
        _lb = list()
        for i in range(nphi):
            _lb.append(i * 100_000_000 + lb)
        zlb = np.concatenate(_lb)

    # zlb = np.concatenate(li)
    zmu = np.mean(Z0, axis=(1, 2))
    zsig = np.std(Z0, axis=(1, 2))
    zmin = np.min(Z0, axis=(1, 2))
    zmax = np.max(Z0, axis=(1, 2))
    Zif = (Z0 - zmin[:, np.newaxis, np.newaxis]) / (zmax - zmin)[
        :, np.newaxis, np.newaxis
    ]

    return (Z0, Zif, zmu, zsig, zmin, zmax, zlb)


def XGC():
    """
    Returns TensorDataset, for input into DataLoader
    """
    Z0, Zif, zmu, zsig, zmin, zmax, zlb = read_f0(
        420, expdir="d3d_coarse_v2_colab", iphi=0
    )
    Zif = Zif.astype(np.float32)
    _, nx, ny = Z0.shape

    fname2 = os.path.join("d3d_coarse_v2_colab", "xgc.mesh.bp")
    with ad2.open(fname2, "r") as f:
        ## r and z positions of XGC mesh nodes
        rz = f.read("rz")
        conn = f.read("nd_connect_list")
    # r = rz[:, 0]
    # z = rz[:, 1]
    # print("Nnodes:", len(rz))

    _rz = np.array(rz[:, 0], dtype=complex)
    _rz.imag = rz[:, 1]

    da = np.zeros_like(rz)  ## list of distance and angle pair
    for inode in zlb:
        dist = np.linalg.norm(_rz[inode] - _rz[0])
        angle = np.angle(_rz[inode] - _rz[0])
        da[inode, 0] = dist
        da[inode, 1] = angle

    lx = list()
    ly = list()
    for i in range(len(zlb)):
        lx.append(Zif[i, np.newaxis, :, :])
        ly.append(da[i, :])

    # Pytorch seems to expect a float32 default datatype.
    X_full, y_full = torch.tensor(lx).squeeze(), torch.tensor(ly, dtype=torch.float32)
    dataset = TensorDataset(X_full, y_full)

    return dataset
