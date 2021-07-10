import math

import matplotlib.pyplot as plt
import torch


def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)

    return onehot


def plot_batch(batch):
    N = batch.shape[0]
    n = math.ceil(math.sqrt(N))
    for i, image in enumerate(batch):
        plt.subplot(n, n, i + 1)
        plt.imshow(image)
        plt.axis("off")
    plt.show()
