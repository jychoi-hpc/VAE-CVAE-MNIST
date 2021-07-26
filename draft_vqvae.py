import argparse
import logging
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from augmenter import augmented_dataset_along_fieldlines
from utils import get_everything_from_adios2
from XGC import XGCDataset

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--log_root", type=str, default="logs")
parser.add_argument(
    "--coordinate",
    type=str,
    default="cartesian",
    choices=["cartesian", "polar"],
)
parser.add_argument("--plot-batch", action="store_true")
parser.add_argument("--train-size", type=float, default=0.8)
parser.add_argument("--augment", action="store_true")
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ts = datetime.now().strftime("%y%m%dT%H%M%S")
logdir = os.path.join(args.log_root, str(ts))
tbdir = os.path.join(logdir, "tb")
# Make logging dirs
if not (os.path.exists(os.path.join(args.log_root))):
    os.mkdir(os.path.join(args.log_root))
if not os.path.exists(logdir):
    os.mkdir(logdir)
    os.mkdir(tbdir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(logdir, "run.log")),
        logging.StreamHandler(sys.stderr),
    ],
)
logging.info(vars(args))
writer = SummaryWriter(log_dir=tbdir)
# This writes the hyperparameters to a file.
with open(os.path.join(logdir, "hyperparameters.yml"), "w") as outfile:
    yaml.dump(vars(args), outfile)

dataset = XGCDataset(
    # extend_angles=args.extend_angles,
    coordinate=args.coordinate,
    # extra_channels=args.extra_channels,
)
logging.info(f"There are {len(dataset)} samples in the dataset.")

# There are N samples in the train dataset.
N = int(args.train_size * len(dataset))
train_data, test_data = random_split(dataset, [N, len(dataset) - N])
logging.info(f"There are {len(train_data)} samples in the train dataset.")
logging.info(f"There are {len(test_data)} samples in the test dataset.")
assert len(train_data) + len(test_data) == len(dataset)

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

if args.augment:
    augmented_dataset = augmented_dataset_along_fieldlines(
        dataset=train_data,
        fieldlines=surf_idx,
    )
    train_data = ConcatDataset((train_data, augmented_dataset))

# arguments_to_dataloader = {"dataset": dataset, "batch_size": args.batch_size}
# if args.balance:
#     all = dataset[:][0]
#     median = torch.median(all, dim=0, keepdim=True).values
#     indiv_loss = recon_error_func(
#         all,
#         median.expand(len(dataset), -1, -1),
#         reduction="none",
#     )  # shape is (16k, 39, 39)
#     sample_weights = torch.mean(indiv_loss, [1, 2])  # average along dim 1, 2
#     sampler = WeightedRandomSampler(
#         list(sample_weights),
#         num_samples=len(dataset),
#         replacement=True,
#     )
#     arguments_to_dataloader["sampler"] = sampler
# else:
#     arguments_to_dataloader["shuffle"] = True
# data_loader = DataLoader(**arguments_to_dataloader)

# if args.plot_batch:
#     first_batch = iter(data_loader).next()[0]
#     plot_batch(first_batch)

# This is just a basic data_loader, without weighted resampling.
train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

"""
TODO: Input your model here
The model should have a encode, and decode function.
I assume decode looks like:
    def decode(random_noise, coord):
        return 39*39 image

"""
from models import YourModel
from vqvae import VQVAE

# model = YourModel().to(device)
model = VQVAE(num_channels=1, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32,
            num_embeddings=512, embedding_dim=16, 
            commitment_cost=0.25, decay=0.99, rescale=None, learndiff=None, 
            shaconv=None, grid=None, conditional=None, decoder_padding=[1,1,0], da_conditional=True,
            decoder_layer_sizes=[]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


def loss_fn(recon_img, img, mu, logvar):
    """
    TODO: You need to change this loss_fn.
    I'm not sure what it should be for VQ-VAE.
    """
    BCE = F.binary_cross_entropy(recon_img, img, reduction="sum")

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


# Print info about number of parameters
logging.info("-" * 50)
num_params = 0
for k, v in model.state_dict().items():
    logging.info("%50s\t%20s\t%10d" % (k, list(v.shape), v.numel()))
    num_params += v.numel()
logging.info("-" * 50)
logging.info("%50s\t%20s\t%10d" % ("Total", "", num_params))


"""
TODO:
The internals of the training loop need to be fixed depending on the loss fcn and output of model(img).
"""
# Training loop
model.train()
for epoch in range(args.epochs):
    logging.info(f"Training Epoch {epoch}/{args.epochs}")
    loss_sum = 0.0
    for iter, (img, coord, nodeid) in enumerate(train_loader):
        nb, nx, ny = img.shape
        img = img.view(nb, 1, nx, ny)
        img = img.to(device)
        coord = coord.to(device)
        
        vq_loss, recon_img, perplexity, dloss = model(img, coord)
        recon_error = F.mse_loss(recon_img, img)
        loss = recon_error + vq_loss
        loss_sum += loss.item()

        # recon_img, mu, logvar = model(img)
        # loss = loss_fn(recon_img, img, mu, logvar)

        writer.add_scalar(
            "Loss/train",
            loss,
            epoch * len(train_loader) + iter,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iter+1)%100 == 0:
            print ('==> %d/%d/%d loss sum: %g'%(epoch, iter, len(train_loader), loss.item()))

writer.flush()


# Eval loop
model.eval()
# A metric accepts 2 tensors of shape (N, H, W) and should return a loss tensor of shape (N,)
metrics = {
    "MSE": lambda x, y: torch.mean(F.mse_loss(x, y, reduction="none"), dim=[1, 2]),
    "BCE": lambda x, y: torch.mean(
        F.binary_cross_entropy(x, y, reduction="none"), dim=[1, 2]
    ),
    "MAX": lambda x, y: torch.amax(torch.abs(x - y), dim=[1, 2]),
}
with torch.no_grad():
    img, coord, nodeid = test_data[:]
    nb, nx, ny = img.shape
    img = img.view(nb, 1, nx, ny)
    img = img.to(device)
    coord = coord.to(device)

    vq_loss, recon_img, perplexity, dloss = model(img, coord)
    nb, nc, nx, ny = img.shape
    assert nc == 1
    img = img.view(nb, nx, ny)
    recon_img = recon_img.view(nb, nx, ny)

    # recon_img, mu, logvar = model(img)
    for name, metric in metrics.items():
        loss = metric(img, recon_img)
        assert loss.shape == (img.shape[0],)

        # Make a histogram of the errors.
        plt.figure()
        plt.title(f"{name} Mean: {torch.mean(loss).item():.5f}")
        plt.hist(loss.tolist())
        plt.axvline(torch.mean(loss).item(), color="k", linestyle="dashed", linewidth=1)
        plt.savefig(
            os.path.join(args.log_root, str(ts), f"{name}_hist.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Make a scatter plot of error vs nodeid.
        plt.figure()
        plt.title(f"{name} Mean: {torch.mean(loss).item():.5f}")
        plt.scatter(nodeid.tolist(), loss.tolist())
        plt.axhline(torch.mean(loss).item(), color="k", linestyle="dashed", linewidth=1)
        plt.xlabel("nodeid")
        plt.savefig(
            os.path.join(args.log_root, str(ts), f"{name}_scatter.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
