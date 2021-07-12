import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models import VAE
from utils import plot_batch
from XGC import XGC


def main(args):

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
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(vars(args))
    writer = SummaryWriter(log_dir=tbdir)
    with open(os.path.join(logdir, "hyperparameters.yml"), "w") as outfile:
        yaml.dump(vars(args), outfile)

    dataset = XGC(extend_angles=args.extend_angles, coordinate=args.coordinate)
    logging.info(f"There are {len(dataset)} samples in the dataset.")

    if args.reconstruction_error == "BCE":
        recon_error_func = F.binary_cross_entropy
    elif args.reconstruction_error == "MSE":
        recon_error_func = F.mse_loss
    else:
        raise ValueError

    arguments_to_dataloader = {"dataset": dataset, "batch_size": args.batch_size}
    if args.balance:
        all = dataset[:][0]
        median = torch.median(all, dim=0, keepdim=True).values
        indiv_loss = recon_error_func(
            all,
            median.expand(len(dataset), -1, -1),
            reduction="none",
        )  # shape is (16k, 39, 39)
        sample_weights = torch.mean(indiv_loss, [1, 2])  # average along dim 1, 2
        sampler = WeightedRandomSampler(
            list(sample_weights),
            num_samples=len(dataset),
            replacement=True,
        )
        arguments_to_dataloader["sampler"] = sampler
    else:
        arguments_to_dataloader["shuffle"] = True
    data_loader = DataLoader(**arguments_to_dataloader)

    if args.plot_batch:
        first_batch = iter(data_loader).next()[0]
        plot_batch(first_batch)

    def loss_fn(recon_x, x, mean, log_var):
        recon_error = recon_error_func(
            recon_x.view(-1),
            x.view(-1),
            reduction="sum",
        )
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (recon_error + KLD) / x.size(0)

    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=2 if args.conditional else 0,
    ).to(device)

    # Print info about number of parameters
    logging.info("-" * 50)
    num_params = 0
    for k, v in vae.state_dict().items():
        logging.info("%50s\t%20s\t%10d" % (k, list(v.shape), v.numel()))
        num_params += v.numel()
    logging.info("-" * 50)
    logging.info("%50s\t%20s\t%10d" % ("Total", "", num_params))

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)

    for epoch in range(args.epochs):
        # tracker_epoch = defaultdict(lambda: defaultdict(dict))
        if args.dry_run and epoch > 0:
            break
        for iteration, (x, y) in enumerate(data_loader):
            vae.train()
            x = x.to(device)
            y = y.to(device)

            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)

            # for i, yi in enumerate(y):
            #     id = len(tracker_epoch)
            #     tracker_epoch[id]["x"] = z[i, 0].item()
            #     tracker_epoch[id]["y"] = z[i, 1].item()
            #     tracker_epoch[id]["label"] = round(yi.item())

            loss = loss_fn(recon_x, x, mean, log_var)
            writer.add_scalar(
                "Loss/train",
                loss,
                epoch * len(data_loader) + iteration,
            )
            logs["loss"].append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % args.print_every == 0 or iteration == len(data_loader) - 1:
                logging.info(
                    "Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                        epoch,
                        args.epochs,
                        iteration,
                        len(data_loader) - 1,
                        loss.item(),
                    )
                )

                with torch.no_grad():
                    vae.eval()
                    # Samples for comparison
                    sample_index = list(range(0, 16395, 16395 // 10))
                    original_images, coord = dataset[sample_index]
                    if args.conditional:
                        # x = original_images
                        recon_images, mean, log_var, z = vae(original_images, coord)
                        # c_sample = torch.linspace(-0.1, 0.1, 7)
                        # c = (torch.arange(0, 10).long().unsqueeze(1) + c_sample).view(-1, 1)
                        # c = c.to(device)
                        # z = torch.randn([c.size(0), args.latent_size]).to(device)
                        # x = vae.inference(z, c=c)
                    else:
                        raise NotImplementedError
                        recon_images = vae(original_images)
                        z = torch.randn([10, args.latent_size]).to(device)
                        x = vae.inference(z)

                    plt.figure()
                    plt.figure(figsize=(5, 30))
                    for p in range(len(original_images)):
                        # Original
                        plt.subplot(len(original_images), 2, 2 * p + 1)
                        if args.conditional:
                            plt.title(
                                "{:s}:({:.2f}, {:.2f})".format(
                                    args.coordinate,
                                    coord[p, 0],
                                    coord[p, 1],
                                ),
                                # color="black",
                                # backgroundcolor="white",
                                # fontsize=8,
                            )
                        plt.imshow(original_images[p].view(39, 39).cpu().data.numpy())
                        plt.colorbar()
                        plt.axis("off")

                        # Recon
                        plt.subplot(len(original_images), 2, 2 * p + 2)
                        plt.title(
                            "{:s}: {:.6f} ".format(
                                args.reconstruction_error,
                                recon_error_func(
                                    original_images[p].view(-1),
                                    recon_images[p],
                                    reduction="sum",
                                ).item(),
                            )
                        )
                        plt.imshow(recon_images[p].view(39, 39).cpu().data.numpy())
                        plt.axis("off")
                        plt.colorbar()

                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            args.log_root,
                            str(ts),
                            "E{:d}I{:d}.png".format(epoch, iteration),
                        ),
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.clf()
                    plt.close("all")

        # df = pd.DataFrame.from_dict(tracker_epoch, orient="index")
        # g = sns.lmplot(
        # x="x",
        # y="y",
        # hue="label",
        # data=df.groupby("label").head(100),
        # fit_reg=False,
        # legend=True,
        # )
        # g.savefig(
        # os.path.join(args.log_root, str(ts), "E{:d}-Dist.png".format(epoch)),
        # dpi=300,
        # )

    writer.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[39 * 39, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 39 * 39])
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--log_root", type=str, default="logs")
    parser.add_argument(
        "--reconstruction_error",
        type=str,
        default="BCE",
        choices=["BCE", "MSE"],
    )
    parser.add_argument(
        "--coordinate",
        type=str,
        default="cartesian",
        choices=["cartesian", "polar"],
    )
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--extend_angles", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--plot-batch", action="store_true")

    args = parser.parse_args()

    main(args)
