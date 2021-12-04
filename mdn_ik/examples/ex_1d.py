import numpy as np
import torch
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from mdn.models import MixtureDensityNetwork


def gen_data(n=512):
    y = np.linspace(-1, 1, n)
    x = 7 * np.sin(5 * y) + 0.5 * y + 0.5 * np.random.randn(*y.shape)
    return x[:,np.newaxis], y[:,np.newaxis]

def plot_data(x, y):
    plt.hist2d(x, y, bins=35)
    plt.xlim(-8, 8)
    plt.ylim(-1, 1)
    plt.axis('off')


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--n-iterations", type=int, default=2000)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    x, y = gen_data()
    x = torch.Tensor(x)
    y = torch.Tensor(y)

    model = MixtureDensityNetwork(1, 1, n_components=3)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    for i in range(args.n_iterations):
        optimizer.zero_grad()
        loss = model.loss(x, y).mean()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            logger.info(f"Iter: {i}\t" + f"Loss: {loss.data:.2f}")

    samples = model.sample(x)
    breakpoint()
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plot_data(x[:,0].numpy(), y[:,0].numpy())
    plt.title("Observed data")
    plt.subplot(1, 2, 2)
    plot_data(x[:,0].numpy(), samples[:,0].numpy())
    plt.title("Sampled data")
    plt.show()
