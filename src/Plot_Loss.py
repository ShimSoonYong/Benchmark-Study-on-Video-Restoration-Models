import torch
import numpy as np
from torch import Tensor
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace

parser = ArgumentParser()
parser.add_argument("--ymin", type=float, default=0.01)
parser.add_argument("--ymax", type=float, default=0.1)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--model", type=str, default='MIMO-UNet')
parser.add_argument("--loss", type=str, default='MSFRLoss')
parser.add_argument("--interval", type=int, default=220)
args: Namespace = parser.parse_args()

losses: list = torch.load('/workspace/logs/TrainLosses.pth')

####################### Average Loss ########################
mean_loss = 0 
mean_losses = []
for i in range(0, len(losses)):
    mean_loss += losses[i]
    if i != 0 and i % args.interval == 0:
        mean_losses.append(mean_loss / args.interval)
        mean_loss = 0

####################### Linear Regression #######################
y: Tensor = torch.tensor(mean_losses[args.start:]).reshape(-1, 1)
X: Tensor = torch.hstack([torch.ones_like(y), torch.arange(len(y)).reshape(-1, 1)])
beta: Tensor = torch.linalg.inv(X.T @ X) @ X.T @ y
y_hat: np.ndarray = (X @ beta).numpy()

####################### Plotting #######################
plt.figure(figsize=(15, 5))
plt.title(f'{args.model} Losses')
plt.plot(y, marker="^", label='Loss')
plt.plot(y_hat, label=f'Trend, the slope: {beta[1].item():.6f}')
plt.ylabel(args.loss)
plt.xlabel(f'Iterations, but inerval={args.interval}')
plt.ylim(args.ymin, args.ymax)
plt.legend()
plt.savefig(fname='/workspace/figures/Model_Loss.png')
plt.close()