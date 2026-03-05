import torch
from torch import Tensor
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace

parser = ArgumentParser()
parser.add_argument("--start", type=int, default=0)
args: Namespace = parser.parse_args()

logs: Tensor = torch.load('/workspace/experiment/MIMOUNet_test/val_log.pt')

####################### Linear Regression #######################
y: Tensor = logs.reshape(-1, 2).clone().detach()
X: Tensor = torch.hstack([torch.ones_like(y[:, 0]).reshape(-1, 1), torch.arange(len(y)).reshape(-1, 1)])
beta: Tensor = torch.linalg.pinv(X.T @ X) @ X.T @ y
y = (X @ beta).numpy()

####################### Plotting #######################
fig , ax = plt.subplots(2, 1, figsize=(15, 5))
ax[0].set_title('MIMO-UNet RMSE')
ax[0].plot(logs[:, 0][args.start:], c="g", marker="o", label='RMSE')
ax[0].plot(y[:, 0][args.start:], label=f'Trend, the slope: {beta[1][0].item():.6f}')
ax[0].set_ylabel('RMSE')
ax[0].legend()
ax[1].set_title('MIMO-UNet PSNR')
ax[1].plot(logs[:, 1][args.start:], c="r", marker="^", label='PSNR')
ax[1].plot(y[:, 1][args.start:], label=f'Trend, the slope: {beta[1][1].item():.6f}')
ax[1].set_ylabel('PSNR')
ax[1].set_xlabel('Epochs, but interval=100')
ax[1].legend()
fig.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.savefig(fname='/workspace/figures/Model_Metrics.png')
plt.close()