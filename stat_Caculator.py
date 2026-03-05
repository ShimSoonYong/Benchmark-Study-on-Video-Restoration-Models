import yaml
import torch
import numpy as np
from utils import calc_psnr_per_slice
from data.data_load import GOPRO_Large
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from scipy.stats._binomtest import BinomTestResult
from scipy.stats import ttest_1samp, binomtest, ttest_rel, wilcoxon

def Input_vs_Target(opt_dir:str) -> None:
    """
    Compare the input and target images from the GOPRO dataset.
    """
    print("="*25, "Input vs. Target", "="*25)
    opt: dict = yaml.safe_load(open(opt_dir, "r"))
    dataset = GOPRO_Large(opt=opt, mode='test')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    holder = []
    for i, (lq, gt, index) in enumerate(loader):
        psnr: float = calc_psnr_per_slice(lq, gt)
        print('Input vs. Target:', psnr)
        holder.append(psnr)
    
    print(f"Average PSNR: {np.mean(holder):.4f} ± {np.std(holder):.4f}")
    print("="*50)
    print()

def metrics_record_stats(val_log_dir:str) -> torch.Tensor:
    val_log = torch.load(val_log_dir)
    print(f"Loaded {len(val_log)} validation logs.")
    print(f'Max PSNR: {val_log[:, 1].max()}, Min PSNR: {val_log[:, 1].min()}')
    print(f'Mean PSNR: {val_log[:, 1].mean()}, Std PSNR: {val_log[:, 1].std()}')
    print(f'Variation Coefficient: {val_log[:, 1].std() / val_log[:, 1].mean() * 100:.2f}%')
    print()

    return val_log

def psnr_ttest(val_log:torch.Tensor, mu0:float) -> None:
    """Perform a t-test to evaluate the statistical significance of the change in PSNR."""

    print("="*25, "One-sample t-test", "="*25)
    print(f"H0 : mu <= {mu0} vs. H1: mu > {mu0}")
    t_statistic, p_value = ttest_1samp(val_log[:, 1], mu0, alternative='greater')
    print(f't-statistic: {t_statistic:.4f}')
    print(f'two-tailed p-value: {p_value:.4f}')

def psnr_ttest_rel(val_log1:torch.Tensor, val_log2:torch.Tensor) -> None:
    """Perform a paired t-test to evaluate the statistical significance of the change in PSNR."""

    print("="*25, "Paired t-test", "="*25)
    print(f"H0 : mu1 <= mu2 vs. H1: mu1 > mu2")
    t_statistic, p_value = ttest_rel(val_log1[:, 1], val_log2[:, 1], alternative='greater')
    print(f't-statistic: {t_statistic:.4f}')
    print(f'two-tailed p-value: {p_value:.4f}')
    print("="*50)
    print()

def psnr_sign_test(val_log:torch.Tensor, mu0:float) -> None:
    """Perform a nonparametric sign test to evaluate the statistical significance of the change in PSNR."""

    print("="*25, "Sign test", "="*25)
    print(f"H0 : mu <= {mu0} vs. H1: mu > {mu0}")
    result: BinomTestResult = binomtest(k=sum([1 if dif > 0 else 0 for dif in val_log[:, 1] - mu0]), 
                                        n=len(val_log), alternative='greater')
    print(f'Sign statistic: {result.statistic * result.n:.4f}')
    print(f'two-tailed p-value: {result.pvalue:.4f}')
    print("="*50)
    print()

def psnr_wilcox_test(val_log1:torch.Tensor, val_log2:torch.Tensor) -> None:
    """Perform a Wilcoxon signed-rank test to evaluate the statistical significance of the change in PSNR."""

    print("="*25, "Wilcoxon signed-rank test", "="*25)
    print(f"H0 : mu1 <= mu2 vs. H1: mu1 > mu2")
    stat, p_value = wilcoxon(val_log1[:, 1], val_log2[:, 1], alternative='greater')
    print(f'Wilcoxon signed-rank statistic: {stat:.4f}')
    print(f'two-tailed p-value: {p_value:.4f}')
    print("="*50)
    print()



if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--mu0", type=float, default=30)
    argparser.add_argument("--opt_dir", type=str, default="/workspace/options/ReBotNet_opts.yaml")
    argparser.add_argument("--val_log1_dir", type=str, default="/workspace/experiment/MIMOUNet_test/val_log.pt")
    argparser.add_argument("--val_log2_dir", type=str, default=None)
    args: Namespace = argparser.parse_args()

    # Input vs. Target (GOPRO reference images)
    Input_vs_Target(args.opt_dir)

    # Load validation metrics records
    val_log1: torch.Tensor = metrics_record_stats(args.val_log1_dir)

    # Perform a t-test to evaluate the statistical significance of the change in PSNR
    psnr_ttest(val_log1, args.mu0)

    # Perform a sign test to evaluate the statistical significance of the change in PSNR
    psnr_sign_test(val_log1, args.mu0)

    if args.val_log2_dir is not None:
        val_log2: torch.Tensor = metrics_record_stats(args.val_log2_dir)
        psnr_ttest_rel(val_log1, val_log2)
        psnr_wilcox_test(val_log1, val_log2)
        