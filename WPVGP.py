# -*- coding: utf-8 -*-
"""
Li & Tan-inspired separable shared-latent multi-output GP + active learning
===========================================================================

M3 runner-compatible module.

Three acquisition strategies:
    1) ACS_FW
    2) BALD
    3) RWPV (regression adaptation of Li & Tan's WPV principle)

Minimal saved outputs ONLY, per scenario:
    1) mape_trajectories_all_strategies.png
    2) metrics_all_strategies.csv
    3) true_vs_predicted_ACS_FW.png
    4) true_vs_predicted_BALD.png
    5) true_vs_predicted_RWPV.png

Expected CSV layout:
    columns 0:7   -> 7 categorical input features
    column 7      -> skipped
    columns 8:    -> response curve outputs (144 expected by default)

Input encoding
--------------
The seven input features are treated as categorical and one-hot encoded
(OneHotEncoder on the string representation of each level, unknown levels
ignored), matching the LT-NP, IndGP and FedMGP workflows. The encoded
inputs are 0/1 indicators and are NOT standardized. The Matern-3/2 ARD
kernel therefore receives one lengthscale per level of every feature, so
the correlation between two different levels a and b of a feature is a
product of per-level factors and does not depend on any arbitrary integer
ordering of the levels.

The runner entry point accepts the exact keyword interface used on M3:
    run_notebook_workflow(
        base_dir=...,
        scenario_files=...,
        acq_rounds=...,
        batch_size=...,
        save_dir=...,
        num_latents=...,
        num_inducing=...,
        train_epochs=...,
        lr=...,
        acs_j=...,
        initial_context_size=...,
        split_seed=...,
        train_seed=...,
        gpu_id=...,
        expected_num_outputs=...,
    )

Unlike the mistakenly supplied ExactGP file, this version actually uses
num_latents and num_inducing through a GPyTorch LMC variational GP.  This is
scalable to a 144-dimensional response curve while preserving a separable
shared-latent input/task covariance interpretation.
"""

from __future__ import annotations

import copy
import gc
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import gpytorch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder


# -----------------------------------------------------------------------------
# Public module constants expected by the runner
# -----------------------------------------------------------------------------
WORKFLOW_VERSION = "2026-10-20-onehot-inputs-v4"
EXPECTED_NUM_OUTPUTS = 144
STRATEGIES = ("ACS_FW", "BALD", "RWPV")

RUN_NOTEBOOK_WORKFLOW_PARAMETERS = (
    "base_dir",
    "scenario_files",
    "acq_rounds",
    "batch_size",
    "save_dir",
    "num_latents",
    "num_inducing",
    "train_epochs",
    "lr",
    "acs_j",
    "initial_context_size",
    "split_seed",
    "train_seed",
    "gpu_id",
    "expected_num_outputs",
    "num_seeds",
)

DEFAULT_BASE_DIR = Path("./paper scenarios")
DEFAULT_SCENARIO_FILES = (
    "scenario_000_HGR.csv",
    "scenario_016_damped.csv",
    "scenario_054_gompertz.csv",
    "scenario_179_sigmoid.csv",
)
DEFAULT_SAVE_DIR = DEFAULT_BASE_DIR / "seprable"

DEFAULT_NUM_SEEDS = 20
TEST_FRACTION = 0.20
N_VIS_CURVES = 9
PREDICT_CHUNK_SIZE = 256
MAPE_EPS = 1e-8
PLOT_DPI = 300
CHOLESKY_JITTER = 1e-4
GRAD_CLIP_NORM = 10.0
EARLY_STOP_PATIENCE = 20
EARLY_STOP_TOL = 1e-4
MIN_TRAIN_EPOCHS = 30
RWPV_RELEVANCE_TEMPERATURE = 1.0
RWPV_RELEVANCE_FLOOR = 0.05


@dataclass(frozen=True)
class RunConfig:
    base_dir: Path
    scenario_files: Tuple[str, ...]
    save_dir: Path
    acq_rounds: int
    batch_size: int
    num_latents: int
    num_inducing: int
    train_epochs: int
    lr: float
    acs_j: int
    initial_context_size: int
    split_seed: int
    train_seed: int
    gpu_id: int
    expected_num_outputs: int
    num_seeds: int = DEFAULT_NUM_SEEDS


# =============================================================================
# Utilities
# =============================================================================

def set_all_seeds(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def clear_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cpu_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def resolve_device(gpu_id: int = 0) -> torch.device:
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if gpu_id < 0 or gpu_id >= n:
            raise ValueError(f"gpu_id={gpu_id}, but only {n} CUDA device(s) are visible.")
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def scenario_output_dir(save_root: Path, scenario_stem: str) -> Path:
    out = save_root / scenario_stem
    out.mkdir(parents=True, exist_ok=True)
    return out


def remove_owned_outputs(output_dir: Path) -> None:
    for name in (
        "mape_trajectories_all_strategies.png",
        "metrics_all_strategies.csv",
        "true_vs_predicted_ACS_FW.png",
        "true_vs_predicted_BALD.png",
        "true_vs_predicted_RWPV.png",
    ):
        p = output_dir / name
        if p.exists():
            p.unlink()


# =============================================================================
# Data
# =============================================================================

def _make_one_hot_encoder() -> OneHotEncoder:
    """OneHotEncoder that works with both new and old scikit-learn APIs."""
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown="ignore")


def load_data(path: Union[str, Path], expected_num_outputs: int):
    """
    X = one-hot encoding of the first 7 columns (categorical features)
    column index 7 is skipped
    Y = columns index 8 onward, original response scale
    """
    path = Path(path)
    df = pd.read_csv(path)

    if df.shape[1] < 9:
        raise ValueError(
            f"{path.name}: expected at least 9 columns "
            "(7 features + 1 skipped + response columns)."
        )

    feature_cols = list(df.columns[:7])
    skipped_col = str(df.columns[7])
    response_cols = list(df.columns[8:])

    if len(response_cols) != int(expected_num_outputs):
        raise ValueError(
            f"{path.name}: expected {expected_num_outputs} response columns from "
            f"df.columns[8:], but found {len(response_cols)}."
        )

    if df[feature_cols].isna().any().any():
        raise ValueError(f"{path.name}: NaN detected in feature columns.")
    if df[response_cols].isna().any().any():
        raise ValueError(f"{path.name}: NaN detected in response columns.")

    # Categorical inputs -> one-hot indicators (no standardization).
    # astype(str) makes every distinct level a separate category, so numeric
    # codes such as 1, 2, 3 are treated as labels rather than magnitudes.
    encoder = _make_one_hot_encoder()
    X_np = encoder.fit_transform(df[feature_cols].astype(str)).astype(np.float64)
    Y_np = df[response_cols].to_numpy(dtype=np.float64)

    X = torch.tensor(X_np, dtype=torch.float32)
    Y = torch.tensor(Y_np, dtype=torch.float32)

    levels_per_feature = {
        str(col): [str(level) for level in cats]
        for col, cats in zip(feature_cols, encoder.categories_)
    }

    metadata = {
        "feature_cols": feature_cols,
        "skipped_col": skipped_col,
        "response_cols": response_cols,
        "n_rows": int(len(df)),
        "n_raw_features": int(len(feature_cols)),
        "n_features": int(X.shape[1]),  # encoded (one-hot) input dimension
        "levels_per_feature": levels_per_feature,
        "n_tasks": int(Y.shape[1]),
    }
    return X, Y, metadata


class ResponseStandardizer:
    """Per-response standardization fitted on the initial context only."""

    def __init__(self, y_context: torch.Tensor):
        mean = y_context.mean(dim=0)
        std = y_context.std(dim=0, unbiased=False)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        self.mean_cpu = mean.detach().cpu()
        self.std_cpu = std.detach().cpu()

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean_cpu.to(y.device)) / self.std_cpu.to(y.device)

    def inverse_mean(self, mean_std: torch.Tensor) -> torch.Tensor:
        return mean_std * self.std_cpu.to(mean_std.device) + self.mean_cpu.to(mean_std.device)

    def inverse_var(self, var_std: torch.Tensor) -> torch.Tensor:
        return var_std * (self.std_cpu.to(var_std.device) ** 2)


# =============================================================================
# Shared-latent variational separable multi-output GP
# =============================================================================

def choose_inducing_points(
    x_train: torch.Tensor,
    num_latents: int,
    num_inducing: int,
    seed: int,
) -> torch.Tensor:
    n = x_train.shape[0]
    m = min(int(num_inducing), n)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    perm = torch.randperm(n, generator=gen)[:m]
    base = x_train.detach().cpu()[perm]
    inducing = base.unsqueeze(0).repeat(int(num_latents), 1, 1)
    return inducing.to(x_train.device)


class LiTanSharedLatentVariationalGP(gpytorch.models.ApproximateGP):
    """
    LMC variational multi-output GP.

    Each latent process uses a Matérn-3/2 ARD kernel over the one-hot encoded
    x (one lengthscale per categorical level). LMC mixing maps the latent GPs
    to the 144 output tasks. With num_latents=1 this gives a rank-one shared
    latent output structure analogous to a separable input/task covariance.
    """

    def __init__(
        self,
        inducing_points: torch.Tensor,
        num_tasks: int,
        num_latents: int,
        n_features: int,
    ):
        batch_shape = torch.Size([int(num_latents)])
        m = inducing_points.shape[-2]

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            m, batch_shape=batch_shape
        )
        base_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        lmc_strategy = gpytorch.variational.LMCVariationalStrategy(
            base_strategy,
            num_tasks=int(num_tasks),
            num_latents=int(num_latents),
            latent_dim=-1,
        )
        super().__init__(lmc_strategy)

        self.num_tasks = int(num_tasks)
        self.num_latents = int(num_latents)

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=1.5,
                ard_num_dims=int(n_features),
                batch_shape=batch_shape,
                lengthscale_constraint=gpytorch.constraints.Interval(1e-3, 1e3),
            ),
            batch_shape=batch_shape,
        )

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=int(num_tasks),
            rank=0,
            has_global_noise=True,
            has_task_noise=True,
            noise_constraint=gpytorch.constraints.GreaterThan(1e-5),
        )

        with torch.no_grad():
            self.likelihood.task_noises = torch.full(
                (int(num_tasks),), 0.05, dtype=inducing_points.dtype, device=inducing_points.device
            )
            self.likelihood.noise = torch.tensor(
                0.01, dtype=inducing_points.dtype, device=inducing_points.device
            )

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_gp(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    cfg: RunConfig,
    seed: int,
    warm_state: Optional[Dict[str, torch.Tensor]] = None,
) -> LiTanSharedLatentVariationalGP:
    set_all_seeds(seed)

    inducing = choose_inducing_points(
        x_train=x_train,
        num_latents=cfg.num_latents,
        num_inducing=cfg.num_inducing,
        seed=seed + 17,
    )
    model = LiTanSharedLatentVariationalGP(
        inducing_points=inducing,
        num_tasks=y_train.shape[1],
        num_latents=cfg.num_latents,
        n_features=x_train.shape[1],
    ).to(x_train.device)

    if warm_state is not None:
        model.load_state_dict(warm_state, strict=True)

    model.train()
    model.likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
    mll = gpytorch.mlls.VariationalELBO(
        model.likelihood,
        model,
        num_data=x_train.shape[0],
    )

    best_loss = float("inf")
    best_state = None
    wait = 0

    with gpytorch.settings.cholesky_jitter(CHOLESKY_JITTER):
        for epoch in range(1, int(cfg.train_epochs) + 1):
            optimizer.zero_grad(set_to_none=True)
            output = model(x_train)
            loss = -mll(output, y_train)

            if loss.ndim > 0:
                loss = loss.sum()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite variational GP loss at epoch {epoch}: "
                    f"{float(loss.detach().cpu())}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            value = float(loss.detach().cpu())
            if value < best_loss - EARLY_STOP_TOL:
                best_loss = value
                best_state = cpu_state_dict(model)
                wait = 0
            elif epoch >= MIN_TRAIN_EPOCHS:
                wait += 1
                if wait >= EARLY_STOP_PATIENCE:
                    break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    model.eval()
    model.likelihood.eval()
    return model


# =============================================================================
# Prediction / metrics
# =============================================================================

def predictive_mean_var_standardized(
    model: LiTanSharedLatentVariationalGP,
    x: torch.Tensor,
    include_noise: bool,
    chunk_size: int = PREDICT_CHUNK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    means, vars_ = [], []
    model.eval()
    model.likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(CHOLESKY_JITTER):
        for start in range(0, len(x), chunk_size):
            xb = x[start:start + chunk_size]
            dist = model(xb)
            if include_noise:
                dist = model.likelihood(dist)
            means.append(dist.mean.detach().cpu())
            vars_.append(dist.variance.detach().cpu())

    if not means:
        empty = torch.empty((0, model.num_tasks), dtype=torch.float32)
        return empty, empty.clone()
    return torch.cat(means, dim=0), torch.cat(vars_, dim=0)


def predict_original_scale(
    model: LiTanSharedLatentVariationalGP,
    x: torch.Tensor,
    scaler: ResponseStandardizer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    m, v = predictive_mean_var_standardized(model, x, include_noise=True)
    return scaler.inverse_mean(m).cpu(), scaler.inverse_var(v).cpu()


def calculate_mape(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    yt = y_true.detach().cpu().float()
    yp = y_pred.detach().cpu().float()
    nz = torch.abs(yt[torch.abs(yt) > 0])
    eps = MAPE_EPS if nz.numel() == 0 else max(MAPE_EPS, float(torch.median(nz)) * 1e-8)
    return float(torch.mean(torch.abs(yp - yt) / torch.clamp(torch.abs(yt), min=eps)).item())


def calculate_mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    yt = y_true.detach().cpu().float()
    yp = y_pred.detach().cpu().float()
    return float(torch.mean((yp - yt) ** 2).item())


def evaluate_test_metrics(model, x_test, y_test, scaler) -> Tuple[float, float]:
    pred, _ = predict_original_scale(model, x_test, scaler)
    return calculate_mape(y_test, pred), calculate_mse(y_test, pred)


# =============================================================================
# Acquisition strategies
# =============================================================================

def latent_marginal_variance(model, x_pool: torch.Tensor) -> torch.Tensor:
    _, v = predictive_mean_var_standardized(model, x_pool, include_noise=False)
    return v.clamp_min(1e-12)


def bald_scores(model, x_pool: torch.Tensor) -> torch.Tensor:
    epi = latent_marginal_variance(model, x_pool)
    task_noise = model.likelihood.task_noises.detach().cpu().reshape(1, -1)
    global_noise = model.likelihood.noise.detach().cpu().reshape(1, 1)
    noise = (task_noise + global_noise).clamp_min(1e-12)
    return 0.5 * torch.log1p(epi / noise).sum(dim=1)


def bald_select_batch(model, x_pool, batch_size: int) -> torch.Tensor:
    s = bald_scores(model, x_pool)
    k = min(int(batch_size), len(s))
    return torch.topk(s, k=k, largest=True).indices.to(x_pool.device)


def acs_fw_select_batch(
    model,
    x_pool: torch.Tensor,
    batch_size: int,
    J: int,
    seed: int,
) -> torch.Tensor:
    M = len(x_pool)
    if M == 0:
        return torch.empty(0, dtype=torch.long, device=x_pool.device)
    k = min(int(batch_size), M)

    mean, var = predictive_mean_var_standardized(model, x_pool, include_noise=True)
    var = var.clamp_min(1e-10)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    eps = torch.randn((M, int(J), mean.shape[1]), generator=gen, dtype=mean.dtype)
    log_norm = torch.log(2.0 * math.pi * var)
    logp = -0.5 * (eps ** 2 + log_norm[:, None, :]).sum(dim=2)
    V = logp / math.sqrt(max(int(J), 1))

    sigma = torch.linalg.vector_norm(V, dim=1).clamp_min(1e-12)
    target = V.sum(dim=0)
    approx = torch.zeros_like(target)
    selected: List[int] = []

    for _ in range(k):
        residual = target - approx
        inner = (V @ residual) / sigma
        if selected:
            inner[torch.tensor(selected, dtype=torch.long)] = -torch.inf
        f = int(torch.argmax(inner).item())
        selected.append(f)

        atom = (sigma.sum() / sigma[f]) * V[f]
        direction = atom - approx
        denom = torch.dot(direction, direction)
        gamma = 0.0 if float(denom) <= 1e-20 else float(
            torch.clamp(torch.dot(residual, direction) / denom, 0.0, 1.0)
        )
        approx = approx + gamma * direction

    return torch.tensor(selected, dtype=torch.long, device=x_pool.device)


def rwpv_scores(model, x_pool: torch.Tensor, y_context_std: torch.Tensor) -> torch.Tensor:
    if len(x_pool) == 0:
        return torch.empty(0)

    mean, epi = predictive_mean_var_standardized(model, x_pool, include_noise=False)
    epi = epi.clamp_min(1e-12)
    yctx = y_context_std.detach().cpu()
    cmean = yctx.mean(dim=0)
    cvar = yctx.var(dim=0, unbiased=False).clamp_min(1e-4)

    sq = (mean - cmean) ** 2 / cvar
    log_rel = -0.5 * (sq + torch.log(2.0 * math.pi * cvar)).mean(dim=1)
    shifted = (log_rel - log_rel.max()) / max(RWPV_RELEVANCE_TEMPERATURE, 1e-8)
    rel = torch.exp(shifted)
    rel = RWPV_RELEVANCE_FLOOR + (1.0 - RWPV_RELEVANCE_FLOOR) * rel
    return epi.mean(dim=1) * rel


def rwpv_select_batch(model, x_pool, y_context_std, batch_size: int) -> torch.Tensor:
    s = rwpv_scores(model, x_pool, y_context_std)
    k = min(int(batch_size), len(s))
    return torch.topk(s, k=k, largest=True).indices.to(x_pool.device)


def select_strategy_batch(
    strategy: str,
    model,
    x_pool: torch.Tensor,
    y_context_std: torch.Tensor,
    cfg: RunConfig,
    seed: int,
) -> torch.Tensor:
    if strategy == "BALD":
        return bald_select_batch(model, x_pool, cfg.batch_size)
    if strategy == "ACS_FW":
        return acs_fw_select_batch(model, x_pool, cfg.batch_size, cfg.acs_j, seed)
    if strategy == "RWPV":
        return rwpv_select_batch(model, x_pool, y_context_std, cfg.batch_size)
    raise ValueError(f"Unknown strategy: {strategy}")


# =============================================================================
# Splitting / experiment
# =============================================================================

def make_split(n_rows: int, split_seed: int, cfg: RunConfig):
    rng = np.random.default_rng(int(split_seed))
    all_idx = np.arange(n_rows)

    n_test = max(N_VIS_CURVES, int(round(TEST_FRACTION * n_rows)))
    max_test = n_rows - cfg.initial_context_size - cfg.batch_size
    n_test = min(n_test, max_test)
    if n_test < N_VIS_CURVES:
        raise ValueError(
            f"N={n_rows} is too small for {N_VIS_CURVES} test curves and "
            f"initial_context_size={cfg.initial_context_size}."
        )

    test_idx = np.sort(rng.choice(all_idx, size=n_test, replace=False))
    train_pool = np.setdiff1d(all_idx, test_idx)
    ctx_idx = np.sort(rng.choice(train_pool, size=cfg.initial_context_size, replace=False))
    pool_idx = np.setdiff1d(train_pool, ctx_idx)

    needed = cfg.acq_rounds * cfg.batch_size
    if len(pool_idx) < needed:
        raise ValueError(
            f"Split seed {split_seed}: pool has {len(pool_idx)} points, but "
            f"{needed} are needed for {cfg.acq_rounds} rounds x batch {cfg.batch_size}."
        )
    return ctx_idx, pool_idx, test_idx


def run_strategy(
    strategy: str,
    replicate: int,
    split_seed: int,
    train_seed: int,
    X: torch.Tensor,
    Y: torch.Tensor,
    ctx_idx: np.ndarray,
    pool_idx: np.ndarray,
    test_idx: np.ndarray,
    initial_state: Dict[str, torch.Tensor],
    scaler: ResponseStandardizer,
    cfg: RunConfig,
    device: torch.device,
    viz_positions: Optional[np.ndarray],
):
    x_ctx = X[ctx_idx].to(device)
    y_ctx_orig = Y[ctx_idx].to(device)
    y_ctx_std = scaler.transform(y_ctx_orig)
    x_pool = X[pool_idx].to(device)
    y_pool = Y[pool_idx].to(device)
    x_test = X[test_idx].to(device)
    y_test = Y[test_idx].detach().cpu()

    # Rebuild the same model shape and restore the paired initial state.
    inducing = choose_inducing_points(x_ctx, cfg.num_latents, cfg.num_inducing, train_seed + 17)
    model = LiTanSharedLatentVariationalGP(
        inducing, y_ctx_std.shape[1], cfg.num_latents, x_ctx.shape[1]
    ).to(device)
    model.load_state_dict(initial_state, strict=True)
    model.eval(); model.likelihood.eval()

    records = []
    mape, mse = evaluate_test_metrics(model, x_test, y_test, scaler)
    records.append({
        "replicate": replicate,
        "split_seed": split_seed,
        "train_seed": train_seed,
        "strategy": strategy,
        "acquisition_step": 0,
        "context_size": len(x_ctx),
        "pool_size": len(x_pool),
        "test_mape": mape,
        "test_mse": mse,
    })

    warm_state = cpu_state_dict(model)

    for acq in range(1, cfg.acq_rounds + 1):
        select_seed = train_seed * 1000 + acq
        selected = select_strategy_batch(
            strategy, model, x_pool, y_ctx_std, cfg, select_seed
        )

        x_ctx = torch.cat([x_ctx, x_pool[selected]], dim=0)
        y_ctx_orig = torch.cat([y_ctx_orig, y_pool[selected]], dim=0)
        y_ctx_std = scaler.transform(y_ctx_orig)

        keep = torch.ones(len(x_pool), dtype=torch.bool, device=device)
        keep[selected] = False
        x_pool = x_pool[keep]
        y_pool = y_pool[keep]

        del model
        clear_cache()
        model = fit_gp(
            x_ctx, y_ctx_std, cfg,
            seed=train_seed * 10_000 + acq,
            warm_state=warm_state,
        )
        warm_state = cpu_state_dict(model)

        mape, mse = evaluate_test_metrics(model, x_test, y_test, scaler)
        records.append({
            "replicate": replicate,
            "split_seed": split_seed,
            "train_seed": train_seed,
            "strategy": strategy,
            "acquisition_step": acq,
            "context_size": len(x_ctx),
            "pool_size": len(x_pool),
            "test_mape": mape,
            "test_mse": mse,
        })
        print(
            f"rep {replicate:02d} | split {split_seed:3d} | {strategy:6s} | "
            f"step {acq:02d}/{cfg.acq_rounds} | MAPE={mape:.6f} | MSE={mse:.6g}"
        )

    viz = None
    if viz_positions is not None:
        pred, _ = predict_original_scale(model, x_test, scaler)
        pos = np.asarray(viz_positions, dtype=int)
        viz = {
            "test_dataset_indices": np.asarray(test_idx, dtype=int)[pos],
            "y_true": y_test[pos].numpy(),
            "y_pred": pred[pos].numpy(),
        }

    del model
    clear_cache()
    return records, viz


def run_replicate(
    replicate: int,
    X: torch.Tensor,
    Y: torch.Tensor,
    cfg: RunConfig,
    device: torch.device,
):
    split_seed = cfg.split_seed + replicate
    train_seed = cfg.train_seed + replicate
    ctx_idx, pool_idx, test_idx = make_split(len(X), split_seed, cfg)
    scaler = ResponseStandardizer(Y[ctx_idx])

    x_ctx = X[ctx_idx].to(device)
    y_ctx_std = scaler.transform(Y[ctx_idx].to(device))
    initial_model = fit_gp(x_ctx, y_ctx_std, cfg, train_seed, warm_state=None)
    initial_state = cpu_state_dict(initial_model)
    del initial_model
    clear_cache()

    viz_positions = None
    if replicate == 0:
        rng = np.random.default_rng(split_seed + 77_777)
        viz_positions = np.sort(rng.choice(len(test_idx), size=N_VIS_CURVES, replace=False))

    all_records = []
    viz_payload = {}
    for strategy in STRATEGIES:
        records, viz = run_strategy(
            strategy=strategy,
            replicate=replicate,
            split_seed=split_seed,
            train_seed=train_seed,
            X=X,
            Y=Y,
            ctx_idx=ctx_idx,
            pool_idx=pool_idx,
            test_idx=test_idx,
            initial_state=initial_state,
            scaler=scaler,
            cfg=cfg,
            device=device,
            viz_positions=viz_positions,
        )
        all_records.extend(records)
        if viz is not None:
            viz_payload[strategy] = viz
    return all_records, viz_payload


# =============================================================================
# Minimal outputs
# =============================================================================

def save_metrics(metrics: pd.DataFrame, output_dir: Path) -> Path:
    p = output_dir / "metrics_all_strategies.csv"
    cols = [
        "replicate", "split_seed", "train_seed", "strategy",
        "acquisition_step", "context_size", "pool_size", "test_mape", "test_mse",
    ]
    metrics[cols].sort_values(["strategy", "replicate", "acquisition_step"]).to_csv(p, index=False)
    return p


def save_mape_plot(metrics: pd.DataFrame, output_dir: Path, cfg: RunConfig) -> Path:
    p = output_dir / "mape_trajectories_all_strategies.png"
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    markers = {"ACS_FW": "o", "BALD": "s", "RWPV": "^"}
    styles = {"ACS_FW": "-", "BALD": "--", "RWPV": "-."}

    for strategy in STRATEGIES:
        d = metrics[metrics.strategy == strategy]
        agg = d.groupby("acquisition_step")["test_mape"].agg(["mean", "std"]).reset_index()
        x = agg.acquisition_step.to_numpy()
        y = agg["mean"].to_numpy()
        s = agg["std"].fillna(0.0).to_numpy()
        line = ax.plot(x, y, marker=markers[strategy], linestyle=styles[strategy], label=strategy)[0]
        if cfg.num_seeds > 1:
            ax.fill_between(x, np.maximum(0, y - s), y + s, alpha=0.12, color=line.get_color())

    ax.set_xlabel("Acquisition Step")
    ax.set_ylabel("Test MAPE")
    ax.set_title("MAPE Trajectories for All Acquisition Strategies")
    ax.set_xticks(np.arange(0, cfg.acq_rounds + 1, 1))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(p, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def save_curve_grid(strategy: str, payload: dict, output_dir: Path) -> Path:
    p = output_dir / f"true_vs_predicted_{strategy}.png"
    yt = np.asarray(payload["y_true"])
    yp = np.asarray(payload["y_pred"])
    ids = np.asarray(payload["test_dataset_indices"])
    response_axis = np.arange(yt.shape[1])

    fig, axes = plt.subplots(3, 3, figsize=(11.0, 8.0))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        ax.plot(response_axis, yt[i], linewidth=1.6, label="True")
        ax.plot(response_axis, yp[i], linewidth=1.4, linestyle="--", label="Predicted")
        ax.set_title(f"Sample {int(ids[i])}", fontsize=9)
        ax.grid(alpha=0.20)
    for ax in axes[6:9]:
        ax.set_xlabel("Response Index")
    for ax in axes[0:7:3]:
        ax.set_ylabel("Response")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"True vs Predicted Response Curves - {strategy}", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(p, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def write_minimal_outputs(metrics, viz_payload, output_dir, cfg) -> List[Path]:
    paths = [save_mape_plot(metrics, output_dir, cfg), save_metrics(metrics, output_dir)]
    for strategy in STRATEGIES:
        if strategy not in viz_payload:
            raise RuntimeError(f"Missing visualization payload for {strategy}.")
        paths.append(save_curve_grid(strategy, viz_payload[strategy], output_dir))
    return paths


# =============================================================================
# Scenario / runner entry points
# =============================================================================

def run_scenario(csv_path: Union[str, Path], cfg: RunConfig, device: torch.device):
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    out = scenario_output_dir(cfg.save_dir, csv_path.stem)
    remove_owned_outputs(out)
    X, Y, meta = load_data(csv_path, cfg.expected_num_outputs)

    levels_summary = ", ".join(
        f"{col}={len(levels)}" for col, levels in meta["levels_per_feature"].items()
    )

    print("\n" + "#" * 92)
    print(f"Scenario             : {csv_path.stem}")
    print(f"Input                : {csv_path}")
    print(f"Output               : {out}")
    print(f"Rows                 : {meta['n_rows']}")
    print(f"Input encoding       : one-hot ({meta['n_raw_features']} categorical features "
          f"-> {meta['n_features']} indicator columns)")
    print(f"Levels per feature   : {levels_summary}")
    print(f"Responses            : {meta['n_tasks']}")
    print(f"Strategies           : {', '.join(STRATEGIES)}")
    print(f"Replicates           : {cfg.num_seeds}")
    print(f"Acquisition rounds   : {cfg.acq_rounds}")
    print(f"Batch size           : {cfg.batch_size}")
    print(f"Latents / inducing   : {cfg.num_latents} / {cfg.num_inducing}")
    print(f"Train epochs / lr    : {cfg.train_epochs} / {cfg.lr}")
    print(f"Device               : {device}")
    print("#" * 92)

    records = []
    viz_payload = {}
    for rep in range(cfg.num_seeds):
        print(f"\n--- replicate {rep + 1}/{cfg.num_seeds} ---")
        rec, viz = run_replicate(rep, X, Y, cfg, device)
        records.extend(rec)
        if rep == 0:
            viz_payload = viz

    metrics = pd.DataFrame(records)
    saved = write_minimal_outputs(metrics, viz_payload, out, cfg)

    final_step = metrics[metrics.acquisition_step == metrics.acquisition_step.max()]
    summary = (
        final_step.groupby("strategy")
        .agg(
            final_mape_mean=("test_mape", "mean"),
            final_mape_std=("test_mape", "std"),
            final_mse_mean=("test_mse", "mean"),
            final_mse_std=("test_mse", "std"),
        )
        .reset_index()
    )
    summary.insert(0, "scenario", csv_path.stem)

    print("\nSaved ONLY:")
    for p in saved:
        print(f"  {p}")
    return summary


def run_notebook_workflow(
    base_dir: Union[str, Path] = DEFAULT_BASE_DIR,
    scenario_files: Optional[Sequence[str]] = None,
    acq_rounds: int = 20,
    batch_size: int = 5,
    save_dir: Union[str, Path] = DEFAULT_SAVE_DIR,
    num_latents: int = 1,
    num_inducing: int = 30,
    train_epochs: int = 300,
    lr: float = 0.03,
    acs_j: int = 50,
    initial_context_size: int = 64,
    split_seed: int = 1,
    train_seed: int = 220,
    gpu_id: int = 0,
    expected_num_outputs: int = EXPECTED_NUM_OUTPUTS,
    num_seeds: int = DEFAULT_NUM_SEEDS,
):
    """
    M3 runner entry point. All arguments in the user's runner are accepted and
    used. `num_seeds` defaults to 20; split seeds are split_seed + replicate
    and training seeds are train_seed + replicate.
    """
    base = Path(base_dir)
    scenarios = tuple(DEFAULT_SCENARIO_FILES if scenario_files is None else scenario_files)
    cfg = RunConfig(
        base_dir=base,
        scenario_files=scenarios,
        save_dir=Path(save_dir),
        acq_rounds=int(acq_rounds),
        batch_size=int(batch_size),
        num_latents=int(num_latents),
        num_inducing=int(num_inducing),
        train_epochs=int(train_epochs),
        lr=float(lr),
        acs_j=int(acs_j),
        initial_context_size=int(initial_context_size),
        split_seed=int(split_seed),
        train_seed=int(train_seed),
        gpu_id=int(gpu_id),
        expected_num_outputs=int(expected_num_outputs),
        num_seeds=int(num_seeds),
    )

    if cfg.acq_rounds < 1 or cfg.batch_size < 1:
        raise ValueError("acq_rounds and batch_size must be positive.")
    if cfg.num_latents < 1 or cfg.num_inducing < 1:
        raise ValueError("num_latents and num_inducing must be positive.")
    if cfg.num_seeds < 1:
        raise ValueError("num_seeds must be positive.")

    device = resolve_device(cfg.gpu_id)

    paths = [cfg.base_dir / name for name in cfg.scenario_files]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing scenario file(s):\n  " + "\n  ".join(missing))

    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    summaries = [run_scenario(p, cfg, device) for p in paths]
    return pd.concat(summaries, ignore_index=True)


def verify_runner_api() -> bool:
    """Return True only if run_notebook_workflow exposes the required M3 API."""
    import inspect

    actual = tuple(inspect.signature(run_notebook_workflow).parameters)
    if actual != RUN_NOTEBOOK_WORKFLOW_PARAMETERS:
        raise RuntimeError(
            "run_notebook_workflow API mismatch. "
            f"Expected {RUN_NOTEBOOK_WORKFLOW_PARAMETERS}, got {actual}."
        )
    return True


def main():
    verify_runner_api()
    summary = run_notebook_workflow()
    print("\nFinal summary:\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()