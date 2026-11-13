# -*- coding: utf-8 -*-
"""
R1-MTGP.py
===========
Baseline exact multitask Gaussian Process + active learning for SMU M3.

Baseline model
--------------
- Exact GPyTorch multitask GP
- Constant multitask mean
- ScaleKernel(RBFKernel(ARD)) x MultitaskKernel(rank=1)
- Per-task observation noise
- Acquisition strategies: BALD and ACS-FW

Data layout
-----------
- Columns 0:7   -> seven categorical input features, one-hot encoded (FedMGP style)
- Column 7      -> intentionally skipped
- Columns 8:    -> response curve (expected: 144 outputs)

Metric indexing
---------------
Metric trajectories (context/pool/test MAPE and MSE) are saved for
acq_step = 0, 1, ..., acq_rounds:
  - acq_step 0  -> GP fitted on the initial context only (before any
                   acquisition); identical for ACS_FW and BALD.
  - acq_step k  -> GP retrained after the k-th acquired batch.
Selection and diversity records start at acq_step 1.

Parallelism
-----------
Work is split into (seed, scenario) jobs and distributed round-robin over
n_gpus worker processes. With GPUs, each worker is pinned to one GPU.
Without GPUs, n_gpus CPU worker processes are used (1 torch thread each).
Passing gpu_id forces a single sequential worker on that GPU.

Checkpointing / resume
----------------------
Every finished (seed, scenario) job is written immediately by the worker to
    <save_root>/_partial/<scenario_stem>_seed<seed>.pkl
If the session dies, rerun the same call with the same save_dir: finished
jobs are loaded from disk and only the missing ones are run. A run-settings
signature is stored in _partial/run_signature.json; resuming with different
settings raises an error (use a new save_dir or resume=False).

M3 entry point
--------------
    run_notebook_workflow(...)
Unknown keyword arguments raise TypeError (no silent ignoring).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import json
import math
import pickle
import queue as _queue
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.multiprocessing as mp
import gpytorch

from sklearn.preprocessing import OneHotEncoder
from gpytorch.settings import fast_pred_var


# =============================================================================
# CONFIGURATION
# =============================================================================

SCENARIO_CSV_DIR = "./Scenarios to Run"

DEFAULT_SCENARIO_FILES = [
    "scenario_000_HGR.csv",
    "scenario_016_damped.csv",
    "scenario_054_gompertz.csv",
    "scenario_179_sigmoid.csv",
]

EXPECTED_NUM_OUTPUTS = 144

INITIAL_CONTEXT_SIZE = 64
DEFAULT_NUM_SEEDS = 16
DEFAULT_SEED_START = 100
DEFAULT_N_WORKERS = 8
DEFAULT_ACQ_ROUNDS = 20
DEFAULT_BATCH_SIZE = 5
DEFAULT_ACS_J = 50

DEFAULT_GP_LR = 0.1
DEFAULT_GP_EPOCHS = 100

# How long the main process waits on the result queue before checking
# whether any worker died silently (seconds).
QUEUE_POLL_SECONDS = 60

VAR_EPS = 1e-8
MAPE_EPS = 1e-8

METRIC_KEYS = (
    "context_mape",
    "pool_mape",
    "test_mape",
    "context_mse",
    "pool_mse",
    "test_mse",
)


# =============================================================================
# REPRODUCIBILITY / MEMORY
# =============================================================================

def set_all_seeds(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))


def clear_torch_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_cpu_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def cpu_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


# =============================================================================
# SCENARIO DISCOVERY / LOADING
# =============================================================================

def _scenario_id_from_name(filename: str) -> int:
    stem = Path(filename).stem
    pieces = stem.split("_")
    if len(pieces) < 2 or not pieces[1].isdigit():
        raise ValueError("Cannot extract scenario id from filename: %s" % filename)
    return int(pieces[1])


def _scenario_key(path: str) -> str:
    """Unique key per scenario file (ids can repeat, e.g. 000_HGR / 000_burnup)."""
    return Path(path).stem


def list_scenario_csvs(folder: str) -> List[Tuple[int, str]]:
    base = Path(folder)
    if not base.is_dir():
        raise RuntimeError("Scenario directory not found: %s" % folder)

    out: List[Tuple[int, str]] = []
    missing: List[str] = []

    for name in DEFAULT_SCENARIO_FILES:
        path = base / name
        if path.is_file():
            out.append((_scenario_id_from_name(name), str(path)))
        else:
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "The GP baseline expects all default scenario CSVs.\nMissing:\n  "
            + "\n  ".join(missing)
        )

    return out


def load_scenario_csv(
    path: str,
    expected_num_outputs: Optional[int] = EXPECTED_NUM_OUTPUTS,
) -> Tuple[torch.Tensor, torch.Tensor, OneHotEncoder, Dict[str, object]]:
    """
    X = first 7 columns, encoded as categorical one-hot indicators (FedMGP style)
    column index 7 is skipped
    Y = columns index 8 onward, original response scale
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("File not found: %s" % path)

    df = pd.read_csv(path)

    if df.shape[1] < 9:
        raise ValueError(
            "%s has %d columns; expected at least 9 "
            "(7 features + 1 skipped + responses)."
            % (os.path.basename(path), df.shape[1])
        )

    feature_cols = list(df.columns[:7])
    skipped_col = str(df.columns[7])
    response_cols = list(df.columns[8:])

    if expected_num_outputs is not None and len(response_cols) != int(expected_num_outputs):
        raise ValueError(
            "%s: expected %d response columns from df.columns[8:], but found %d."
            % (os.path.basename(path), int(expected_num_outputs), len(response_cols))
        )

    # Match FedMGP: interpret each of the first seven features as a
    # categorical label, even when the CSV uses numeric category codes.
    # Convert values to strings and one-hot encode; never standardize X.
    X_df = df[feature_cols].astype(str)

    Y_df = df[response_cols].apply(pd.to_numeric, errors="coerce")
    if Y_df.isna().any().any():
        bad = int(Y_df.isna().sum().sum())
        raise ValueError(
            "%s: %d non-numeric/missing values found in response columns."
            % (os.path.basename(path), bad)
        )

    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    X_np = encoder.fit_transform(X_df).astype(np.float64)
    Y_np = Y_df.to_numpy(dtype=np.float64)

    X = torch.tensor(X_np, dtype=torch.float32)
    Y = torch.tensor(Y_np, dtype=torch.float32)

    metadata: Dict[str, object] = {
        "filename": os.path.basename(path),
        "n_rows": int(len(df)),
        "feature_cols": feature_cols,
        "skipped_col": skipped_col,
        "response_cols": response_cols,
        "n_features": int(X.shape[1]),
        "n_tasks": int(Y.shape[1]),
    }
    return X, Y, encoder, metadata


# =============================================================================
# SPLIT
# =============================================================================

def make_seed_split(
    X: torch.Tensor,
    Y: torch.Tensor,
    seed: int,
    test_frac: float = 0.5,
    initial_context_size: int = INITIAL_CONTEXT_SIZE,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    Fixed-test active-learning split. The test set never participates in
    acquisition. The remaining rows form the initial context and pool.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not (0.0 < float(test_frac) < 1.0):
        raise ValueError("test_frac/val_frac must lie strictly between 0 and 1.")

    initial_context_size = int(initial_context_size)
    N = len(X)
    n_test = int(round(float(test_frac) * N))
    n_test = max(1, min(n_test, N - initial_context_size - 1))

    if N - n_test < initial_context_size + 1:
        raise ValueError(
            "Not enough non-test rows for initial context and acquisition pool."
        )

    rng = np.random.default_rng(int(seed))
    all_idx = np.arange(N, dtype=int)

    test_idx = np.sort(rng.choice(all_idx, size=n_test, replace=False))
    active_idx = np.setdiff1d(all_idx, test_idx)

    ctx_idx = np.sort(
        rng.choice(active_idx, size=initial_context_size, replace=False)
    )
    pool_idx = np.setdiff1d(active_idx, ctx_idx)

    Xd = X.to(device)
    Yd = Y.to(device)

    return {
        "x_ctx": Xd[ctx_idx],
        "y_ctx": Yd[ctx_idx],
        "x_pool": Xd[pool_idx],
        "y_pool": Yd[pool_idx],
        "x_test": Xd[test_idx],
        "y_test": Yd[test_idx],
        "ctx_idx": ctx_idx.copy(),
        "pool_idx": pool_idx.copy(),
        "test_idx": test_idx.copy(),
    }


def clone_active_data(active_data: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in active_data.items():
        if torch.is_tensor(value):
            out[key] = value.clone()
        elif isinstance(value, np.ndarray):
            out[key] = value.copy()
        else:
            out[key] = deepcopy(value)
    return out


# =============================================================================
# EXACT MULTITASK GP
# =============================================================================

class MultitaskGPModel(gpytorch.models.ExactGP):
    """RBF(ARD) x rank-1 multitask covariance, with per-task noise."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        n_features: int,
    ):
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=train_y.shape[1],
            has_global_noise=False,
            has_task_noise=True,
        )
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(),
            num_tasks=train_y.shape[1],
        )

        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=int(n_features)
                )
            ),
            num_tasks=train_y.shape[1],
            rank=1,
        )

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(
            mean_x,
            covar_x,
        )


def train_gp(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    lr: float = DEFAULT_GP_LR,
    epochs: int = DEFAULT_GP_EPOCHS,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> MultitaskGPModel:
    if seed is not None:
        set_all_seeds(int(seed))

    model = MultitaskGPModel(
        x_train,
        y_train,
        n_features=x_train.shape[1],
    ).to(x_train.device)

    model.train()
    model.likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(
        model.likelihood,
        model,
    )

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, int(epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        output = model(x_train)
        loss = -mll(output, y_train)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite GP training loss at epoch %d." % epoch
            )

        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = cpu_state_dict(model)

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == int(epochs)):
            print(
                "    epoch %4d/%d | negative MLL = %.6f"
                % (epoch, int(epochs), loss_value),
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    model.likelihood.eval()
    return model


def restore_gp_from_state(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    state: Dict[str, torch.Tensor],
) -> MultitaskGPModel:
    model = MultitaskGPModel(
        x_train,
        y_train,
        n_features=x_train.shape[1],
    ).to(x_train.device)
    model.load_state_dict(state)
    model.eval()
    model.likelihood.eval()
    return model


# =============================================================================
# PREDICTION / METRICS
# =============================================================================

@torch.no_grad()
def predictive_mean_var(
    model: MultitaskGPModel,
    x: torch.Tensor,
    chunk_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(x) == 0:
        n_tasks = int(model.train_targets.shape[-1])
        empty = torch.empty((0, n_tasks), dtype=torch.float32)
        return empty, empty.clone()

    model.eval()
    model.likelihood.eval()

    means: List[torch.Tensor] = []
    variances: List[torch.Tensor] = []

    with fast_pred_var():
        for start in range(0, len(x), int(chunk_size)):
            xb = x[start:start + int(chunk_size)]
            pred = model.likelihood(model(xb))
            means.append(pred.mean.detach().cpu())
            variances.append(pred.variance.detach().cpu())

    return torch.cat(means, dim=0), torch.cat(variances, dim=0)


def calculate_mape_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = MAPE_EPS,
) -> float:
    denom = np.maximum(np.abs(y_true), float(eps))
    return float(np.mean(np.abs(y_pred - y_true) / denom))


def calculate_mse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def evaluate_gp(
    model: MultitaskGPModel,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[float, float]:
    if len(x) == 0:
        return float("nan"), float("nan")

    mean_cpu, _ = predictive_mean_var(model, x)
    y_np = to_cpu_np(y)
    pred_np = mean_cpu.numpy()

    return (
        calculate_mape_np(y_np, pred_np),
        calculate_mse_np(y_np, pred_np),
    )


# =============================================================================
# BALD
# =============================================================================

@torch.no_grad()
def bald_score_gp(
    model: MultitaskGPModel,
    x_candidates: torch.Tensor,
    chunk_size: int = 128,
) -> torch.Tensor:
    """
    Gaussian-regression BALD:
        0.5 * sum_t [log predictive_variance_t - log noise_variance_t]
    """
    if len(x_candidates) == 0:
        return torch.empty(0, device=x_candidates.device)

    model.eval()
    model.likelihood.eval()

    noise_vars = model.likelihood.task_noises.detach().clamp_min(VAR_EPS)
    noise_vars_cpu = noise_vars.cpu()

    scores: List[torch.Tensor] = []

    with fast_pred_var():
        for start in range(0, len(x_candidates), int(chunk_size)):
            xb = x_candidates[start:start + int(chunk_size)]
            pred = model.likelihood(model(xb))
            var_pred = pred.variance.detach().cpu().clamp_min(VAR_EPS)

            score = 0.5 * (
                torch.log(var_pred)
                - torch.log(noise_vars_cpu).unsqueeze(0)
            ).sum(dim=1)

            scores.append(score)

    return torch.cat(scores, dim=0).to(x_candidates.device)


def bald_select_batch_gp(
    model: MultitaskGPModel,
    x_pool: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, float]:
    scores = bald_score_gp(model, x_pool)
    k = min(int(batch_size), len(scores))

    if k <= 0:
        return torch.empty(0, dtype=torch.long, device=x_pool.device), float("nan")

    top = torch.topk(scores, k=k, largest=True).indices
    return top, float(scores[top].mean().detach().cpu())


# =============================================================================
# ACS-FW
# =============================================================================

@torch.no_grad()
def acs_fw_select_batch_gp(
    model: MultitaskGPModel,
    x_pool: torch.Tensor,
    batch_size: int,
    J: int = DEFAULT_ACS_J,
) -> torch.Tensor:
    """
    ACS-FW on the GP predictive marginals. V and Frank-Wolfe algebra are
    kept on CPU to reduce accelerator memory pressure.
    """
    model.eval()
    model.likelihood.eval()

    M = len(x_pool)
    k = min(int(batch_size), M)
    J = int(J)

    if k <= 0:
        return torch.empty(0, dtype=torch.long, device=x_pool.device)
    if J <= 0:
        raise ValueError("ACS-FW J must be positive.")

    V = torch.zeros((M, J), dtype=torch.float32)
    sigma_m = torch.zeros(M, dtype=torch.float32)

    for m in range(M):
        xm = x_pool[m:m + 1]

        with fast_pred_var():
            pred = model.likelihood(model(xm))

        mean_pred = pred.mean.squeeze(0).detach().cpu()
        var_pred = pred.variance.squeeze(0).detach().cpu().clamp_min(VAR_EPS)
        std_pred = torch.sqrt(var_pred)

        dist = torch.distributions.Normal(mean_pred, std_pred)
        y_samples = dist.sample((J,))
        log_probs = dist.log_prob(y_samples).sum(dim=-1)
        v_m = log_probs / math.sqrt(float(J))

        V[m] = v_m
        sigma_m[m] = torch.linalg.vector_norm(v_m, ord=2)

    sigma_safe = sigma_m.clamp_min(VAR_EPS)
    sigma_total = sigma_safe.sum()

    w = torch.zeros(M, dtype=torch.float32)
    v_L = V.sum(dim=0)
    v_Lw = torch.zeros(J, dtype=torch.float32)

    for _ in range(k):
        residual = v_L - v_Lw
        inner = torch.mv(V, residual) / sigma_safe
        f = int(torch.argmax(inner).item())

        u_f = (sigma_total / sigma_safe[f]) * V[f]
        d = u_f - v_Lw
        denom = torch.dot(d, d)

        if float(denom) > VAR_EPS:
            gamma = torch.clamp(
                torch.dot(residual, d) / denom,
                min=0.0,
                max=1.0,
            )
        else:
            gamma = torch.tensor(0.0)

        w = (1.0 - gamma) * w
        w[f] += gamma * (sigma_total / sigma_safe[f])
        v_Lw = (1.0 - gamma) * v_Lw + gamma * u_f

    positive = torch.nonzero(w > 0, as_tuple=False).squeeze(1)

    if len(positive) >= k:
        local = torch.topk(w[positive], k=k, largest=True).indices
        selected_cpu = positive[local]
    else:
        selected_list = positive.tolist()
        already = set(selected_list)
        for idx in torch.argsort(sigma_m, descending=True).tolist():
            if idx not in already:
                selected_list.append(int(idx))
                already.add(int(idx))
            if len(selected_list) == k:
                break
        selected_cpu = torch.tensor(selected_list, dtype=torch.long)

    return selected_cpu.to(x_pool.device)


# =============================================================================
# ACTIVE-LEARNING HELPERS
# =============================================================================

def batch_diversity(x_selected: torch.Tensor) -> Dict[str, float]:
    n = len(x_selected)

    if n < 2:
        return {
            "avg_pairwise_distance": 0.0,
            "min_pairwise_distance": 0.0,
            "max_pairwise_distance": 0.0,
            "batch_size": int(n),
        }

    x = to_cpu_np(x_selected).astype(float)
    diff = x[:, None, :] - x[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(n, k=1)
    vals = d[iu]

    return {
        "avg_pairwise_distance": float(np.mean(vals)),
        "min_pairwise_distance": float(np.min(vals)),
        "max_pairwise_distance": float(np.max(vals)),
        "batch_size": int(n),
    }


def reveal_selected(
    active: Dict[str, object],
    selected: torch.Tensor,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    x_pool = active["x_pool"]
    y_pool = active["y_pool"]
    pool_idx = active["pool_idx"]

    assert torch.is_tensor(x_pool)
    assert torch.is_tensor(y_pool)
    assert isinstance(pool_idx, np.ndarray)

    selected = selected.to(x_pool.device, dtype=torch.long)

    x_sel = x_pool[selected]
    y_sel = y_pool[selected]
    selected_original = pool_idx[to_cpu_np(selected).astype(int)]

    active["x_ctx"] = torch.cat([active["x_ctx"], x_sel], dim=0)
    active["y_ctx"] = torch.cat([active["y_ctx"], y_sel], dim=0)
    active["ctx_idx"] = np.concatenate(
        [active["ctx_idx"], selected_original.astype(int)]
    )

    keep = torch.ones(len(x_pool), dtype=torch.bool, device=x_pool.device)
    keep[selected] = False

    active["x_pool"] = x_pool[keep]
    active["y_pool"] = y_pool[keep]
    active["pool_idx"] = pool_idx[to_cpu_np(keep).astype(bool)]

    return active, {
        "x_selected": x_sel,
        "y_selected": y_sel,
        "selected_original_indices": selected_original,
    }


def _empty_strategy_record() -> Dict[str, object]:
    return {
        "context_mape": [],
        "pool_mape": [],
        "test_mape": [],
        "context_mse": [],
        "pool_mse": [],
        "test_mse": [],
        "n_context": [],
        "diversity": [],
        "selected_original_indices": [],
        "score_mean": [],
    }


def _append_metrics(
    record: Dict[str, object],
    model: MultitaskGPModel,
    active: Dict[str, object],
) -> Tuple[float, float, float]:
    ctx_mape, ctx_mse = evaluate_gp(model, active["x_ctx"], active["y_ctx"])
    pool_mape, pool_mse = evaluate_gp(model, active["x_pool"], active["y_pool"])
    test_mape, test_mse = evaluate_gp(model, active["x_test"], active["y_test"])

    record["context_mape"].append(float(ctx_mape))
    record["pool_mape"].append(float(pool_mape))
    record["test_mape"].append(float(test_mape))

    record["context_mse"].append(float(ctx_mse))
    record["pool_mse"].append(float(pool_mse))
    record["test_mse"].append(float(test_mse))

    record["n_context"].append(int(len(active["x_ctx"])))

    return float(ctx_mape), float(pool_mape), float(test_mape)


def run_strategy_for_seed(
    strategy: str,
    initial_active: Dict[str, object],
    initial_state: Dict[str, torch.Tensor],
    seed: int,
    lr: float,
    epochs: int,
    acq_rounds: int,
    batch_size: int,
    acs_j: int,
    tag: str = "",
) -> Tuple[Dict[str, object], MultitaskGPModel]:
    if strategy not in ("ACS_FW", "BALD"):
        raise ValueError("Unknown strategy: %s" % strategy)

    active = clone_active_data(initial_active)
    record = _empty_strategy_record()

    x_ctx = active["x_ctx"]
    y_ctx = active["y_ctx"]
    assert torch.is_tensor(x_ctx)
    assert torch.is_tensor(y_ctx)

    model = restore_gp_from_state(x_ctx, y_ctx, initial_state)

    ctx_mape, pool_mape, test_mape = _append_metrics(record, model, active)
    print(
        "%s[seed %d] [%s] acq %02d/%02d | "
        "ctx MAPE %.6f | pool MAPE %.6f | TEST MAPE %.6f  (initial context)"
        % (tag, int(seed), strategy, 0, int(acq_rounds),
           ctx_mape, pool_mape, test_mape),
        flush=True,
    )

    for acq in range(1, int(acq_rounds) + 1):
        x_pool = active["x_pool"]
        assert torch.is_tensor(x_pool)

        if len(x_pool) == 0:
            break

        if strategy == "BALD":
            selected, score_mean = bald_select_batch_gp(
                model,
                x_pool,
                batch_size,
            )
        else:
            selected = acs_fw_select_batch_gp(
                model,
                x_pool,
                batch_size,
                J=acs_j,
            )
            score_mean = float("nan")

        if len(selected) == 0:
            break

        selected_x = x_pool[selected]
        div = batch_diversity(selected_x)

        active, selected_info = reveal_selected(active, selected)

        record["diversity"].append({
            "strategy": strategy,
            "batch_number": int(acq),
            **div,
        })
        record["selected_original_indices"].append(
            [
                int(v)
                for v in np.asarray(
                    selected_info["selected_original_indices"]
                ).tolist()
            ]
        )
        record["score_mean"].append(float(score_mean))

        del model
        clear_torch_cache()

        round_seed = 100000 + int(seed) * 100 + int(acq)
        model = train_gp(
            active["x_ctx"],
            active["y_ctx"],
            lr=lr,
            epochs=epochs,
            seed=round_seed,
            verbose=False,
        )

        ctx_mape, pool_mape, test_mape = _append_metrics(record, model, active)

        print(
            "%s[seed %d] [%s] acq %02d/%02d | "
            "ctx MAPE %.6f | pool MAPE %.6f | TEST MAPE %.6f"
            % (tag, int(seed), strategy, int(acq), int(acq_rounds),
               ctx_mape, pool_mape, test_mape),
            flush=True,
        )

    return record, model


def run_one_seed(
    scenario_id: int,
    scenario_path: str,
    seed: int,
    lr: float = DEFAULT_GP_LR,
    epochs: int = DEFAULT_GP_EPOCHS,
    acq_rounds: int = DEFAULT_ACQ_ROUNDS,
    batch: int = DEFAULT_BATCH_SIZE,
    val_frac: float = 0.5,
    acs_j: int = DEFAULT_ACS_J,
    expected_num_outputs: Optional[int] = EXPECTED_NUM_OUTPUTS,
    initial_context_size: int = INITIAL_CONTEXT_SIZE,
    device: Optional[torch.device] = None,
    tag: str = "",
) -> Dict[str, object]:
    """Run both acquisition strategies for one scenario and one seed."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    set_all_seeds(int(seed))

    X_cpu, Y_cpu, _scaler, metadata = load_scenario_csv(
        scenario_path,
        expected_num_outputs=expected_num_outputs,
    )

    split = make_seed_split(
        X_cpu,
        Y_cpu,
        seed=int(seed),
        test_frac=float(val_frac),
        initial_context_size=int(initial_context_size),
        device=device,
    )

    # One pristine initial GP; both strategies start from the same state.
    initial_model = train_gp(
        split["x_ctx"],
        split["y_ctx"],
        lr=lr,
        epochs=epochs,
        seed=500000 + int(seed),
        verbose=False,
    )
    initial_state = cpu_state_dict(initial_model)

    del initial_model
    clear_torch_cache()

    acs_record, acs_model = run_strategy_for_seed(
        "ACS_FW", split, initial_state, seed=seed, lr=lr, epochs=epochs,
        acq_rounds=acq_rounds, batch_size=batch, acs_j=acs_j, tag=tag,
    )

    bald_record, bald_model = run_strategy_for_seed(
        "BALD", split, initial_state, seed=seed, lr=lr, epochs=epochs,
        acq_rounds=acq_rounds, batch_size=batch, acs_j=acs_j, tag=tag,
    )

    # Nine fixed test examples for predicted-vs-true plots.
    rng = np.random.default_rng(int(seed) + 999)
    n_viz = min(9, len(split["x_test"]))
    viz_idx = np.sort(
        rng.choice(np.arange(len(split["x_test"])), size=n_viz, replace=False)
    )

    x_viz = split["x_test"][
        torch.as_tensor(viz_idx, dtype=torch.long, device=device)
    ]
    y_viz = split["y_test"][
        torch.as_tensor(viz_idx, dtype=torch.long, device=device)
    ]

    acs_mean, acs_var = predictive_mean_var(acs_model, x_viz)
    bald_mean, bald_var = predictive_mean_var(bald_model, x_viz)

    viz = {
        "indices": viz_idx.astype(int).tolist(),
        "ACS_FW": {
            "y_true": to_cpu_np(y_viz),
            "y_pred": acs_mean.numpy(),
            "y_var": acs_var.numpy(),
        },
        "BALD": {
            "y_true": to_cpu_np(y_viz),
            "y_pred": bald_mean.numpy(),
            "y_var": bald_var.numpy(),
        },
    }

    del acs_model, bald_model
    clear_torch_cache()

    return {
        "scenario_id": int(scenario_id),
        "scenario_file": os.path.basename(scenario_path),
        "scenario_key": _scenario_key(scenario_path),
        "seed": int(seed),
        "metadata": metadata,
        "ACS_FW": acs_record,
        "BALD": bald_record,
        "viz": viz,
        "runtime_sec": float(time.time() - t0),
    }


# =============================================================================
# CHECKPOINTING
# =============================================================================

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _partial_path(partial_dir: str, scenario_path: str, seed: int) -> str:
    return os.path.join(
        partial_dir,
        "%s_seed%d.pkl" % (_scenario_key(scenario_path), int(seed)),
    )


def _save_partial(partial_dir: str, scenario_path: str, result: Dict[str, object]) -> str:
    path = _partial_path(partial_dir, scenario_path, int(result["seed"]))
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)  # atomic: never leaves a half-written .pkl
    return path


def _load_partial(path: str) -> Optional[Dict[str, object]]:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _check_run_signature(partial_dir: str, signature: Dict[str, object], resume: bool) -> None:
    sig_path = os.path.join(partial_dir, "run_signature.json")
    if resume and os.path.isfile(sig_path):
        with open(sig_path, "r") as f:
            old = json.load(f)
        if old != signature:
            raise RuntimeError(
                "Existing checkpoints in %s were produced with different "
                "settings.\n  existing: %s\n  current : %s\n"
                "Use a new save_dir, or pass resume=False to overwrite."
                % (partial_dir, old, signature)
            )
    with open(sig_path, "w") as f:
        json.dump(signature, f, indent=2)


# =============================================================================
# AGGREGATION / SAVING
# =============================================================================

def _default_save_root(scenario_dir: str) -> str:
    return os.path.join(scenario_dir, "GP_base_results")


def _series_matrix(
    seed_results: Sequence[Dict[str, object]],
    strategy: str,
    key: str,
    L: int,
) -> np.ndarray:
    rows = []
    for result in seed_results:
        values = list(result[strategy][key])
        padded = values[:L] + [np.nan] * max(0, L - len(values))
        rows.append(padded[:L])
    return np.asarray(rows, dtype=float)


def _aggregate_vector(mat: np.ndarray) -> Dict[str, np.ndarray]:
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(mat, axis=0)
        median = np.nanmedian(mat, axis=0)
        std = np.nanstd(mat, axis=0, ddof=1)

    n = np.sum(np.isfinite(mat), axis=0)
    se = np.divide(
        std,
        np.sqrt(np.maximum(n, 1)),
        out=np.zeros_like(std),
        where=n > 0,
    )
    ci = 1.959963984540054 * se

    return {
        "mean": mean,
        "median": median,
        "std": std,
        "n": n,
        "ci_low": mean - ci,
        "ci_high": mean + ci,
    }


def _save_per_seed_results(
    sid: int,
    seed_results: Sequence[Dict[str, object]],
    out_dir: str,
) -> None:
    metric_rows: List[Dict[str, object]] = []
    selection_rows: List[Dict[str, object]] = []
    diversity_rows: List[Dict[str, object]] = []

    def _at(values: Sequence[float], i: int) -> float:
        return values[i] if i < len(values) else np.nan

    for result in seed_results:
        seed = int(result["seed"])

        for strategy in ("ACS_FW", "BALD"):
            rec = result[strategy]
            L = max(
                len(rec["context_mape"]),
                len(rec["test_mape"]),
            )

            for i in range(L):
                metric_rows.append({
                    "scenario": int(sid),
                    "seed": seed,
                    "strategy": strategy,
                    "acq_step": i,
                    "n_context": _at(rec["n_context"], i),
                    "context_mape": _at(rec["context_mape"], i),
                    "pool_mape": _at(rec["pool_mape"], i),
                    "test_mape": _at(rec["test_mape"], i),
                    "context_mse": _at(rec["context_mse"], i),
                    "pool_mse": _at(rec["pool_mse"], i),
                    "test_mse": _at(rec["test_mse"], i),
                })

            for i, selected in enumerate(rec["selected_original_indices"]):
                selection_rows.append({
                    "scenario": int(sid),
                    "seed": seed,
                    "strategy": strategy,
                    "acq_step": i + 1,
                    "selected_original_indices": ";".join(
                        str(int(v)) for v in selected
                    ),
                    "score_mean": _at(rec["score_mean"], i),
                })

            for div in rec["diversity"]:
                diversity_rows.append({
                    "scenario": int(sid),
                    "seed": seed,
                    **div,
                })

    pd.DataFrame(metric_rows).to_csv(
        os.path.join(out_dir, "per_seed_metrics.csv"), index=False,
    )
    pd.DataFrame(selection_rows).to_csv(
        os.path.join(out_dir, "selected_points.csv"), index=False,
    )
    pd.DataFrame(diversity_rows).to_csv(
        os.path.join(out_dir, "per_seed_diversity.csv"), index=False,
    )

    runtime_rows = [
        {"seed": int(r["seed"]), "runtime_sec": r.get("runtime_sec", np.nan)}
        for r in seed_results
    ]
    pd.DataFrame(runtime_rows).to_csv(
        os.path.join(out_dir, "per_seed_runtime.csv"), index=False,
    )


def _save_aggregate_metrics(
    sid: int,
    seed_results: Sequence[Dict[str, object]],
    acq_rounds: int,
    out_dir: str,
    initial_context_size: int = INITIAL_CONTEXT_SIZE,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    n_steps = int(acq_rounds) + 1

    for strategy in ("ACS_FW", "BALD"):
        for metric in METRIC_KEYS:
            mat = _series_matrix(seed_results, strategy, metric, n_steps)
            agg = _aggregate_vector(mat)

            for step in range(n_steps):
                rows.append({
                    "scenario": int(sid),
                    "strategy": strategy,
                    "metric": metric,
                    "acq_step": step,
                    "n_context": int(initial_context_size),
                    "mean": agg["mean"][step],
                    "median": agg["median"][step],
                    "std": agg["std"][step],
                    "n": int(agg["n"][step]),
                    "ci_low_95": agg["ci_low"][step],
                    "ci_high_95": agg["ci_high"][step],
                })

    df = pd.DataFrame(rows)

    if seed_results:
        for strategy in ("ACS_FW", "BALD"):
            n_ctx = list(seed_results[0][strategy].get("n_context", []))
            for step, n in enumerate(n_ctx):
                mask = (df["strategy"] == strategy) & (df["acq_step"] == step)
                df.loc[mask, "n_context"] = int(n)

    df.to_csv(os.path.join(out_dir, "aggregate_metrics.csv"), index=False)
    return df


def _save_diversity_aggregates(
    sid: int,
    seed_results: Sequence[Dict[str, object]],
    acq_rounds: int,
    out_dir: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for strategy in ("ACS_FW", "BALD"):
        for stat in (
            "avg_pairwise_distance",
            "min_pairwise_distance",
            "max_pairwise_distance",
        ):
            matrix_rows = []
            for result in seed_results:
                vals = [float(d[stat]) for d in result[strategy]["diversity"]]
                vals = vals[:acq_rounds] + [np.nan] * max(0, acq_rounds - len(vals))
                matrix_rows.append(vals[:acq_rounds])

            mat = np.asarray(matrix_rows, dtype=float)
            agg = _aggregate_vector(mat)

            for step in range(int(acq_rounds)):
                rows.append({
                    "scenario": int(sid),
                    "strategy": strategy,
                    "statistic": stat,
                    "acq_step": step + 1,
                    "mean": agg["mean"][step],
                    "median": agg["median"][step],
                    "std": agg["std"][step],
                    "n": int(agg["n"][step]),
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "aggregate_diversity.csv"), index=False)
    return df


def _plot_metric(
    aggregate_df: pd.DataFrame,
    metric: str,
    statistic: str,
    ylabel: str,
    title: str,
    save_path: str,
) -> None:
    plt.figure(figsize=(7.0, 4.5))

    for strategy in ("ACS_FW", "BALD"):
        d = aggregate_df[
            (aggregate_df["strategy"] == strategy)
            & (aggregate_df["metric"] == metric)
        ].sort_values("acq_step")

        if d.empty:
            continue

        plt.plot(
            d["acq_step"].to_numpy(),
            d[statistic].to_numpy(),
            marker="o",
            label=strategy,
        )

    plt.xlabel("Acquisition Step (0 = initial context)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(
        np.arange(
            int(aggregate_df["acq_step"].min()),
            int(aggregate_df["acq_step"].max()) + 1,
            dtype=int,
        )
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_diversity(diversity_df: pd.DataFrame, save_path: str) -> None:
    plt.figure(figsize=(7.0, 4.5))

    for strategy in ("ACS_FW", "BALD"):
        d = diversity_df[
            (diversity_df["strategy"] == strategy)
            & (diversity_df["statistic"] == "avg_pairwise_distance")
        ].sort_values("acq_step")

        if d.empty:
            continue

        plt.plot(
            d["acq_step"].to_numpy(),
            d["mean"].to_numpy(),
            marker="o",
            label=strategy,
        )

    plt.xlabel("Acquisition Step")
    plt.ylabel("Mean Batch Pairwise Distance")
    plt.title("Acquisition-Batch Diversity")
    if not diversity_df.empty:
        plt.xticks(
            np.arange(1, int(diversity_df["acq_step"].max()) + 1, dtype=int)
        )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def _save_viz(first_result: Dict[str, object], out_dir: str) -> None:
    viz = first_result.get("viz")
    if not isinstance(viz, dict):
        return

    rows: List[Dict[str, object]] = []

    for strategy in ("ACS_FW", "BALD"):
        block = viz[strategy]
        y_true = np.asarray(block["y_true"])
        y_pred = np.asarray(block["y_pred"])
        y_var = np.asarray(block["y_var"])

        for s in range(y_true.shape[0]):
            for t in range(y_true.shape[1]):
                rows.append({
                    "strategy": strategy,
                    "sample_slot": int(s),
                    "test_local_index": int(viz["indices"][s]),
                    "response_index": int(t),
                    "y_true": float(y_true[s, t]),
                    "y_pred": float(y_pred[s, t]),
                    "y_var": float(y_var[s, t]),
                })

        fig, axes = plt.subplots(3, 3, figsize=(10, 8))
        axes = axes.ravel()

        for s, ax in enumerate(axes):
            if s >= y_true.shape[0]:
                ax.axis("off")
                continue

            x = np.arange(y_true.shape[1])
            ax.plot(x, y_true[s], label="True")
            ax.plot(x, y_pred[s], label="Predicted")
            ax.set_xlabel("Response Index")
            ax.set_ylabel("Response")
            ax.grid(alpha=0.25)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=2)

        fig.suptitle("%s: True vs Predicted (first saved seed)" % strategy)
        fig.tight_layout(rect=[0, 0.04, 1, 0.96])
        fig.savefig(
            os.path.join(out_dir, "pred_vs_true_%s.png" % strategy),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "pred_vs_true_samples.csv"), index=False,
    )


def save_scenario_results(
    sid: int,
    scenario_path: str,
    seed_results: Sequence[Dict[str, object]],
    acq_rounds: int,
    save_root: str,
    initial_context_size: int = INITIAL_CONTEXT_SIZE,
) -> pd.DataFrame:
    scenario_name = Path(scenario_path).stem
    out_dir = _ensure_dir(os.path.join(save_root, scenario_name))

    _save_per_seed_results(sid, seed_results, out_dir)
    agg = _save_aggregate_metrics(
        sid, seed_results, acq_rounds, out_dir,
        initial_context_size=initial_context_size,
    )
    div = _save_diversity_aggregates(sid, seed_results, acq_rounds, out_dir)

    _plot_metric(agg, "context_mape", "mean", "Mean Context MAPE",
                 "%s: Mean Context MAPE" % scenario_name,
                 os.path.join(out_dir, "mean_context_mape.png"))
    _plot_metric(agg, "test_mape", "mean", "Mean Fixed-Test MAPE",
                 "%s: Mean Fixed-Test MAPE" % scenario_name,
                 os.path.join(out_dir, "mean_test_mape.png"))
    _plot_metric(agg, "test_mape", "median", "Median Fixed-Test MAPE",
                 "%s: Median Fixed-Test MAPE" % scenario_name,
                 os.path.join(out_dir, "median_test_mape.png"))
    _plot_metric(agg, "context_mse", "mean", "Mean Context MSE",
                 "%s: Mean Context MSE" % scenario_name,
                 os.path.join(out_dir, "mean_context_mse.png"))
    _plot_metric(agg, "test_mse", "mean", "Mean Fixed-Test MSE",
                 "%s: Mean Fixed-Test MSE" % scenario_name,
                 os.path.join(out_dir, "mean_test_mse.png"))
    _plot_diversity(div, os.path.join(out_dir, "mean_batch_diversity.png"))

    if seed_results:
        _save_viz(seed_results[0], out_dir)

    summary_rows: List[Dict[str, object]] = []
    final_step = int(acq_rounds)

    for strategy in ("ACS_FW", "BALD"):
        for metric in ("test_mape", "test_mse", "context_mape", "context_mse"):
            for step in (0, final_step):
                d = agg[
                    (agg["strategy"] == strategy)
                    & (agg["metric"] == metric)
                    & (agg["acq_step"] == step)
                ]

                if not d.empty:
                    summary_rows.append({
                        "scenario": int(sid),
                        "scenario_file": os.path.basename(scenario_path),
                        "strategy": strategy,
                        "metric": metric,
                        "acq_step": int(step),
                        "final_step": final_step,
                        "mean": float(d.iloc[0]["mean"]),
                        "median": float(d.iloc[0]["median"]),
                        "std": float(d.iloc[0]["std"]),
                        "n": int(d.iloc[0]["n"]),
                    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "final_summary.csv"), index=False)

    metadata = seed_results[0]["metadata"] if seed_results else {}
    pd.DataFrame([
        {
            "scenario_file": os.path.basename(scenario_path),
            "rows": metadata.get("n_rows", np.nan),
            "feature_count": metadata.get("n_features", np.nan),
            "response_count": metadata.get("n_tasks", np.nan),
            "skipped_column": metadata.get("skipped_col", ""),
            "feature_columns": ";".join(metadata.get("feature_cols", [])),
        }
    ]).to_csv(os.path.join(out_dir, "data_layout.csv"), index=False)

    print("Saved scenario results to: %s" % out_dir, flush=True)
    return summary_df


# =============================================================================
# PARALLEL WORKER
# =============================================================================

def _worker_proc(
    rank: int,
    tasks: Sequence[Tuple[int, int, str]],
    lr: float,
    epochs: int,
    acq_rounds: int,
    batch: int,
    val_frac: float,
    acs_j: int,
    expected_num_outputs: Optional[int],
    initial_context_size: int,
    partial_dir: str,
    result_queue,
) -> None:
    """
    tasks: list of (seed, scenario_id, scenario_path).
    Each finished task is checkpointed to disk by the worker itself, so it
    survives even if the main (notebook) process dies.
    """
    try:
        if torch.cuda.is_available():
            gpu = int(rank) % torch.cuda.device_count()
            torch.cuda.set_device(gpu)
            device = torch.device("cuda:%d" % gpu)
        else:
            device = torch.device("cpu")

        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        tag = "[w%d] " % int(rank)
        print(
            "%sdevice=%s | %d jobs: %s"
            % (tag, str(device), len(tasks),
               ["%s/seed%d" % (_scenario_key(p), s) for s, _, p in tasks]),
            flush=True,
        )

        for seed, sid, path in tasks:
            result = run_one_seed(
                scenario_id=sid,
                scenario_path=path,
                seed=int(seed),
                lr=lr,
                epochs=epochs,
                acq_rounds=acq_rounds,
                batch=batch,
                val_frac=val_frac,
                acs_j=acs_j,
                expected_num_outputs=expected_num_outputs,
                initial_context_size=initial_context_size,
                device=device,
                tag=tag,
            )
            _save_partial(partial_dir, path, result)
            result_queue.put(("__result__", int(rank), path, int(seed), result))
            clear_torch_cache()

    except Exception:
        result_queue.put(("__error__", int(rank), traceback.format_exc()))


# =============================================================================
# M3 / NOTEBOOK ENTRY POINT
# =============================================================================

def run_notebook_workflow(
    n_gpus: int = DEFAULT_N_WORKERS,
    n_seeds: Optional[int] = DEFAULT_NUM_SEEDS,
    scenario_dir: str = SCENARIO_CSV_DIR,
    lr: float = DEFAULT_GP_LR,
    epochs: int = DEFAULT_GP_EPOCHS,
    acq_rounds: int = DEFAULT_ACQ_ROUNDS,
    batch: int = DEFAULT_BATCH_SIZE,
    val_frac: float = 0.5,
    save_dir: Optional[str] = None,
    scenario_ids: Optional[Sequence[int]] = None,
    acs_j: int = DEFAULT_ACS_J,
    expected_num_outputs: Optional[int] = EXPECTED_NUM_OUTPUTS,
    initial_context_size: int = INITIAL_CONTEXT_SIZE,
    seed_start: int = DEFAULT_SEED_START,
    resume: bool = True,
    # Compatibility aliases:
    base_dir: Optional[str] = None,
    scenario_files: Optional[Sequence[str]] = None,
    batch_size: Optional[int] = None,
    train_epochs: Optional[int] = None,
    gpu_id: Optional[int] = None,
) -> pd.DataFrame:
    """
    M3-compatible entry point.

    n_gpus   : number of parallel workers. With GPUs, capped at the number of
               visible GPUs; without GPUs, that many CPU worker processes.
    n_seeds  : seeds seed_start .. seed_start + n_seeds - 1 (default 100..115).
    gpu_id   : if given, forces ONE sequential worker on that GPU.
    resume   : reuse finished (seed, scenario) checkpoints in save_dir.

    Aliases: base_dir -> scenario_dir, batch_size -> batch,
             train_epochs -> epochs.
    Unknown keyword arguments raise TypeError.
    """
    if base_dir is not None:
        scenario_dir = str(base_dir)
    if batch_size is not None:
        batch = int(batch_size)
    if train_epochs is not None:
        epochs = int(train_epochs)

    if int(acq_rounds) < 1:
        raise ValueError("acq_rounds must be >= 1.")
    if int(batch) < 1:
        raise ValueError("batch/batch_size must be >= 1.")
    if int(epochs) < 1:
        raise ValueError("epochs must be >= 1.")
    if float(lr) <= 0:
        raise ValueError("lr must be > 0.")
    if int(acs_j) < 1:
        raise ValueError("acs_j must be >= 1.")
    if int(initial_context_size) < 1:
        raise ValueError("initial_context_size must be >= 1.")

    base = Path(scenario_dir)

    if scenario_files is None:
        selected_files = list_scenario_csvs(str(base))
    else:
        selected_files = []
        for name in scenario_files:
            p = Path(name)
            if not p.is_absolute():
                p = base / p
            if not p.is_file():
                raise FileNotFoundError(str(p))
            selected_files.append((_scenario_id_from_name(p.name), str(p)))

    if scenario_ids is not None:
        keep = set(int(v) for v in scenario_ids)
        selected_files = [
            (sid, path) for sid, path in selected_files if int(sid) in keep
        ]

    if not selected_files:
        raise RuntimeError("No scenario files selected.")

    keys = [_scenario_key(p) for _, p in selected_files]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate scenario files in selection: %s" % keys)

    # Validate every file before launching long jobs.
    for sid, path in selected_files:
        header = pd.read_csv(path, nrows=0)
        n_outputs = len(header.columns[8:])
        if expected_num_outputs is not None and n_outputs != int(expected_num_outputs):
            raise ValueError(
                "%s: expected %d response columns from df.columns[8:], "
                "but found %d."
                % (os.path.basename(path), int(expected_num_outputs), int(n_outputs))
            )

    total_seeds = DEFAULT_NUM_SEEDS if n_seeds is None else int(n_seeds)
    if total_seeds < 1:
        raise ValueError("n_seeds must be >= 1.")

    full_seed_list = list(range(int(seed_start), int(seed_start) + total_seeds))

    available_gpus = torch.cuda.device_count()

    if gpu_id is not None:
        n_use = 1
        if torch.cuda.is_available():
            torch.cuda.set_device(int(gpu_id))
    elif available_gpus == 0:
        # CPU-only: n_gpus CPU worker processes.
        n_use = max(1, int(n_gpus))
    else:
        n_use = min(max(1, int(n_gpus)), int(available_gpus))

    if save_dir is None or str(save_dir).strip() == "":
        save_root = _default_save_root(str(base))
    else:
        save_root = str(save_dir)

    _ensure_dir(save_root)
    partial_dir = _ensure_dir(os.path.join(save_root, "_partial"))

    signature = {
        "lr": float(lr),
        "epochs": int(epochs),
        "acq_rounds": int(acq_rounds),
        "batch": int(batch),
        "val_frac": float(val_frac),
        "acs_j": int(acs_j),
        "expected_num_outputs": expected_num_outputs,
        "initial_context_size": int(initial_context_size),
        "feature_encoding": "onehot_first7_categorical_string_v1",
    }
    _check_run_signature(partial_dir, signature, resume=bool(resume))

    # ---------------------------------------------------------------- resume
    by_key: Dict[str, List[Dict[str, object]]] = {
        _scenario_key(p): [] for _, p in selected_files
    }
    todo: List[Tuple[int, int, str]] = []
    n_loaded = 0

    for seed in full_seed_list:
        for sid, path in selected_files:
            cp = _partial_path(partial_dir, path, seed)
            r = _load_partial(cp) if (resume and os.path.isfile(cp)) else None
            if r is not None:
                by_key[_scenario_key(path)].append(r)
                n_loaded += 1
            else:
                todo.append((int(seed), int(sid), str(path)))

    n_use = max(1, min(n_use, len(todo))) if todo else 1

    print("=" * 88)
    print("GP_base: exact multitask GP baseline")
    print("=" * 88)
    print("Scenario directory : %s" % str(base))
    print("Save root          : %s" % save_root)
    print("GPUs visible       : %d" % int(available_gpus))
    print("Workers used       : %d (%s)"
          % (int(n_use), "GPU" if available_gpus > 0 else "CPU"))
    print("Seeds              : %s" % full_seed_list)
    print("Jobs total         : %d (%d scenarios x %d seeds)"
          % (len(selected_files) * len(full_seed_list),
             len(selected_files), len(full_seed_list)))
    print("Jobs from ckpt     : %d" % n_loaded)
    print("Jobs to run        : %d" % len(todo))
    print("Acquisitions       : %d (metrics saved for steps 0..%d)"
          % (int(acq_rounds), int(acq_rounds)))
    print("Batch size         : %d" % int(batch))
    print("Initial context    : %d" % int(initial_context_size))
    print("GP lr / epochs     : %g / %d" % (float(lr), int(epochs)))
    print("ACS-FW J           : %d" % int(acs_j))
    print("Expected outputs   : %s" % str(expected_num_outputs))
    print("Scenarios:")
    for _, path in selected_files:
        print("  - %s" % os.path.basename(path))
    print("=" * 88, flush=True)

    # ------------------------------------------------------------------- run
    if todo and n_use == 1:
        if torch.cuda.is_available():
            chosen_gpu = int(gpu_id) if gpu_id is not None else 0
            torch.cuda.set_device(chosen_gpu)
            device = torch.device("cuda:%d" % chosen_gpu)
        else:
            device = torch.device("cpu")

        for j, (seed, sid, path) in enumerate(todo, start=1):
            result = run_one_seed(
                scenario_id=sid,
                scenario_path=path,
                seed=seed,
                lr=lr,
                epochs=epochs,
                acq_rounds=acq_rounds,
                batch=batch,
                val_frac=val_frac,
                acs_j=acs_j,
                expected_num_outputs=expected_num_outputs,
                initial_context_size=initial_context_size,
                device=device,
            )
            _save_partial(partial_dir, path, result)
            by_key[_scenario_key(path)].append(result)
            print(
                "Progress: %d/%d jobs (%s, seed %d, %.1f min)"
                % (j, len(todo), _scenario_key(path), seed,
                   result["runtime_sec"] / 60.0),
                flush=True,
            )
            clear_torch_cache()

    elif todo:
        # Round-robin over (seed, scenario) jobs for good load balance.
        worker_tasks: List[List[Tuple[int, int, str]]] = [[] for _ in range(n_use)]
        for i, task in enumerate(todo):
            worker_tasks[i % n_use].append(task)

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        procs = []

        for rank in range(n_use):
            p = ctx.Process(
                target=_worker_proc,
                args=(
                    rank,
                    worker_tasks[rank],
                    float(lr),
                    int(epochs),
                    int(acq_rounds),
                    int(batch),
                    float(val_frac),
                    int(acs_j),
                    expected_num_outputs,
                    int(initial_context_size),
                    partial_dir,
                    result_queue,
                ),
            )
            p.start()
            procs.append(p)

        expected_per_rank = [len(t) for t in worker_tasks]
        received_per_rank = [0] * n_use
        received = 0
        errors: List[Tuple[int, str]] = []

        try:
            while received < len(todo) and not errors:
                try:
                    item = result_queue.get(timeout=QUEUE_POLL_SECONDS)
                except _queue.Empty:
                    silent_dead = [
                        r for r, p in enumerate(procs)
                        if not p.is_alive()
                        and received_per_rank[r] < expected_per_rank[r]
                    ]
                    if not silent_dead:
                        continue
                    # Last chance to drain anything flushed just before exit.
                    try:
                        item = result_queue.get(timeout=10)
                    except _queue.Empty:
                        for r in silent_dead:
                            errors.append((
                                r,
                                "Worker %d exited (exitcode %s) after %d/%d jobs "
                                "without reporting an error (likely killed: OOM "
                                "or segfault)."
                                % (r, procs[r].exitcode,
                                   received_per_rank[r], expected_per_rank[r]),
                            ))
                        break

                if item[0] == "__error__":
                    errors.append((item[1], item[2]))
                    break

                _, rank, path, seed, result = item
                by_key[_scenario_key(path)].append(result)
                received_per_rank[int(rank)] += 1
                received += 1

                print(
                    "M3 progress: %d/%d jobs (w%d: %s, seed %d, %.1f min)"
                    % (received, len(todo), int(rank), _scenario_key(path),
                       int(seed), result["runtime_sec"] / 60.0),
                    flush=True,
                )
        finally:
            if errors:
                for p in procs:
                    if p.is_alive():
                        p.terminate()
            for p in procs:
                p.join()

        if errors:
            for rank, err in errors:
                print("\nWorker %s failed:\n%s" % (str(rank), str(err)), flush=True)
            raise RuntimeError(
                "At least one worker failed. Finished jobs are checkpointed in "
                "%s; rerun the same call to resume." % partial_dir
            )

    # ------------------------------------------------------------------ save
    all_summary_frames: List[pd.DataFrame] = []

    for sid, path in selected_files:
        results = sorted(by_key[_scenario_key(path)], key=lambda r: int(r["seed"]))

        if len(results) != len(full_seed_list):
            raise RuntimeError(
                "Scenario %s has %d/%d seeds."
                % (os.path.basename(path), len(results), len(full_seed_list))
            )

        summary_df = save_scenario_results(
            sid=int(sid),
            scenario_path=path,
            seed_results=results,
            acq_rounds=int(acq_rounds),
            save_root=save_root,
            initial_context_size=int(initial_context_size),
        )
        all_summary_frames.append(summary_df)

    combined = (
        pd.concat(all_summary_frames, ignore_index=True)
        if all_summary_frames
        else pd.DataFrame()
    )

    combined.to_csv(
        os.path.join(save_root, "all_scenarios_final_summary.csv"), index=False,
    )

    config = pd.DataFrame([
        {"parameter": "module", "value": "GP_base"},
        {"parameter": "model", "value": "Exact multitask GP"},
        {"parameter": "kernel", "value": "Scale(RBF-ARD) x MultitaskKernel(rank=1)"},
        {"parameter": "has_global_noise", "value": False},
        {"parameter": "has_task_noise", "value": True},
        {"parameter": "scenario_dir", "value": str(base)},
        {"parameter": "scenario_files", "value": ";".join(os.path.basename(p) for _, p in selected_files)},
        {"parameter": "expected_num_outputs", "value": expected_num_outputs},
        {"parameter": "n_seeds", "value": total_seeds},
        {"parameter": "seeds", "value": str(full_seed_list)},
        {"parameter": "n_workers", "value": int(n_use)},
        {"parameter": "worker_device", "value": "GPU" if available_gpus > 0 else "CPU"},
        {"parameter": "initial_context_size", "value": int(initial_context_size)},
        {"parameter": "fixed_test_fraction", "value": float(val_frac)},
        {"parameter": "acq_rounds", "value": int(acq_rounds)},
        {"parameter": "metric_steps_saved", "value": "0..%d (0 = initial context)" % int(acq_rounds)},
        {"parameter": "batch_size", "value": int(batch)},
        {"parameter": "acs_j", "value": int(acs_j)},
        {"parameter": "gp_lr", "value": float(lr)},
        {"parameter": "gp_epochs", "value": int(epochs)},
    ])
    config.to_csv(os.path.join(save_root, "run_config.csv"), index=False)

    print("\n" + "=" * 88)
    print("GP_base COMPLETE")
    print("Results saved under:")
    print(save_root)
    print("=" * 88, flush=True)

    return combined


def main() -> None:
    run_notebook_workflow()


if __name__ == "__main__":
    main()