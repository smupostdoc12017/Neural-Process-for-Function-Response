# -*- coding: utf-8 -*-
"""
IndGP.py

Standalone implementation of the Independent Gaussian Process (IndGP) baseline
described by Chung & Al Kontar: one independent scalar GP per response-curve
coordinate, with no cross-output covariance/shared latent GP.

This is an implementation of the IndGP baseline idea, not a copy of unreleased
benchmark-driver code from the authors.

Data preparation follows the supplied NP workflow exactly:
  - CSV directory: ./paper scenarios
  - first 7 columns: categorical features
  - OneHotEncoder(handle_unknown="ignore")
  - all remaining columns: response curve
  - default validation fraction: 0.5
  - initial context: min(64, len(train_idx)//2)
  - acquisition pool: remaining training observations
  - BALD and ACS_FW start from the same split for each seed

The module is self-contained with respect to the FedMGP repository. It requires
only standard scientific Python packages plus torch/gpytorch.
"""

import os, re, math, time, traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import gpytorch
import torch.multiprocessing as mp
from sklearn.preprocessing import OneHotEncoder

SCENARIO_CSV_DIR = "./paper scenarios"
MODEL_NAME = "IndGP"
STRATEGIES = ("ACS_FW", "BALD")


def list_scenario_csvs(folder):
    out = []
    pat = re.compile(
        r"^scenario_(\d+)_(?:damped|sigmoid|HGR|d_sigmoid|double(?:_sigmoid)?|sinusoid|burnup|gompertz)\.csv$",
        re.IGNORECASE,
    )
    if not os.path.isdir(folder):
        raise RuntimeError(f"Scenario directory not found: {folder}")
    for fn in os.listdir(folder):
        m = pat.match(fn)
        if m:
            out.append((int(m.group(1)), os.path.join(folder, fn)))
    return sorted(out, key=lambda z: (z[0], z[1]))


def load_scenario_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.shape[1] < 8:
        raise ValueError(
            f"Expected at least 8 columns (7 categorical features + responses), "
            f"got {df.shape[1]}"
        )
    feat_cols = df.columns[:7]
    resp_cols = df.columns[7:]

    try:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")

    X = ohe.fit_transform(df[feat_cols].astype(str)).astype(np.float32)
    Y = df[resp_cols].to_numpy(dtype=np.float32)

    if not np.isfinite(Y).all():
        raise ValueError(f"Non-finite response values found in {path}")

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
        ohe,
    )


class ScalarExactGP(gpytorch.models.ExactGP):
    """One scalar-output GP. Each response coordinate receives its own instance."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def _fit_one_gp(x, y, epochs=300, lr=0.03):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ScalarExactGP(x, y, likelihood)
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = -mll(output, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite IndGP training loss.")
        loss.backward()
        optimizer.step()

    return model, likelihood


def fit_indgp(x_ctx, y_ctx, train_epochs=300, lr=0.03):
    """
    Fit T mutually independent GPs for Y in R^{N x T}.
    No parameter or covariance is shared across output coordinates.
    """
    models = []
    for t in range(y_ctx.shape[1]):
        model, likelihood = _fit_one_gp(
            x_ctx, y_ctx[:, t], epochs=train_epochs, lr=lr
        )
        models.append((model, likelihood))
    return models


@torch.no_grad()
def predict_indgp(models, x):
    means, latent_vars, obs_vars, noise_vars = [], [], [], []

    for model, likelihood in models:
        model.eval()
        likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            latent = model(x)
            observed = likelihood(latent)

        means.append(observed.mean)
        latent_vars.append(latent.variance.clamp_min(1e-12))
        obs_vars.append(observed.variance.clamp_min(1e-12))
        noise = likelihood.noise.detach().clamp_min(1e-12)
        noise_vars.append(torch.ones_like(observed.mean) * noise)

    return (
        torch.stack(means, dim=1),
        torch.stack(latent_vars, dim=1),
        torch.stack(obs_vars, dim=1),
        torch.stack(noise_vars, dim=1),
    )


def calculate_mape(y_true, y_pred, eps=1e-8):
    denom = torch.clamp(torch.abs(y_true), min=eps)
    return torch.mean(torch.abs(y_pred - y_true) / denom).item()


@torch.no_grad()
def bald_scores(models, x_pool):
    """
    Gaussian information-gain score summed over independent outputs:
        1/2 sum_t log(1 + Var[f_t(x)|D] / sigma_t^2).
    """
    _, latent_var, _, noise_var = predict_indgp(models, x_pool)
    return 0.5 * torch.log1p(latent_var / noise_var).sum(dim=1)


def bald_select_batch(models, x_pool, k):
    scores = bald_scores(models, x_pool)
    return torch.topk(scores, min(int(k), len(x_pool))).indices


@torch.no_grad()
def acs_fw_select_batch(models, x_pool, k):
    """
    GP-compatible Frank-Wolfe coreset acquisition.

    Each candidate is represented by its independent-GP predictive mean and
    log observation variance across response coordinates. The standard
    Frank-Wolfe coreset construction then approximates the full pool embedding.
    """
    mu, _, obs_var, _ = predict_indgp(models, x_pool)

    scale = mu.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    V = torch.cat([mu / scale, torch.log(obs_var.clamp_min(1e-12))], dim=1)
    V = V / math.sqrt(max(1, V.shape[1]))

    sigma = torch.linalg.vector_norm(V, dim=1)
    s_tot = sigma.sum()
    weights = torch.zeros(len(x_pool), dtype=V.dtype, device=V.device)
    vL = V.sum(dim=0)
    vLw = torch.zeros_like(vL)
    selected = []

    for _ in range(min(int(k), len(x_pool))):
        residual = vL - vLw
        inner = torch.full(
            (len(x_pool),), -torch.inf, dtype=V.dtype, device=V.device
        )
        good = sigma > 1e-12
        inner[good] = V[good].matmul(residual) / sigma[good]

        if selected:
            inner[torch.tensor(selected, dtype=torch.long, device=V.device)] = -torch.inf

        f = int(torch.argmax(inner).item())
        if not torch.isfinite(inner[f]):
            break

        selected.append(f)
        u = (s_tot / sigma[f]) * V[f]
        d = u - vLw
        denom = torch.dot(d, d)

        if denom > 0:
            gamma = torch.clamp(torch.dot(residual, d) / denom, 0.0, 1.0)
        else:
            gamma = torch.tensor(0.0, dtype=V.dtype, device=V.device)

        weights *= (1.0 - gamma)
        weights[f] += gamma * (s_tot / sigma[f])
        vLw = (1.0 - gamma) * vLw + gamma * u

    need = min(int(k), len(x_pool)) - len(selected)
    if need > 0:
        order = torch.argsort(sigma, descending=True).tolist()
        selected.extend([i for i in order if i not in selected][:need])

    return torch.tensor(selected, dtype=torch.long, device=x_pool.device)


def _select_batch(models, x_pool, strategy, batch):
    if strategy == "BALD":
        return bald_select_batch(models, x_pool, batch)
    if strategy == "ACS_FW":
        return acs_fw_select_batch(models, x_pool, batch)
    raise ValueError(f"Unknown strategy: {strategy}")


def _evaluate(models, x_ctx, y_ctx, x_pool, y_pool, x_val, y_val):
    pred_ctx, _, _, _ = predict_indgp(models, x_ctx)
    pred_val, _, _, _ = predict_indgp(models, x_val)

    if len(x_pool):
        pred_pool, _, _, _ = predict_indgp(models, x_pool)
        pool_mape = calculate_mape(y_pool, pred_pool)
    else:
        pool_mape = np.nan

    return {
        "context_mape": calculate_mape(y_ctx, pred_ctx),
        "pool_mape": pool_mape,
        "validation_mape": calculate_mape(y_val, pred_val),
    }


def _run_strategy(
    X, Y, init_ctx, init_pool, val_idx, strategy, seed,
    acq_rounds, batch, train_epochs, lr,
):
    x_ctx, y_ctx = X[init_ctx].clone(), Y[init_ctx].clone()
    x_pool, y_pool = X[init_pool].clone(), Y[init_pool].clone()
    x_val, y_val = X[val_idx], Y[val_idx]

    torch.manual_seed(seed)
    np.random.seed(seed)

    models = fit_indgp(x_ctx, y_ctx, train_epochs=train_epochs, lr=lr)

    rows = []
    metric = _evaluate(models, x_ctx, y_ctx, x_pool, y_pool, x_val, y_val)
    rows.append({
        "acq_step": 0,
        "context_size": len(x_ctx),
        "pool_size": len(x_pool),
        **metric,
    })

    for r in range(1, int(acq_rounds) + 1):
        if len(x_pool) < batch:
            break

        idx = _select_batch(models, x_pool, strategy, batch)
        x_sel, y_sel = x_pool[idx], y_pool[idx]

        x_ctx = torch.cat([x_ctx, x_sel], dim=0)
        y_ctx = torch.cat([y_ctx, y_sel], dim=0)

        keep = torch.ones(len(x_pool), dtype=torch.bool)
        keep[idx.cpu()] = False
        x_pool, y_pool = x_pool[keep], y_pool[keep]

        # ExactGP must be rebuilt because the training set has changed.
        torch.manual_seed(seed + 1000 * r)
        np.random.seed(seed + 1000 * r)
        models = fit_indgp(
            x_ctx, y_ctx, train_epochs=train_epochs, lr=lr
        )

        metric = _evaluate(
            models, x_ctx, y_ctx, x_pool, y_pool, x_val, y_val
        )
        rows.append({
            "acq_step": r,
            "context_size": len(x_ctx),
            "pool_size": len(x_pool),
            **metric,
        })

        print(
            f"[Seed {seed}] [{strategy:6s}] Acq {r:02d} | "
            f"Context MAPE: {metric['context_mape']:.4f} | "
            f"Pool MAPE: {metric['pool_mape']:.4f} | "
            f"Validation MAPE: {metric['validation_mape']:.4f}",
            flush=True,
        )

    return rows


def run_single_experiment(
    seed,
    path,
    acq_rounds=20,
    batch=5,
    val_frac=0.5,
    train_epochs=300,
    lr=0.03,
):
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    torch.manual_seed(seed)
    np.random.seed(seed)

    X, Y, _ = load_scenario_csv(path)

    N = len(X)
    val_n = int(val_frac * N)
    val_idx = np.random.choice(N, val_n, replace=False)
    train_idx = np.setdiff1d(np.arange(N), val_idx)

    init_ctx_size = min(64, len(train_idx) // 2)
    init_ctx = np.random.choice(train_idx, init_ctx_size, replace=False)
    init_pool = np.setdiff1d(train_idx, init_ctx)

    result = {}
    for strategy in STRATEGIES:
        result[strategy] = _run_strategy(
            X, Y, init_ctx, init_pool, val_idx, strategy, seed,
            acq_rounds, batch, train_epochs, lr,
        )
    return result


def _worker(args):
    sid, path, seed, kwargs = args
    try:
        return (
            sid,
            path,
            seed,
            run_single_experiment(seed, path, **kwargs),
            None,
        )
    except Exception:
        return sid, path, seed, None, traceback.format_exc()


def _save_scenario(sid, seed_results, save_root):
    outdir = Path(save_root) / f"scenario_{sid:03d}"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed, result in sorted(seed_results, key=lambda z: z[0]):
        for strategy in STRATEGIES:
            for rec in result[strategy]:
                rows.append({
                    "scenario": sid,
                    "seed": seed,
                    "strategy": strategy,
                    **rec,
                })

    per = pd.DataFrame(rows)
    per.to_csv(
        outdir / f"scenario_{sid:03d}_per_seed_mape.csv",
        index=False,
    )

    agg = []
    for (strategy, step), g in per.groupby(["strategy", "acq_step"], sort=True):
        vals = g["validation_mape"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        n = len(vals)

        if n:
            mean = float(np.mean(vals))
            median = float(np.median(vals))
            sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            se = sd / math.sqrt(n)
            lo = mean - 1.959963984540054 * se
            hi = mean + 1.959963984540054 * se
        else:
            mean = median = lo = hi = np.nan

        agg.append({
            "model": MODEL_NAME,
            "strategy": strategy,
            "round": int(step),
            "context_size": int(g["context_size"].iloc[0]),
            "pool_size": int(g["pool_size"].iloc[0]),
            "mean_test_mape": mean,
            "median_test_mape": median,
            "ci_lower": lo,
            "ci_upper": hi,
            "n_seeds": n,
        })

    pd.DataFrame(agg).to_csv(
        outdir / f"scenario_{sid:03d}_trajectory.csv",
        index=False,
    )

    return str(outdir)


def run_notebook_workflow(
    n_cpu_workers=8,
    n_seeds=16,
    scenario_dir=SCENARIO_CSV_DIR,
    save_dir=None,
    scenario_ids=None,
    acq_rounds=20,
    batch=5,
    val_frac=0.5,
    train_epochs=300,
    lr=0.03,
):
    """
    M3 CPU runner.

    Defaults follow the supplied NP workflow's data split:
      seeds 100..115
      50% validation
      initial context = min(64, len(train_idx)//2)
      pool = remaining train observations

    Both BALD and ACS_FW use exactly the same split for a given seed.
    """
    if n_seeds != 16:
        raise ValueError(
            "This runner is configured for 16 seeds: 100 through 115."
        )

    scenario_files = list_scenario_csvs(scenario_dir)

    if scenario_ids:
        wanted = set(scenario_ids)
        scenario_files = [
            item for item in scenario_files if item[0] in wanted
        ]

    if not scenario_files:
        raise RuntimeError(
            f"No matching scenario files found in {scenario_dir}"
        )

    if save_dir is None:
        scratch = os.environ.get(
            "SCRATCH", f"/scratch/{os.environ.get('USER', 'user')}"
        )
        save_dir = str(
            Path(scratch)
            / "IndGP_runs"
            / time.strftime("%Y%m%d_%H%M%S")
        )

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    seeds = list(range(100, 116))
    kwargs = {
        "acq_rounds": acq_rounds,
        "batch": batch,
        "val_frac": val_frac,
        "train_epochs": train_epochs,
        "lr": lr,
    }

    jobs = [
        (sid, path, seed, kwargs)
        for sid, path in scenario_files
        for seed in seeds
    ]

    by_scenario = {sid: [] for sid, _ in scenario_files}

    print(f"Model: {MODEL_NAME}")
    print(f"CPU workers: {n_cpu_workers}")
    print(f"Seeds: {seeds}")
    print(f"Scenarios: {[p for _, p in scenario_files]}")
    print(f"Save root: {save_dir}")
    print(
        "Data convention: first 7 columns = categorical features; "
        "remaining columns = response curve."
    )

    ctx = mp.get_context("spawn")

    with ctx.Pool(processes=int(n_cpu_workers)) as pool:
        for sid, path, seed, result, error in pool.imap_unordered(
            _worker, jobs
        ):
            if error is not None:
                raise RuntimeError(
                    f"IndGP failed: scenario {sid}, seed {seed}\n{error}"
                )

            by_scenario[sid].append((seed, result))
            print(
                f"[IndGP] scenario {sid:03d} | seed {seed} complete",
                flush=True,
            )

    for sid, _ in scenario_files:
        location = _save_scenario(
            sid, by_scenario[sid], save_dir
        )
        print(f"Saved: {location}")

    print(f"IndGP complete. Root: {save_dir}")
    return save_dir
