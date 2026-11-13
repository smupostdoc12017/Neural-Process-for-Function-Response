from __future__ import annotations
import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial import cKDTree
try:
    from twinning import twin as _twin
    _HAS_TWINNING = True
except Exception:
    _twin = None
    _HAS_TWINNING = False
BASE_DIR = Path('./paper scenarios')
SCENARIO_FILES = ['scenario_000_HGR.csv', 'scenario_016_damped.csv', 'scenario_054_gompertz.csv', 'scenario_179_sigmoid.csv']
RESULTS_ROOT = BASE_DIR / 'GP_TwinGP_results'
EXPECTED_NUM_OUTPUTS = None
N_FEATURES = 7
DEFAULT_NUM_SEEDS = 8
DEFAULT_CPU_WORKERS = 8
SEEDS = tuple(range(100, 100 + DEFAULT_NUM_SEEDS))
CPU_WORKERS = DEFAULT_CPU_WORKERS
TEST_FRACTION = 0.2
INITIAL_CONTEXT_SIZE = 64
NUM_ACQUISITIONS = 20
ACQUISITION_BATCH_SIZE = 5
ACS_J = 50
LOCAL_POINTS = 25
GLOBAL_TWIN_TRIES = 5
LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
LAMBDA_VALIDATION_POINTS = 8
GLOBAL_ALPHA = 2.0
HYPEROPT_MAXITER = 20
HYPEROPT_PCA_COMPONENTS = 6
GLOBAL_SELECTION_PCA_COMPONENTS = 3
MIN_NUGGET = 1e-06
MAX_NUGGET = 0.2
JITTER = 1e-09
MAPE_EPS = 1e-08
STRATEGIES = ('ACS_FW', 'BALD')
CURVE_PLOT_SEED = 100
N_CURVE_PLOTS = 9

def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))

def _safe_std(a: np.ndarray, axis=0, keepdims=False) -> np.ndarray:
    s = np.std(a, axis=axis, keepdims=keepdims)
    return np.where(s < 1e-12, 1.0, s)

def _stable_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), MAPE_EPS)
    return float(np.mean(np.abs(y_true - y_pred) / denom))

def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

def _minmax_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.min(x, axis=0)
    hi = np.max(x, axis=0)
    span = hi - lo
    span = np.where(span < 1e-12, 1.0, span)
    return (lo, span)

def _minmax_apply(x: np.ndarray, lo: np.ndarray, span: np.ndarray) -> np.ndarray:
    return (x - lo) / span

def _pca_scores(y: np.ndarray, n_components: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2 or len(y) == 0:
        return np.empty((len(y), 0), dtype=np.float64)
    yz = (y - np.mean(y, axis=0, keepdims=True)) / _safe_std(y, axis=0, keepdims=True)
    k = int(min(n_components, yz.shape[0] - 1, yz.shape[1]))
    if k <= 0:
        return np.empty((len(y), 0), dtype=np.float64)
    try:
        u, s, _ = np.linalg.svd(yz, full_matrices=False)
        scores = u[:, :k] * s[:k]
    except np.linalg.LinAlgError:
        return np.empty((len(y), 0), dtype=np.float64)
    lo = scores.min(axis=0, keepdims=True)
    span = scores.max(axis=0, keepdims=True) - lo
    span = np.where(span < 1e-12, 1.0, span)
    return (scores - lo) / span

def _maximin_indices(z: np.ndarray, k: int, seed: int) -> np.ndarray:
    n = len(z)
    k = int(min(max(1, k), n))
    if k == n:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(seed)
    centroid = np.mean(z, axis=0)
    d0 = np.sum((z - centroid) ** 2, axis=1)
    first = int(np.argmin(d0 + rng.uniform(0.0, 1e-12, size=n)))
    selected = [first]
    min_d2 = np.sum((z - z[first]) ** 2, axis=1)
    min_d2[first] = -np.inf
    while len(selected) < k:
        nxt = int(np.argmax(min_d2))
        selected.append(nxt)
        d2 = np.sum((z - z[nxt]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
        min_d2[np.asarray(selected, dtype=int)] = -np.inf
    return np.asarray(selected, dtype=int)

def _representative_global_indices(x: np.ndarray, y: np.ndarray, g_num: int, seed: int, twins: int=GLOBAL_TWIN_TRIES) -> np.ndarray:
    n = len(x)
    g_num = int(min(max(2, g_num), n))
    if g_num >= n:
        return np.arange(n, dtype=int)
    scores = _pca_scores(y, GLOBAL_SELECTION_PCA_COMPONENTS)
    z = np.hstack([x, scores]) if scores.shape[1] else x.copy()
    best = None
    best_energy_proxy = np.inf
    if _HAS_TWINNING:
        r = max(2, int(round(n / float(g_num))))
        for attempt in range(max(1, int(twins))):
            try:
                np.random.seed(seed + attempt)
                idx = np.asarray(_twin(z, r=r), dtype=int).ravel()
                idx = np.unique(idx[(idx >= 0) & (idx < n)])
                if len(idx) == 0:
                    continue
                if len(idx) > g_num:
                    idx = idx[_maximin_indices(z[idx], g_num, seed + attempt)]
                elif len(idx) < g_num:
                    remaining = np.setdiff1d(np.arange(n), idx, assume_unique=False)
                    need = g_num - len(idx)
                    if len(idx):
                        d2 = np.min(np.sum((z[remaining, None, :] - z[idx][None, :, :]) ** 2, axis=2), axis=1)
                        fill = remaining[np.argsort(d2)[-need:]]
                    else:
                        fill = remaining[_maximin_indices(z[remaining], need, seed)]
                    idx = np.unique(np.concatenate([idx, fill]))[:g_num]
                d2_all = np.min(np.sum((z[:, None, :] - z[idx][None, :, :]) ** 2, axis=2), axis=1)
                proxy = float(np.mean(np.sqrt(np.maximum(d2_all, 0.0))))
                if proxy < best_energy_proxy:
                    best_energy_proxy = proxy
                    best = idx
            except Exception:
                break
    if best is None or len(best) != g_num:
        best = _maximin_indices(z, g_num, seed)
    return np.asarray(best, dtype=int)

def _global_kernel(xa: np.ndarray, xb: np.ndarray, ell: np.ndarray, alpha: float=GLOBAL_ALPHA) -> np.ndarray:
    xa = np.asarray(xa, dtype=np.float64)
    xb = np.asarray(xb, dtype=np.float64)
    ell = np.maximum(np.asarray(ell, dtype=np.float64), 1e-06)
    diff = np.abs(xa[:, None, :] - xb[None, :, :]) / ell[None, None, :]
    return np.exp(-np.sum(diff ** float(alpha), axis=2))

def _wendland_kernel(xa: np.ndarray, xb: np.ndarray, theta_l: float) -> np.ndarray:
    xa = np.asarray(xa, dtype=np.float64)
    xb = np.asarray(xb, dtype=np.float64)
    d = xa.shape[1]
    q = math.floor(d / 2.0) + 3
    theta_l = max(float(theta_l), 1e-08)
    dist = np.sqrt(np.maximum(np.sum((xa[:, None, :] - xb[None, :, :]) ** 2, axis=2), 0.0))
    u = dist / theta_l
    one_minus = np.maximum(0.0, 1.0 - u)
    return (q * u + 1.0) * one_minus ** q

def _covering_radius(x_domain: np.ndarray, x_global: np.ndarray) -> float:
    tree = cKDTree(x_global)
    dist, _ = tree.query(x_domain, k=1)
    return max(float(np.max(dist)), 1e-06)

def _initial_lengthscales(xg: np.ndarray) -> np.ndarray:
    d = xg.shape[1]
    ell = np.empty(d, dtype=np.float64)
    for j in range(d):
        vals = np.abs(xg[:, None, j] - xg[None, :, j])
        vals = vals[np.triu_indices_from(vals, k=1)]
        vals = vals[vals > 1e-10]
        ell[j] = np.median(vals) if len(vals) else 0.5
    return np.clip(ell, 0.03, 2.0)

def _fit_global_hyperparameters(xg: np.ndarray, yg: np.ndarray, initial: Optional[Tuple[np.ndarray, float]]=None, maxiter: int=HYPEROPT_MAXITER) -> Tuple[np.ndarray, float]:
    xg = np.asarray(xg, dtype=np.float64)
    yg = np.asarray(yg, dtype=np.float64)
    d = xg.shape[1]
    if initial is None:
        ell0 = _initial_lengthscales(xg)
        eta0 = 0.001
    else:
        ell0 = np.clip(np.asarray(initial[0], dtype=float), 0.03, 3.0)
        eta0 = float(np.clip(initial[1], MIN_NUGGET, MAX_NUGGET))
    z = _pca_scores(yg, HYPEROPT_PCA_COMPONENTS)
    if z.shape[1] == 0:
        z = ((yg - yg.mean(axis=0, keepdims=True)) / _safe_std(yg, axis=0, keepdims=True))[:, :1]
    z = z - np.mean(z, axis=0, keepdims=True)
    n, q = z.shape
    lower = np.log(np.r_[np.full(d, 0.02), MIN_NUGGET])
    upper = np.log(np.r_[np.full(d, 3.0), MAX_NUGGET])
    p0 = np.log(np.r_[ell0, eta0])
    p0 = np.minimum(np.maximum(p0, lower), upper)

    def objective(p: np.ndarray) -> float:
        ell = np.exp(p[:d])
        eta = float(np.exp(p[d]))
        k = _global_kernel(xg, xg, ell)
        k.flat[::n + 1] += eta + JITTER
        try:
            cf = cho_factor(k, lower=True, check_finite=False)
            alpha = cho_solve(cf, z, check_finite=False)
            quad = float(np.sum(z * alpha))
            logdet = 2.0 * float(np.sum(np.log(np.diag(cf[0]))))
            return 0.5 * (quad + q * logdet + n * q * math.log(2.0 * math.pi))
        except Exception:
            return 1e+100
    try:
        res = minimize(objective, p0, method='L-BFGS-B', bounds=list(zip(lower, upper)), options={'maxiter': int(maxiter), 'ftol': 1e-07})
        p = res.x if np.all(np.isfinite(res.x)) else p0
    except Exception:
        p = p0
    ell = np.exp(p[:d])
    eta = float(np.exp(p[d]))
    return (np.clip(ell, 0.02, 3.0), float(np.clip(eta, MIN_NUGGET, MAX_NUGGET)))

@dataclass
class _PreparedGlobalBlock:
    lam: float
    Ainv: np.ndarray
    Ainv_Yg: np.ndarray
    Ainv_ones: np.ndarray

class TwinGPVectorSurrogate:

    def __init__(self, local_points: int=LOCAL_POINTS, lambda_grid: Sequence[float]=LAMBDA_GRID, lambda_validation_points: int=LAMBDA_VALIDATION_POINTS, seed: int=0, hyperopt_maxiter: int=HYPEROPT_MAXITER):
        self.local_points = int(local_points)
        self.lambda_grid = tuple((float(v) for v in lambda_grid))
        self.lambda_validation_points = int(lambda_validation_points)
        self.seed = int(seed)
        self.hyperopt_maxiter = int(hyperopt_maxiter)
        self.x_ctx = None
        self.y_ctx = None
        self.x_domain = None
        self.global_idx = None
        self.local_candidate_idx = None
        self.xg = None
        self.yg = None
        self.ell = None
        self.eta = None
        self.theta_l = None
        self.lam = None
        self._tree = None
        self._prepared = None
        self.global_tau2 = None

    def fit(self, x_ctx: np.ndarray, y_ctx: np.ndarray, x_domain: np.ndarray, initial_hyperparams: Optional[Tuple[np.ndarray, float]]=None) -> 'TwinGPVectorSurrogate':
        self.x_ctx = np.asarray(x_ctx, dtype=np.float64)
        self.y_ctx = np.asarray(y_ctx, dtype=np.float64)
        self.x_domain = np.asarray(x_domain, dtype=np.float64)
        n, d = self.x_ctx.shape
        if n < 4:
            raise ValueError('TwinGPVectorSurrogate requires at least 4 context points.')
        paper_g = int(round(min(50 * d, max(math.sqrt(len(self.x_domain)), 10 * d))))
        reserve_local = min(self.local_points, max(2, n // 3))
        g_num = min(paper_g, max(2, n - reserve_local))
        self.global_idx = _representative_global_indices(self.x_ctx, self.y_ctx, g_num=g_num, seed=self.seed)
        self.xg = self.x_ctx[self.global_idx]
        self.yg = self.y_ctx[self.global_idx]
        mask = np.ones(n, dtype=bool)
        mask[self.global_idx] = False
        self.local_candidate_idx = np.where(mask)[0]
        if len(self.local_candidate_idx):
            self._tree = cKDTree(self.x_ctx[self.local_candidate_idx])
        else:
            self._tree = None
        self.ell, self.eta = _fit_global_hyperparameters(self.xg, self.yg, initial=initial_hyperparams, maxiter=self.hyperopt_maxiter)
        self.theta_l = _covering_radius(self.x_domain, self.xg)
        self.lam = self._tune_lambda()
        self._prepared = self._prepare_global_block(self.lam)
        self.global_tau2 = self._estimate_global_tau2()
        return self

    @property
    def hyperparams(self) -> Tuple[np.ndarray, float]:
        return (self.ell.copy(), float(self.eta))

    def _prepare_global_block(self, lam: float) -> _PreparedGlobalBlock:
        g = len(self.xg)
        Ggg = _global_kernel(self.xg, self.xg, self.ell)
        Lgg = _wendland_kernel(self.xg, self.xg, self.theta_l)
        A = (1.0 - lam) * Ggg + lam * Lgg
        A.flat[::g + 1] += self.eta + JITTER
        cf = cho_factor(A, lower=True, check_finite=False)
        Ainv = cho_solve(cf, np.eye(g), check_finite=False)
        Ainv_Yg = Ainv @ self.yg
        Ainv_ones = Ainv @ np.ones(g, dtype=np.float64)
        return _PreparedGlobalBlock(lam=float(lam), Ainv=Ainv, Ainv_Yg=Ainv_Yg, Ainv_ones=Ainv_ones)

    def _local_indices_for_point(self, xstar: np.ndarray, exclude_context_idx: Optional[int]=None) -> np.ndarray:
        if self._tree is None or len(self.local_candidate_idx) == 0:
            return np.empty(0, dtype=int)
        k = min(len(self.local_candidate_idx), self.local_points + (1 if exclude_context_idx is not None else 0))
        _, pos = self._tree.query(np.asarray(xstar), k=k)
        pos = np.atleast_1d(pos).astype(int)
        idx = self.local_candidate_idx[pos]
        if exclude_context_idx is not None:
            idx = idx[idx != int(exclude_context_idx)]
        return idx[:self.local_points]

    def _predict_one_given_prepared(self, xstar: np.ndarray, prepared: _PreparedGlobalBlock, exclude_context_idx: Optional[int]=None, need_mean: bool=True, use_global_tau_for_variance: bool=False) -> Tuple[Optional[np.ndarray], np.ndarray, float]:
        lam = prepared.lam
        xstar2 = np.asarray(xstar, dtype=np.float64).reshape(1, -1)
        local_idx = self._local_indices_for_point(xstar, exclude_context_idx)
        xl = self.x_ctx[local_idx]
        yl = self.y_ctx[local_idx]
        rg = (1.0 - lam) * _global_kernel(xstar2, self.xg, self.ell).ravel() + lam * _wendland_kernel(xstar2, self.xg, self.theta_l).ravel()
        Ainv = prepared.Ainv
        Ainv_rg = Ainv @ rg
        if len(local_idx) == 0:
            z1 = prepared.Ainv_ones
            ZY = prepared.Ainv_Yg
            denom = float(np.sum(z1))
            denom = denom if abs(denom) > 1e-12 else 1.0
            mu = np.sum(ZY, axis=0) / denom
            alpha = ZY - z1[:, None] * mu[None, :]
            mean = mu + rg @ alpha if need_mean else None
            residual = self.yg - mu[None, :]
            tau2 = np.sum(residual * alpha, axis=0) / max(1, len(self.xg))
            tau2 = np.maximum(tau2, 1e-12)
            latent_base = max(1.0 - float(rg @ Ainv_rg), 1e-12)
            if use_global_tau_for_variance and self.global_tau2 is not None:
                tau2 = self.global_tau2
            pred_var = np.maximum(tau2 * (latent_base + self.eta), 1e-12)
            return (mean, pred_var, latent_base)
        B = (1.0 - lam) * _global_kernel(self.xg, xl, self.ell) + lam * _wendland_kernel(self.xg, xl, self.theta_l)
        D = (1.0 - lam) * _global_kernel(xl, xl, self.ell) + lam * _wendland_kernel(xl, xl, self.theta_l)
        l = len(local_idx)
        D.flat[::l + 1] += self.eta + JITTER
        AinvB = Ainv @ B
        S = D - B.T @ AinvB
        S = 0.5 * (S + S.T)
        S.flat[::l + 1] += JITTER
        try:
            cfS = cho_factor(S, lower=True, check_finite=False)
        except Exception:
            S.flat[::l + 1] += 1e-07
            cfS = cho_factor(S, lower=True, check_finite=False)
        rl = (1.0 - lam) * _global_kernel(xstar2, xl, self.ell).ravel() + lam * _wendland_kernel(xstar2, xl, self.theta_l).ravel()
        temp_r = rl - B.T @ Ainv_rg
        ql = cho_solve(cfS, temp_r, check_finite=False)
        qg = Ainv_rg - AinvB @ ql
        latent_base = max(1.0 - float(rg @ qg + rl @ ql), 1e-12)
        if not need_mean and use_global_tau_for_variance and (self.global_tau2 is not None):
            pred_var = np.maximum(self.global_tau2 * (latent_base + self.eta), 1e-12)
            return (None, pred_var, latent_base)
        tempY = yl - B.T @ prepared.Ainv_Yg
        zlY = cho_solve(cfS, tempY, check_finite=False)
        zgY = prepared.Ainv_Yg - AinvB @ zlY
        temp1 = np.ones(l) - B.T @ prepared.Ainv_ones
        zl1 = cho_solve(cfS, temp1, check_finite=False)
        zg1 = prepared.Ainv_ones - AinvB @ zl1
        denom = float(np.sum(zg1) + np.sum(zl1))
        denom = denom if abs(denom) > 1e-12 else 1.0
        mu = (np.sum(zgY, axis=0) + np.sum(zlY, axis=0)) / denom
        alpha_g = zgY - zg1[:, None] * mu[None, :]
        alpha_l = zlY - zl1[:, None] * mu[None, :]
        mean = mu + rg @ alpha_g + rl @ alpha_l if need_mean else None
        res_g = self.yg - mu[None, :]
        res_l = yl - mu[None, :]
        tau2 = (np.sum(res_g * alpha_g, axis=0) + np.sum(res_l * alpha_l, axis=0)) / max(1, len(self.xg) + l)
        tau2 = np.maximum(tau2, 1e-12)
        if use_global_tau_for_variance and self.global_tau2 is not None:
            tau2 = self.global_tau2
        pred_var = np.maximum(tau2 * (latent_base + self.eta), 1e-12)
        return (mean, pred_var, latent_base)

    def _estimate_global_tau2(self) -> np.ndarray:
        prep = self._prepared
        z1 = prep.Ainv_ones
        ZY = prep.Ainv_Yg
        denom = float(np.sum(z1))
        denom = denom if abs(denom) > 1e-12 else 1.0
        mu = np.sum(ZY, axis=0) / denom
        alpha = ZY - z1[:, None] * mu[None, :]
        residual = self.yg - mu[None, :]
        tau2 = np.sum(residual * alpha, axis=0) / max(1, len(self.xg))
        return np.maximum(tau2, 1e-12)

    def _tune_lambda(self) -> float:
        if len(self.lambda_grid) == 1:
            return float(self.lambda_grid[0])
        nonglobal = np.asarray(self.local_candidate_idx, dtype=int)
        if len(nonglobal) < 2:
            return 0.5
        rng = np.random.default_rng(self.seed + 91777)
        vnum = min(self.lambda_validation_points, len(nonglobal))
        val_idx = rng.choice(nonglobal, size=vnum, replace=False)
        scale = _safe_std(self.y_ctx, axis=0, keepdims=False)
        best_lam = 0.5
        best_err = np.inf
        for lam in self.lambda_grid:
            try:
                prep = self._prepare_global_block(float(lam))
                errs = []
                for idx in val_idx:
                    mean, _, _ = self._predict_one_given_prepared(self.x_ctx[idx], prep, exclude_context_idx=int(idx), need_mean=True, use_global_tau_for_variance=False)
                    errs.append(np.mean(((self.y_ctx[idx] - mean) / scale) ** 2))
                err = float(np.mean(errs))
                if np.isfinite(err) and err < best_err:
                    best_err = err
                    best_lam = float(lam)
            except Exception:
                continue
        return best_lam

    def predict(self, xstar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xstar = np.asarray(xstar, dtype=np.float64)
        means = np.empty((len(xstar), self.y_ctx.shape[1]), dtype=np.float64)
        vars_ = np.empty_like(means)
        for i, x in enumerate(xstar):
            mean, var, _ = self._predict_one_given_prepared(x, self._prepared, need_mean=True, use_global_tau_for_variance=False)
            means[i] = mean
            vars_[i] = var
        return (means, vars_)

    def acquisition_variances(self, xstar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xstar = np.asarray(xstar, dtype=np.float64)
        n = len(xstar)
        task_var = np.empty((n, self.y_ctx.shape[1]), dtype=np.float64)
        latent_base = np.empty(n, dtype=np.float64)
        for i, x in enumerate(xstar):
            _, v, b = self._predict_one_given_prepared(x, self._prepared, need_mean=False, use_global_tau_for_variance=True)
            task_var[i] = v
            latent_base[i] = b
        return (task_var, latent_base)

def _bald_select_batch(surrogate: TwinGPVectorSurrogate, x_pool: np.ndarray, batch_size: int) -> np.ndarray:
    _, latent = surrogate.acquisition_variances(x_pool)
    t = surrogate.y_ctx.shape[1]
    scores = 0.5 * t * np.log1p(np.maximum(latent, 1e-12) / surrogate.eta)
    k = min(int(batch_size), len(scores))
    return np.argsort(scores)[-k:][::-1].astype(int)

def _acs_fw_select_batch(surrogate: TwinGPVectorSurrogate, x_pool: np.ndarray, batch_size: int, J: int=ACS_J, seed: int=0) -> np.ndarray:
    pred_var, _ = surrogate.acquisition_variances(x_pool)
    pred_var = np.maximum(pred_var, 1e-12)
    M, T = pred_var.shape
    if M == 0:
        return np.empty(0, dtype=int)
    k_select = min(int(batch_size), M)
    J = int(J)
    rng = np.random.default_rng(seed)
    V = np.zeros((M, J), dtype=np.float64)
    chunk = 128
    log2pi = math.log(2.0 * math.pi)
    for start in range(0, M, chunk):
        stop = min(M, start + chunk)
        v = pred_var[start:stop]
        z = rng.standard_normal((J, stop - start, T))
        lp = -0.5 * (z * z + log2pi + np.log(v)[None, :, :])
        V[start:stop] = np.sum(lp, axis=2).T / math.sqrt(J)
    sigma_m = np.linalg.norm(V, axis=1)
    sigma_m = np.maximum(sigma_m, 1e-08)
    sigma_total = float(np.sum(sigma_m))
    v_L = np.sum(V, axis=0)
    v_Lw = np.zeros(J, dtype=np.float64)
    selected: List[int] = []
    for _ in range(k_select):
        residual = v_L - v_Lw
        inner = V @ residual / sigma_m
        if selected:
            inner[np.asarray(selected, dtype=int)] = -np.inf
        f = int(np.argmax(inner))
        selected.append(f)
        u_f = sigma_total / sigma_m[f] * V[f]
        direction = u_f - v_Lw
        denom = float(direction @ direction)
        if denom > 1e-12:
            gamma = float(np.clip(residual @ direction / denom, 0.0, 1.0))
        else:
            gamma = 0.0
        v_Lw = (1.0 - gamma) * v_Lw + gamma * u_f
    return np.asarray(selected, dtype=int)

def _infer_common_response_block(df: pd.DataFrame, feature_count: int=N_FEATURES) -> Tuple[int, List[str], Optional[str]]:
    if df.shape[1] <= int(feature_count):
        raise ValueError('Dataset does not contain any columns after the seven input features.')
    tail = df.iloc[:, int(feature_count):]
    numeric_tail = tail.apply(pd.to_numeric, errors='coerce')
    complete = numeric_tail.notna().all(axis=0).to_numpy(dtype=bool)
    complete_positions = np.flatnonzero(complete)
    if len(complete_positions) == 0:
        raise ValueError('Could not find any response column that is numeric and present for every data point after the seven feature columns.')
    first_complete = int(complete_positions[0])
    response_start = int(feature_count) + first_complete
    common_length = 0
    for is_complete in complete[first_complete:]:
        if not bool(is_complete):
            break
        common_length += 1
    if common_length < 1:
        raise ValueError('The inferred common response-curve length is zero.')
    response_cols = list(df.columns[response_start:response_start + common_length])
    skipped_col: Optional[str] = None
    if response_start > int(feature_count):
        skipped = list(df.columns[int(feature_count):response_start])
        skipped_col = ';'.join((str(v) for v in skipped))
    return (response_start, response_cols, skipped_col)

def _standardize_responses_like_np(y: np.ndarray, eps: float=1e-08) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    y_mean = np.mean(y, axis=0, keepdims=True)
    y_std = np.std(y, axis=0, ddof=1, keepdims=True)
    y_std = np.where(np.isfinite(y_std) & (y_std >= float(eps)), y_std, float(eps))
    y_standardized = (y - y_mean) / y_std
    return (y_standardized, y_mean, y_std)

def _unstandardize_responses(y_standardized: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    return np.asarray(y_standardized, dtype=np.float64) * np.asarray(y_std, dtype=np.float64) + np.asarray(y_mean, dtype=np.float64)

def _one_hot_encode_features(X_raw_df: pd.DataFrame) -> np.ndarray:
    blocks: List[np.ndarray] = []
    for col in X_raw_df.columns:
        values = X_raw_df[col].astype(str).to_numpy()
        levels = np.unique(values)
        blocks.append((values[:, None] == levels[None, :]).astype(np.float64))
    return np.hstack(blocks)

def _load_scenario(csv_path: Path, expected_num_outputs: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    del expected_num_outputs
    df = pd.read_csv(csv_path)
    if df.shape[1] <= N_FEATURES:
        raise ValueError(f'{csv_path.name}: expected seven input features plus response columns, found only {df.shape[1]} columns.')
    X_raw_df = df.iloc[:, :N_FEATURES]
    if X_raw_df.isna().any().any():
        bad = int(X_raw_df.isna().sum().sum())
        raise ValueError(f'{csv_path.name}: {bad} missing values found in the first seven feature columns.')
    X = _one_hot_encode_features(X_raw_df)
    print(f'{csv_path.name}: one-hot encoded {N_FEATURES} categorical features -> Xdim={X.shape[1]}', flush=True)
    response_start, response_cols, skipped_col = _infer_common_response_block(df, feature_count=N_FEATURES)
    Y_df = df[response_cols].apply(pd.to_numeric, errors='coerce')
    if Y_df.isna().any().any():
        raise RuntimeError(f'{csv_path.name}: internal error: inferred common response block still contains missing/non-numeric values.')
    Y_raw = Y_df.to_numpy(dtype=np.float64)
    Y, y_mean, y_std = _standardize_responses_like_np(Y_raw)
    if not np.all(np.isfinite(X)):
        raise ValueError(f'{csv_path.name}: non-finite values found in the first seven features.')
    if not np.all(np.isfinite(Y)):
        raise ValueError(f'{csv_path.name}: non-finite values found after response standardization.')
    skipped_text = f', skipped={skipped_col}' if skipped_col else ''
    print(f'{csv_path.name}: inferred common response-curve length={Y.shape[1]}, response_start={response_start}{skipped_text}', flush=True)
    return (X, Y, y_mean, y_std)

def _split_train_test_indices(n: int, seed: int, test_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(float(test_fraction) * n)))
    n_test = min(n - 1, n_test)
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    return (train_idx, test_idx)

def _initial_context_pool(train_idx: np.ndarray, seed: int, initial_context_size: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(train_idx) < initial_context_size + 1:
        raise ValueError(f'Training split has {len(train_idx)} rows but initial context requires {initial_context_size}.')
    rng = np.random.default_rng(seed + 1000003)
    ctx = np.sort(rng.choice(train_idx, size=int(initial_context_size), replace=False))
    pool = np.setdiff1d(train_idx, ctx, assume_unique=False)
    return (ctx, pool)

def _run_strategy_one_seed(X_scaled: np.ndarray, Y: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, init_ctx_idx: np.ndarray, init_pool_idx: np.ndarray, strategy: str, seed: int, num_acquisitions: int, batch_size: int, acs_j: int, local_points: int, hyperopt_maxiter: int) -> Tuple[List[Dict[str, float]], np.ndarray]:
    ctx_idx = init_ctx_idx.copy()
    pool_idx = init_pool_idx.copy()
    rows: List[Dict[str, float]] = []
    previous_hyperparams = None
    final_test_pred = None
    for round_id in range(int(num_acquisitions) + 1):
        surrogate = TwinGPVectorSurrogate(local_points=local_points, seed=seed * 1000 + round_id, hyperopt_maxiter=hyperopt_maxiter)
        surrogate.fit(X_scaled[ctx_idx], Y[ctx_idx], X_scaled[train_idx], initial_hyperparams=previous_hyperparams)
        previous_hyperparams = surrogate.hyperparams
        test_pred_standardized, _ = surrogate.predict(X_scaled[test_idx])
        test_true_original = _unstandardize_responses(Y[test_idx], y_mean, y_std)
        test_pred_original = _unstandardize_responses(test_pred_standardized, y_mean, y_std)
        final_test_pred = test_pred_original
        rows.append({'seed': int(seed), 'strategy': strategy, 'acquisition_round': int(round_id), 'context_size': int(len(ctx_idx)), 'pool_size': int(len(pool_idx)), 'test_mape': _stable_mape(test_true_original, test_pred_original), 'test_mse': _mse(test_true_original, test_pred_original)})
        print(f"seed={seed:3d} | {strategy:6s} | round={round_id:02d} | ctx={len(ctx_idx):3d} | pool={len(pool_idx):3d} | MAPE={rows[-1]['test_mape']:.6f} | MSE={rows[-1]['test_mse']:.6f} | g={len(surrogate.global_idx):2d} | l={local_points:2d} | lambda={surrogate.lam:.2f}")
        if round_id >= int(num_acquisitions) or len(pool_idx) == 0:
            break
        x_pool = X_scaled[pool_idx]
        k = min(int(batch_size), len(pool_idx))
        if strategy == 'BALD':
            selected_pos = _bald_select_batch(surrogate, x_pool, k)
        elif strategy == 'ACS_FW':
            selected_pos = _acs_fw_select_batch(surrogate, x_pool, k, J=acs_j, seed=seed * 100000 + round_id)
        else:
            raise ValueError(f'Unknown strategy: {strategy}')
        selected_pos = np.unique(np.asarray(selected_pos, dtype=int))
        if len(selected_pos) < k:
            missing = k - len(selected_pos)
            remaining_pos = np.setdiff1d(np.arange(len(pool_idx)), selected_pos)
            if len(selected_pos):
                chosen_x = x_pool[selected_pos]
                d2 = np.min(np.sum((x_pool[remaining_pos, None, :] - chosen_x[None, :, :]) ** 2, axis=2), axis=1)
                fill = remaining_pos[np.argsort(d2)[-missing:]]
            else:
                fill = remaining_pos[:missing]
            selected_pos = np.concatenate([selected_pos, fill])
        selected_pos = selected_pos[:k]
        selected_global_idx = pool_idx[selected_pos]
        ctx_idx = np.concatenate([ctx_idx, selected_global_idx])
        keep = np.ones(len(pool_idx), dtype=bool)
        keep[selected_pos] = False
        pool_idx = pool_idx[keep]
    return (rows, final_test_pred)

def _save_mape_trajectory_plot(metrics: pd.DataFrame, out_path: Path, scenario_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    for strategy in STRATEGIES:
        sub = metrics[metrics['strategy'] == strategy]
        grp = sub.groupby('acquisition_round')['test_mape']
        mean = grp.mean()
        std = grp.std().fillna(0.0)
        x = mean.index.to_numpy(dtype=int)
        y = mean.to_numpy(dtype=float)
        s = std.reindex(mean.index).to_numpy(dtype=float)
        ax.plot(x, y, marker='o', linewidth=1.8, markersize=4, label=strategy)
        ax.fill_between(x, y - s, y + s, alpha=0.15)
    ax.set_xlabel('Acquisition round')
    ax.set_ylabel('Test MAPE')
    ax.set_title(f'{scenario_name}: Test MAPE Trajectories')
    ax.set_xticks(np.arange(metrics['acquisition_round'].min(), metrics['acquisition_round'].max() + 1, 1))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)

def _save_curve_grid(y_true: np.ndarray, y_pred: np.ndarray, selected_test_positions: np.ndarray, out_path: Path, scenario_name: str, strategy: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    axes = axes.ravel()
    x_curve = np.arange(y_true.shape[1])
    for ax, pos in zip(axes, selected_test_positions):
        ax.plot(x_curve, y_true[pos], linewidth=1.7, label='True')
        ax.plot(x_curve, y_pred[pos], linewidth=1.5, linestyle='--', label='Predicted')
        ax.set_title(f'Test curve {int(pos)}')
        ax.set_xlabel('Response index')
        ax.set_ylabel('Response')
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)
    fig.suptitle(f'{scenario_name} - {strategy}: True vs Predicted', y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)

def _run_one_seed_worker(seed: int, X: np.ndarray, Y: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, test_fraction: float, initial_context_size: int, num_acquisitions: int, batch_size: int, acs_j: int, local_points: int, hyperopt_maxiter: int) -> Tuple[int, List[Dict[str, float]], Optional[Dict[str, object]]]:
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    seed = int(seed)
    n = len(X)
    train_idx, test_idx = _split_train_test_indices(n, seed, test_fraction)
    required = int(initial_context_size) + int(num_acquisitions) * int(batch_size)
    if len(train_idx) < required:
        raise ValueError(f'seed={seed}: training split has {len(train_idx)} rows, but the configured active-learning budget needs at least {required}.')
    lo, span = _minmax_fit(X[train_idx])
    Xs = _minmax_apply(X, lo, span)
    init_ctx_idx, init_pool_idx = _initial_context_pool(train_idx, seed, initial_context_size)
    seed_rows: List[Dict[str, float]] = []
    final_predictions: Dict[str, np.ndarray] = {}
    for strategy in STRATEGIES:
        rows, final_pred = _run_strategy_one_seed(X_scaled=Xs, Y=Y, y_mean=y_mean, y_std=y_std, train_idx=train_idx, test_idx=test_idx, init_ctx_idx=init_ctx_idx, init_pool_idx=init_pool_idx, strategy=strategy, seed=seed, num_acquisitions=num_acquisitions, batch_size=batch_size, acs_j=acs_j, local_points=local_points, hyperopt_maxiter=hyperopt_maxiter)
        seed_rows.extend(rows)
        final_predictions[strategy] = final_pred
    curve_payload: Optional[Dict[str, object]] = None
    if seed == int(CURVE_PLOT_SEED):
        rng = np.random.default_rng(seed + 77777)
        k = min(N_CURVE_PLOTS, len(test_idx))
        selected_positions = np.sort(rng.choice(len(test_idx), size=k, replace=False))
        if k < N_CURVE_PLOTS:
            raise ValueError('Test split has fewer than 9 curves; cannot make requested 3x3 grid.')
        curve_payload = {'y_true': _unstandardize_responses(Y[test_idx], y_mean, y_std).copy(), 'positions': selected_positions, 'predictions': {name: pred.copy() for name, pred in final_predictions.items()}}
    print(f'Completed seed trajectory: seed={seed}', flush=True)
    return (seed, seed_rows, curve_payload)

def _run_scenario(csv_path: Path, scenario_dir: Path, seeds: Sequence[int], expected_num_outputs: Optional[int], test_fraction: float, initial_context_size: int, num_acquisitions: int, batch_size: int, acs_j: int, local_points: int, hyperopt_maxiter: int, cpu_workers: int=CPU_WORKERS) -> None:
    X, Y, y_mean, y_std = _load_scenario(csv_path, expected_num_outputs)
    scenario_name = csv_path.stem
    scenario_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple((int(s) for s in seeds))
    if len(seeds) == 0:
        raise ValueError('seeds cannot be empty.')
    workers = min(max(1, int(cpu_workers)), len(seeds))
    all_rows: List[Dict[str, float]] = []
    curve_payload = None
    print(f'Running {len(seeds)} seeds with {workers} parallel CPU workers (one numerical thread per worker).', flush=True)
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        future_to_seed = {executor.submit(_run_one_seed_worker, seed, X, Y, y_mean, y_std, float(test_fraction), int(initial_context_size), int(num_acquisitions), int(batch_size), int(acs_j), int(local_points), int(hyperopt_maxiter)): seed for seed in seeds}
        for future in as_completed(future_to_seed):
            seed = int(future_to_seed[future])
            try:
                returned_seed, rows, payload = future.result()
            except Exception as exc:
                raise RuntimeError(f'TwinGP worker failed for {scenario_name}, seed {seed}. Original error: {exc}') from exc
            if int(returned_seed) != seed:
                raise RuntimeError(f'Worker seed mismatch: expected {seed}, got {returned_seed}.')
            all_rows.extend(rows)
            if payload is not None:
                curve_payload = payload
    metrics = pd.DataFrame(all_rows)
    metrics = metrics[['seed', 'strategy', 'acquisition_round', 'context_size', 'pool_size', 'test_mape', 'test_mse']].sort_values(['seed', 'strategy', 'acquisition_round'], kind='mergesort').reset_index(drop=True)
    metrics.to_csv(scenario_dir / 'metrics_trajectories.csv', index=False)
    _save_mape_trajectory_plot(metrics, scenario_dir / 'test_mape_trajectories.png', scenario_name)
    if curve_payload is None:
        raise RuntimeError(f'Curve plot seed {CURVE_PLOT_SEED} was not included in seeds={list(seeds)}.')
    for strategy in STRATEGIES:
        _save_curve_grid(curve_payload['y_true'], curve_payload['predictions'][strategy], curve_payload['positions'], scenario_dir / f'{strategy}_true_vs_predicted_9.png', scenario_name, strategy)
    print(f'Completed and saved scenario: {scenario_name}', flush=True)

def main(base_dir: Path=BASE_DIR, save_dir: Path=RESULTS_ROOT, scenario_files: Sequence[str]=SCENARIO_FILES, seeds: Sequence[int]=SEEDS, expected_num_outputs: Optional[int]=EXPECTED_NUM_OUTPUTS, test_fraction: float=TEST_FRACTION, initial_context_size: int=INITIAL_CONTEXT_SIZE, acq_rounds: int=NUM_ACQUISITIONS, batch_size: int=ACQUISITION_BATCH_SIZE, acs_j: int=ACS_J, local_points: int=LOCAL_POINTS, hyperopt_maxiter: int=HYPEROPT_MAXITER, cpu_workers: int=CPU_WORKERS) -> None:
    base_dir = Path(base_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if CURVE_PLOT_SEED not in [int(s) for s in seeds]:
        raise ValueError(f'seeds must include {CURVE_PLOT_SEED} because that seed is used for the requested nine final true-vs-predicted curve plots.')
    print('TwinGP-inspired global-local GP active learning')
    print(f'Base directory : {base_dir}')
    print(f'Results root   : {save_dir}')
    print(f"Twinning pkg   : {('available' if _HAS_TWINNING else 'not installed; maximin fallback')}")
    print(f'Seeds          : {list(seeds)}')
    print(f'CPU workers    : {min(int(cpu_workers), len(seeds))}')
    print('Outputs        : inferred dynamically from the common response-curve block')
    print(f'AL budget      : initial={initial_context_size}, rounds={acq_rounds}, batch={batch_size}')
    for filename in scenario_files:
        csv_path = base_dir / filename
        if not csv_path.exists():
            raise FileNotFoundError(f'Scenario file not found: {csv_path}')
        scenario_dir = save_dir / csv_path.stem
        print('\n' + '=' * 80)
        print(f'Scenario: {csv_path.name}')
        print('=' * 80)
        _run_scenario(csv_path=csv_path, scenario_dir=scenario_dir, seeds=seeds, expected_num_outputs=expected_num_outputs, test_fraction=test_fraction, initial_context_size=initial_context_size, num_acquisitions=acq_rounds, batch_size=batch_size, acs_j=acs_j, local_points=local_points, hyperopt_maxiter=hyperopt_maxiter, cpu_workers=cpu_workers)

def run_notebook_workflow(gpu_id: Optional[int]=None, base_dir: str=str(BASE_DIR), save_dir: Optional[str]=None, scenario_files: Sequence[str]=SCENARIO_FILES, seeds: Sequence[int]=SEEDS, expected_num_outputs: Optional[int]=EXPECTED_NUM_OUTPUTS, test_fraction: float=TEST_FRACTION, initial_context_size: int=INITIAL_CONTEXT_SIZE, acq_rounds: int=NUM_ACQUISITIONS, batch_size: int=ACQUISITION_BATCH_SIZE, acs_j: int=ACS_J, local_points: int=LOCAL_POINTS, hyperopt_maxiter: int=HYPEROPT_MAXITER, cpu_workers: int=CPU_WORKERS, **_ignored) -> None:
    if gpu_id is not None:
        print(f'gpu_id={gpu_id} received. GP_TwinGP uses CPU NumPy/SciPy block linear algebra; the argument is kept only for runner compatibility.')
    out = Path(save_dir) if save_dir is not None else Path(base_dir) / 'GP_TwinGP_results'
    return main(base_dir=Path(base_dir), save_dir=out, scenario_files=scenario_files, seeds=tuple((int(s) for s in seeds)), expected_num_outputs=expected_num_outputs, test_fraction=float(test_fraction), initial_context_size=int(initial_context_size), acq_rounds=int(acq_rounds), batch_size=int(batch_size), acs_j=int(acs_j), local_points=int(local_points), hyperopt_maxiter=int(hyperopt_maxiter), cpu_workers=int(cpu_workers))
if __name__ == '__main__':
    main()
