# -*- coding: utf-8 -*-
"""
FedMGP Multi-Output Gaussian Process + Batch Active Learning
============================================================

Implements the hierarchical multi-output GP and federated variational
inference of

    Chung, S. and Al Kontar, R. (2024). "Federated Multi-Output Gaussian
    Processes." Technometrics, 66(1), 90-103.

and uses it as the surrogate inside a BALD / ACS-FW batch active-learning
loop.

Mapping of the paper onto this problem
--------------------------------------
* Each response-curve coordinate t = 1..T is one FedMGP *unit* f_t(x).
* x is the one-hot (optionally min-max scaled) vector of the 7 input features.
* All units are generated from I shared latent GPs g_i(x) through
  unit-specific Gaussian convolution-process (CP) kernels:

    k_{g_i g_i}(w, w') = exp(-1/2 (w-w')^T S_i^{-1} (w-w'))                  (17)

    k_{f_t g_i}(x, w)  = v_ti sqrt(|S_i| / |R_ti + S_i|)
                         exp(-1/2 (x-w)^T (R_ti + S_i)^{-1} (x-w))           (18)

    k_{f_t f_t}(x, x') = sum_i v_ti^2 sqrt(|S_i| / |2R_ti + S_i|)
                         exp(-1/2 (x-x')^T (2R_ti + S_i)^{-1} (x-x'))

  (Eq. 18 in the paper writes v^2 in the cross-covariance; v in the cross
  term and v^2 in the auto term is the self-consistent convolution form and
  is the same model family under v -> sqrt(v).)  S_i and R_ti are diagonal.
  Internally v_ti is reparameterised through the unit's prior variance
  a_ti^2 = v_ti^2 sqrt(|S_i| / |2R_ti + S_i|), which stays O(1) when the
  one-hot input has many dimensions.

* J pseudo-inputs W are shared by all units.  q(g) = N(mu, L L^T) is the
  global variational posterior over the pseudo-targets g = g(W).
* The ELBO (Eq. 13) decomposes over units:

      ELBO = sum_t E_{q(f_t)}[log p(y_t | f_t)] - KL(q(g) || p(g | W))

* Global parameters   theta   = {W, S, mu, L}
  Personal parameters phi_t   = {R_t, v_t, sigma_t}

Training (TRAINING_MODE)
------------------------
"federated"   : Algorithm 2.  Each round the server broadcasts theta; every
                unit runs LOCAL_STEPS steps on its local risk R_t (Eq. 14)
                over its own copy of theta (SGD) and its own phi_t (Adam);
                the server averages the local pseudo-gradients (Eq. 15,
                p_t = 1/T) and updates theta with them (SERVER_OPT: Adam on
                Delta = FedAdam, or theta <- theta - eta*Delta).  All T
                clients are run as one batched computation.

                Note on LOCAL_STEPS: with E = 1, the averaged pseudo-gradient
                equals the exact gradient of the global ELBO, so the fit
                matches centralized training.  Here every client is a single
                response coordinate, so clients are very heterogeneous and
                E > 1 causes client drift: on synthetic sigmoid curves E = 5
                roughly doubled test MSE at the same number of rounds.
"centralized" : Adam directly on the same ELBO (the paper's CenMGP-style
                reference optimiser for the same model).

Prediction is the personalised predictive (Eq. 16).

Active learning
---------------
BALD   : exact joint mutual information between the T-dimensional response
         y(x) and the latent outputs f(x) under the FedMGP posterior,
             0.5 [ logdet(Sigma_f(x) + Sigma_n) - logdet(Sigma_n) ],
         where Sigma_f(x) = diag(Omega(x)) + V(x) M V(x)^T carries the
         cross-unit correlation induced by the shared latent functions.
         Computed with the matrix determinant lemma in O(T (IJ)^2).
ACS-FW : Pinsler et al. (2019) Frank-Wolfe batch construction with random
         projections built from SHARED posterior samples g_j ~ q(g), so the
         projection inner products encode correlation between pool points.
         L_n(g_j) = E_{y ~ p(y|x_n, D)} [log p(y | x_n, g_j)] is closed form.
         Projections are centred over samples (theta-independent constants
         do not change the log posterior).

Saved output per scenario
-------------------------
1) mape_trajectories_all_strategies.png   (mean over seeds +/- 1 s.e.)
2) acquisition_metrics_all_strategies.csv
       seed, strategy, acquisition_round, context_size, pool_size,
       test_mape, test_mse
   (`seed` is the split seed.)
3) true_vs_predicted_9_curves_<STRATEGY>.png   (first seed, final model)

Data layout
-----------
columns 0-6: input features; column 7: skipped; response curve: the block of
consecutive, fully numeric, fully observed columns starting at column 8
(RESPONSE_START_INDEX).

Split (per seed)
----------------
test   = round(TEST_FRACTION * n) rows, never seen during training/acquisition
context= INITIAL_CONTEXT_SIZE rows
pool   = remaining rows (or POOL_SIZE of them if set)
The pool must be strictly larger than NUM_ACQUISITIONS * ACQUISITION_BATCH_SIZE
so the strategies do not end on an identical training set.

Responses are standardised with the INITIAL CONTEXT mean/std of each seed
(no test-set leakage).  Metrics are computed on the original scale.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import math
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import torch
import torch.nn.functional as F

from sklearn.preprocessing import OneHotEncoder


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("./paper scenarios")

SCENARIO_FILES = [
    "scenario_000_HGR.csv",
    "scenario_016_damped.csv",
    "scenario_054_gompertz.csv",
    "scenario_179_sigmoid.csv",
]

RESULTS_ROOT = BASE_DIR / "fedmgp_results"

STRATEGIES = ("ACS_FW", "BALD")

# --- split / AL budget ---
INITIAL_CONTEXT_SIZE = 140
NUM_ACQUISITIONS = 30
ACQUISITION_BATCH_SIZE = 5
TEST_FRACTION = 0.20
POOL_SIZE = None              # None -> every row not in test or context

# --- FedMGP model ---
NUM_LATENTS = 1               # I  (paper: 1 in simulations, 2 in case study)
NUM_INDUCING = 30             # J
CONTINUOUS_FEATURES = ()      # names/indices of the 7 inputs to min-max scale
                              # instead of one-hot encode

# --- training ---
TRAINING_MODE = "federated"   # "federated" (Algorithm 2) or "centralized"
TRAIN_EPOCHS = 1000           # max total local gradient steps per fit
LOCAL_STEPS = 1               # E: local steps per round (see note in docstring)
TRAIN_LR = 0.05               # Adam lr (centralized; personal phi_t when federated)
CLIENT_OPT = "sgd"            # local optimiser for the theta copies: "sgd" | "adam"
CLIENT_SGD_LR = 0.05          # local SGD lr for theta copies
SERVER_OPT = "adam"           # "adam" (FedAdam on Delta) | "sgd" (theta -= eta*Delta)
SERVER_LR = 0.05              # server step size (eta when SERVER_OPT="sgd")
EARLY_STOP_PATIENCE = 50      # in local gradient steps (converted to rounds)
EARLY_STOP_MIN_EPOCHS = 75    # in local gradient steps
EARLY_STOP_TOL = 1e-4         # on the per-observation negative ELBO

# --- acquisition ---
ACS_J = 50                    # posterior samples for ACS-FW projections
ACS_MAX_ITERS_FACTOR = 10     # Frank-Wolfe iterations <= factor * batch size
PREDICT_CHUNK_SIZE = 128

# --- reproducibility ---
TRAIN_SEED = 220
DEFAULT_NUM_SEEDS = 20
NP_SPLIT_SEEDS = (
    1, 7, 11, 19, 23,
    31, 37, 41, 55, 70,
    99, 105, 121, 135, 160,
    202, 82, 114, 303, 154,
)

EXPECTED_NUM_OUTPUTS = None   # if set, assert the inferred curve length
RESPONSE_START_INDEX = 8      # columns 0-6 inputs, column 7 skipped, curve from 8

# --- numerics ---
DTYPE = torch.float64
JITTER = 1e-6
MAPE_EPS = 1e-8
LOG_2PI = math.log(2.0 * math.pi)


# =============================================================================
# UTILITIES
# =============================================================================

def clear_torch_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def softplus_inv(x: float) -> float:
    return float(x + math.log(-math.expm1(-x)))


def _pos(raw: torch.Tensor, floor: float) -> torch.Tensor:
    return F.softplus(raw) + floor


# =============================================================================
# DATA LOADING
# =============================================================================

def validate_requested_files() -> None:
    missing = [n for n in SCENARIO_FILES if not (BASE_DIR / n).is_file()]
    if missing:
        expected = "\n".join(str(BASE_DIR / n) for n in SCENARIO_FILES)
        raise FileNotFoundError(
            "One or more requested scenario files are missing.\n\n"
            "Missing:\n" + "\n".join(missing) + "\n\n"
            f"The code expects these paths:\n{expected}"
        )


def _infer_common_response_block(df: pd.DataFrame, search_start: int):
    if df.shape[1] <= search_start:
        raise ValueError(f"Dataset has no columns at or after index {search_start}.")

    numeric_tail = df.iloc[:, search_start:].apply(pd.to_numeric, errors="coerce")
    complete = numeric_tail.notna().all(axis=0).to_numpy(dtype=bool)
    positions = np.flatnonzero(complete)
    if len(positions) == 0:
        raise ValueError(
            "No response column is numeric and observed for every row."
        )

    first = int(positions[0])
    length = 0
    for ok in complete[first:]:
        if not ok:
            break
        length += 1

    start = search_start + first
    return start, list(df.columns[start:start + length])


def _resolve_feature_cols(feature_cols, spec):
    out = []
    for item in spec:
        if isinstance(item, (int, np.integer)):
            out.append(feature_cols[int(item)])
        elif item in feature_cols:
            out.append(item)
        else:
            raise ValueError(f"Unknown continuous feature: {item!r}")
    return out


def load_data(path: Path):
    """Return X (encoded inputs), Y_raw (original-scale curves), metadata."""
    df = pd.read_csv(path)
    if df.shape[1] < 8:
        raise ValueError(f"{path.name} has only {df.shape[1]} columns.")

    feature_cols = list(df.columns[:7])
    cont_cols = _resolve_feature_cols(feature_cols, CONTINUOUS_FEATURES)
    cat_cols = [c for c in feature_cols if c not in cont_cols]

    blocks = []
    if cat_cols:
        try:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
        blocks.append(enc.fit_transform(df[cat_cols].astype(str)))
    if cont_cols:
        c = df[cont_cols].apply(pd.to_numeric, errors="raise").to_numpy(float)
        lo, hi = c.min(axis=0), c.max(axis=0)
        blocks.append((c - lo) / np.where(hi > lo, hi - lo, 1.0))
    X_np = np.concatenate(blocks, axis=1)

    start, response_cols = _infer_common_response_block(df, RESPONSE_START_INDEX)
    if start != RESPONSE_START_INDEX:
        warnings.warn(
            f"{path.name}: column {RESPONSE_START_INDEX} is not fully numeric; "
            f"response block starts at column {start} instead."
        )
    Y_np = df[response_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if np.isnan(Y_np).any():
        raise RuntimeError(f"{path.name}: response block has missing values.")

    if EXPECTED_NUM_OUTPUTS is not None and Y_np.shape[1] != EXPECTED_NUM_OUTPUTS:
        raise ValueError(
            f"{path.name}: inferred curve length {Y_np.shape[1]} != "
            f"expected_num_outputs={EXPECTED_NUM_OUTPUTS}."
        )

    X = torch.tensor(X_np, dtype=DTYPE)
    Y_raw = torch.tensor(Y_np, dtype=DTYPE)

    metadata = {
        "feature_cols": feature_cols,
        "continuous_cols": cont_cols,
        "response_start_index": int(start),
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "n_tasks": int(Y_raw.shape[1]),
    }
    print(
        f"{path.name}: rows={len(df)}, encoded Xdim={X.shape[1]}, "
        f"curve length={Y_raw.shape[1]}, response_start={start}",
        flush=True,
    )
    return X, Y_raw, metadata


# =============================================================================
# SPLIT
# =============================================================================

def create_al_split(
    n_samples: int,
    split_seed: int,
    context_size: int,
    test_fraction: float,
    acquisition_budget: int,
    pool_size=None,
):
    n_test = int(round(test_fraction * n_samples))
    if n_test < 9:
        raise ValueError(
            f"Test set would have {n_test} rows; the 3 x 3 curve grid needs 9. "
            "Increase test_fraction."
        )

    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(n_samples)

    test_idx = np.sort(perm[:n_test])
    rest = perm[n_test:]
    ctx_idx = np.sort(rest[:context_size])
    pool_idx = rest[context_size:]
    if pool_size is not None:
        pool_idx = pool_idx[: int(pool_size)]
    pool_idx = np.sort(pool_idx)

    if len(pool_idx) <= acquisition_budget:
        raise ValueError(
            f"Pool has {len(pool_idx)} rows but the acquisition budget is "
            f"{acquisition_budget}. The pool must be strictly larger, otherwise "
            "every strategy ends on the same training set. Reduce "
            "test_fraction / initial_context_size / acq_rounds / batch_size."
        )
    return ctx_idx, pool_idx, test_idx


# =============================================================================
# FEDMGP MODEL
# =============================================================================
#
# theta tensors carry a leading "copy" dimension B:
#   B = 1 : the server's global parameters (broadcast to every unit)
#   B = T : one local copy per unit (client) during federated local updates
# phi tensors always carry a leading unit dimension T.

THETA_KEYS = ("W", "raw_S", "mu", "raw_L")
PHI_KEYS = ("raw_R", "log_a", "raw_noise")


def init_fedmgp_params(x_train, num_units, num_latents, num_inducing, seed):
    """Initial global theta (B=1) and personal phi."""
    n, d = x_train.shape
    J = min(num_inducing, n)
    I = num_latents

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(n, generator=gen)[:J]
    W = x_train.detach().cpu()[perm].clone().to(x_train)

    kw = dict(dtype=x_train.dtype, device=x_train.device)

    raw_L = torch.zeros(I, J, J, **kw)
    raw_L.diagonal(dim1=-2, dim2=-1).fill_(softplus_inv(1.0))

    theta = {
        "W": W.unsqueeze(0),                                      # [1,J,d]
        "raw_S": torch.full((1, I, d), softplus_inv(1.0), **kw),  # [1,I,d]
        "mu": torch.zeros(1, I, J, **kw),                         # [1,I,J]
        "raw_L": raw_L.unsqueeze(0),                              # [1,I,J,J]
    }
    phi = {
        "raw_R": torch.full((num_units, I, d), softplus_inv(0.1), **kw),
        "log_a": torch.full((num_units, I), -0.5 * math.log(I), **kw),
        "raw_noise": torch.full((num_units,), softplus_inv(0.1), **kw),
    }
    return theta, phi


def _constrained(theta, phi):
    S = _pos(theta["raw_S"], 1e-4)                               # [B,I,d]
    raw_L = theta["raw_L"]
    L = torch.tril(raw_L, -1) + torch.diag_embed(
        _pos(raw_L.diagonal(dim1=-2, dim2=-1), 1e-5)
    )                                                            # [B,I,J,J]
    R = _pos(phi["raw_R"], 1e-6)                                 # [T,I,d]
    a2 = torch.exp(2.0 * phi["log_a"])                           # [T,I]
    noise = _pos(phi["raw_noise"], 1e-6)                         # [T]
    return S, L, R, a2, noise


def _sqdist(a, b):
    """Squared distances between rows of a [...,N,d] and b [...,M,d]."""
    d2 = (
        (a * a).sum(-1, keepdim=True)
        - 2.0 * a @ b.transpose(-1, -2)
        + (b * b).sum(-1).unsqueeze(-2)
    )
    return d2.clamp_min(0.0)


def _latent_prior(theta, S):
    """Cholesky of K_gg per copy/latent: [B,I,J,J]."""
    W = theta["W"]                                               # [B,J,d]
    Ws = W.unsqueeze(1) / S.sqrt().unsqueeze(2)                  # [B,I,J,d]
    K = torch.exp(-0.5 * _sqdist(Ws, Ws))
    J = K.shape[-1]
    K = K + JITTER * torch.eye(J, dtype=K.dtype, device=K.device)
    return torch.linalg.cholesky(K)


def fedmgp_marginals(theta, phi, X):
    """
    Variational marginals q(f_t(x_n)) (Eq. 12) for every unit t and row n.

    Returns
    -------
    mean   [T,N] : V_t mu
    omega  [T,N] : diag(Omega_t) = k_tt(x,x) - k_tg K_gg^{-1} k_gt
    shared [T,N] : diag(V_t M V_t^T)
    A      [T,I,J,N] : K_gg^{-1} K_g,f_t   (V_t^T, split by latent)
    Lg     [B,I,J,J] : chol(K_gg)
    """
    S, L, R, a2, _ = _constrained(theta, phi)
    W = theta["W"]
    T = R.shape[0]

    Lg = _latent_prior(theta, S)                                 # [B,I,J,J]

    RS = R + S                                                   # [T,I,d]
    twoRS = 2.0 * R + S
    log_norm = (
        0.5 * torch.log(a2)
        - 0.25 * torch.log(S / twoRS).sum(-1)                    # -> log v
        + 0.5 * torch.log(S / RS).sum(-1)
    )                                                            # [T,I]

    scale = RS.sqrt().unsqueeze(2)                               # [T,I,1,d]
    Xs = X.unsqueeze(0).unsqueeze(0) / scale                     # [T,I,N,d]
    Wsc = W.unsqueeze(1) / scale                                 # [T,I,J,d]
    Kfg = torch.exp(log_norm[..., None, None] - 0.5 * _sqdist(Xs, Wsc))  # [T,I,N,J]

    Lg_t = Lg.expand(T, *Lg.shape[1:])
    tmp = torch.linalg.solve_triangular(
        Lg_t, Kfg.transpose(-1, -2), upper=False
    )                                                            # [T,I,J,N]
    A = torch.linalg.solve_triangular(
        Lg_t.transpose(-1, -2), tmp, upper=True
    )                                                            # [T,I,J,N]

    mu = theta["mu"].expand(T, *theta["mu"].shape[1:])           # [T,I,J]
    L_t = L.expand(T, *L.shape[1:])

    mean = torch.einsum("tijn,tij->tn", A, mu)
    kff = a2.sum(-1, keepdim=True)                               # [T,1]
    omega = (kff - (tmp * tmp).sum(-2).sum(1)).clamp_min(1e-10)
    LtA = L_t.transpose(-1, -2) @ A                              # [T,I,J,N]
    shared = (LtA * LtA).sum(-2).sum(1)
    return mean, omega, shared, A, Lg


def kl_q_p(theta, Lg):
    """KL(q(g) || p(g|W)) summed over latents, per copy: [B]."""
    raw_L = theta["raw_L"]
    L = torch.tril(raw_L, -1) + torch.diag_embed(
        _pos(raw_L.diagonal(dim1=-2, dim2=-1), 1e-5)
    )
    mu = theta["mu"]
    J = mu.shape[-1]

    LgiL = torch.linalg.solve_triangular(Lg, L, upper=False)
    Lgimu = torch.linalg.solve_triangular(Lg, mu.unsqueeze(-1), upper=False)

    trace = (LgiL * LgiL).sum((-1, -2))
    maha = (Lgimu * Lgimu).sum((-1, -2))
    logdet_p = 2.0 * torch.log(Lg.diagonal(dim1=-2, dim2=-1)).sum(-1)
    logdet_q = 2.0 * torch.log(L.diagonal(dim1=-2, dim2=-1)).sum(-1)
    kl = 0.5 * (trace + maha - J + logdet_p - logdet_q)          # [B,I]
    return kl.sum(-1)


def expected_loglik(theta, phi, X, Y):
    """E_{q(f_t)}[log p(y_t | f_t)] per unit: [T].  Y is [N,T]."""
    mean, omega, shared, _, Lg = fedmgp_marginals(theta, phi, X)
    noise = _pos(phi["raw_noise"], 1e-6).unsqueeze(-1)           # [T,1]
    yT = Y.transpose(0, 1)
    el = -0.5 * (
        LOG_2PI + torch.log(noise)
        + ((yT - mean) ** 2 + omega + shared) / noise
    )
    return el.sum(-1), Lg


def negative_elbo_per_obs(theta, phi, X, Y):
    """Global objective (-Eq. 13) / (N*T), evaluated with B = 1 theta."""
    el, Lg = expected_loglik(theta, phi, X, Y)
    kl = kl_q_p(theta, Lg)
    N, T = Y.shape
    return -(el.sum() - kl.sum()) / (N * T)


def federated_local_objective(theta_local, phi, X, Y):
    """
    Sum over clients of the local risk (Eq. 14), each on its own theta copy:
        R_t = -(1/p_t) E_q(f_t)[log p(y_t|f_t)] + KL(q(g)||p(g)),  p_t = 1/T
    Gradients w.r.t. copy t and phi_t equal those of R_t alone.
    Scaled by 1/(N*T) for a stable learning rate.
    """
    el, Lg = expected_loglik(theta_local, phi, X, Y)            # [T]
    kl = kl_q_p(theta_local, Lg)                                 # [T]
    N, T = Y.shape
    return (-T * el + kl).sum() / (N * T)


def _clone_params(d):
    return {k: v.detach().clone() for k, v in d.items()}


class FedMGP:
    """Container for a fitted FedMGP (global theta with B=1, personal phi)."""

    def __init__(self, theta, phi):
        self.theta = _clone_params(theta)
        self.phi = _clone_params(phi)

    @property
    def noise(self):
        return _pos(self.phi["raw_noise"], 1e-6)

    def L(self):
        raw_L = self.theta["raw_L"][0]
        return torch.tril(raw_L, -1) + torch.diag_embed(
            _pos(raw_L.diagonal(dim1=-2, dim2=-1), 1e-5)
        )

    @torch.no_grad()
    def marginals(self, x, chunk_size=None):
        """Chunked marginals: mean, omega, shared each [N,T]; A [T,I,J,N]."""
        chunk_size = chunk_size or PREDICT_CHUNK_SIZE
        out = {"mean": [], "omega": [], "shared": [], "A": []}
        for s in range(0, len(x), chunk_size):
            mean, omega, shared, A, _ = fedmgp_marginals(
                self.theta, self.phi, x[s:s + chunk_size]
            )
            out["mean"].append(mean.T)
            out["omega"].append(omega.T)
            out["shared"].append(shared.T)
            out["A"].append(A)
        return (
            torch.cat(out["mean"], 0),
            torch.cat(out["omega"], 0),
            torch.cat(out["shared"], 0),
            torch.cat(out["A"], -1),
        )

    @torch.no_grad()
    def predict(self, x):
        """Personalised predictive (Eq. 16): mean and variance, [N,T]."""
        mean, omega, shared, _ = self.marginals(x)
        return mean, omega + shared + self.noise.unsqueeze(0)


def train_fedmgp(x_train, y_train, seed, mode=None, epochs=None, lr=None):
    mode = mode or TRAINING_MODE
    epochs = int(epochs or TRAIN_EPOCHS)
    lr = float(lr or TRAIN_LR)
    if mode not in ("federated", "centralized"):
        raise ValueError(f"Unknown training mode: {mode}")

    set_all_seeds(seed)
    T = y_train.shape[1]
    theta, phi = init_fedmgp_params(
        x_train, T, NUM_LATENTS, NUM_INDUCING, seed
    )
    for v in phi.values():
        v.requires_grad_(True)

    step_per_eval = LOCAL_STEPS if mode == "federated" else 1
    n_evals = max(1, math.ceil(epochs / step_per_eval))
    patience = max(1, math.ceil(EARLY_STOP_PATIENCE / step_per_eval))
    min_evals = max(1, math.ceil(EARLY_STOP_MIN_EPOCHS / step_per_eval))

    best = (float("inf"), None, None)
    wait = 0
    history = []

    if mode == "centralized":
        for v in theta.values():
            v.requires_grad_(True)
        opt = torch.optim.Adam(list(theta.values()) + list(phi.values()), lr=lr)
    else:
        # Clients: local copies of theta (overwritten by every broadcast) and
        # personal phi_t.  phi_t always uses Adam with persistent state.
        local = {
            k: v.detach().expand(T, *v.shape[1:]).clone().requires_grad_(True)
            for k, v in theta.items()
        }
        phi_opt = torch.optim.Adam(list(phi.values()), lr=lr)
        if CLIENT_OPT == "adam":
            local_opt = torch.optim.Adam(list(local.values()), lr=lr)
        else:
            local_opt = torch.optim.SGD(list(local.values()), lr=CLIENT_SGD_LR)
        # Server: theta <- theta - eta * Delta (Algorithm 2, line 10), or an
        # Adam step that treats Delta as a gradient (FedOpt / FedAdam,
        # Reddi et al. 2021).  The aggregation itself is always Eq. (15).
        server_params = [theta[k].requires_grad_(False) for k in THETA_KEYS]
        server_opt = (
            torch.optim.Adam(server_params, lr=SERVER_LR)
            if SERVER_OPT == "adam" else
            torch.optim.SGD(server_params, lr=SERVER_LR)
        )

    for it in range(1, n_evals + 1):
        if mode == "centralized":
            opt.zero_grad(set_to_none=True)
            loss = negative_elbo_per_obs(theta, phi, x_train, y_train)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {it}")
            loss.backward()
            opt.step()
            with torch.no_grad():
                obj = float(negative_elbo_per_obs(theta, phi, x_train, y_train))
        else:
            # --- broadcast (line 2) ---
            with torch.no_grad():
                for k in THETA_KEYS:
                    local[k].copy_(theta[k].expand_as(local[k]))
            # --- client_update, all clients in parallel (lines 3-8) ---
            for _ in range(LOCAL_STEPS):
                local_opt.zero_grad(set_to_none=True)
                phi_opt.zero_grad(set_to_none=True)
                loss = federated_local_objective(local, phi, x_train, y_train)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss in round {it}")
                loss.backward()
                local_opt.step()
                phi_opt.step()
            # --- cloud_update: Delta = sum_t p_t (theta - theta_t), Eq. (15) ---
            with torch.no_grad():
                for k in THETA_KEYS:
                    theta[k].grad = (theta[k] - local[k]).mean(0, keepdim=True)
                server_opt.step()
                for k in THETA_KEYS:
                    theta[k].grad = None
                obj = float(negative_elbo_per_obs(theta, phi, x_train, y_train))

        history.append(obj)
        if obj < best[0] - EARLY_STOP_TOL:
            best = (obj, _clone_params(theta), _clone_params(phi))
            wait = 0
        else:
            wait += 1

        if it == 1 or it % max(1, 25 // step_per_eval) == 0:
            unit = "round" if mode == "federated" else "step"
            print(f"  {unit} {it:4d}/{n_evals} | -ELBO/obs = {obj:.5f} | best = {best[0]:.5f}")

        if it >= min_evals and wait >= patience:
            print(f"  early stop at {it}; best -ELBO/obs = {best[0]:.5f}")
            break

    if best[1] is None:
        best = (history[-1], _clone_params(theta), _clone_params(phi))
    return FedMGP(best[1], best[2]), {"history": history, "best": best[0]}


# =============================================================================
# METRICS
# =============================================================================

def calculate_mape(y_true, y_pred) -> float:
    """Elementwise MAPE as a fraction (0.05 = 5%)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), MAPE_EPS)))


def evaluate_test_metrics(model, x, y_raw, y_mean, y_std):
    mean_std, _ = model.predict(x)
    y_pred = (mean_std * y_std + y_mean).cpu().numpy()
    y_true = y_raw.cpu().numpy()
    return calculate_mape(y_true, y_pred), float(np.mean((y_true - y_pred) ** 2))


# =============================================================================
# ACQUISITION: JOINT BALD
# =============================================================================

def _feature_factors(model, A):
    """B_n = V(x_n) blockdiag(L): [N,T,I*J]."""
    L = model.L()                                                # [I,J,J]
    B = torch.einsum("tijn,ijk->ntik", A, L)
    N, T, I, J = B.shape
    return B.reshape(N, T, I * J)


@torch.no_grad()
def bald_scores(model, x_pool):
    noise = model.noise                                          # [T]
    scores = []
    for s in range(0, len(x_pool), PREDICT_CHUNK_SIZE):
        _, omega, _, A = model.marginals(x_pool[s:s + PREDICT_CHUNK_SIZE])
        D = omega + noise.unsqueeze(0)                           # [n,T]
        B = _feature_factors(model, A)                           # [n,T,K]
        K = B.shape[-1]
        G = torch.eye(K, dtype=B.dtype, device=B.device) + (
            B.transpose(-1, -2) @ (B / D.unsqueeze(-1))
        )
        logdet_G = torch.linalg.slogdet(G)[1]
        scores.append(0.5 * (torch.log(D / noise).sum(-1) + logdet_G))
    return torch.cat(scores)


def bald_select_batch(model, x_pool, batch_size):
    scores = bald_scores(model, x_pool)
    k = min(batch_size, len(scores))
    return torch.topk(scores, k=k).indices


# =============================================================================
# ACQUISITION: ACS-FW (shared posterior projections)
# =============================================================================

@torch.no_grad()
def acs_projections(model, x_pool, num_samples, generator):
    """
    L_hat [N, num_samples]:  L_n(g_j) = E_{y~p(y|x_n,D)} log p(y | x_n, g_j),
    with g_j ~ q(g) SHARED across all pool points, centred over j and scaled
    by 1/sqrt(num_samples).
    """
    mu = model.theta["mu"][0]                                    # [I,J]
    L = model.L()                                                # [I,J,J]
    eps = torch.randn(
        (num_samples, *mu.shape), generator=generator,
        dtype=mu.dtype, device="cpu",
    ).to(mu.device)
    g = mu.unsqueeze(0) + (L.unsqueeze(0) @ eps.unsqueeze(-1)).squeeze(-1)  # [S,I,J]

    noise = model.noise
    rows = []
    for s in range(0, len(x_pool), PREDICT_CHUNK_SIZE):
        mean, omega, shared, A = model.marginals(x_pool[s:s + PREDICT_CHUNK_SIZE])
        cond_var = omega + noise.unsqueeze(0)                    # [n,T]
        pred_var = cond_var + shared
        cond_mean = torch.einsum("tijn,sij->snt", A, g)          # [S,n,T]
        ell = -0.5 * (
            torch.log(2.0 * math.pi * cond_var).unsqueeze(0)
            + ((mean.unsqueeze(0) - cond_mean) ** 2 + pred_var.unsqueeze(0))
            / cond_var.unsqueeze(0)
        ).sum(-1)                                                # [S,n]
        ell = ell - ell.mean(0, keepdim=True)
        rows.append(ell.T / math.sqrt(num_samples))
    return torch.cat(rows, 0)


@torch.no_grad()
def acs_fw_select_batch(model, x_pool, batch_size, num_samples, generator):
    Lhat = acs_projections(model, x_pool, num_samples, generator)  # [N,S]
    N = Lhat.shape[0]
    b = min(batch_size, N)

    sigmas = Lhat.norm(dim=1).clamp_min(1e-12)
    sigma = sigmas.sum()
    L_full = Lhat.sum(0)

    w = torch.zeros(N, dtype=Lhat.dtype, device=Lhat.device)
    Lw = torch.zeros_like(L_full)

    for _ in range(ACS_MAX_ITERS_FACTOR * b):
        resid = L_full - Lw
        f = int(torch.argmax((Lhat @ resid) / sigmas))
        u = (sigma / sigmas[f]) * Lhat[f]
        d = u - Lw
        denom = float(d @ d)
        gamma = 0.0 if denom <= 1e-18 else float((d @ resid) / denom)
        gamma = min(max(gamma, 0.0), 1.0)
        w = (1.0 - gamma) * w
        w[f] += gamma * float(sigma / sigmas[f])
        Lw = (1.0 - gamma) * Lw + gamma * u
        if int((w > 0).sum()) >= b:
            break

    support = torch.nonzero(w > 0).flatten()
    if len(support) > b:
        support = support[torch.topk(w[support], b).indices]
    if len(support) < b:
        mask = torch.ones(N, dtype=torch.bool, device=w.device)
        mask[support] = False
        rest = torch.nonzero(mask).flatten()
        fill = rest[torch.topk(sigmas[rest], b - len(support)).indices]
        support = torch.cat([support, fill])
    return support


# =============================================================================
# PLOTTING
# =============================================================================

def save_all_strategy_mape_plot(scenario_name, seed_metrics_df, output_dir):
    fig, ax = plt.subplots(figsize=(10, 7))
    n_seeds = seed_metrics_df["seed"].nunique()
    for strategy in STRATEGIES:
        g = (
            seed_metrics_df[seed_metrics_df["strategy"] == strategy]
            .groupby("acquisition_round")["test_mape"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        se = (g["std"] / np.sqrt(g["count"])).fillna(0.0)
        ax.plot(g["acquisition_round"], g["mean"], marker="o", linewidth=2, label=strategy)
        ax.fill_between(g["acquisition_round"], g["mean"] - se, g["mean"] + se, alpha=0.2)

    ax.set_xlabel("Acquisition round")
    ax.set_ylabel(f"Test MAPE (mean ± 1 s.e. over {n_seeds} seeds)")
    ax.set_title(f"{scenario_name}: Test MAPE trajectories")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mape_trajectories_all_strategies.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_true_vs_predicted_grid(scenario_name, strategy_name, y_true_9, y_pred_9,
                                row_ids_9, output_dir):
    axis = np.arange(1, y_true_9.shape[1] + 1)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    axes = axes.ravel()
    for j, ax in enumerate(axes):
        ax.plot(axis, y_true_9[j], linewidth=2, label="True")
        ax.plot(axis, y_pred_9[j], linewidth=2, linestyle="--", label="Predicted")
        ax.set_title(f"Test row {int(row_ids_9[j])}")
        ax.grid(alpha=0.25)
        if j // 3 == 2:
            ax.set_xlabel("Response coordinate")
        if j % 3 == 0:
            ax.set_ylabel("Response")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(f"{scenario_name} - {strategy_name}: True vs Predicted (final model)",
                 y=0.995, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_dir / f"true_vs_predicted_9_curves_{strategy_name}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# ONE STRATEGY RUN
# =============================================================================

def run_strategy(strategy_name, X_all, Y_std, Y_raw, ctx_idx, pool_idx, test_idx,
                 y_mean, y_std, initial_model, train_seed, curve_positions=None,
                 scenario_name="", grid_dir=None):
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    print("\n" + "=" * 78 + f"\nRUNNING {strategy_name}\n" + "=" * 78)

    ctx = list(ctx_idx)
    pool = np.asarray(pool_idx).copy()
    x_test, y_test_raw = X_all[test_idx], Y_raw[test_idx]

    gen = torch.Generator(device="cpu").manual_seed(int(train_seed) + 7_777)
    model = initial_model
    trajectory = []

    def log_step(step):
        mape, mse = evaluate_test_metrics(model, x_test, y_test_raw, y_mean, y_std)
        trajectory.append({
            "strategy": strategy_name, "acquisition_round": step,
            "context_size": len(ctx), "pool_size": len(pool),
            "test_mape": mape, "test_mse": mse,
        })
        print(f"Round {step:3d} | context={len(ctx):4d} | pool={len(pool):4d} | "
              f"test={len(test_idx):4d} | MAPE={mape:.6f} | MSE={mse:.6f}")

    log_step(0)

    for acq in range(1, NUM_ACQUISITIONS + 1):
        x_pool = X_all[pool]
        if strategy_name == "BALD":
            sel = bald_select_batch(model, x_pool, ACQUISITION_BATCH_SIZE)
        else:
            sel = acs_fw_select_batch(model, x_pool, ACQUISITION_BATCH_SIZE, ACS_J, gen)
        sel = sel.cpu().numpy()

        ctx.extend(pool[sel].tolist())
        pool = np.delete(pool, sel)

        del model
        clear_torch_cache()
        model, _ = train_fedmgp(X_all[ctx], Y_std[ctx], seed=train_seed + acq)
        log_step(acq)

    if grid_dir is not None and curve_positions is not None:
        x_c = x_test[torch.as_tensor(curve_positions, device=X_all.device)]
        pred, _ = model.predict(x_c)
        y_pred_9 = (pred * y_std + y_mean).cpu().numpy()
        y_true_9 = y_test_raw[torch.as_tensor(curve_positions, device=X_all.device)].cpu().numpy()
        save_true_vs_predicted_grid(scenario_name, strategy_name, y_true_9, y_pred_9,
                                    np.asarray(test_idx)[curve_positions], grid_dir)

    del model
    clear_torch_cache()
    return trajectory


# =============================================================================
# ONE SCENARIO
# =============================================================================

def run_scenario(csv_path: Path, device: torch.device, num_seeds: int = None):
    scenario_name = csv_path.stem
    scenario_dir = RESULTS_ROOT / scenario_name
    num_seeds = int(num_seeds or DEFAULT_NUM_SEEDS)

    if num_seeds > len(NP_SPLIT_SEEDS):
        raise ValueError(f"num_seeds={num_seeds} > {len(NP_SPLIT_SEEDS)} defined split seeds.")

    if scenario_dir.exists():
        shutil.rmtree(scenario_dir)
    scenario_dir.mkdir(parents=True, exist_ok=True)

    print("\n\n" + "#" * 88)
    print(f"SCENARIO: {scenario_name}\nINPUT   : {csv_path}\nOUTPUT  : {scenario_dir}\nSEEDS   : {num_seeds}")
    print("#" * 88)

    X_cpu, Y_raw_cpu, meta = load_data(csv_path)
    X_all = X_cpu.to(device)
    Y_raw = Y_raw_cpu.to(device)

    budget = NUM_ACQUISITIONS * ACQUISITION_BATCH_SIZE
    rows = []

    for k, split_seed in enumerate(NP_SPLIT_SEEDS[:num_seeds]):
        train_seed = TRAIN_SEED + 1000 * k
        print("\n" + "=" * 88)
        print(f"SEED {k + 1}/{num_seeds} | split_seed={split_seed} | train_seed={train_seed}")
        print("=" * 88)

        ctx_idx, pool_idx, test_idx = create_al_split(
            X_all.shape[0], split_seed, INITIAL_CONTEXT_SIZE,
            TEST_FRACTION, budget, POOL_SIZE,
        )

        # Standardise with initial-context statistics only.
        y_ctx = Y_raw[ctx_idx]
        y_mean = y_ctx.mean(0, keepdim=True)
        y_std = y_ctx.std(0, keepdim=True).clamp_min(1e-8)
        Y_std = (Y_raw - y_mean) / y_std

        curve_positions = None
        if k == 0:
            rng = np.random.default_rng(split_seed + 90_009)
            curve_positions = np.sort(rng.choice(len(test_idx), size=9, replace=False))

        initial_model, _ = train_fedmgp(X_all[ctx_idx], Y_std[ctx_idx], seed=train_seed)

        for strategy in STRATEGIES:
            traj = run_strategy(
                strategy, X_all, Y_std, Y_raw, ctx_idx, pool_idx, test_idx,
                y_mean, y_std, FedMGP(initial_model.theta, initial_model.phi),
                train_seed, curve_positions, scenario_name,
                grid_dir=scenario_dir if k == 0 else None,
            )
            for r in traj:
                r["seed"] = int(split_seed)
            rows.extend(traj)

        del initial_model
        clear_torch_cache()

    seed_metrics_df = pd.DataFrame(rows)
    save_all_strategy_mape_plot(scenario_name, seed_metrics_df, scenario_dir)

    metrics_df = seed_metrics_df[
        ["seed", "strategy", "acquisition_round", "context_size",
         "pool_size", "test_mape", "test_mse"]
    ].sort_values(["seed", "strategy", "acquisition_round"]).reset_index(drop=True)
    metrics_df.to_csv(scenario_dir / "acquisition_metrics_all_strategies.csv", index=False)

    print("\nSaved:\n  - mape_trajectories_all_strategies.png\n"
          "  - acquisition_metrics_all_strategies.csv\n"
          + "".join(f"  - true_vs_predicted_9_curves_{s}.png\n" for s in STRATEGIES))

    del X_all, Y_raw
    clear_torch_cache()
    metrics_df.insert(0, "scenario", scenario_name)
    return metrics_df


# =============================================================================
# MAIN
# =============================================================================

def main(num_seeds: int = None):
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    validate_requested_files()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 88)
    print(f"FedMGP (Chung & Al Kontar 2024) + active learning | training: {TRAINING_MODE}")
    print("=" * 88)
    print(f"Base directory : {BASE_DIR}\nResults root   : {RESULTS_ROOT}\nDevice         : {device}")

    all_metrics = [
        run_scenario(BASE_DIR / f, device, num_seeds) for f in SCENARIO_FILES
    ]
    print("\n" + "=" * 88 + "\nALL SCENARIOS COMPLETE\n" + "=" * 88)
    print(f"Results saved under:\n{RESULTS_ROOT}")
    clear_torch_cache()
    return pd.concat(all_metrics, ignore_index=True)


# =============================================================================
# JUPYTER / M3 ENTRY POINT
# =============================================================================

def run_notebook_workflow(
    base_dir="./paper scenarios",
    scenario_files=None,
    acq_rounds=30,
    batch_size=5,
    save_dir=None,
    num_latents=1,
    num_inducing=30,
    train_epochs=1000,
    lr=0.05,
    acs_j=50,
    initial_context_size=140,
    split_seed=None,
    train_seed=220,
    gpu_id=0,
    expected_num_outputs=None,
    test_fraction=0.20,
    num_seeds=20,
    # --- new ---
    training_mode="federated",
    local_steps=1,
    server_lr=None,
    client_opt="sgd",
    client_sgd_lr=0.05,
    server_opt="adam",
    pool_size=None,
    continuous_features=(),
    response_start_index=8,
):
    """
    Jupyter / SMU-M3 entry point.  Existing calls keep working.

    New arguments
    -------------
    training_mode       "federated" (Algorithm 2) or "centralized"
    local_steps         E, local steps per communication round
    client_opt          local optimiser for theta copies: "sgd" or "adam"
    client_sgd_lr       local SGD step size for theta copies
    server_opt          "adam" (FedAdam on Delta) or "sgd" (theta -= eta*Delta)
    server_lr           server step size; None -> same as lr
    pool_size           cap on the acquisition pool (None = all remaining rows)
    continuous_features names/indices of inputs to scale instead of one-hot
    response_start_index first response column (default 8: column 7 skipped)

    `train_epochs` is the total number of local gradient steps per fit; in
    federated mode that is ceil(train_epochs / local_steps) rounds.
    `split_seed` is ignored (split seeds come from NP_SPLIT_SEEDS).
    """
    global BASE_DIR, SCENARIO_FILES, RESULTS_ROOT
    global INITIAL_CONTEXT_SIZE, NUM_ACQUISITIONS, ACQUISITION_BATCH_SIZE
    global TEST_FRACTION, POOL_SIZE, CONTINUOUS_FEATURES
    global NUM_LATENTS, NUM_INDUCING
    global TRAIN_EPOCHS, TRAIN_LR, ACS_J, TRAINING_MODE, LOCAL_STEPS, SERVER_LR
    global CLIENT_OPT, CLIENT_SGD_LR, SERVER_OPT
    global TRAIN_SEED, EXPECTED_NUM_OUTPUTS, DEFAULT_NUM_SEEDS, RESPONSE_START_INDEX

    if split_seed is not None:
        warnings.warn("split_seed is ignored; split seeds come from NP_SPLIT_SEEDS.")

    BASE_DIR = Path(base_dir)
    SCENARIO_FILES = list(scenario_files) if scenario_files is not None else [
        "scenario_000_HGR.csv", "scenario_016_damped.csv",
        "scenario_054_gompertz.csv", "scenario_179_sigmoid.csv",
    ]
    RESULTS_ROOT = Path(save_dir) if save_dir is not None else BASE_DIR / "fedmgp_results"

    INITIAL_CONTEXT_SIZE = int(initial_context_size)
    NUM_ACQUISITIONS = int(acq_rounds)
    ACQUISITION_BATCH_SIZE = int(batch_size)
    TEST_FRACTION = float(test_fraction)
    POOL_SIZE = None if pool_size is None else int(pool_size)
    CONTINUOUS_FEATURES = tuple(continuous_features)
    NUM_LATENTS = int(num_latents)
    NUM_INDUCING = int(num_inducing)
    TRAIN_EPOCHS = int(train_epochs)
    TRAIN_LR = float(lr)
    ACS_J = int(acs_j)
    TRAINING_MODE = str(training_mode)
    LOCAL_STEPS = int(local_steps)
    SERVER_LR = float(lr if server_lr is None else server_lr)
    CLIENT_OPT = str(client_opt)
    CLIENT_SGD_LR = float(client_sgd_lr)
    SERVER_OPT = str(server_opt)
    TRAIN_SEED = int(train_seed)
    DEFAULT_NUM_SEEDS = int(num_seeds)
    EXPECTED_NUM_OUTPUTS = None if expected_num_outputs is None else int(expected_num_outputs)
    RESPONSE_START_INDEX = int(response_start_index)

    checks = [
        (INITIAL_CONTEXT_SIZE >= 2, "initial_context_size must be >= 2."),
        (NUM_ACQUISITIONS >= 1, "acq_rounds must be >= 1."),
        (ACQUISITION_BATCH_SIZE >= 1, "batch_size must be >= 1."),
        (0.0 < TEST_FRACTION < 1.0, "test_fraction must be in (0, 1)."),
        (NUM_LATENTS >= 1, "num_latents must be >= 1."),
        (NUM_INDUCING >= 2, "num_inducing must be >= 2."),
        (TRAIN_EPOCHS >= 1, "train_epochs must be >= 1."),
        (TRAIN_LR > 0, "lr must be > 0."),
        (ACS_J >= 2, "acs_j must be >= 2."),
        (TRAINING_MODE in ("federated", "centralized"), "training_mode must be 'federated' or 'centralized'."),
        (LOCAL_STEPS >= 1, "local_steps must be >= 1."),
        (SERVER_LR > 0, "server_lr must be > 0."),
        (CLIENT_OPT in ("sgd", "adam"), "client_opt must be 'sgd' or 'adam'."),
        (SERVER_OPT in ("sgd", "adam"), "server_opt must be 'sgd' or 'adam'."),
        (CLIENT_SGD_LR > 0, "client_sgd_lr must be > 0."),
    ]
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)

    set_all_seeds(TRAIN_SEED)

    if torch.cuda.is_available():
        n_visible = torch.cuda.device_count()
        gpu_id = 0 if gpu_id is None else int(gpu_id)
        if not 0 <= gpu_id < n_visible:
            raise ValueError(f"gpu_id={gpu_id} invalid; visible CUDA devices: {n_visible}.")
        torch.cuda.set_device(gpu_id)
        print(f"Using GPU cuda:{gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
    else:
        print("CUDA is not available; running on CPU.")

    print("\nConfigured input files:")
    for f in SCENARIO_FILES:
        print(f"  {BASE_DIR / f}")
    print(f"\nResults directory  : {RESULTS_ROOT}")
    print(f"Acquisition        : {NUM_ACQUISITIONS} rounds x {ACQUISITION_BATCH_SIZE}")
    print(f"Initial context    : {INITIAL_CONTEXT_SIZE}")
    print(f"Test fraction      : {TEST_FRACTION}")
    print(f"Pool size cap      : {POOL_SIZE}")
    print(f"Latents I / J      : {NUM_LATENTS} / {NUM_INDUCING}")
    print(f"Training           : {TRAINING_MODE}, {TRAIN_EPOCHS} steps, lr={TRAIN_LR}, "
          f"E={LOCAL_STEPS}, client={CLIENT_OPT}, server={SERVER_OPT}(lr={SERVER_LR})\n")

    return main(num_seeds=DEFAULT_NUM_SEEDS)


if __name__ == "__main__":
    set_all_seeds(TRAIN_SEED)
    main()