# -*- coding: utf-8 -*-
"""
Restored Older Neural Process workflow.

The Jupyter/M3 entry point run_notebook_workflow() uses the two intended
active-learning strategies independently for every seed:
    - ACS-FW
    - BALD

The original standalone random-acquisition main is retained below only as
legacy reference and is not executed when this module is imported.

    - Initial context size: 140
    - Acquisition rounds: 30
    - Acquisition batch size: 5 (unchanged)
    - Initial pool size: 30 * 5 = 150
    - All remaining observations are used as validation because the older
      workflow uses validation for early stopping.
    - No separate test holdout is used.
    - Round 0 MAPE is explicitly recorded before the first acquisition.

For a 468-point dataset:
    140 context + 150 pool + 178 validation = 468

The older NP mechanics are preserved:
    - columns A-G are categorical inputs
    - column H is skipped
    - columns I onward are response curves
    - one-hot encoding
    - global response standardization before the seed-specific split
    - NP hidden size 256, latent size 32
    - LR = 1e-3
    - 600 epochs
    - random conditioning subset during NP training
    - full labeled context used as the training target
    - Gaussian NLL summed over response dimensions
    - KL annealing 0 -> 1 over 100 epochs
    - softplus predictive variance
    - validation-MSE early stopping
    - model retrained from scratch after every acquisition
    - uniform random acquisition without replacement
    - original 20 seeds

IMPORTANT:
The validation set is used only for early stopping/model selection.
Validation points are never added to the acquisition pool.
"""

from __future__ import annotations

import copy
import glob
import os
import random
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from torch.distributions import Normal, kl_divergence


# ============================================================
# Configuration
# ============================================================

# -----------------------------
# Data file
# -----------------------------
DATA_FILE = r"./integrated_fuel_only_dataset.xlsx"

# For Colab/HGR, for example:
# DATA_FILE = "/content/upsampled_data_HGR w header.csv"


# -----------------------------
# Split / active-learning settings
# -----------------------------
INIT_CONTEXT_SIZE = 140

NUM_ACTIVE_ROUNDS = 30
ACQ_BATCH_SIZE = 5

# Exactly enough pool points for 30 full acquisitions of 5 points.
INITIAL_POOL_SIZE = NUM_ACTIVE_ROUNDS * ACQ_BATCH_SIZE  # 150

# There is intentionally NO separate test holdout in this requested version.
# Validation size is calculated automatically as:
#
#     N - INIT_CONTEXT_SIZE - INITIAL_POOL_SIZE
#
# Thus for N=468:
#
#     468 - 140 - 150 = 178 validation observations.


# -----------------------------
# Seeds -- unchanged from older workflow
# -----------------------------
SEEDS = [
    1, 7, 11, 19, 23,
    31, 37, 41, 55, 70,
    99, 105, 121, 135, 160,
    202, 82, 114, 303, 154,
]


# -----------------------------
# Neural Process settings -- unchanged
# -----------------------------
NP_EPOCHS = 600
NP_Z_DIM = 32
NP_HIDDEN = 256
NP_LR = 1e-3
NP_PATIENCE = 3
NP_EVAL_EVERY = 50
NP_DROP = 0.0
NP_BATCH_SIZE = 16

KL_WEIGHT_MAX = 1.0
KL_ANNEAL_EPOCHS = 100

NP_MIN_CONTEXT_SIZE = 4
NP_MAX_CONTEXT_FRAC = 0.90


# -----------------------------
# Evaluation / output
# -----------------------------
MAPE_DENOM_EPS = 1e-8
RUN_TAG = "older_np_random_140ctx_30rounds"
CHECKPOINT_EVERY_SEED = True


# ============================================================
# Step 0: Reproducibility
# ============================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Preserve older workflow behavior.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def set_np_fit_seed(seed: int, context_round: int) -> None:
    """Make each seed/trajectory-point NP fit reproducible."""
    fit_seed = 100000 + seed * 1000 + context_round

    random.seed(fit_seed)
    np.random.seed(fit_seed)
    torch.manual_seed(fit_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(fit_seed)
        torch.cuda.manual_seed_all(fit_seed)


# ============================================================
# Step 1: Data loading -- older layout preserved
# ============================================================

def read_csv_with_encoding_fallback(path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "cp1252", "latin1", "ISO-8859-1"]
    last_error = None

    for encoding in encodings:
        try:
            print(f"Trying encoding: {encoding}")
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is None:
        raise RuntimeError("CSV reading failed before an encoding was attempted.")

    raise UnicodeDecodeError(
        last_error.encoding,
        last_error.object,
        last_error.start,
        last_error.end,
        "Could not read CSV with common encodings.",
    )


def load_design_response_file(
    path: str,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    OneHotEncoder,
    pd.Index,
    pd.Index,
    pd.DataFrame,
]:
    """
    Older data-layout assumption:
        columns A-G : categorical inputs
        column  H   : skipped
        columns I+  : response curve
    """
    extension = os.path.splitext(path)[1].lower()

    if extension in [".xlsx", ".xls"]:
        dataframe = pd.read_excel(path)
    elif extension == ".csv":
        dataframe = read_csv_with_encoding_fallback(path)
    else:
        raise ValueError(
            f"Unsupported file type: {extension}. Use .xlsx, .xls, or .csv."
        )

    if dataframe.shape[1] < 9:
        raise ValueError(
            "Expected at least 9 columns: 7 features + skipped column + responses."
        )

    feature_columns = dataframe.columns[:7]
    response_columns = dataframe.columns[8:]

    print("\n" + "!" * 70)
    print("VERIFY COLUMN LAYOUT (older workflow: 7 features, 1 skipped, responses):")
    print(f"  Feature columns: {list(feature_columns)}")
    print(f"  Skipped column:  {dataframe.columns[7]}")
    print(
        f"  First responses: {list(response_columns[:5])} ... "
        f"({len(response_columns)} response columns total)"
    )
    print("!" * 70 + "\n")

    try:
        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
        )
    except TypeError:
        encoder = OneHotEncoder(
            sparse=False,
            handle_unknown="ignore",
        )

    encoded_x = encoder.fit_transform(
        dataframe[feature_columns].astype(str)
    )

    response_frame = dataframe[response_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if response_frame.isna().any().any():
        number_missing = int(response_frame.isna().sum().sum())
        print(
            f"Warning: {number_missing} NaN values after numeric conversion."
        )
        response_frame = (
            response_frame.ffill(axis=1)
            .bfill(axis=1)
            .fillna(0.0)
        )

    x_tensor = torch.tensor(
        encoded_x,
        dtype=torch.float32,
    )
    y_tensor = torch.tensor(
        response_frame.values,
        dtype=torch.float32,
    )

    return (
        x_tensor,
        y_tensor,
        encoder,
        feature_columns,
        response_columns,
        dataframe,
    )


# ============================================================
# Step 2: Older response standardization + requested split
# ============================================================

def standardize_responses(
    y: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Preserve the older workflow's global, per-response-position
    standardization.
    """
    y_mean = y.mean(dim=0, keepdim=True)
    y_std = y.std(dim=0, keepdim=True).clamp(min=eps)

    standardized = (y - y_mean) / y_std

    return standardized, y_mean, y_std


def unstandardize(
    standardized_y: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> torch.Tensor:
    return standardized_y * y_std + y_mean


def create_active_learning_split(
    x: torch.Tensor,
    y: torch.Tensor,
    init_context_size: int = INIT_CONTEXT_SIZE,
    initial_pool_size: int = INITIAL_POOL_SIZE,
    seed: int = 100,
    device: torch.device | None = None,
    fixed_holdout_indices=None,
) -> Dict[str, object]:
    """
    Requested split:

        context    = 140
        pool       = 30 * 5 = 150
        validation = everything else

    For N=468:
        140 context + 150 pool + 178 validation = 468

    Validation is required by the older NP training loop for early stopping,
    so the remaining observations are kept as validation rather than added
    to the pool.

    Context, pool, and validation are mutually disjoint.
    """
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    x = x.to(device)
    y = y.to(device)

    number_points = len(x)

    required_non_validation = init_context_size + initial_pool_size
    fixed_holdout_indices = np.asarray(
        [] if fixed_holdout_indices is None else fixed_holdout_indices,
        dtype=np.int64,
    )
    if len(np.unique(fixed_holdout_indices)) != len(fixed_holdout_indices):
        raise ValueError("Fixed holdout indices must be unique")
    if np.any((fixed_holdout_indices < 0) | (fixed_holdout_indices >= number_points)):
        raise ValueError("Fixed holdout indices are outside the dataset")
    eligible_indices = np.setdiff1d(np.arange(number_points), fixed_holdout_indices)

    if len(eligible_indices) <= required_non_validation:
        raise ValueError(
            f"Dataset has {number_points} observations, but the requested "
            f"configuration requires {init_context_size} context + "
            f"{initial_pool_size} pool = {required_non_validation} points "
            f"plus at least one validation point."
        )

    validation_size = len(eligible_indices) - required_non_validation

    rng = np.random.default_rng(seed)
    all_indices = eligible_indices

    # Preserve a fixed validation set for this seed.
    validation_indices = rng.choice(
        all_indices,
        size=validation_size,
        replace=False,
    )

    acquisition_eligible_indices = np.setdiff1d(
        all_indices,
        validation_indices,
    )

    if len(acquisition_eligible_indices) != required_non_validation:
        raise RuntimeError("Unexpected split-size mismatch.")

    # Initial labeled context.
    context_indices = rng.choice(
        acquisition_eligible_indices,
        size=init_context_size,
        replace=False,
    )

    # Every other acquisition-eligible point becomes the pool.
    pool_indices = np.setdiff1d(
        acquisition_eligible_indices,
        context_indices,
    )

    if len(pool_indices) != initial_pool_size:
        raise RuntimeError(
            f"Expected initial pool size {initial_pool_size}, "
            f"got {len(pool_indices)}."
        )

    return {
        "x_ctx": x[context_indices],
        "y_ctx": y[context_indices],

        "x_pool": x[pool_indices],
        "y_pool": y[pool_indices],

        "x_val": x[validation_indices],
        "y_val": y[validation_indices],

        "ctx_idx": context_indices,
        "pool_idx": pool_indices,
        "val_idx": validation_indices,
        "plot_idx": fixed_holdout_indices.copy(),
        "x_plot": x[fixed_holdout_indices],
        "y_plot": y[fixed_holdout_indices],
    }


# ============================================================
# Step 3: Neural Process -- older architecture preserved
# ============================================================

class NPResidualBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        drop: float = 0.0,
    ):
        super().__init__()

        self.fc1 = nn.Linear(dimension, dimension)
        self.ln1 = nn.LayerNorm(dimension)

        self.fc2 = nn.Linear(dimension, dimension)
        self.ln2 = nn.LayerNorm(dimension)

        self.activation = nn.ReLU()

        self.dropout = (
            nn.Dropout(drop)
            if drop > 0
            else nn.Identity()
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:

        output = self.activation(
            self.ln1(
                self.fc1(inputs)
            )
        )

        output = self.dropout(output)

        output = self.ln2(
            self.fc2(output)
        )

        return self.activation(
            output + inputs
        )


class Encoder(nn.Module):
    """
    Older mean-pooled set encoder for the global latent Gaussian.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        z_dim: int = NP_Z_DIM,
        drop: float = NP_DROP,
        hidden: int = NP_HIDDEN,
    ):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Linear(
                x_dim + y_dim,
                hidden,
            ),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        self.residual_1 = NPResidualBlock(
            hidden,
            drop,
        )
        self.residual_2 = NPResidualBlock(
            hidden,
            drop,
        )

        self.mean_network = nn.Linear(
            hidden,
            z_dim,
        )

        self.log_variance_network = nn.Linear(
            hidden,
            z_dim,
        )

    def embed(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:

        representation = torch.cat(
            [x, y],
            dim=-1,
        )

        representation = self.pre(
            representation
        )

        representation = self.residual_1(
            representation
        )

        representation = self.residual_2(
            representation
        )

        return representation

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        point_embeddings = self.embed(
            x,
            y,
        )

        pooled_embedding = point_embeddings.mean(
            dim=0,
            keepdim=True,
        )

        latent_mean = self.mean_network(
            pooled_embedding
        )

        latent_log_variance = (
            self.log_variance_network(
                pooled_embedding
            )
        )

        return (
            latent_mean,
            latent_log_variance,
        )


class Decoder(nn.Module):
    """
    Older Gaussian response-curve decoder.

    Note: predictive variance uses softplus exactly as in the older
    implementation, not exp().
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        z_dim: int = NP_Z_DIM,
        drop: float = NP_DROP,
        hidden: int = NP_HIDDEN,
    ):
        super().__init__()

        self.output_dimension = y_dim

        self.pre = nn.Sequential(
            nn.Linear(
                x_dim + z_dim,
                hidden,
            ),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        self.residual_1 = NPResidualBlock(
            hidden,
            drop,
        )

        self.residual_2 = NPResidualBlock(
            hidden,
            drop,
        )

        self.post = nn.Linear(
            hidden,
            y_dim * 2,
        )

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        if (
            z.shape[0] == 1
            and x.shape[0] > 1
        ):
            z = z.expand(
                x.shape[0],
                -1,
            )

        representation = torch.cat(
            [x, z],
            dim=-1,
        )

        representation = self.pre(
            representation
        )

        representation = self.residual_1(
            representation
        )

        representation = self.residual_2(
            representation
        )

        output = self.post(
            representation
        )

        predicted_mean = output[
            :,
            : self.output_dimension,
        ]

        predicted_variance = (
            F.softplus(
                output[
                    :,
                    self.output_dimension :,
                ]
            )
            + 1e-6
        )

        return (
            predicted_mean,
            predicted_variance,
        )


class NeuralProcess(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        z_dim: int = NP_Z_DIM,
        drop: float = NP_DROP,
        hidden: int = NP_HIDDEN,
    ):
        super().__init__()

        self.encoder = Encoder(
            x_dim=x_dim,
            y_dim=y_dim,
            z_dim=z_dim,
            drop=drop,
            hidden=hidden,
        )

        self.decoder = Decoder(
            x_dim=x_dim,
            y_dim=y_dim,
            z_dim=z_dim,
            drop=drop,
            hidden=hidden,
        )

    def latent_distribution(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
    ) -> Normal:

        latent_mean, latent_log_variance = (
            self.encoder(
                x_context,
                y_context,
            )
        )

        latent_std = (
            torch.exp(
                0.5
                * latent_log_variance.clamp(
                    -10,
                    10,
                )
            )
            + 1e-6
        )

        return Normal(
            latent_mean,
            latent_std,
        )

    def forward(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_target: torch.Tensor,
        mix: float = 0.0,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Normal,
    ]:

        latent_distribution = (
            self.latent_distribution(
                x_context,
                y_context,
            )
        )

        latent_sample = (
            latent_distribution.rsample()
        )

        latent = (
            (1.0 - mix)
            * latent_distribution.mean
            + mix
            * latent_sample
        )

        predicted_mean, predicted_variance = (
            self.decoder(
                x_target,
                latent,
            )
        )

        return (
            predicted_mean,
            predicted_variance,
            latent_distribution,
        )


# ============================================================
# Step 4: NP training -- older mechanics preserved
# ============================================================

def train_neural_process(
    model: NeuralProcess,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_validation: torch.Tensor,
    y_validation: torch.Tensor,
    epochs: int = NP_EPOCHS,
    learning_rate: float = NP_LR,
    evaluate_every: int = NP_EVAL_EVERY,
    patience: int = NP_PATIENCE,
    kl_weight_max: float = KL_WEIGHT_MAX,
    kl_anneal_epochs: int = KL_ANNEAL_EPOCHS,
    tolerance: float = 1e-4,
    batch_size: int = NP_BATCH_SIZE,
    minimum_context_size: int = NP_MIN_CONTEXT_SIZE,
    maximum_context_fraction: float = NP_MAX_CONTEXT_FRAC,
) -> NeuralProcess:
    """
    Preserve older NP training:

    - The full labeled context is the target set.
    - A random SUBSET of the labeled context conditions the NP at each step.
    - Gaussian NLL is SUMMED over response dimensions, then averaged over rows.
    - Global-latent KL is annealed from 0 to KL_WEIGHT_MAX.
    - Early stopping uses validation MSE with posterior-mean latent.
    """
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    number_context_points = len(
        x_context
    )

    steps_per_epoch = max(
        1,
        number_context_points // batch_size,
    )

    best_validation_loss = float(
        "inf"
    )

    best_state = None
    wait = 0

    minimum_size = max(
        1,
        min(
            minimum_context_size,
            number_context_points - 1,
        ),
    )

    maximum_size = max(
        minimum_size + 1,
        min(
            int(
                maximum_context_fraction
                * number_context_points
            ),
            number_context_points,
        ),
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        # Older latent sampling schedule:
        # stochastic early -> posterior mean late.
        latent_mix = (
            1.0
            - (epoch - 1)
            / max(
                1,
                epochs - 1,
            )
        )

        # Older KL annealing:
        # 0 -> KL_WEIGHT_MAX over KL_ANNEAL_EPOCHS.
        kl_weight = (
            kl_weight_max
            * min(
                1.0,
                epoch
                / max(
                    1,
                    kl_anneal_epochs,
                ),
            )
        )

        for _ in range(
            steps_per_epoch
        ):
            sampled_context_size = int(
                np.random.randint(
                    minimum_size,
                    maximum_size,
                )
            )

            permutation = torch.randperm(
                number_context_points,
                device=x_context.device,
            )

            conditioning_indices = (
                permutation[
                    :sampled_context_size
                ]
            )

            x_conditioning = x_context[
                conditioning_indices
            ]

            y_conditioning = y_context[
                conditioning_indices
            ]

            optimizer.zero_grad()

            (
                predicted_mean,
                predicted_variance,
                latent_distribution,
            ) = model(
                x_conditioning,
                y_conditioning,
                x_context,
                mix=latent_mix,
            )

            negative_log_likelihood = (
                0.5
                * (
                    torch.log(
                        predicted_variance
                    )
                    + (
                        y_context
                        - predicted_mean
                    ) ** 2
                    / predicted_variance
                )
            )

            # Older scaling: sum over response dimensions,
            # then average over observations.
            negative_log_likelihood = (
                negative_log_likelihood
                .sum(dim=1)
                .mean()
            )

            prior = Normal(
                torch.zeros_like(
                    latent_distribution.mean
                ),
                torch.ones_like(
                    latent_distribution.stddev
                ),
            )

            latent_kl = (
                kl_divergence(
                    latent_distribution,
                    prior,
                ).sum()
                / max(
                    1,
                    len(x_context),
                )
            )

            loss = (
                negative_log_likelihood
                + kl_weight
                * latent_kl
            )

            loss.backward()
            optimizer.step()

        if (
            epoch
            % evaluate_every
            == 0
        ):
            model.eval()

            with torch.no_grad():
                (
                    validation_mean,
                    _,
                    _,
                ) = model(
                    x_context,
                    y_context,
                    x_validation,
                    mix=0.0,
                )

                validation_mse = (
                    torch.mean(
                        (
                            validation_mean
                            - y_validation
                        ) ** 2
                    )
                    .item()
                )

            print(
                f"  NP ep {epoch:04d} "
                f"| val MSE: {validation_mse:.6f} "
                f"| kl_w: {kl_weight:.2f}"
            )

            if (
                validation_mse
                < best_validation_loss
                - tolerance
            ):
                best_validation_loss = (
                    validation_mse
                )

                best_state = copy.deepcopy(
                    model.state_dict()
                )

                wait = 0

            else:
                wait += 1

                if wait >= patience:
                    print(
                        "  NP early stop."
                    )
                    break

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    return model


def fit_neural_process(
    active_data: Dict[str, object],
    x_dim: int,
    y_dim: int,
    seed: int,
    context_round: int,
    device: torch.device,
) -> NeuralProcess:
    """
    Reinitialize and retrain the NP from scratch for one trajectory point.
    """
    set_np_fit_seed(
        seed,
        context_round,
    )

    model = NeuralProcess(
        x_dim=x_dim,
        y_dim=y_dim,
        z_dim=NP_Z_DIM,
        drop=NP_DROP,
        hidden=NP_HIDDEN,
    ).to(device)

    return train_neural_process(
        model=model,
        x_context=active_data["x_ctx"],       # type: ignore[arg-type]
        y_context=active_data["y_ctx"],       # type: ignore[arg-type]
        x_validation=active_data["x_val"],    # type: ignore[arg-type]
        y_validation=active_data["y_val"],    # type: ignore[arg-type]
    )


# ============================================================
# Step 5: Prediction + MAPE
# ============================================================

def calculate_mape(
    y_true: torch.Tensor,
    y_predicted: torch.Tensor,
    eps: float = MAPE_DENOM_EPS,
) -> float:

    denominator = torch.clamp(
        torch.abs(y_true),
        min=eps,
    )

    return float(
        torch.mean(
            torch.abs(
                (
                    y_true
                    - y_predicted
                )
                / denominator
            )
        ).item()
    )


@torch.no_grad()
def predict_np_mean(
    model: NeuralProcess,
    active_data: Dict[str, object],
    x_target: torch.Tensor,
) -> torch.Tensor:
    model.eval()

    x_context = active_data[
        "x_ctx"
    ]
    y_context = active_data[
        "y_ctx"
    ]

    if not isinstance(
        x_context,
        torch.Tensor,
    ):
        raise TypeError(
            "active_data['x_ctx'] must be a tensor."
        )

    if not isinstance(
        y_context,
        torch.Tensor,
    ):
        raise TypeError(
            "active_data['y_ctx'] must be a tensor."
        )

    latent_distribution = (
        model.latent_distribution(
            x_context,
            y_context,
        )
    )

    predicted_mean, _ = (
        model.decoder(
            x_target,
            latent_distribution.mean,
        )
    )

    return predicted_mean


@torch.no_grad()
def evaluate_set_mape(
    model: NeuralProcess,
    active_data: Dict[str, object],
    x_key: str,
    y_key: str,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> float:

    x_evaluation = active_data[
        x_key
    ]
    y_evaluation = active_data[
        y_key
    ]

    if not isinstance(
        x_evaluation,
        torch.Tensor,
    ):
        raise TypeError(
            f"active_data['{x_key}'] "
            "must be a tensor."
        )

    if not isinstance(
        y_evaluation,
        torch.Tensor,
    ):
        raise TypeError(
            f"active_data['{y_key}'] "
            "must be a tensor."
        )

    if len(x_evaluation) == 0:
        return float("nan")

    predicted_standardized = (
        predict_np_mean(
            model,
            active_data,
            x_evaluation,
        )
    )

    # Older workflow evaluated MAPE on the ORIGINAL response scale.
    y_true_original = unstandardize(
        y_evaluation,
        y_mean,
        y_std,
    )

    y_predicted_original = unstandardize(
        predicted_standardized,
        y_mean,
        y_std,
    )

    return calculate_mape(
        y_true_original,
        y_predicted_original,
    )


def record_mape_trajectory(
    model: NeuralProcess,
    active_data: Dict[str, object],
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    trajectory: Dict[
        str,
        List[float],
    ],
    round_id: int,
) -> Dict[
    str,
    List[float],
]:
    """
    Record MAPE at the current labeled-context state.

    round_id = 0 is the INITIAL / PRE-ACQUISITION MAPE.
    """

    context_mape = (
        evaluate_set_mape(
            model,
            active_data,
            "x_ctx",
            "y_ctx",
            y_mean,
            y_std,
        )
    )

    pool_mape = (
        evaluate_set_mape(
            model,
            active_data,
            "x_pool",
            "y_pool",
            y_mean,
            y_std,
        )
    )

    validation_mape = (
        evaluate_set_mape(
            model,
            active_data,
            "x_val",
            "y_val",
            y_mean,
            y_std,
        )
    )

    x_context = active_data[
        "x_ctx"
    ]
    x_pool = active_data[
        "x_pool"
    ]

    if not isinstance(
        x_context,
        torch.Tensor,
    ):
        raise TypeError(
            "active_data['x_ctx'] "
            "must be a tensor."
        )

    if not isinstance(
        x_pool,
        torch.Tensor,
    ):
        raise TypeError(
            "active_data['x_pool'] "
            "must be a tensor."
        )

    trajectory[
        "round"
    ].append(
        float(round_id)
    )

    trajectory[
        "context_size"
    ].append(
        float(
            len(x_context)
        )
    )

    trajectory[
        "pool_size"
    ].append(
        float(
            len(x_pool)
        )
    )

    trajectory[
        "context_mape"
    ].append(
        context_mape
    )

    trajectory[
        "pool_mape"
    ].append(
        pool_mape
    )

    trajectory[
        "validation_mape"
    ].append(
        validation_mape
    )

    prefix = (
        "INITIAL / PRE-ACQUISITION"
        if round_id == 0
        else f"ROUND {round_id}"
    )

    print(
        f"  {prefix}: "
        f"ctx_size={len(x_context)} "
        f"pool_size={len(x_pool)} | "
        f"ctx MAPE={context_mape:.4f} "
        f"pool MAPE={pool_mape:.4f} "
        f"val MAPE={validation_mape:.4f}"
    )

    return trajectory


# ============================================================
# Step 6: Uniform random acquisition -- unchanged
# ============================================================

def select_random_pool_points(
    active_data: Dict[str, object],
    batch_size: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """
    Older acquisition rule:
    uniform random selection without replacement.
    """
    x_pool = active_data[
        "x_pool"
    ]

    if not isinstance(
        x_pool,
        torch.Tensor,
    ):
        raise TypeError(
            "active_data['x_pool'] must be a tensor."
        )

    pool_size = len(
        x_pool
    )

    if pool_size == 0:
        raise ValueError(
            "Cannot select from an empty pool."
        )

    if pool_size < batch_size:
        raise ValueError(
            f"Pool has only {pool_size} points, "
            f"but batch size is {batch_size}."
        )

    selected_numpy = rng.choice(
        np.arange(pool_size),
        size=batch_size,
        replace=False,
    )

    return torch.tensor(
        selected_numpy,
        dtype=torch.long,
        device=x_pool.device,
    )


def reveal_and_update_context(
    active_data: Dict[str, object],
    selected_pool_indices: torch.Tensor,
) -> Tuple[
    Dict[str, object],
    Dict[str, object],
]:
    """
    Reveal selected pool labels, append them to the labeled context,
    and remove them from the acquisition pool.
    """
    x_context = active_data[
        "x_ctx"
    ]
    y_context = active_data[
        "y_ctx"
    ]
    x_pool = active_data[
        "x_pool"
    ]
    y_pool = active_data[
        "y_pool"
    ]

    context_indices = active_data[
        "ctx_idx"
    ]
    pool_indices = active_data[
        "pool_idx"
    ]

    if not all(
        isinstance(value, torch.Tensor)
        for value in [
            x_context,
            y_context,
            x_pool,
            y_pool,
        ]
    ):
        raise TypeError(
            "Context and pool entries must be tensors."
        )

    if not isinstance(
        context_indices,
        np.ndarray,
    ):
        raise TypeError(
            "active_data['ctx_idx'] must be a NumPy array."
        )

    if not isinstance(
        pool_indices,
        np.ndarray,
    ):
        raise TypeError(
            "active_data['pool_idx'] must be a NumPy array."
        )

    selected = selected_pool_indices.to(
        device=x_pool.device,
        dtype=torch.long,
    )

    selected_x = x_pool[
        selected
    ]
    selected_y = y_pool[
        selected
    ]

    selected_original_indices = (
        pool_indices[
            selected
            .detach()
            .cpu()
            .numpy()
        ]
    )

    keep_mask = torch.ones(
        len(x_pool),
        dtype=torch.bool,
        device=x_pool.device,
    )

    keep_mask[
        selected
    ] = False

    active_data[
        "x_ctx"
    ] = torch.cat(
        [
            x_context,
            selected_x,
        ],
        dim=0,
    )

    active_data[
        "y_ctx"
    ] = torch.cat(
        [
            y_context,
            selected_y,
        ],
        dim=0,
    )

    active_data[
        "x_pool"
    ] = x_pool[
        keep_mask
    ]

    active_data[
        "y_pool"
    ] = y_pool[
        keep_mask
    ]

    active_data[
        "ctx_idx"
    ] = np.concatenate(
        [
            context_indices,
            selected_original_indices,
        ]
    )

    active_data[
        "pool_idx"
    ] = pool_indices[
        keep_mask
        .detach()
        .cpu()
        .numpy()
    ]

    return (
        active_data,
        {
            "x_selected": selected_x,
            "y_selected": selected_y,
            "selected_pool_indices": selected,
            "selected_original_indices": selected_original_indices,
        },
    )


# ============================================================
# Step 7: Reporting helpers
# ============================================================

def summarize_trajectories(
    all_trajectories: pd.DataFrame,
) -> pd.DataFrame:

    return (
        all_trajectories
        .groupby(
            "round",
            as_index=False,
        )
        .agg(
            context_size_mean=(
                "context_size",
                "mean",
            ),
            pool_size_mean=(
                "pool_size",
                "mean",
            ),

            context_mape_mean=(
                "context_mape",
                "mean",
            ),
            pool_mape_mean=(
                "pool_mape",
                "mean",
            ),
            validation_mape_mean=(
                "validation_mape",
                "mean",
            ),

            context_mape_median=(
                "context_mape",
                "median",
            ),
            pool_mape_median=(
                "pool_mape",
                "median",
            ),
            validation_mape_median=(
                "validation_mape",
                "median",
            ),

            context_mape_std=(
                "context_mape",
                "std",
            ),
            pool_mape_std=(
                "pool_mape",
                "std",
            ),
            validation_mape_std=(
                "validation_mape",
                "std",
            ),

            num_seeds=(
                "seed",
                "nunique",
            ),
        )
    )


def build_initial_mape_table(
    all_trajectories: pd.DataFrame,
) -> pd.DataFrame:
    """
    Explicit table of PRE-ACQUISITION (round 0) MAPE.
    """
    initial = (
        all_trajectories[
            all_trajectories[
                "round"
            ] == 0
        ][
            [
                "seed",
                "round",
                "context_size",
                "pool_size",
                "context_mape",
                "pool_mape",
                "validation_mape",
            ]
        ]
        .copy()
        .sort_values(
            "seed"
        )
    )

    initial = initial.rename(
        columns={
            "round": "pre_acquisition_round",
            "context_mape": "initial_context_mape",
            "pool_mape": "initial_pool_mape",
            "validation_mape": "initial_validation_mape",
        }
    )

    return initial


def build_final_seed_summary(
    all_trajectories: pd.DataFrame,
) -> pd.DataFrame:

    ordered = (
        all_trajectories
        .sort_values(
            [
                "seed",
                "round",
            ]
        )
    )

    initial = (
        ordered
        .groupby(
            "seed",
            as_index=False,
        )
        .first()
    )

    final = (
        ordered
        .groupby(
            "seed",
            as_index=False,
        )
        .last()
    )

    summary = pd.DataFrame({
        "seed": initial["seed"],

        "initial_context_mape":
            initial["context_mape"],

        "final_context_mape":
            final["context_mape"],

        "initial_pool_mape":
            initial["pool_mape"],

        # At round 30 the pool is empty by design,
        # so final_pool_mape is expected to be NaN.
        "final_pool_mape":
            final["pool_mape"],

        "initial_validation_mape":
            initial["validation_mape"],

        "final_validation_mape":
            final["validation_mape"],
    })

    return summary


def plot_mean_mape_trajectory(
    summary: pd.DataFrame,
) -> None:

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        summary["round"],
        summary[
            "context_mape_mean"
        ],
        marker="o",
        label="Mean Context MAPE",
    )

    plt.plot(
        summary["round"],
        summary[
            "pool_mape_mean"
        ],
        marker="s",
        label="Mean Pool MAPE (moving pool)",
    )

    plt.plot(
        summary["round"],
        summary[
            "validation_mape_mean"
        ],
        marker="^",
        label="Mean Validation MAPE",
    )

    plt.xlabel(
        "Active-learning round"
    )

    plt.ylabel(
        "Mean MAPE across seeds"
    )

    plt.title(
        "Older Neural Process + Random Acquisition: Mean MAPE"
    )

    plt.xticks(
        summary["round"]
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_median_mape_trajectory(
    summary: pd.DataFrame,
) -> None:

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        summary["round"],
        summary[
            "context_mape_median"
        ],
        marker="o",
        label="Median Context MAPE",
    )

    plt.plot(
        summary["round"],
        summary[
            "pool_mape_median"
        ],
        marker="s",
        label="Median Pool MAPE (moving pool)",
    )

    plt.plot(
        summary["round"],
        summary[
            "validation_mape_median"
        ],
        marker="^",
        label="Median Validation MAPE",
    )

    plt.xlabel(
        "Active-learning round"
    )

    plt.ylabel(
        "Median MAPE across seeds"
    )

    plt.title(
        "Older Neural Process + Random Acquisition: Median MAPE"
    )

    plt.xticks(
        summary["round"]
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


def save_results_to_excel(
    excel_path: str,
    all_trajectories: pd.DataFrame,
    summary_trajectory: pd.DataFrame,
    initial_mape: pd.DataFrame,
    final_seed_summary: pd.DataFrame,
    selected_points: pd.DataFrame,
    config: pd.DataFrame,
) -> None:

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
    ) as writer:

        all_trajectories.to_excel(
            writer,
            sheet_name="all_seed_trajectories",
            index=False,
        )

        summary_trajectory.to_excel(
            writer,
            sheet_name="summary_trajectory",
            index=False,
        )

        initial_mape.to_excel(
            writer,
            sheet_name="initial_mape_round0",
            index=False,
        )

        final_seed_summary.to_excel(
            writer,
            sheet_name="final_seed_summary",
            index=False,
        )

        selected_points.to_excel(
            writer,
            sheet_name="selected_points",
            index=False,
        )

        config.to_excel(
            writer,
            sheet_name="config",
            index=False,
        )


# ============================================================
# Step 8: Main pipeline
# ============================================================

def _legacy_random_main() -> None:

    run_start = time.time()

    data_directory = os.path.dirname(
        DATA_FILE
    )

    output_directory = os.path.join(
        data_directory,
        RUN_TAG,
    )

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    csv_files = glob.glob(
        os.path.join(
            data_directory,
            "*.csv",
        )
    )

    print(
        "CSV files in folder:"
    )

    for file_path in csv_files:
        print(
            " ",
            os.path.basename(
                file_path
            ),
        )

    if not os.path.exists(
        DATA_FILE
    ):
        raise FileNotFoundError(
            f"Could not find data file: {DATA_FILE}"
        )

    (
        x_tensor,
        y_tensor,
        _encoder,
        feature_columns,
        response_columns,
        _raw_dataframe,
    ) = load_design_response_file(
        DATA_FILE
    )

    print(
        f"Loaded data: "
        f"X {tuple(x_tensor.shape)}, "
        f"Y {tuple(y_tensor.shape)}"
    )

    number_points = len(
        x_tensor
    )

    validation_size = (
        number_points
        - INIT_CONTEXT_SIZE
        - INITIAL_POOL_SIZE
    )

    if validation_size <= 0:
        raise ValueError(
            "Not enough observations for the requested "
            "context + pool + validation split."
        )

    print(
        "\nRequested split:"
    )

    print(
        f"  Total observations:   {number_points}"
    )

    print(
        f"  Initial context:      {INIT_CONTEXT_SIZE}"
    )

    print(
        f"  Initial pool:         {INITIAL_POOL_SIZE}"
    )

    print(
        f"  Validation:           {validation_size}"
    )

    print(
        "  Separate test set:    0"
    )

    print(
        f"  Acquisition rounds:   {NUM_ACTIVE_ROUNDS}"
    )

    print(
        f"  Batch size:           {ACQ_BATCH_SIZE}"
    )

    print(
        "  Round 0 MAPE:         YES "
        "(before first acquisition)"
    )

    expected_final_context = (
        INIT_CONTEXT_SIZE
        + NUM_ACTIVE_ROUNDS
        * ACQ_BATCH_SIZE
    )

    expected_final_pool = (
        INITIAL_POOL_SIZE
        - NUM_ACTIVE_ROUNDS
        * ACQ_BATCH_SIZE
    )

    print(
        f"  Final context size:   {expected_final_context}"
    )

    print(
        f"  Final pool size:      {expected_final_pool}"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nUsing device: {device}"
    )

    x_tensor = x_tensor.to(
        device
    )

    y_tensor = y_tensor.to(
        device
    )

    # PRESERVED FROM OLDER WORKFLOW:
    # standardize once before the seed-specific split.
    (
        y_standardized,
        y_mean,
        y_std,
    ) = standardize_responses(
        y_tensor
    )

    y_mean = y_mean.to(
        device
    )

    y_std = y_std.to(
        device
    )

    print(
        "\nOlder NP settings preserved:"
    )

    print(
        f"  NP epochs:             {NP_EPOCHS}"
    )

    print(
        f"  NP learning rate:      {NP_LR}"
    )

    print(
        f"  NP hidden size:        {NP_HIDDEN}"
    )

    print(
        f"  NP latent size:        {NP_Z_DIM}"
    )

    print(
        f"  NP batch size:         {NP_BATCH_SIZE}"
    )

    print(
        f"  KL max weight:         {KL_WEIGHT_MAX}"
    )

    print(
        f"  KL anneal epochs:      {KL_ANNEAL_EPOCHS}"
    )

    print(
        f"  Early-stop patience:   {NP_PATIENCE}"
    )

    print(
        "  Acquisition rule:     uniform random without replacement"
    )

    all_seed_trajectories: List[
        pd.DataFrame
    ] = []

    all_selected_points: List[
        Dict[str, object]
    ] = []

    for seed in SEEDS:

        print(
            "\n"
            + "#"
            * 80
        )

        print(
            f"Starting older NP workflow for seed = {seed}"
        )

        print(
            "#"
            * 80
        )

        set_global_seed(
            seed
        )

        acquisition_rng = (
            np.random.default_rng(
                seed
            )
        )

        active_data = (
            create_active_learning_split(
                x_tensor,
                y_standardized,
                init_context_size=INIT_CONTEXT_SIZE,
                initial_pool_size=INITIAL_POOL_SIZE,
                seed=seed,
                device=device,
            )
        )

        x_context = active_data[
            "x_ctx"
        ]

        x_pool = active_data[
            "x_pool"
        ]

        x_validation = active_data[
            "x_val"
        ]

        if not all(
            isinstance(
                value,
                torch.Tensor,
            )
            for value in [
                x_context,
                x_pool,
                x_validation,
            ]
        ):
            raise TypeError(
                "Malformed active-learning split."
            )

        print(
            "\nAL split for this seed:"
        )

        print(
            f"  Context: "
            f"X {tuple(x_context.shape)} "
            "(labeled)"
        )

        print(
            f"  Pool:    "
            f"X {tuple(x_pool.shape)} "
            "(labels hidden from training)"
        )

        print(
            f"  Val:     "
            f"X {tuple(x_validation.shape)} "
            "(early stopping only)"
        )

        x_dimension = (
            x_context.shape[1]
        )

        y_context = active_data[
            "y_ctx"
        ]

        if not isinstance(
            y_context,
            torch.Tensor,
        ):
            raise TypeError(
                "active_data['y_ctx'] "
                "must be a tensor."
            )

        y_dimension = (
            y_context.shape[1]
        )

        trajectory: Dict[
            str,
            List[float],
        ] = {
            "round": [],
            "context_size": [],
            "pool_size": [],
            "context_mape": [],
            "pool_mape": [],
            "validation_mape": [],
        }

        # ====================================================
        # ROUND 0 -- PRE-ACQUISITION BASELINE
        # ====================================================
        print(
            "\n"
            + "="
            * 70
        )

        print(
            "ROUND 0: PRE-ACQUISITION BASELINE"
        )

        print(
            "="
            * 70
        )

        model = (
            fit_neural_process(
                active_data=active_data,
                x_dim=x_dimension,
                y_dim=y_dimension,
                seed=seed,
                context_round=0,
                device=device,
            )
        )

        trajectory = (
            record_mape_trajectory(
                model=model,
                active_data=active_data,
                y_mean=y_mean,
                y_std=y_std,
                trajectory=trajectory,
                round_id=0,
            )
        )

        # ====================================================
        # 30 ACQUISITION ROUNDS
        # ====================================================
        for active_round in range(
            1,
            NUM_ACTIVE_ROUNDS + 1,
        ):

            print(
                "\n"
                + "="
                * 70
            )

            print(
                f"Seed {seed} | "
                f"Active-learning round "
                f"{active_round}/"
                f"{NUM_ACTIVE_ROUNDS}"
            )

            print(
                "="
                * 70
            )

            current_pool = (
                active_data[
                    "x_pool"
                ]
            )

            if not isinstance(
                current_pool,
                torch.Tensor,
            ):
                raise TypeError(
                    "active_data['x_pool'] "
                    "must be a tensor."
                )

            if (
                len(current_pool)
                < ACQ_BATCH_SIZE
            ):
                raise RuntimeError(
                    f"Pool has only "
                    f"{len(current_pool)} points "
                    f"before round {active_round}. "
                    "The requested configuration "
                    "should support all 30 rounds."
                )

            selected = (
                select_random_pool_points(
                    active_data=active_data,
                    batch_size=ACQ_BATCH_SIZE,
                    rng=acquisition_rng,
                )
            )

            pool_indices = (
                active_data[
                    "pool_idx"
                ]
            )

            if not isinstance(
                pool_indices,
                np.ndarray,
            ):
                raise TypeError(
                    "active_data['pool_idx'] "
                    "must be a NumPy array."
                )

            selected_local = (
                selected
                .detach()
                .cpu()
                .numpy()
            )

            selected_original = (
                pool_indices[
                    selected_local
                ]
            )

            print(
                "  Randomly selected "
                "pool indices: "
                f"{selected_local.tolist()}"
            )

            print(
                "  Corresponding "
                "original row indices: "
                f"{selected_original.tolist()}"
            )

            for (
                local_index,
                original_index,
            ) in zip(
                selected_local,
                selected_original,
            ):
                all_selected_points.append({
                    "seed": seed,
                    "active_round":
                        active_round,
                    "selection_rule":
                        "uniform_random_without_replacement",
                    "selected_pool_index_before_update":
                        int(local_index),
                    "selected_original_index":
                        int(original_index),
                })

            (
                active_data,
                _selected_batch,
            ) = reveal_and_update_context(
                active_data,
                selected,
            )

            # Older workflow: RETRAIN FROM SCRATCH after acquisition.
            model = (
                fit_neural_process(
                    active_data=active_data,
                    x_dim=x_dimension,
                    y_dim=y_dimension,
                    seed=seed,
                    context_round=active_round,
                    device=device,
                )
            )

            trajectory = (
                record_mape_trajectory(
                    model=model,
                    active_data=active_data,
                    y_mean=y_mean,
                    y_std=y_std,
                    trajectory=trajectory,
                    round_id=active_round,
                )
            )

        trajectory_frame = (
            pd.DataFrame(
                trajectory
            )
        )

        trajectory_frame[
            "seed"
        ] = seed

        all_seed_trajectories.append(
            trajectory_frame
        )

        print(
            "\nTrajectory for this seed:"
        )

        print(
            trajectory_frame
        )

        if CHECKPOINT_EVERY_SEED:

            checkpoint_frame = (
                pd.concat(
                    all_seed_trajectories,
                    ignore_index=True,
                )
            )

            checkpoint_frame.to_csv(
                os.path.join(
                    output_directory,
                    "checkpoint_trajectories.csv",
                ),
                index=False,
            )

            # Explicitly save round-0 MAPE after every completed seed.
            build_initial_mape_table(
                checkpoint_frame
            ).to_csv(
                os.path.join(
                    output_directory,
                    "checkpoint_initial_mape_round0.csv",
                ),
                index=False,
            )

            pd.DataFrame(
                all_selected_points
            ).to_csv(
                os.path.join(
                    output_directory,
                    "checkpoint_selected_points.csv",
                ),
                index=False,
            )

            print(
                f"  [checkpoint written "
                f"after seed {seed}]"
            )

    # ========================================================
    # Aggregate / save
    # ========================================================

    all_trajectories_frame = (
        pd.concat(
            all_seed_trajectories,
            ignore_index=True,
        )
    )

    summary_trajectory_frame = (
        summarize_trajectories(
            all_trajectories_frame
        )
    )

    initial_mape_frame = (
        build_initial_mape_table(
            all_trajectories_frame
        )
    )

    final_seed_summary_frame = (
        build_final_seed_summary(
            all_trajectories_frame
        )
    )

    selected_points_frame = (
        pd.DataFrame(
            all_selected_points
        )
    )

    config_rows = [
        {
            "parameter": "method",
            "value": RUN_TAG,
        },
        {
            "parameter": "surrogate",
            "value": "neural_process",
        },
        {
            "parameter": "acquisition_method",
            "value":
                "uniform_random_without_replacement",
        },
        {
            "parameter":
                "initial_context_size",
            "value":
                INIT_CONTEXT_SIZE,
        },
        {
            "parameter":
                "initial_pool_size",
            "value":
                INITIAL_POOL_SIZE,
        },
        {
            "parameter":
                "validation_size",
            "value":
                validation_size,
        },
        {
            "parameter":
                "separate_test_size",
            "value":
                0,
        },
        {
            "parameter":
                "num_active_rounds",
            "value":
                NUM_ACTIVE_ROUNDS,
        },
        {
            "parameter":
                "acq_batch_size",
            "value":
                ACQ_BATCH_SIZE,
        },
        {
            "parameter":
                "round0_pre_acquisition_mape",
            "value":
                True,
        },
        {
            "parameter":
                "num_seeds",
            "value":
                len(SEEDS),
        },
        {
            "parameter":
                "seeds",
            "value":
                str(SEEDS),
        },
        {
            "parameter":
                "np_epochs",
            "value":
                NP_EPOCHS,
        },
        {
            "parameter":
                "np_z_dim",
            "value":
                NP_Z_DIM,
        },
        {
            "parameter":
                "np_hidden",
            "value":
                NP_HIDDEN,
        },
        {
            "parameter":
                "np_learning_rate",
            "value":
                NP_LR,
        },
        {
            "parameter":
                "np_patience",
            "value":
                NP_PATIENCE,
        },
        {
            "parameter":
                "np_eval_every",
            "value":
                NP_EVAL_EVERY,
        },
        {
            "parameter":
                "np_dropout",
            "value":
                NP_DROP,
        },
        {
            "parameter":
                "np_batch_size",
            "value":
                NP_BATCH_SIZE,
        },
        {
            "parameter":
                "kl_weight_max",
            "value":
                KL_WEIGHT_MAX,
        },
        {
            "parameter":
                "kl_anneal_epochs",
            "value":
                KL_ANNEAL_EPOCHS,
        },
        {
            "parameter":
                "np_training_loss",
            "value":
                "Gaussian NLL summed over response dimensions + annealed global-latent KL",
        },
        {
            "parameter":
                "np_conditioning_scheme",
            "value":
                "random subset of labeled context predicts full labeled context",
        },
        {
            "parameter":
                "np_early_stopping",
            "value":
                "validation MSE at posterior-mean latent",
        },
        {
            "parameter":
                "response_standardization",
            "value":
                "global per-response-position standardization before split",
        },
        {
            "parameter":
                "predictive_variance",
            "value":
                "softplus(decoder variance head) + 1e-6",
        },
        {
            "parameter":
                "data_file",
            "value":
                DATA_FILE,
        },
        {
            "parameter":
                "feature_columns",
            "value":
                str(
                    list(
                        feature_columns
                    )
                ),
        },
        {
            "parameter":
                "response_column_count",
            "value":
                len(
                    response_columns
                ),
        },
        {
            "parameter":
                "total_runtime_hours",
            "value":
                (
                    time.time()
                    - run_start
                )
                / 3600.0,
        },
    ]

    config_frame = pd.DataFrame(
        config_rows
    )

    # Main CSV outputs.
    all_trajectories_frame.to_csv(
        os.path.join(
            output_directory,
            "all_seed_trajectories.csv",
        ),
        index=False,
    )

    summary_trajectory_frame.to_csv(
        os.path.join(
            output_directory,
            "summary_trajectory.csv",
        ),
        index=False,
    )

    # Explicit initial/pre-acquisition MAPE output.
    initial_mape_frame.to_csv(
        os.path.join(
            output_directory,
            "initial_mape_round0.csv",
        ),
        index=False,
    )

    final_seed_summary_frame.to_csv(
        os.path.join(
            output_directory,
            "final_seed_summary.csv",
        ),
        index=False,
    )

    selected_points_frame.to_csv(
        os.path.join(
            output_directory,
            "selected_points.csv",
        ),
        index=False,
    )

    # Excel workbook.
    excel_path = os.path.join(
        output_directory,
        f"{RUN_TAG}.xlsx",
    )

    save_results_to_excel(
        excel_path=excel_path,
        all_trajectories=all_trajectories_frame,
        summary_trajectory=summary_trajectory_frame,
        initial_mape=initial_mape_frame,
        final_seed_summary=final_seed_summary_frame,
        selected_points=selected_points_frame,
        config=config_frame,
    )

    print(
        "\nSaved outputs under:"
    )

    print(
        f"  {output_directory}"
    )

    print(
        "\nInitial / pre-acquisition MAPE "
        "(round 0):"
    )

    print(
        initial_mape_frame
    )

    print(
        "\nSummary MAPE trajectory "
        "across seeds:"
    )

    print(
        summary_trajectory_frame
    )

    plot_mean_mape_trajectory(
        summary_trajectory_frame
    )

    plot_median_mape_trajectory(
        summary_trajectory_frame
    )

    print(
        "\nCurrent working directory:"
    )

    print(
        os.getcwd()
    )


# The legacy random-acquisition standalone main is intentionally NOT executed.
# Use run_notebook_workflow() below; this is the experiment entry point used on M3.


# =============================================================================
# NOTEBOOK / M3 CPU-PARALLEL INTERFACE
# =============================================================================
# This interface is intentionally separate from main() above.  It makes this
# older NP workflow callable from the notebook runner used on M3:
#
#     run_notebook_workflow(
#         n_cpu_workers=8,
#         n_seeds=20,
#         acq_rounds=30,
#         initial_context_size=140,
#         lr=1e-3,
#         save_dir="...",
#     )
#
# Older NP mechanics are preserved:
#   * global response standardization before the seed-specific split
#   * random conditioning subset during NP fitting
#   * retrain from scratch at every trajectory point
#   * uniform-random acquisition without replacement
#   * round 0 MAPE before the first acquisition
#
# For each scenario, the acquisition pool is exactly:
#
#     acq_rounds * batch
#
# so 30 rounds x 5 points = 150 initial pool points.
#
# Every remaining observation is validation-only (early stopping).

import re as _runner_re
from pathlib import Path as _RunnerPath


RUNNER_SCENARIO_DIR = "./paper scenarios"


def _runner_list_scenario_csvs(folder: str):
    """Find the scenario CSVs used by the M3 paper workflow."""
    folder = os.path.abspath(os.path.expanduser(folder))

    if not os.path.isdir(folder):
        raise RuntimeError(f"Scenario directory not found: {folder}")

    pattern = _runner_re.compile(
        r"^scenario_(\d+)_(?:damped|sigmoid|HGR|d_sigmoid|double(?:_sigmoid)?|"
        r"sinusoid|burnup|gompertz)\.csv$",
        _runner_re.IGNORECASE,
    )

    found = []
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            found.append(
                (
                    int(match.group(1)),
                    os.path.join(folder, filename),
                )
            )

    # Sort by numeric scenario id and then filename so duplicate IDs such as
    # scenario_000_burnup / scenario_000_sigmoid remain deterministic.
    return sorted(found, key=lambda item: (item[0], os.path.basename(item[1]).lower()))


def _runner_read_dataframe(path: str) -> pd.DataFrame:
    extension = os.path.splitext(path)[1].lower()

    if extension in (".xlsx", ".xls"):
        return pd.read_excel(path)

    if extension == ".csv":
        return read_csv_with_encoding_fallback(path)

    raise ValueError(
        f"Unsupported file type {extension!r} for {path}. "
        "Use .csv, .xlsx, or .xls."
    )


def _runner_load_design_response_file(path: str):
    """
    Load a scenario robustly while preserving the older 7-categorical-feature
    input convention.

    The historical workflow used:
        A:G = inputs, H = skipped, I: = responses.

    The newer paper-scenario CSVs commonly use:
        first 7 columns = inputs, remaining 83 columns = responses.

    To avoid silently losing one response position, this loader detects the
    common 83-output layouts:
        total columns = 90 -> responses start at column index 7
        total columns = 91 -> column index 7 is skipped, responses start at 8

    For any other shape, it falls back to the historical "skip column H" layout.
    """
    dataframe = _runner_read_dataframe(path)

    if dataframe.shape[1] < 8:
        raise ValueError(
            f"{path} has only {dataframe.shape[1]} columns; expected at least "
            "7 categorical inputs plus response columns."
        )

    feature_columns = dataframe.columns[:7]
    number_columns = dataframe.shape[1]

    if number_columns - 7 == 83:
        response_start = 7
        skipped_column = None
    elif number_columns - 8 == 83:
        response_start = 8
        skipped_column = dataframe.columns[7]
    else:
        if number_columns < 9:
            raise ValueError(
                "Could not infer response columns and the historical layout "
                "(7 features + 1 skipped column + responses) is unavailable."
            )
        response_start = 8
        skipped_column = dataframe.columns[7]

    response_columns = dataframe.columns[response_start:]

    try:
        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
        )
    except TypeError:
        encoder = OneHotEncoder(
            sparse=False,
            handle_unknown="ignore",
        )

    encoded_x = encoder.fit_transform(
        dataframe[feature_columns].astype(str)
    )

    response_frame = dataframe[response_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if response_frame.isna().any().any():
        number_missing = int(response_frame.isna().sum().sum())
        print(
            f"Warning: {number_missing} NaNs after numeric conversion in "
            f"{os.path.basename(path)}; applying row-wise ffill/bfill then 0."
        )
        response_frame = (
            response_frame
            .ffill(axis=1)
            .bfill(axis=1)
            .fillna(0.0)
        )

    x_tensor = torch.tensor(encoded_x, dtype=torch.float32)
    y_tensor = torch.tensor(response_frame.values, dtype=torch.float32)

    layout_description = (
        "7 feature columns + responses"
        if skipped_column is None
        else f"7 feature columns + skipped '{skipped_column}' + responses"
    )

    return (
        x_tensor,
        y_tensor,
        encoder,
        feature_columns,
        response_columns,
        dataframe,
        layout_description,
    )


def _runner_fit_np(
    active_data,
    x_dim: int,
    y_dim: int,
    seed: int,
    context_round: int,
    device: torch.device,
    learning_rate: float,
    epochs: int,
):
    """Same old NP fit, but with notebook-supplied LR/epoch settings."""
    set_np_fit_seed(seed, context_round)

    model = NeuralProcess(
        x_dim=x_dim,
        y_dim=y_dim,
        z_dim=NP_Z_DIM,
        drop=NP_DROP,
        hidden=NP_HIDDEN,
    ).to(device)

    return train_neural_process(
        model=model,
        x_context=active_data["x_ctx"],
        y_context=active_data["y_ctx"],
        x_validation=active_data["x_val"],
        y_validation=active_data["y_val"],
        epochs=epochs,
        learning_rate=learning_rate,
        evaluate_every=NP_EVAL_EVERY,
        patience=NP_PATIENCE,
        kl_weight_max=KL_WEIGHT_MAX,
        kl_anneal_epochs=KL_ANNEAL_EPOCHS,
        batch_size=NP_BATCH_SIZE,
        minimum_context_size=NP_MIN_CONTEXT_SIZE,
        maximum_context_fraction=NP_MAX_CONTEXT_FRAC,
    )



# =============================================================================
# NOTEBOOK RUNNER ACQUISITION STRATEGIES: ACS-FW + BALD
# =============================================================================
# These selectors are used only by run_notebook_workflow().  They preserve the
# restored NP architecture/training/split mechanics above while replacing the
# temporary uniform-random notebook acquisition with the two intended active-
# learning strategies.  Both strategies start from the same seed-specific
# initial context/pool/validation split and then evolve independently.

_RUNNER_ACS_J = 50
_RUNNER_BALD_SAMPLES = 50
_RUNNER_ACQ_EPS = 1e-8


@torch.no_grad()
def _runner_bald_scores(
    model: NeuralProcess,
    active_data: Dict[str, object],
    num_samples: int = _RUNNER_BALD_SAMPLES,
) -> torch.Tensor:
    """
    NP BALD score used in the earlier project workflow.

    For every candidate x, draw latent samples z ~ q(z | context), decode a
    Gaussian response distribution for each z, and approximate

        I(y; z | x, C) = H[p(y | x, C)] - E_z H[p(y | x, z)].

    The marginal entropy is approximated with a moment-matched Gaussian over
    the latent-conditioned predictive Gaussians, exactly as in the earlier NP
    BALD implementation.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1 for BALD.")

    x_context = active_data["x_ctx"]
    y_context = active_data["y_ctx"]
    x_pool = active_data["x_pool"]

    if not all(
        isinstance(value, torch.Tensor)
        for value in (x_context, y_context, x_pool)
    ):
        raise TypeError("Context and pool entries must be tensors.")

    model.eval()
    device = x_pool.device
    scores: List[float] = []
    log_2pi = float(np.log(2.0 * np.pi))

    for x_candidate in x_pool:
        x_candidate = x_candidate.unsqueeze(0)

        latent_mean, latent_log_variance = model.encoder(
            x_context,
            y_context,
        )
        latent_std = (
            torch.exp(
                0.5 * latent_log_variance.clamp(-10.0, 10.0)
            )
            + 1e-6
        )

        latent_samples = Normal(
            latent_mean,
            latent_std,
        ).rsample((num_samples,)).squeeze(1)

        repeated_x = x_candidate.expand(num_samples, -1)
        predictive_mean, predictive_variance = model.decoder(
            repeated_x,
            latent_samples,
        )
        predictive_variance = predictive_variance.clamp_min(
            _RUNNER_ACQ_EPS
        )

        # E_z[H(y | x, z)] for diagonal Gaussian predictions.
        conditional_entropy = 0.5 * (
            log_2pi
            + torch.log(predictive_variance)
            + 1.0
        ).sum(dim=1).mean()

        # H(y | x, C) using the moment-matched Gaussian approximation to the
        # latent mixture.
        mixture_mean = predictive_mean.mean(dim=0)
        mixture_variance = (
            (predictive_variance + predictive_mean ** 2).mean(dim=0)
            - mixture_mean ** 2
        ).clamp_min(_RUNNER_ACQ_EPS)

        mixture_entropy = 0.5 * (
            log_2pi
            + torch.log(mixture_variance)
            + 1.0
        ).sum()

        scores.append(
            float((mixture_entropy - conditional_entropy).item())
        )

    return torch.tensor(
        scores,
        dtype=torch.float32,
        device=device,
    )


def _runner_bald_select_batch(
    model: NeuralProcess,
    active_data: Dict[str, object],
    batch_size: int,
    num_samples: int = _RUNNER_BALD_SAMPLES,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = _runner_bald_scores(
        model=model,
        active_data=active_data,
        num_samples=num_samples,
    )

    k = min(int(batch_size), len(scores))
    if k <= 0:
        return (
            torch.empty(0, dtype=torch.long, device=scores.device),
            scores,
        )

    selected = torch.topk(
        scores,
        k=k,
        largest=True,
    ).indices

    return selected, scores


@torch.no_grad()
def _runner_acs_fw_select_batch(
    model: NeuralProcess,
    active_data: Dict[str, object],
    batch_size: int,
    J: int = _RUNNER_ACS_J,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ACS-FW batch selection for the restored Neural Process.

    This follows the earlier NP ACS-FW implementation: sample J latent draws,
    build one log-likelihood feature vector V_m per pool candidate, and apply
    the Frank-Wolfe sparse approximation.  To preserve the requested fixed
    acquisition batch size, if the positive FW support contains fewer than k
    distinct candidates, the batch is deterministically supplemented with the
    largest-sigma candidates not already selected.
    """
    if J < 1:
        raise ValueError("J must be >= 1 for ACS-FW.")

    x_context = active_data["x_ctx"]
    y_context = active_data["y_ctx"]
    x_pool = active_data["x_pool"]

    if not all(
        isinstance(value, torch.Tensor)
        for value in (x_context, y_context, x_pool)
    ):
        raise TypeError("Context and pool entries must be tensors.")

    model.eval()
    device = x_pool.device
    number_candidates = len(x_pool)
    k = min(int(batch_size), number_candidates)

    if k <= 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )

    latent_mean, latent_log_variance = model.encoder(
        x_context,
        y_context,
    )
    latent_std = (
        torch.exp(
            0.5 * latent_log_variance.clamp(-10.0, 10.0)
        )
        + 1e-6
    )
    latent_samples = Normal(
        latent_mean,
        latent_std,
    ).sample((J,)).squeeze(1)

    V = torch.zeros(
        (number_candidates, J),
        dtype=torch.float32,
        device=device,
    )
    sigma = torch.zeros(
        number_candidates,
        dtype=torch.float32,
        device=device,
    )

    for candidate_index in range(number_candidates):
        x_candidate = x_pool[candidate_index].unsqueeze(0)
        repeated_x = x_candidate.expand(J, -1)

        predictive_mean, predictive_variance = model.decoder(
            repeated_x,
            latent_samples,
        )
        predictive_variance = predictive_variance.clamp_min(
            _RUNNER_ACQ_EPS
        )

        predictive_dist = Normal(
            predictive_mean,
            torch.sqrt(predictive_variance),
        )
        sampled_y = predictive_dist.sample()
        log_probabilities = predictive_dist.log_prob(
            sampled_y
        ).sum(dim=-1)

        V[candidate_index] = (
            log_probabilities / np.sqrt(float(J))
        )
        sigma[candidate_index] = torch.linalg.vector_norm(
            V[candidate_index],
            ord=2,
        )

    sigma_safe = sigma.clamp_min(_RUNNER_ACQ_EPS)
    sigma_total = sigma_safe.sum()

    weights = torch.zeros(
        number_candidates,
        dtype=torch.float32,
        device=device,
    )
    v_L = V.sum(dim=0)
    v_Lw = torch.zeros_like(v_L)

    for _ in range(k):
        residual = v_L - v_Lw
        inner_products = torch.mv(V, residual) / sigma_safe
        selected_index = int(torch.argmax(inner_products).item())

        u_f = (
            sigma_total / sigma_safe[selected_index]
        ) * V[selected_index]
        direction = u_f - v_Lw
        denominator = torch.dot(direction, direction)

        if float(denominator.item()) > _RUNNER_ACQ_EPS:
            gamma = torch.clamp(
                torch.dot(residual, direction) / denominator,
                min=0.0,
                max=1.0,
            )
        else:
            gamma = torch.tensor(
                0.0,
                dtype=torch.float32,
                device=device,
            )

        weights = (1.0 - gamma) * weights
        weights[selected_index] += (
            gamma * sigma_total / sigma_safe[selected_index]
        )
        v_Lw = (1.0 - gamma) * v_Lw + gamma * u_f

    positive_support = torch.nonzero(
        weights > 0,
        as_tuple=False,
    ).squeeze(1)

    if len(positive_support) >= k:
        local_top = torch.topk(
            weights[positive_support],
            k=k,
            largest=True,
        ).indices
        selected = positive_support[local_top]
    else:
        selected_list = positive_support.detach().cpu().tolist()
        already_selected = set(int(value) for value in selected_list)

        for candidate_index in torch.argsort(
            sigma,
            descending=True,
        ).detach().cpu().tolist():
            candidate_index = int(candidate_index)
            if candidate_index not in already_selected:
                selected_list.append(candidate_index)
                already_selected.add(candidate_index)
            if len(selected_list) == k:
                break

        selected = torch.tensor(
            selected_list,
            dtype=torch.long,
            device=device,
        )

    return selected, weights


def _runner_select_strategy_batch(
    strategy: str,
    model: NeuralProcess,
    active_data: Dict[str, object],
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if strategy == "BALD":
        return _runner_bald_select_batch(
            model=model,
            active_data=active_data,
            batch_size=batch_size,
            num_samples=_RUNNER_BALD_SAMPLES,
        )

    if strategy == "ACS_FW":
        return _runner_acs_fw_select_batch(
            model=model,
            active_data=active_data,
            batch_size=batch_size,
            J=_RUNNER_ACS_J,
        )

    raise ValueError(
        f"Unknown acquisition strategy: {strategy!r}. "
        "Expected 'ACS_FW' or 'BALD'."
    )

def _runner_run_one_seed(
    seed: int,
    x_tensor: torch.Tensor,
    y_standardized: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    initial_context_size: int,
    initial_pool_size: int,
    acq_rounds: int,
    batch: int,
    learning_rate: float,
    epochs: int,
    fixed_plot_indices,
    plot_seed_values,
):
    """
    Run BOTH acquisition strategies independently for one seed on CPU.

    Fair-comparison rule:
      * ACS_FW and BALD receive the exact same initial context, pool, and
        validation split for this seed.
      * Each strategy then evolves its own copy of the active-learning state.
      * The NP is retrained from scratch after every acquisition, exactly as in
        the restored workflow.
      * Round 0 is evaluated separately for each strategy, but because the
        split and fit seed are identical, the two round-0 states are directly
        comparable.
    """
    device = torch.device("cpu")

    # Each worker is kept single-threaded; parallelism is across workers.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    set_global_seed(seed)

    x_cpu = x_tensor.to(device)
    y_std_cpu = y_standardized.to(device)
    y_mean_cpu = y_mean.to(device)
    y_sd_cpu = y_std.to(device)

    # One seed-specific split, cloned for both strategies.
    initial_active_data = create_active_learning_split(
        x=x_cpu,
        y=y_std_cpu,
        init_context_size=initial_context_size,
        initial_pool_size=initial_pool_size,
        seed=seed,
        device=device,
        fixed_holdout_indices=fixed_plot_indices,
    )

    x_dimension = int(initial_active_data["x_ctx"].shape[1])
    y_dimension = int(initial_active_data["y_ctx"].shape[1])

    all_trajectory_records = []
    all_selected_points = []
    all_curve_records = []

    for strategy in ("ACS_FW", "BALD"):
        # Independent AL state, identical starting split.
        active_data = copy.deepcopy(initial_active_data)

        trajectory = {
            "round": [],
            "context_size": [],
            "pool_size": [],
            "context_mape": [],
            "pool_mape": [],
            "validation_mape": [],
        }

        print(
            f"\n[Seed {seed}] Starting independent {strategy} trajectory",
            flush=True,
        )

        # -------------------------------------------------------------
        # ROUND 0: same initial data for ACS_FW and BALD.
        # -------------------------------------------------------------
        model = _runner_fit_np(
            active_data=active_data,
            x_dim=x_dimension,
            y_dim=y_dimension,
            seed=seed,
            context_round=0,
            device=device,
            learning_rate=learning_rate,
            epochs=epochs,
        )

        trajectory = record_mape_trajectory(
            model=model,
            active_data=active_data,
            y_mean=y_mean_cpu,
            y_std=y_sd_cpu,
            trajectory=trajectory,
            round_id=0,
        )

        # -------------------------------------------------------------
        # Acquisition rounds 1..R for this strategy only.
        # -------------------------------------------------------------
        for active_round in range(1, acq_rounds + 1):
            current_pool = active_data["x_pool"]

            if not isinstance(current_pool, torch.Tensor):
                raise TypeError("active_data['x_pool'] must be a tensor.")

            if len(current_pool) < batch:
                raise RuntimeError(
                    f"Seed {seed} | {strategy}: pool has only "
                    f"{len(current_pool)} points before round {active_round}; "
                    f"batch={batch}. The runner constructs the initial pool "
                    "as acq_rounds * batch, so this indicates an internal "
                    "split/acquisition mismatch."
                )

            selected, acquisition_values = _runner_select_strategy_batch(
                strategy=strategy,
                model=model,
                active_data=active_data,
                batch_size=batch,
            )

            if len(selected) != batch:
                raise RuntimeError(
                    f"Seed {seed} | {strategy} | round {active_round}: "
                    f"expected {batch} selected points, got {len(selected)}."
                )

            pool_indices = active_data["pool_idx"]
            if not isinstance(pool_indices, np.ndarray):
                raise TypeError(
                    "active_data['pool_idx'] must be a NumPy array."
                )

            selected_local = selected.detach().cpu().numpy()
            selected_original = pool_indices[selected_local]

            if strategy == "BALD":
                selected_scores = acquisition_values[selected]
                score_lookup = selected_scores.detach().cpu().numpy()
            else:
                score_lookup = acquisition_values[selected].detach().cpu().numpy()

            for local_index, original_index, acquisition_value in zip(
                selected_local,
                selected_original,
                score_lookup,
            ):
                all_selected_points.append(
                    {
                        "seed": int(seed),
                        "strategy": strategy,
                        "active_round": int(active_round),
                        "selection_rule": strategy,
                        "selected_pool_index_before_update": int(local_index),
                        "selected_original_index": int(original_index),
                        "acquisition_value": float(acquisition_value),
                    }
                )

            active_data, _ = reveal_and_update_context(
                active_data,
                selected,
            )

            # Restored behavior: reinitialize + retrain from scratch after
            # every acquisition.  The fit seed is held the same across
            # strategies for a given seed/round; only the acquired data differ.
            model = _runner_fit_np(
                active_data=active_data,
                x_dim=x_dimension,
                y_dim=y_dimension,
                seed=seed,
                context_round=active_round,
                device=device,
                learning_rate=learning_rate,
                epochs=epochs,
            )

            trajectory = record_mape_trajectory(
                model=model,
                active_data=active_data,
                y_mean=y_mean_cpu,
                y_std=y_sd_cpu,
                trajectory=trajectory,
                round_id=active_round,
            )

        # Fixed nine experiments, shared by ALL seeds and BOTH acquisition strategies.
        # Save curves for the first n_plot_seeds completed seeds only.
        # These observations are never used in context, pool, or early stopping.
        if int(seed) in plot_seed_values:
            x_plot = active_data["x_plot"]
            y_plot = active_data["y_plot"]
            plot_idx = active_data["plot_idx"]
            model.eval()
            with torch.no_grad():
                pred_scaled = predict_np_mean(model, active_data, x_plot)
                true_values = unstandardize(y_plot, y_mean_cpu, y_sd_cpu).cpu().numpy()
                pred_values = unstandardize(pred_scaled, y_mean_cpu, y_sd_cpu).cpu().numpy()
            for plot_order, original_index in enumerate(plot_idx):
                for response_position, (true_value, predicted_value) in enumerate(
                    zip(true_values[plot_order], pred_values[plot_order])
                ):
                    all_curve_records.append({
                        "seed": int(seed),
                        "strategy": strategy,
                        "acquisition_round": int(acq_rounds),
                        "plot_order": int(plot_order),
                        "original_row_index": int(original_index),
                        "response_position": int(response_position),
                        "true": float(true_value),
                        "predicted": float(predicted_value),
                    })

        trajectory_frame = pd.DataFrame(trajectory)
        trajectory_frame["seed"] = int(seed)
        trajectory_frame["strategy"] = strategy

        all_trajectory_records.extend(
            trajectory_frame.to_dict(orient="records")
        )

    return (
        all_trajectory_records,
        all_selected_points,
        all_curve_records,
    )

def _runner_seed_chunk_worker(args):
    """
    One multiprocessing worker.  A worker receives several seeds and processes
    them sequentially, which is appropriate for 20 seeds across 8 CPU workers.
    """
    (
        worker_rank,
        cpu_id,
        seeds_subset,
        scenario_path,
        initial_context_size,
        initial_pool_size,
        acq_rounds,
        batch,
        learning_rate,
        epochs,
        fixed_plot_indices,
        plot_seed_values,
    ) = args

    # Force CPU execution even if the M3 node exposes GPUs.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if hasattr(os, "sched_setaffinity") and cpu_id is not None:
        try:
            os.sched_setaffinity(0, {int(cpu_id)})
        except (OSError, ValueError):
            pass

    print(
        f"[CPU worker {worker_rank}"
        + (f" | core {cpu_id}" if cpu_id is not None else "")
        + f"] {os.path.basename(scenario_path)} | seeds {seeds_subset}",
        flush=True,
    )

    (
        x_tensor,
        y_tensor,
        _encoder,
        _feature_columns,
        _response_columns,
        _dataframe,
        _layout,
    ) = _runner_load_design_response_file(scenario_path)

    y_standardized, y_mean, y_std = standardize_responses(y_tensor)

    trajectories = []
    selected = []
    curves = []

    for seed in seeds_subset:
        trajectory_records, selected_records, curve_records = _runner_run_one_seed(
            seed=seed,
            x_tensor=x_tensor,
            y_standardized=y_standardized,
            y_mean=y_mean,
            y_std=y_std,
            initial_context_size=initial_context_size,
            initial_pool_size=initial_pool_size,
            acq_rounds=acq_rounds,
            batch=batch,
            learning_rate=learning_rate,
            epochs=epochs,
            fixed_plot_indices=fixed_plot_indices,
            plot_seed_values=plot_seed_values,
        )
        trajectories.extend(trajectory_records)
        selected.extend(selected_records)
        curves.extend(curve_records)

    return {
        "worker_rank": worker_rank,
        "scenario_path": scenario_path,
        "trajectories": trajectories,
        "selected_points": selected,
        "validation_curves": curves,
    }


def _runner_make_seed_chunks(seed_list, number_workers: int):
    """Balanced chunks: for 20 seeds / 8 workers => 3,3,3,3,2,2,2,2."""
    chunks = [[] for _ in range(number_workers)]
    for index, seed in enumerate(seed_list):
        chunks[index % number_workers].append(seed)
    return chunks


def _runner_save_trajectory_plot(
    summary: pd.DataFrame,
    statistic: str,
    save_path: str,
    title_prefix: str,
):
    if statistic not in ("mean", "median"):
        raise ValueError("statistic must be 'mean' or 'median'")

    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    metric_specs = (
        (f"context_mape_{statistic}", "Context MAPE", "o"),
        (f"pool_mape_{statistic}", "Pool MAPE", "s"),
        (f"validation_mape_{statistic}", "Validation MAPE", "^"),
    )

    if "strategy" in summary.columns:
        for strategy in ("ACS_FW", "BALD"):
            strategy_summary = summary[
                summary["strategy"] == strategy
            ].sort_values("round")

            if strategy_summary.empty:
                continue

            for metric, label, marker in metric_specs:
                if metric in strategy_summary.columns:
                    ax.plot(
                        strategy_summary["round"],
                        strategy_summary[metric],
                        marker=marker,
                        markersize=3,
                        linewidth=1.0,
                        label=f"{strategy} — {label}",
                    )
    else:
        for metric, label, marker in metric_specs:
            if metric in summary.columns:
                ax.plot(
                    summary["round"],
                    summary[metric],
                    marker=marker,
                    markersize=3,
                    linewidth=1.0,
                    label=label,
                )

    ax.set_xlabel("Active-learning round (0 = pre-acquisition)")
    ax.set_ylabel(f"{statistic.title()} MAPE")
    ax.set_title(f"{title_prefix} — {statistic.title()} MAPE trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _runner_summarize_by_strategy(
    trajectories: pd.DataFrame,
) -> pd.DataFrame:
    parts = []
    for strategy in ("ACS_FW", "BALD"):
        subset = trajectories[
            trajectories["strategy"] == strategy
        ].copy()
        if subset.empty:
            continue
        part = summarize_trajectories(subset)
        part.insert(0, "strategy", strategy)
        parts.append(part)

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)


def _runner_initial_mape_by_strategy(
    trajectories: pd.DataFrame,
) -> pd.DataFrame:
    parts = []
    for strategy in ("ACS_FW", "BALD"):
        subset = trajectories[
            trajectories["strategy"] == strategy
        ].copy()
        if subset.empty:
            continue
        part = build_initial_mape_table(subset)
        part.insert(1, "strategy", strategy)
        parts.append(part)

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)


def _runner_final_seed_summary_by_strategy(
    trajectories: pd.DataFrame,
) -> pd.DataFrame:
    parts = []
    for strategy in ("ACS_FW", "BALD"):
        subset = trajectories[
            trajectories["strategy"] == strategy
        ].copy()
        if subset.empty:
            continue
        part = build_final_seed_summary(subset)
        part.insert(1, "strategy", strategy)
        parts.append(part)

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)


def _runner_build_attached_format_csv(
    trajectories: pd.DataFrame,
    scenario_path: str,
    seed_list,
) -> pd.DataFrame:
    """
    Build the exact compact schema used by the attached comparison CSV:

        scenario | seed_order | acq_step | strategy | context_mape | pool_mape

    Round 0 remains available in all_seed_trajectories.csv but is intentionally
    omitted here because the attached comparison file starts at acquisition 1.
    """
    stem = _RunnerPath(scenario_path).stem
    match = _runner_re.match(
        r"^scenario_(\d+)_",
        stem,
        _runner_re.IGNORECASE,
    )
    if match is None:
        raise ValueError(
            f"Could not extract numeric scenario id from {stem!r}."
        )

    scenario_id = int(match.group(1))
    seed_to_order = {
        int(seed): int(order)
        for order, seed in enumerate(seed_list)
    }

    compact = trajectories.loc[
        trajectories["round"] >= 1,
        [
            "seed",
            "round",
            "strategy",
            "context_mape",
            "pool_mape",
        ],
    ].copy()

    compact["scenario"] = scenario_id
    compact["seed_order"] = compact["seed"].map(seed_to_order)

    if compact["seed_order"].isna().any():
        missing = sorted(
            compact.loc[
                compact["seed_order"].isna(),
                "seed",
            ].unique().tolist()
        )
        raise ValueError(
            f"Could not map these seeds to seed_order: {missing}"
        )

    compact["seed_order"] = compact["seed_order"].astype(int)
    compact["acq_step"] = compact["round"].astype(int)

    compact = compact[
        [
            "scenario",
            "seed_order",
            "acq_step",
            "strategy",
            "context_mape",
            "pool_mape",
        ]
    ].sort_values(
        [
            "scenario",
            "seed_order",
            "strategy",
            "acq_step",
        ]
    ).reset_index(drop=True)

    return compact

def _runner_plot_validation_curves(curves: pd.DataFrame, output_dir: str) -> None:
    """One 3x3 plot per strategy, with pointwise mean predictions across seeds."""
    plot_dir = os.path.join(output_dir, "true_vs_predicted_validation")
    os.makedirs(plot_dir, exist_ok=True)
    for strategy, group in curves.groupby("strategy", sort=True):
        fig, axes = plt.subplots(3, 3, figsize=(9, 7), squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        experiment_ids = sorted(group["original_row_index"].unique())
        for plot_order, experiment_id in enumerate(experiment_ids):
            ax = axes.flat[plot_order]
            ax.set_visible(True)
            curve = group[group["original_row_index"] == experiment_id].sort_values("response_position")
            ax.plot(curve["response_position"], curve["true"], label="True", color="black", linewidth=1.6)
            ax.plot(curve["response_position"], curve["predicted_mean"], label="Predicted (seed mean)", color="tab:blue", linewidth=1.5, linestyle="--")
            ax.set_title(f"Experiment {int(experiment_id)} (n={int(curve['n_seeds'].min())})", fontsize=10)
            ax.set_xlabel("Response position")
            ax.grid(alpha=0.2)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.97))
        fig.suptitle(f"True vs predicted | {strategy} | pointwise seed average", y=1.01)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        for extension in ("pdf", "png"):
            fig.savefig(os.path.join(plot_dir, f"nine_curves_seed_average_{strategy}.{extension}"),
                        dpi=600 if extension == "png" else None, bbox_inches="tight")
        plt.close(fig)


def _runner_save_scenario_outputs(
    scenario_path: str,
    trajectories: pd.DataFrame,
    selected_points: pd.DataFrame,
    validation_curves: pd.DataFrame,
    save_root: str,
    initial_context_size: int,
    initial_pool_size: int,
    acq_rounds: int,
    batch: int,
    learning_rate: float,
    epochs: int,
    seed_list,
):
    stem = _RunnerPath(scenario_path).stem
    scenario_output_dir = os.path.join(save_root, stem)
    os.makedirs(scenario_output_dir, exist_ok=True)

    # Preserve plotted values for reproducibility and downstream analysis.
    validation_curves.to_csv(
        os.path.join(scenario_output_dir, "final_validation_true_vs_predicted.csv"),
        index=False,
    )
    # Verify eight distinct seed predictions per experiment/strategy/response point.
    group_keys = ["strategy", "original_row_index", "response_position"]
    counts = validation_curves.groupby(group_keys)["seed"].nunique()
    expected_plot_seeds = int(validation_curves["seed"].nunique())
    if not (counts == expected_plot_seeds).all():
        raise RuntimeError("Fixed validation curves do not have complete seed coverage")
    averaged_curves = (
        validation_curves.groupby(group_keys, as_index=False)
        .agg(true=("true", "mean"),
             predicted_mean=("predicted", "mean"),
             predicted_std=("predicted", "std"),
             n_seeds=("seed", "nunique"))
    )
    averaged_curves.to_csv(
        os.path.join(scenario_output_dir, "nine_validation_true_vs_pred_seed_averaged.csv"),
        index=False,
    )
    _runner_plot_validation_curves(averaged_curves, scenario_output_dir)

    summary = _runner_summarize_by_strategy(trajectories)
    initial_mape = _runner_initial_mape_by_strategy(trajectories)
    final_seed_summary = _runner_final_seed_summary_by_strategy(trajectories)
    attached_format = _runner_build_attached_format_csv(
        trajectories=trajectories,
        scenario_path=scenario_path,
        seed_list=seed_list,
    )

    scenario_match = _runner_re.match(
        r"^scenario_(\d+)_",
        stem,
        _runner_re.IGNORECASE,
    )
    if scenario_match is None:
        raise ValueError(
            f"Could not extract numeric scenario id from {stem!r}."
        )
    scenario_id = int(scenario_match.group(1))

    config = pd.DataFrame(
        [
            {"parameter": "surrogate", "value": "neural_process"},
            {"parameter": "acquisition_methods", "value": "ACS_FW;BALD"},
            {"parameter": "strategy_execution", "value": "independent_from_same_seed_split"},
            {"parameter": "acs_fw_J", "value": _RUNNER_ACS_J},
            {"parameter": "bald_latent_samples", "value": _RUNNER_BALD_SAMPLES},
            {"parameter": "initial_context_size", "value": initial_context_size},
            {"parameter": "initial_pool_size", "value": initial_pool_size},
            {"parameter": "num_active_rounds", "value": acq_rounds},
            {"parameter": "acq_batch_size", "value": batch},
            {"parameter": "round0_pre_acquisition_mape", "value": True},
            {"parameter": "num_seeds", "value": len(seed_list)},
            {"parameter": "seeds", "value": str(list(seed_list))},
            {"parameter": "np_epochs", "value": epochs},
            {"parameter": "np_learning_rate", "value": learning_rate},
            {"parameter": "np_hidden", "value": NP_HIDDEN},
            {"parameter": "np_z_dim", "value": NP_Z_DIM},
            {"parameter": "np_batch_size", "value": NP_BATCH_SIZE},
            {"parameter": "kl_weight_max", "value": KL_WEIGHT_MAX},
            {"parameter": "kl_anneal_epochs", "value": KL_ANNEAL_EPOCHS},
            {"parameter": "scenario_file", "value": scenario_path},
        ]
    )

    trajectories.to_csv(
        os.path.join(scenario_output_dir, "all_seed_trajectories.csv"),
        index=False,
    )
    summary.to_csv(
        os.path.join(scenario_output_dir, "summary_trajectory.csv"),
        index=False,
    )
    initial_mape.to_csv(
        os.path.join(scenario_output_dir, "initial_mape_round0.csv"),
        index=False,
    )
    final_seed_summary.to_csv(
        os.path.join(scenario_output_dir, "final_seed_summary.csv"),
        index=False,
    )
    selected_points.to_csv(
        os.path.join(scenario_output_dir, "selected_points.csv"),
        index=False,
    )
    config.to_csv(
        os.path.join(scenario_output_dir, "config.csv"),
        index=False,
    )

    # Primary compact output: same six columns/order as the attached file.
    attached_name = f"scenario_{scenario_id:03d}_per_seed_mape.csv"
    attached_format.to_csv(
        os.path.join(scenario_output_dir, attached_name),
        index=False,
    )

    _runner_save_trajectory_plot(
        summary,
        statistic="mean",
        save_path=os.path.join(scenario_output_dir, "mean_mape_trajectory.png"),
        title_prefix=stem,
    )
    _runner_save_trajectory_plot(
        summary,
        statistic="median",
        save_path=os.path.join(scenario_output_dir, "median_mape_trajectory.png"),
        title_prefix=stem,
    )

    try:
        save_results_to_excel(
            excel_path=os.path.join(
                scenario_output_dir,
                f"{stem}_older_np_ACSFW_BALD.xlsx",
            ),
            all_trajectories=trajectories,
            summary_trajectory=summary,
            initial_mape=initial_mape,
            final_seed_summary=final_seed_summary,
            selected_points=selected_points,
            config=config,
        )
    except Exception as exc:
        print(
            f"Warning: Excel workbook was not written for {stem}: {exc}. "
            "All CSV outputs were still saved.",
            flush=True,
        )

    print(
        f"Saved {stem}: independent ACS_FW + BALD trajectories, rounds "
        f"0 through {acq_rounds}, under {scenario_output_dir}",
        flush=True,
    )
    print(
        f"  Attached-format CSV: {attached_name}",
        flush=True,
    )

    return {
        "scenario": stem,
        "output_dir": scenario_output_dir,
        "summary": summary,
        "initial_mape": initial_mape,
        "comparison_trajectory": attached_format,
        "validation_curves_csv": os.path.join(scenario_output_dir, "final_validation_true_vs_predicted.csv"),
    }

def run_notebook_workflow(
    n_cpu_workers: int = 8,
    n_seeds: int = 20,
    scenario_dir: str = RUNNER_SCENARIO_DIR,
    acq_rounds: int = 30,
    initial_context_size: int = 140,
    batch: int = 5,
    lr: float = 1e-3,
    epochs: int = NP_EPOCHS,
    save_dir: str | None = None,
    scenario_ids=None,
    n_plot_seeds: int = 8,
    plot_selection_seed: int = 20261013,
):
    """
    Jupyter/M3 entry point expected by the user's runner.

    Default experiment:
        8 CPU workers
        20 historical seeds
        140 initial context points
        round 0 MAPE before acquisition
        30 acquisition rounds
        5 points per round, independently selected by ACS_FW and BALD
        150-point initial acquisition pool
        all remaining observations = validation
        LR = 1e-3

    Returns a dictionary keyed by scenario filename stem.
    """

    if n_cpu_workers < 1:
        raise ValueError("n_cpu_workers must be >= 1.")

    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1.")

    if n_seeds > len(SEEDS):
        raise ValueError(
            f"This restored workflow contains {len(SEEDS)} historical seeds; "
            f"requested n_seeds={n_seeds}."
        )

    if acq_rounds < 1:
        raise ValueError("acq_rounds must be >= 1.")

    if initial_context_size < 1:
        raise ValueError("initial_context_size must be >= 1.")

    if batch < 1:
        raise ValueError("batch must be >= 1.")

    if not (1 <= n_plot_seeds <= n_seeds):
        raise ValueError("n_plot_seeds must be between 1 and n_seeds")
    seed_list = list(SEEDS[:n_seeds])
    plot_seed_values = tuple(int(v) for v in seed_list[:n_plot_seeds])
    initial_pool_size = int(acq_rounds * batch)

    scenario_files = _runner_list_scenario_csvs(scenario_dir)

    if scenario_ids is not None:
        keep = set(int(value) for value in scenario_ids)
        scenario_files = [
            item for item in scenario_files
            if item[0] in keep
        ]

    if not scenario_files:
        raise RuntimeError(
            f"No matching scenario CSVs found in {scenario_dir!r}"
            + (
                f" for scenario_ids={scenario_ids}"
                if scenario_ids is not None
                else ""
            )
        )

    if save_dir is None or str(save_dir).strip() == "":
        save_root = os.path.join(
            os.environ.get("SCRATCH", os.getcwd()),
            "older_np_ACSFW_BALD_results",
            time.strftime("%Y%m%d_%H%M%S"),
        )
    else:
        save_root = os.path.abspath(
            os.path.expanduser(str(save_dir))
        )

    os.makedirs(save_root, exist_ok=True)

    # Determine CPUs visible to the current M3 allocation.
    if hasattr(os, "sched_getaffinity"):
        available_cpu_ids = sorted(os.sched_getaffinity(0))
    else:
        available_cpu_ids = list(range(os.cpu_count() or 1))

    number_workers = min(
        int(n_cpu_workers),
        len(seed_list),
        len(available_cpu_ids),
    )

    if number_workers < 1:
        raise RuntimeError("No CPU worker is available.")

    cpu_ids = available_cpu_ids[:number_workers]
    seed_chunks = _runner_make_seed_chunks(
        seed_list,
        number_workers,
    )

    print("\nRestored older NP workflow — independent ACS_FW + BALD")
    print(f"  scenario_dir:          {scenario_dir}")
    print(f"  scenarios:             {len(scenario_files)}")
    print(f"  CPU workers:           {number_workers}")
    print(f"  CPU IDs:               {cpu_ids}")
    print(f"  seeds:                 {seed_list}")
    print(f"  curve averaging seeds: {plot_seed_values}")
    print(f"  initial context:       {initial_context_size}")
    print(f"  initial pool:          {initial_pool_size}")
    print(f"  acquisition rounds:    {acq_rounds}")
    print(f"  batch size:            {batch}")
    print("  acquisition methods:   ACS_FW + BALD (independent per seed)")
    print(f"  learning rate:         {lr}")
    print("  round 0 MAPE:          YES")
    print(f"  save root:             {save_root}")

    # Validate every scenario before launching the expensive jobs.
    for sid, scenario_path in scenario_files:
        (
            x_check,
            y_check,
            _enc,
            _features,
            _responses,
            _df,
            layout,
        ) = _runner_load_design_response_file(scenario_path)

        required = initial_context_size + initial_pool_size + 9 + 1

        if len(x_check) < required:
            raise ValueError(
                f"{os.path.basename(scenario_path)} has {len(x_check)} rows, but "
                f"this experiment needs at least {required}: "
                f"{initial_context_size} context + {initial_pool_size} pool + "
                "at least 1 validation row."
            )

        print(
            f"  {os.path.basename(scenario_path)}: "
            f"N={len(x_check)}, Xdim={x_check.shape[1]}, Ydim={y_check.shape[1]}, "
            f"validation={len(x_check)-initial_context_size-initial_pool_size}, "
            f"layout={layout}"
        )

    results_by_scenario = {}

    # Process scenarios one at a time.  Within each scenario, seed chunks run
    # concurrently across the requested CPU workers.
    for sid, scenario_path in scenario_files:
        stem = _RunnerPath(scenario_path).stem

        print("\n" + "=" * 88)
        print(f"SCENARIO: {stem}")
        print("=" * 88)

        # Choose nine original dataset rows ONCE, outside all seed-specific splits.
        # Every seed excludes these rows from context/pool/early stopping.
        scenario_frame = _runner_read_dataframe(scenario_path)
        if len(scenario_frame) < initial_context_size + initial_pool_size + 10:
            raise ValueError("Insufficient observations for nine shared holdouts")
        plot_rng = np.random.default_rng(int(plot_selection_seed))
        fixed_plot_indices = np.sort(plot_rng.choice(len(scenario_frame), size=9, replace=False))
        print(f"  Shared nine experiment row indices: {fixed_plot_indices.tolist()}")
        tasks = []
        for rank, seeds_subset in enumerate(seed_chunks):
            if not seeds_subset:
                continue

            tasks.append(
                (
                    rank,
                    cpu_ids[rank],
                    seeds_subset,
                    scenario_path,
                    initial_context_size,
                    initial_pool_size,
                    acq_rounds,
                    batch,
                    lr,
                    epochs,
                    fixed_plot_indices,
                    plot_seed_values,
                )
            )

        ctx = torch.multiprocessing.get_context("spawn")

        with ctx.Pool(processes=len(tasks)) as pool:
            worker_results = pool.map(
                _runner_seed_chunk_worker,
                tasks,
            )

        trajectory_records = []
        selected_records = []
        curve_records = []

        for worker_result in worker_results:
            trajectory_records.extend(
                worker_result["trajectories"]
            )
            selected_records.extend(
                worker_result["selected_points"]
            )
            curve_records.extend(worker_result["validation_curves"])

        pd.DataFrame({"plot_order": np.arange(9), "original_row_index": fixed_plot_indices}).to_csv(
            os.path.join(save_root, f"{stem}_nine_shared_experiment_ids.csv"), index=False
        )
        trajectories = pd.DataFrame(trajectory_records)
        selected_points = pd.DataFrame(selected_records)
        validation_curves = pd.DataFrame(curve_records)

        if trajectories.empty:
            raise RuntimeError(
                f"No trajectory results were returned for {stem}."
            )

        # Keep deterministic order in all saved files.
        trajectories = trajectories.sort_values(
            ["seed", "strategy", "round"]
        ).reset_index(drop=True)

        if not selected_points.empty:
            selected_points = selected_points.sort_values(
                ["seed", "strategy", "active_round", "selected_pool_index_before_update"]
            ).reset_index(drop=True)

        results_by_scenario[stem] = _runner_save_scenario_outputs(
            scenario_path=scenario_path,
            trajectories=trajectories,
            selected_points=selected_points,
            validation_curves=validation_curves,
            save_root=save_root,
            initial_context_size=initial_context_size,
            initial_pool_size=initial_pool_size,
            acq_rounds=acq_rounds,
            batch=batch,
            learning_rate=lr,
            epochs=epochs,
            seed_list=seed_list,
        )

    print("\nAll requested scenarios completed.")
    return results_by_scenario

