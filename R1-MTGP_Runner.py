# -*- coding: utf-8 -*-
"""
Runner for R1_MTGP.py (R1-MTGP): 5 scenarios, 16 seeds, 8 parallel workers.
"""

import sys, importlib

MODULE_DIR = "./Scenarios to Run"
MODULE_NAME = "R1_MTGP"

if MODULE_DIR in sys.path:
    sys.path.remove(MODULE_DIR)
sys.path.insert(0, MODULE_DIR)

if MODULE_NAME in sys.modules:
    del sys.modules[MODULE_NAME]

base = importlib.import_module(MODULE_NAME)

print("Imported from:", base.__file__)
print("Has run_notebook_workflow:", hasattr(base, "run_notebook_workflow"))
print("Expected outputs:", base.EXPECTED_NUM_OUTPUTS)

assert hasattr(base, "run_notebook_workflow")
assert base.EXPECTED_NUM_OUTPUTS == 144


# Verify the scenario files and their response-column counts
from pathlib import Path
import pandas as pd

BASE_DIR = Path("./Scenarios to Run")
SCENARIO_FILES = [
    "scenario_000_burnup.csv",
    "scenario_000_HGR.csv",
    "scenario_016_damped.csv",
    "scenario_054_gompertz.csv",
    "scenario_179_sigmoid.csv",
]

for filename in SCENARIO_FILES:
    path = BASE_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0)
    n_outputs = len(header.columns[8:])
    print(f"{filename}: FOUND | response columns = {n_outputs}")
    if n_outputs != 144:
        raise ValueError(
            f"{filename}: expected 144 response columns, found {n_outputs}"
        )


import torch
print("GPUs visible:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
%matplotlib inline


# Do NOT pass gpu_id (forces one sequential worker), num_latents,
# num_inducing, split_seed or train_seed (the module raises TypeError).
summary = base.run_notebook_workflow(
    base_dir=str(BASE_DIR),
    scenario_files=SCENARIO_FILES,
    n_gpus=8,                  # 8 parallel workers (CPU workers if no GPU is visible)
    n_seeds=16,                # seeds 100..115, 2 per worker
    acq_rounds=20,
    batch_size=5,
    epochs=300,                # train_epochs also accepted as an alias
    lr=0.03,
    acs_j=50,
    initial_context_size=64,
    expected_num_outputs=144,
    save_dir="./base results 16",
)

summary
