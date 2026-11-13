#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WPVGP runner converted from WPVGP Runner.ipynb.

The notebook markdown refers to an older FedMGP workflow, but its
executable cells import and run the WPVGP module.
"""

def main():
    # --- Notebook markdown cell 0 ---
    # # FedMGP Shared-Latent GP — 144-output SMU M3 workflow
    #
    # This version expects **144 response columns** from `df.columns[8:]`.
    #
    # Place `fedmgp_shared_latent_active_learning_m3_144.py` in `./paper scenarios`.

    # --- Notebook code cell 1 ---
    import sys, importlib

    MODULE_DIR = "./paper scenarios"
    MODULE_NAME = "WPVGP"

    if MODULE_DIR in sys.path:
        sys.path.remove(MODULE_DIR)
    sys.path.insert(0, MODULE_DIR)

    if MODULE_NAME in sys.modules:
        del sys.modules[MODULE_NAME]

    import WPVGP as separable
    importlib.reload(separable)

    print("Imported from:", separable.__file__)
    print("Has run_notebook_workflow:", hasattr(separable, "run_notebook_workflow"))
    print("Expected outputs:", separable.EXPECTED_NUM_OUTPUTS)

    assert hasattr(separable, "run_notebook_workflow")
    assert separable.EXPECTED_NUM_OUTPUTS == 144

    # --- Notebook code cell 2 ---
    # Verify the four exact files and their response-column counts
    from pathlib import Path
    import pandas as pd

    BASE_DIR = Path("./paper scenarios")  # change to the folder you have put the data files
    # comment out your desired data file
    SCENARIO_FILES = [
       # "Scenario_000_burnupp.csv",  
       # "scenario_000_HGR.csv",
       # "scenario_016_damped.csv",
       # "scenario_054_gompertz.csv",
       # "scenario_179_sigmoid.csv",
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

    # --- Notebook code cell 3 ---
    import torch
    print("GPUs visible:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    # Notebook-only %matplotlib inline magic omitted in the .py script.

    # --- Notebook code cell 4 ---
    summary = separable.run_notebook_workflow(
        base_dir="./paper scenarios",
        scenario_files=[
           # "Scenario_000_burnupp.csv",
           # "scenario_000_HGR.csv",
            "scenario_016_damped.csv",
           # "scenario_054_gompertz.csv",
           # "scenario_179_sigmoid.csv",
        ],
        acq_rounds=20,
        batch_size=5,
        save_dir="./paper scenarios/WPVGP results", # change to your desired folder to keep the results
        num_latents=1,
        num_inducing=30,
        train_epochs=300,
        lr=0.03,
        acs_j=50,
        initial_context_size=140,
        split_seed=1,
        train_seed=220,
        gpu_id=0,
        expected_num_outputs=144,
        num_seeds=16,
    )

    print(summary)  # Display the summary previously shown by Jupyter.


if __name__ == "__main__":
    main()
