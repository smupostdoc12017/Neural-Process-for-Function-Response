# -*- coding: utf-8 -*-
"""FedMGP experiment runner converted from FedMGP Runner.ipynb.

The original notebook code, arguments, and scenario selections are retained.
Jupyter-only magics are omitted. The main guard prevents automatic re-execution
when Python multiprocessing starts new workers.
"""


def main():
    # Notebook notes:
    # # FedMGP Shared-Latent GP — 144-output SMU M3 workflow
    #
    # This version expects **144 response columns** from `df.columns[8:]`.
    #
    # Place `fedmgp_shared_latent_active_learning_m3_144.py` in `./paper scenarios`.

    # Cell 1
    import sys, importlib

    MODULE_DIR = "./paper scenarios"
    MODULE_NAME = "FedMGP"

    if MODULE_DIR in sys.path:
        sys.path.remove(MODULE_DIR)
    sys.path.insert(0, MODULE_DIR)

    if MODULE_NAME in sys.modules:
        del sys.modules[MODULE_NAME]

    import FedMGP as fedmgp
    importlib.reload(fedmgp)

    print("Imported from:", fedmgp.__file__)
    print("Has run_notebook_workflow:", hasattr(fedmgp, "run_notebook_workflow"))
    print("Expected outputs:", fedmgp.EXPECTED_NUM_OUTPUTS)

    assert hasattr(fedmgp, "run_notebook_workflow")
    #assert fedmgp.EXPECTED_NUM_OUTPUTS == 144

    # Cell 2
    # Verify the four exact files and their response-column counts
    from pathlib import Path
    import pandas as pd

    BASE_DIR = Path("./paper scenarios") # change this to the place you put the data files. Comment out the scenario you want to run. 
    SCENARIO_FILES = [
       # "Scenario_000_burnupp.csv",
      #  "scenario_000_HGR.csv",
      #  "scenario_016_damped.csv",
       # "scenario_054_gompertz.csv",
       # "scenario_179_sigmoid.csv"
        #,
    ]

    for filename in SCENARIO_FILES:
        path = BASE_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        header = pd.read_csv(path, nrows=0)
        n_outputs = len(header.columns[7:])
        print(f"{filename}: FOUND | response columns = {n_outputs}")
        if n_outputs != 145:
            raise ValueError(
                f"{filename}: expected 145 response columns, found {n_outputs}"
            )

    # Cell 3
    import torch
    print("GPUs visible:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Cell 4
    summary = fedmgp.run_notebook_workflow(
        base_dir="./paper scenarios",
        scenario_files=[
         #  "Scenario_000_burnupp.csv"
          #  "scenario_000_HGR.csv",
            "scenario_016_damped.csv",
            "scenario_054_gompertz.csv",
            "scenario_179_sigmoid.csv"
            #,
        ],
        acq_rounds=20,
        batch_size=5,
        save_dir="./paper scenarios/synthetic",   # change this to your own folder
        num_latents=1,
        num_inducing=30,
        acs_j=50,
        initial_context_size=140,
        train_seed=220,
        gpu_id=0,
        response_start_index=7,
        expected_num_outputs=145,
    )


if __name__ == "__main__":
    main()
