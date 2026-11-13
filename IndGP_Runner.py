# -*- coding: utf-8 -*-
"""IndGP runner converted from IndGP Runner(1).ipynb.

Experimental settings are preserved from the notebook.
Run: python IndGP_Runner_Damped.py
"""

def main():

    import sys
    import importlib
    from pathlib import Path

    # ============================================================
    # IMPORT IndGP MODULE
    # ============================================================

    MODULE_DIR = "./paper scenarios"
    MODULE_NAME = "IndGP"

    if MODULE_DIR in sys.path:
        sys.path.remove(MODULE_DIR)

    sys.path.insert(0, MODULE_DIR)

    if MODULE_NAME in sys.modules:
        del sys.modules[MODULE_NAME]

    IndGP = importlib.import_module(MODULE_NAME)

    print("Imported from:", IndGP.__file__, flush=True)

    print(
        "Has run_notebook_workflow:",
        hasattr(IndGP, "run_notebook_workflow")
    )

    assert hasattr(IndGP, "run_notebook_workflow")

    # ============================================================
    # VERIFY BURNUP DATASET
    # ============================================================

    BASE_DIR = Path(MODULE_DIR)

    SCENARIO = "scenario_016_damped.csv"     # choose your desired scenario

    scenario_path = BASE_DIR / SCENARIO

    if not scenario_path.is_file():
        raise FileNotFoundError(scenario_path)

    print("Dataset found:", scenario_path)

    # ============================================================
    # RUN IndGP
    # 8 seeds -> 8 parallel CPU workers
    # ============================================================

    summary = IndGP.run_notebook_workflow(

        # Parallelization
        n_cpu_workers=8,
        n_seeds=16,

        # Dataset
        scenario_dir=str(BASE_DIR),
        scenario_files=[SCENARIO],

        # Active learning
        acq_rounds=20,
        batch=5,

        # Validation split
        val_frac=0.5,

        # Independent GP training
        train_epochs=300,
        lr=0.03,

        # Output directory
        save_dir="./paper scenarios/IndGP results",   # change to you desired folder
    )

    print("Results saved to:", summary)



if __name__ == "__main__":
    main()
