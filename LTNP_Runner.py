# -*- coding: utf-8 -*-
"""LTNP active-learning runner converted from LTNP Runner .ipynb.

Run as a standalone Python script. Adjust the selected scenarios and output
folder before running, if desired.
"""


def main():
    # ============================================================
    # Import and reload final_np
    # ============================================================

    import sys
    import os
    import importlib

    sys.path.insert(0, os.getcwd())

    import LTNP
    importlib.reload(LTNP)

    # Check available CPUs
    print("Logical CPUs visible:", os.cpu_count())

    if hasattr(os, "sched_getaffinity"):
        print(
            "CPU cores available:",
            sorted(os.sched_getaffinity(0))
        )



    # ============================================================
    # SELECT SCENARIOS
    # Comment out any scenario you do not want to run.
    # ============================================================
    # Comment out your desired scenario to run
    selected_scenarios = [   
        # "Scenario_000_burnup.csv",
        #scenario_000_HGR.csv",
        # "Scenario_016_damped.csv",
         "Scenario_054_gompertz.csv",
        # "Scenario_179_sigmoid.csv",
    ]


    # ============================================================
    # Filter scenarios by their filenames
    # ============================================================

    original_list_scenarios = LTNP._runner_list_scenario_csvs

    def filtered_scenarios(folder):

        all_scenarios = original_list_scenarios(folder)

        selected_names = {
            name.lower() for name in selected_scenarios
        }

        return [
            (sid, path)
            for sid, path in all_scenarios
            if os.path.basename(path).lower() in selected_names
        ]


    LTNP._runner_list_scenario_csvs = filtered_scenarios


    # ============================================================
    # Run NP active learning
    # ============================================================

    results = LTNP.run_notebook_workflow(
        n_cpu_workers=8,
        n_seeds=16,
        n_plot_seeds=8,
        acq_rounds=30,
        initial_context_size=140,
        lr=1e-3,
        save_dir="./Results/LTNP results",  # Change to your desired folder
    )


if __name__ == "__main__":
    main()
