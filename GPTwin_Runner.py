# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:50:13 2026

"""

# -*- coding: utf-8 -*-
"""Run TwinGP on the Burnup scenario using eight parallel CPU workers.

Place GP_TwinGP_parallel8.py in MODULE_DIR (below) and the scenario CSV
in BASE_DIR. The runner uses the module's own preprocessing implementation.
"""

import sys
from pathlib import Path

MODULE_DIR = Path("./paper scenarios")
BASE_DIR = MODULE_DIR
SAVE_DIR = Path("./Results/Twin GP results")


def main():
    # Ensure that the intended module is imported rather than an older copy.
    sys.path.insert(0, str(MODULE_DIR))

    import GPTwin

    print(f"TwinGP module imported from: {GPTwin.__file__}", flush=True)

    GPTwin.run_notebook_workflow(
        base_dir=str(BASE_DIR),
        save_dir=str(SAVE_DIR),
        scenario_files=["scenario_000_burnup.csv"],
        seeds=tuple(range(100, 108)),
        cpu_workers=8,
        initial_context_size=140,
        acq_rounds=20,
        batch_size=5,
        acs_j=50,
        hyperopt_maxiter=20,
        expected_num_outputs=None,
    )


if __name__ == "__main__":
    main()
