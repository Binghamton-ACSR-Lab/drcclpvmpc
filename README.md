# Distributionally Robust Chance Constraint - Linear Parameter Varying Model predictive control (DRCC-LPVMPC)

## Installation

It is recommended to create a new conda environment:

```
conda create --name drcc_lpvmpc python=3.12.3
conda activate drcc_lpvmpc
```

To install drcc_lpvmpc:

```
cd drcc_lpvmpc/
pip install -e .
```

## Run drcc_lpvmpc

```
cd main
python drcc_lpvmpc_main.py
```

## MPC Location
You can refer to the mpc in mpc/dynamics_drccmpc_rhunc.py

## Modify the main file
First, refer to the main/config/config.yaml

Change useDRCC to False and then run the main file again, this will lead to control the vehicle with pure quasi-lpvmpc controller.
Change add_disturbance to True and then run the main file again, this will add additive random noise to the vehicle location which simulate the disturbance from other factors e.g. localizations, sensors, etc.

