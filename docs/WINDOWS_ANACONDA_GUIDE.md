# Windows and Anaconda Guide

## 1. Extract the project

Extract the ZIP into Documents. Open the folder and confirm that `README.md`, `pyproject.toml`, `src`, and `tests` are visible.

## 2. Open Anaconda Prompt

Open Windows Start, type `Anaconda Prompt`, and open it.

## 3. Enter the project folder

```bat
cd /d "C:\Users\YOUR_NAME\Documents\FleetRL_Warehouse_v0.1.0"
```

Run `dir` and confirm that `pyproject.toml` appears.

## 4. Create the environment

```bat
conda create -n fleetrl python=3.11 -y
conda activate fleetrl
```

## 5. Install

```bat
python -m pip install --upgrade pip
python -m pip install -e .[all]
```

## 6. Test

```bat
python -m pytest -v
```

## 7. Run the immediate simulation

```bat
fleetrl simulate --robots 3 --tasks 5
```

## 8. Train and evaluate Q-learning

```bat
fleetrl train-tabular --algorithm q_learning --episodes 1500 --robots 2 --tasks 2 --max-steps 100 --output artifacts/q_learning.json --history artifacts/q_learning.csv
fleetrl evaluate --model artifacts/q_learning.json --robots 2 --tasks 2 --max-steps 100
fleetrl simulate --model artifacts/q_learning.json --robots 2 --tasks 2 --max-steps 100
```

## 9. Jupyter Notebook

```bat
conda install notebook ipykernel -y
python -m ipykernel install --user --name fleetrl --display-name "Python (fleetrl)"
jupyter notebook
```

Open `notebooks/quickstart.ipynb` and select the `Python (fleetrl)` kernel.
