# Step-by-Step Project Plan

## Stage 1 — Verify the software foundation

Goal: prove that the repository installs and the simulator behaves correctly.

```bash
python -m pip install -e .[all]
python -m pytest -v
fleetrl simulate --robots 2 --tasks 2 --max-steps 100
```

Deliverable: eight passing tests and a completed two-robot delivery episode.

## Stage 2 — Train tabular baselines

Train and save Q-learning and SARSA models. Inspect their CSV histories and replay the learned policies.

Deliverable: model JSON files, learning curves, and evaluation metrics.

## Stage 3 — Train deep policies

Use PPO and A2C with the centralized multi-discrete environment. Use DDPG and SAC with the continuous high-level motion environment.

Deliverable: saved SB3 models and TensorBoard logs.

## Stage 4 — Perform fair comparison

Use at least five seeds, equal interaction budgets where practical, and report reward, completion rate, collisions, and training time.

```bash
python scripts/benchmark_tabular.py --episodes 1500 --seeds 7 17 27 37 47
```

Deliverable: benchmark CSV, JSON summary, plots, and a short findings section.

## Stage 5 — Install ROS 2 and Gazebo

Use Ubuntu 24.04, ROS 2 Jazzy, and Gazebo Harmonic. Build the included Colcon workspace and launch the three-robot world.

Deliverable: Gazebo screenshot, ROS graph screenshot, and a recorded demonstration.

## Stage 6 — Deploy a learned policy

Supply a trained Q-table to each ROS policy executor. Verify that the safety supervisor overrides forward motion when lidar detects a nearby obstacle.

Deliverable: policy-versus-safety topic traces and collision-free delivery demonstration.

## Stage 7 — Publish on GitHub and add to CV

Upload the repository, verify GitHub Actions, create release `v0.1.0`, add screenshots/results, and use the wording in `CV_ENTRY.md`.
