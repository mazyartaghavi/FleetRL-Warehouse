# Troubleshooting

## `fleetrl` is not recognized

Activate the environment and reinstall the project:

```bash
python -m pip install -e .[all]
```

You can always use:

```bash
python -m fleetrl_warehouse.cli simulate --robots 2 --tasks 2
```

## `ModuleNotFoundError: stable_baselines3`

```bash
python -m pip install -e .[deep]
```

## The learned tabular policy performs poorly

Increase episodes, keep training and evaluation environment sizes identical, and evaluate across seeds. A model trained for two robots cannot be loaded as a three-robot policy.

## The ANSI screen flickers in Jupyter

Run the `fleetrl simulate` command in Anaconda Prompt. Use `render_rgb()` for notebook visualization.

## `colcon` is not recognized

ROS 2 is not installed or the ROS environment is not sourced. On Ubuntu:

```bash
source /opt/ros/jazzy/setup.bash
```

## Gazebo opens but robots do not move

Inspect `/fleet/state`, odometry, raw velocity, and safe velocity topics. See `ROS2_UBUNTU_SETUP.md`.
