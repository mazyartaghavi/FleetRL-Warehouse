# Validation Report

## Automated validation

The dependency-light Python core was validated with:

```bash
python -m pytest -v
```

Expected result:

```text
8 passed
```

Validated behavior:

- deterministic seeded reset;
- simultaneous collision blocking;
- pickup and delivery lifecycle;
- centralized observation dimensions;
- ANSI and RGB rendering;
- Q-learning training and model round-trip;
- SARSA training execution.

## Sample Q-learning benchmark

Command:

```bash
fleetrl train-tabular --algorithm q_learning --episodes 1500 --eval-episodes 20 --robots 2 --tasks 2 --max-steps 100 --seed 7
```

Observed CPU result in the build environment:

```json
{
  "mean_reward": 31.67099999999998,
  "mean_completion_rate": 0.8,
  "mean_collisions": 0.05,
  "mean_steps": 59.3
}
```

This is a deterministic software validation point, not a statistical comparison across algorithms. Research conclusions require multiple seeds and confidence intervals.

## ROS 2 validation boundary

ROS 2 Python files are syntax-checked in this release. Full Gazebo execution must be validated on Ubuntu 24.04 with ROS 2 Jazzy and Gazebo Harmonic because the build environment does not contain ROS 2 or a 3D display server.
