# FleetRL-Warehouse

**Cooperative Multi-Robot Reinforcement Learning for Warehouse Logistics with ROS 2 and Gazebo**

FleetRL-Warehouse is a portfolio-grade research and engineering project in which a fleet of differential-drive mobile robots cooperates to collect packages, avoid collisions, and deliver orders in a warehouse. It provides a fast Python simulator for learning and benchmarking, plus a ROS 2 / Gazebo deployment layer for robot communication, task allocation, policy execution, and independent safety supervision.

![Warehouse layout](docs/assets/warehouse_layout.png)

## What this repository demonstrates

- Multi-agent reinforcement learning and cooperative reward design
- Independent Q-learning and SARSA implemented from first principles
- PPO and A2C with centralized multi-discrete actions
- DDPG and SAC with continuous velocity-like actions
- Centralized training with decentralized execution
- Collision handling, task assignment, pickup/drop interactions, and metrics
- ROS 2 nodes, topics, parameters, launch files, and Gazebo simulation assets
- Reproducible tests, benchmark scripts, Docker support, and GitHub Actions

## Two execution paths

### Path A — Fast Python simulator

Runs on Windows, macOS, or Linux. Use it to learn the environment, train policies, compare algorithms, and create results quickly.

### Path B — ROS 2 and Gazebo fleet

Runs best on Ubuntu 24.04 with ROS 2 Jazzy and Gazebo Harmonic. Use it to demonstrate robot middleware, odometry, lidar, velocity commands, namespaces, policy deployment, and a safety filter.

## Repository structure

```text
FleetRL_Warehouse/
├── src/fleetrl_warehouse/
│   ├── envs/                 # Core simulator and Gymnasium wrappers
│   ├── algorithms/           # Q-learning, SARSA, PPO/A2C/DDPG/SAC entry points
│   ├── evaluation/           # Reproducible metrics
│   └── cli.py                # `fleetrl` command
├── ros2_ws/src/fleetrl_warehouse_ros/
│   ├── fleetrl_warehouse_ros/# ROS 2 nodes
│   ├── launch/               # One-command fleet launch
│   ├── worlds/               # Gazebo warehouse
│   └── models/               # Differential-drive robot with lidar
├── configs/                  # Experiment configuration
├── notebooks/                # Beginner-friendly demonstration
├── scripts/                  # Direct Python entry points
├── tests/                    # Unit tests
├── docs/                     # Architecture, algorithms, setup, CV wording
├── docker/                   # Core and ROS 2 containers
└── .github/workflows/        # Automatic tests
```

## Quick start on Windows with Anaconda

Open Anaconda Prompt in this extracted project folder.

```bat
conda create -n fleetrl python=3.11 -y
conda activate fleetrl
python -m pip install --upgrade pip
python -m pip install -e .[all]
python -m pytest -v
```

Run an obstacle-aware non-learning baseline first:

```bat
fleetrl simulate --robots 2 --tasks 2 --max-steps 100 --delay 0.08
```

Train Q-learning:

```bat
fleetrl train-tabular --algorithm q_learning --episodes 1500 --robots 2 --tasks 2 --max-steps 100 --output artifacts/q_learning.json --history artifacts/q_learning.csv
```

Evaluate and visualize the trained policy:

```bat
fleetrl evaluate --model artifacts/q_learning.json --robots 2 --tasks 2 --max-steps 100
fleetrl simulate --model artifacts/q_learning.json --robots 2 --tasks 2 --max-steps 100
```

Train SARSA:

```bat
fleetrl train-tabular --algorithm sarsa --episodes 1500 --robots 2 --tasks 2 --max-steps 100 --output artifacts/sarsa.json --history artifacts/sarsa.csv
```

Train deep RL:

```bat
fleetrl train-deep --algorithm ppo --timesteps 100000 --output artifacts/ppo_warehouse
fleetrl train-deep --algorithm a2c --timesteps 100000 --output artifacts/a2c_warehouse
fleetrl train-deep --algorithm ddpg --timesteps 150000 --output artifacts/ddpg_warehouse
fleetrl train-deep --algorithm sac --timesteps 150000 --output artifacts/sac_warehouse
```

PPO/A2C use a centralized `MultiDiscrete` action containing one action per robot. DDPG/SAC use a continuous vector containing `[linear, angular]` commands for each robot.

## ROS 2 / Gazebo quick start

Recommended host:

- Ubuntu 24.04
- ROS 2 Jazzy Jalisco
- Gazebo Harmonic

After installing ROS 2, Gazebo, `ros_gz`, and Colcon:

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
ros2 launch fleetrl_warehouse_ros warehouse_fleet.launch.py
```

The launch starts three simulated robots, a central task coordinator, one policy executor per robot, one safety supervisor per robot, and ROS–Gazebo topic bridges.

Detailed installation: [docs/ROS2_UBUNTU_SETUP.md](docs/ROS2_UBUNTU_SETUP.md)

Development sequence: [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)

## Algorithms

| Algorithm | Action space | Implementation | Main purpose |
|---|---|---|---|
| Q-learning | Discrete | From scratch | Off-policy tabular baseline |
| SARSA | Discrete | From scratch | On-policy tabular baseline |
| PPO | MultiDiscrete | Stable-Baselines3 | Stable cooperative deep policy |
| A2C | MultiDiscrete | Stable-Baselines3 | Synchronous actor-critic baseline |
| DDPG | Continuous | Stable-Baselines3 | Deterministic velocity control |
| SAC | Continuous | Stable-Baselines3 | Entropy-regularized velocity control |

See [docs/ALGORITHMS.md](docs/ALGORITHMS.md) for the state, actions, reward, and evaluation protocol.

## Sample verified result

A deterministic CPU run with seed 7, two robots, two tasks, 1,500 Q-learning episodes, and 20 evaluation episodes produced:

- Mean completion rate: **0.80**
- Mean collisions: **0.05 per episode**
- Mean episode steps: **59.3**
- Mean reward: **31.671**

These are sample results for the included small benchmark and are not claims about all seeds or warehouse sizes. Re-run the command above to reproduce or challenge them.

![Q-learning curve](docs/assets/q_learning_curve.png)

## Safety design

The learned policy publishes a raw velocity command. A separate ROS 2 safety supervisor reads lidar data and can stop forward motion before the command reaches the simulated drive controller. This makes the safety layer independent of the RL algorithm.

## Tests

```bash
python -m pytest -v
```

The lightweight unit suite checks deterministic resets, collisions, pickup/delivery transitions, observation size, tabular updates, model serialization, and rendering.

In a two-robot/two-task warehouse benchmark, SARSA achieved 95% task completion with 0.05 mean collisions over the final 100 episodes, compared with 93% completion and 0.12 mean collisions for Q-learning.

## Limitations

- The fast simulator uses grid dynamics rather than full rigid-body physics.
- The ROS 2 high-level policy adapter discretizes Gazebo positions to grid cells.
- Deep models require additional training time and are not bundled as universal pretrained policies.
- The Gazebo launch files require Ubuntu/ROS testing on the target machine; automated CI validates the Python core, not GPU rendering.

## License

MIT License.

