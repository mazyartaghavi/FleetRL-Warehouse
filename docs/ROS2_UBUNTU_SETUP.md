# ROS 2 and Gazebo Setup for Beginners

## Recommended platform

Use Ubuntu 24.04 with ROS 2 Jazzy and Gazebo Harmonic. The Python simulator works on Windows, but the realistic Gazebo demonstration is easiest and most reproducible on Ubuntu.

Suitable choices:

1. Ubuntu 24.04 installed directly or in dual boot — best graphics performance.
2. Ubuntu 24.04 virtual machine — simpler separation, but slower 3D graphics.
3. WSL2 — possible, but ROS networking and Gazebo graphics add complexity; not recommended for a first ROS project.

## Install ROS 2 Jazzy

Follow the official ROS 2 Jazzy Ubuntu binary installation instructions. After installation, verify:

```bash
source /opt/ros/jazzy/setup.bash
ros2 --help
```

## Install project dependencies

```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-desktop \
  ros-jazzy-ros-gz \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool
```

Initialize rosdep if needed:

```bash
sudo rosdep init
rosdep update
```

## Install the Python learning package

From the repository root:

```bash
python3 -m pip install --user -e .
```

For deep RL:

```bash
python3 -m pip install --user -e .[deep]
```

## Build the ROS workspace

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## Launch the warehouse fleet

```bash
ros2 launch fleetrl_warehouse_ros warehouse_fleet.launch.py
```

Gazebo should open with three robots and warehouse shelves. The fleet coordinator assigns tasks. Policy executors produce commands, and safety supervisors filter them using lidar.

## Deploy a trained Q-table

Train on the same machine or copy a model into the repository:

```bash
fleetrl train-tabular --algorithm q_learning --episodes 1500 --robots 3 --tasks 5 --output artifacts/q_learning.json
```

Edit `ros2_ws/src/fleetrl_warehouse_ros/launch/warehouse_fleet.launch.py` and add the absolute model path to each `policy_executor` parameter dictionary:

```python
{"robot_name": name, "robot_id": robot_id, "model_path": "/absolute/path/artifacts/q_learning.json"}
```

Rebuild and relaunch:

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch fleetrl_warehouse_ros warehouse_fleet.launch.py
```

## Inspect the ROS graph

In another terminal:

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 node list
ros2 topic list
ros2 topic echo /fleet/state
ros2 topic hz /robot_0/scan
```

## Stop

Press `Ctrl+C` in the launch terminal.

## Troubleshooting

### Gazebo cannot find `fleet_bot`

Confirm that you sourced the workspace after building:

```bash
source ros2_ws/install/setup.bash
```

### No robot motion

Inspect topics:

```bash
ros2 topic echo /robot_0/cmd_vel_raw
ros2 topic echo /robot_0/cmd_vel
ros2 topic echo /robot_0/odom
```

### Lidar topic is missing

Run:

```bash
gz topic -l | grep scan
```

Gazebo topic names can differ across package revisions. Update the scan bridge argument in the launch file to match the discovered topic.

### Graphics are slow in a virtual machine

Run Gazebo server without the GUI by changing `gz_args` from `-r -v 2` to `-s -r -v 2`, or use native Ubuntu.
