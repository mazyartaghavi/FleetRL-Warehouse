# Uploading the Project to GitHub

## Recommended repository name

```text
FleetRL-Warehouse-ROS2
```

## Browser method

1. Create a new public repository on GitHub without adding a README.
2. Open the extracted project folder on your laptop.
3. Select the contents inside the folder, not the enclosing folder and not the ZIP.
4. On GitHub, choose **Add file → Upload files**.
5. Drag the selected contents into the page.
6. Use commit message: `Add cooperative warehouse multi-robot RL project`.
7. Commit to `main`.

The repository root should directly show:

```text
src
ros2_ws
configs
docs
notebooks
tests
README.md
pyproject.toml
```

## Do not upload

The `.gitignore` excludes generated models, experiment runs, caches, ROS build folders, and virtual environments. Never upload private keys or secrets.

## Verify GitHub Actions

Open the **Actions** tab. The `Python tests` workflow should finish with eight passing tests and a green check.

## Suggested repository topics

```text
reinforcement-learning
multi-agent-reinforcement-learning
ros2
gazebo
mobile-robots
warehouse-robotics
q-learning
sarsa
ppo
sac
```
