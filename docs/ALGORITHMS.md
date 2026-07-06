# Algorithms and MDP Design

## State

The local tabular state for robot `i` is:

```text
(x_i, y_i, carrying_i, clipped_dx_to_goal, clipped_dy_to_goal,
 clipped_dx_to_nearest_robot, clipped_dy_to_nearest_robot)
```

The centralized deep-RL observation concatenates normalized robot positions, carrying flags, current targets, task pickup/delivery coordinates, task statuses, and elapsed-time fraction.

## Actions

Discrete actions:

```text
0 WAIT
1 NORTH
2 SOUTH
3 WEST
4 EAST
5 INTERACT (pickup or drop)
```

Continuous actions contain two values per robot:

```text
[v_x, v_y] in [-1, 1]^2
```

The lightweight environment maps them to safe grid primitives. In ROS 2, they correspond conceptually to differential-drive linear and angular velocities.

## Shared reward

```text
-0.05       each timestep
+0.08       for each unit of distance progress toward current goal
-1.00       each collision conflict
-0.20       invalid wall/shelf action
+1.00       successful pickup
+10.00      successful delivery
+2.00       extra simultaneous-delivery cooperation bonus
+20.00      all tasks completed
```

## Q-learning

Off-policy target:

```text
Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]
```

## SARSA

On-policy target:

```text
Q(s,a) <- Q(s,a) + alpha [r + gamma Q(s',a') - Q(s,a)]
```

## PPO and A2C

PPO and A2C use the centralized multi-discrete wrapper. The policy outputs one discrete component per robot. PPO is the preferred robust deep baseline; A2C provides a simpler synchronous actor-critic comparison.

## DDPG and SAC

DDPG and SAC use the continuous wrapper. DDPG provides deterministic control, while SAC encourages exploration using entropy regularization.

## Evaluation metrics

- team episode reward;
- task completion rate;
- deliveries per episode;
- collisions per episode;
- invalid-action count;
- episode length;
- sample efficiency;
- wall-clock training time.

Use at least five random seeds for research-grade comparisons. The included sample result uses one fixed seed solely as a reproducible software demonstration.
