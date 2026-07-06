from fleetrl_warehouse.envs.warehouse_core import Action, WarehouseCore


def test_reset_is_seed_reproducible():
    env = WarehouseCore(n_robots=3, n_tasks=3, seed=11)
    first = [r.position for r in env.robots]
    env.reset(seed=11)
    second = [r.position for r in env.robots]
    assert first == second


def test_collision_blocks_same_target():
    env = WarehouseCore(width=8, height=8, n_robots=2, n_tasks=1, obstacles=set(), seed=1)
    env.robots[0].position = (2, 2)
    env.robots[1].position = (4, 2)
    _, reward, _, _, info = env.step([Action.EAST, Action.WEST])
    assert env.robots[0].position == (2, 2)
    assert env.robots[1].position == (4, 2)
    assert info["collisions"] == 1
    assert reward < 0


def test_pickup_and_delivery_flow():
    env = WarehouseCore(width=8, height=8, n_robots=1, n_tasks=1, obstacles=set(), seed=2)
    task = env.tasks[0]
    env.robots[0].position = task.pickup
    env.step([Action.INTERACT])
    assert env.robots[0].carrying_task == task.task_id
    env.robots[0].position = task.delivery
    _, reward, terminated, _, info = env.step([Action.INTERACT])
    assert terminated
    assert info["deliveries"] == 1
    assert reward > 20


def test_global_vector_shape_is_stable():
    env = WarehouseCore(n_robots=3, n_tasks=5)
    expected = 3 * 5 + 5 * 5 + 1
    assert env.global_vector().shape == (expected,)


def test_shortest_path_baseline_completes_small_scenario():
    env = WarehouseCore(n_robots=2, n_tasks=2, max_steps=100, seed=7)
    for _ in range(env.max_steps):
        actions = [env.shortest_path_action(i) for i in range(env.n_robots)]
        _, _, terminated, truncated, _ = env.step(actions)
        if terminated or truncated:
            break
    assert env.total_deliveries == 2
