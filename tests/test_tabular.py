from fleetrl_warehouse.algorithms.tabular import IndependentTabularFleet, TabularConfig
from fleetrl_warehouse.envs.warehouse_core import WarehouseCore


def test_q_learning_training_and_roundtrip(tmp_path):
    env = WarehouseCore(width=8, height=8, n_robots=2, n_tasks=2, max_steps=15, obstacles=set(), seed=4)
    agent = IndependentTabularFleet(2, TabularConfig(algorithm="q_learning", seed=4))
    history = agent.train(env, episodes=3)
    assert len(history) == 3
    path = tmp_path / "agent.json"
    agent.save(path)
    loaded = IndependentTabularFleet.load(path)
    assert loaded.n_robots == 2
    assert len(loaded.q_tables[0]) > 0


def test_sarsa_training_executes():
    env = WarehouseCore(width=8, height=8, n_robots=2, n_tasks=2, max_steps=10, obstacles=set(), seed=5)
    agent = IndependentTabularFleet(2, TabularConfig(algorithm="sarsa", seed=5))
    history = agent.train(env, episodes=2)
    assert len(history) == 2
