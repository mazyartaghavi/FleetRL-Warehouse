from fleetrl_warehouse.envs.warehouse_core import WarehouseCore


def test_render_outputs():
    env = WarehouseCore(width=8, height=8, n_robots=2, n_tasks=2)
    ansi = env.render_ansi()
    rgb = env.render_rgb(cell_size=8)
    assert "Legend" in ansi
    assert rgb.shape == (64, 64, 3)
