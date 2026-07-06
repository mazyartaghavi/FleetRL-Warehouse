"""Compare Q-learning and SARSA across multiple random seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

from fleetrl_warehouse.algorithms.tabular import IndependentTabularFleet, TabularConfig
from fleetrl_warehouse.envs.warehouse_core import WarehouseCore
from fleetrl_warehouse.evaluation.runner import evaluate_tabular


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--tasks", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--output", default="artifacts/tabular_benchmark.csv")
    args = parser.parse_args()

    rows: list[dict[str, float | str | int]] = []
    for algorithm in ("q_learning", "sarsa"):
        for seed in args.seeds:
            env = WarehouseCore(
                n_robots=args.robots,
                n_tasks=args.tasks,
                max_steps=args.max_steps,
                seed=seed,
            )
            agent = IndependentTabularFleet(
                args.robots,
                TabularConfig(algorithm=algorithm, seed=seed),
            )
            agent.train(env, episodes=args.episodes)
            metrics = evaluate_tabular(
                agent,
                env,
                episodes=args.eval_episodes,
                seed=seed + 10_000,
            )
            rows.append({"algorithm": algorithm, "seed": seed, **metrics})
            print(json.dumps(rows[-1]))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for algorithm in ("q_learning", "sarsa"):
        selected = [row for row in rows if row["algorithm"] == algorithm]
        summary[algorithm] = {
            metric: {
                "mean": mean(float(row[metric]) for row in selected),
                "population_std": pstdev(float(row[metric]) for row in selected),
            }
            for metric in ("mean_reward", "mean_completion_rate", "mean_collisions", "mean_steps")
        }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(output), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
