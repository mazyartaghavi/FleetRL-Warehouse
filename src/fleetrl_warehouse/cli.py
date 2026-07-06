"""Command-line interface for training, evaluation, and simulation."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from fleetrl_warehouse.algorithms.tabular import IndependentTabularFleet, TabularConfig
from fleetrl_warehouse.envs.warehouse_core import Action, WarehouseCore
from fleetrl_warehouse.evaluation.runner import evaluate_tabular


def _env_from_args(args: argparse.Namespace) -> WarehouseCore:
    return WarehouseCore(
        width=args.width,
        height=args.height,
        n_robots=args.robots,
        n_tasks=args.tasks,
        max_steps=args.max_steps,
        seed=args.seed,
    )


def train_tabular(args: argparse.Namespace) -> None:
    env = _env_from_args(args)
    agent = IndependentTabularFleet(
        env.n_robots,
        TabularConfig(
            algorithm=args.algorithm,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon_decay=args.epsilon_decay,
            seed=args.seed,
        ),
    )
    history = agent.train(env, episodes=args.episodes)
    agent.save(args.output)
    history_path = Path(args.history)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    metrics = evaluate_tabular(agent, env, episodes=args.eval_episodes, seed=args.seed + 10_000)
    print(json.dumps({"model": args.output, "history": args.history, "evaluation": metrics}, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    env = _env_from_args(args)
    agent = IndependentTabularFleet.load(args.model)
    metrics = evaluate_tabular(agent, env, episodes=args.eval_episodes, seed=args.seed)
    print(json.dumps(metrics, indent=2))


def simulate(args: argparse.Namespace) -> None:
    env = _env_from_args(args)
    agent = IndependentTabularFleet.load(args.model) if args.model else None
    env.reset(seed=args.seed)
    total_reward = 0.0
    for _ in range(env.max_steps):
        print("\033[2J\033[H", end="")
        print(env.render_ansi())
        print(f"step={env.step_count} deliveries={env.total_deliveries}/{env.n_tasks} reward={total_reward:.2f}")
        if agent:
            states = [env.local_state(i) for i in range(env.n_robots)]
            actions = agent.act(states, explore=False)
        else:
            # Deterministic shortest-path baseline useful before a model is trained.
            actions = [int(env.shortest_path_action(i)) for i in range(env.n_robots)]
        _, reward, terminated, truncated, _ = env.step(actions)
        total_reward += reward
        time.sleep(args.delay)
        if terminated or truncated:
            break
    print(env.render_ansi())
    print(f"Finished: deliveries={env.total_deliveries}/{env.n_tasks}, reward={total_reward:.2f}")


def train_deep(args: argparse.Namespace) -> None:
    from fleetrl_warehouse.algorithms.deep import train_sb3

    train_sb3(
        args.algorithm,
        total_timesteps=args.timesteps,
        output=args.output,
        env_config={
            "width": args.width,
            "height": args.height,
            "n_robots": args.robots,
            "n_tasks": args.tasks,
            "max_steps": args.max_steps,
            "seed": args.seed,
        },
        seed=args.seed,
    )
    print(json.dumps({"algorithm": args.algorithm, "model": args.output}, indent=2))


def common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--robots", type=int, default=3)
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fleetrl", description="Cooperative warehouse robot RL")
    sub = parser.add_subparsers(dest="command", required=True)

    tab = sub.add_parser("train-tabular", help="Train independent Q-learning or SARSA")
    common(tab)
    tab.add_argument("--algorithm", choices=["q_learning", "sarsa"], default="q_learning")
    tab.add_argument("--episodes", type=int, default=500)
    tab.add_argument("--eval-episodes", type=int, default=20)
    tab.add_argument("--learning-rate", type=float, default=0.15)
    tab.add_argument("--gamma", type=float, default=0.97)
    tab.add_argument("--epsilon-decay", type=float, default=0.995)
    tab.add_argument("--output", default="artifacts/q_learning.json")
    tab.add_argument("--history", default="artifacts/q_learning_history.csv")
    tab.set_defaults(func=train_tabular)

    deep = sub.add_parser("train-deep", help="Train PPO, A2C, DDPG, or SAC with SB3")
    common(deep)
    deep.add_argument("--algorithm", choices=["ppo", "a2c", "ddpg", "sac"], default="ppo")
    deep.add_argument("--timesteps", type=int, default=100_000)
    deep.add_argument("--output", default="artifacts/ppo_warehouse")
    deep.set_defaults(func=train_deep)

    ev = sub.add_parser("evaluate", help="Evaluate a saved tabular model")
    common(ev)
    ev.add_argument("--model", required=True)
    ev.add_argument("--eval-episodes", type=int, default=20)
    ev.set_defaults(func=evaluate)

    sim = sub.add_parser("simulate", help="Render an ANSI warehouse simulation")
    common(sim)
    sim.add_argument("--model", default=None)
    sim.add_argument("--delay", type=float, default=0.08)
    sim.set_defaults(func=simulate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
