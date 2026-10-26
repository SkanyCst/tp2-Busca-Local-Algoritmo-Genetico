#!/usr/bin/env python3
"""Benchmark da Tarefa T2: Comparação empírica das vizinhanças e ganho incremental."""

import argparse
import csv
import os
import random
import sys
import time
from typing import Dict, List, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.problem import MaintenanceProblem
from src.evaluator import Evaluator
from src.operators import get_swap_neighbors, get_insertion_neighbors, get_two_opt_neighbors
from src.algorithms.greedy_descent import greedy_descent_restart

NEIGHBORHOODS = {
    "swap": get_swap_neighbors,
    "insertion": get_insertion_neighbors,
    "2opt": get_two_opt_neighbors,
}

INSTANCES = {
    "pequena": "manutencao_pequena.txt",
    "media": "manutencao_media.txt",
    "grande": "manutencao_grande.txt",
}


def benchmark_incremental_speedup(filepath: str, n_samples: int = 10000, seed: int = 42) -> Dict[str, float]:
    """Mede a aceleração (speedup) do tempo de parede da avaliação incremental vs completa."""
    problem = MaintenanceProblem(filepath)
    rng = random.Random(seed)
    n = problem.n
    pi = list(range(1, n + 1))
    rng.shuffle(pi)
    base_cost = problem.h(pi)

    moves = []
    for _ in range(n_samples):
        i, j = sorted(rng.sample(range(n), 2))
        nb = pi[:]
        nb[i], nb[j] = nb[j], nb[i]
        moves.append((nb, i))

    t0 = time.perf_counter()
    for nb, _ in moves:
        _ = problem.h(nb)
    t_complete = time.perf_counter() - t0

    t0 = time.perf_counter()
    for nb, start_idx in moves:
        _ = problem.h_incremental(nb, base_cost, start_idx)
    t_incremental = time.perf_counter() - t0

    speedup = t_complete / t_incremental if t_incremental > 0 else 1.0
    return {
        "t_complete": t_complete,
        "t_incremental": t_incremental,
        "speedup": speedup,
        "time_per_eval_complete_us": (t_complete / n_samples) * 1e6,
        "time_per_eval_incremental_us": (t_incremental / n_samples) * 1e6,
    }


def run_t2_comparison(budget: int = 20000, seed: int = 123) -> List[Dict[str, Any]]:
    results = []
    print("\n" + "=" * 80)
    print("TAREFA T2 — COMPARAÇÃO DE VIZINHANÇAS E AVALIAÇÃO INCREMENTAL")
    print("=" * 80)

    for inst_name, filename in INSTANCES.items():
        filepath = os.path.join("data", filename)
        if not os.path.exists(filepath):
            filepath = os.path.join("..", "data", filename)
        problem = MaintenanceProblem(filepath)

        inc_data = benchmark_incremental_speedup(filepath, n_samples=10000, seed=seed)
        print(f"\n>> Instância: {inst_name.upper()} (n={problem.n}) | Orçamento fixo: {budget:,}")
        print(
            f"   [Speedup Incremental Swap] Completo: {inc_data['time_per_eval_complete_us']:.2f} µs/eval | "
            f"Incremental: {inc_data['time_per_eval_incremental_us']:.2f} µs/eval -> Speedup: {inc_data['speedup']:.2f}x"
        )

        for hood_name, hood_fn in NEIGHBORHOODS.items():
            ev = Evaluator(problem, max_evaluations=budget)
            rng = random.Random(seed)

            t_start = time.perf_counter()
            best_pi, best_cost, stats = greedy_descent_restart(ev, hood_fn, variant="best", rng=rng)
            elapsed = time.perf_counter() - t_start

            row = {
                "instance": inst_name,
                "n": problem.n,
                "neighborhood": hood_name,
                "budget": budget,
                "best_cost": best_cost,
                "completed_descents": stats.completed_descents,
                "distinct_local_optima": stats.distinct_local_optima,
                "improvement_steps": stats.improvement_steps,
                "elapsed_seconds": elapsed,
                "incremental_speedup": inc_data["speedup"],
            }
            results.append(row)
            print(
                f"   * {hood_name.ljust(9)}: Melhor h = {best_cost:10.1f} | "
                f"Descidas = {stats.completed_descents:3d} | "
                f"Passos = {stats.improvement_steps:4d} | "
                f"Tempo = {elapsed:6.3f}s"
            )

    return results


def save_t2_csv(results: List[Dict[str, Any]], output_path: str = "data/t2_neighborhoods_comparison.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    keys = list(results[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[OK] Resultados salvos em: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Tarefa T2")
    parser.add_argument("--budget", type=int, default=20000, help="Orçamento por vizinhança")
    parser.add_argument("--seed", type=int, default=123, help="Semente pseudo-aleatória")
    parser.add_argument("--csv", type=str, default="data/t2_neighborhoods_comparison.csv", help="Caminho do CSV de saída")
    args = parser.parse_args()

    res = run_t2_comparison(budget=args.budget, seed=args.seed)
    save_t2_csv(res, args.csv)
