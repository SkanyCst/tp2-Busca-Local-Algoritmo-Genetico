#!/usr/bin/env python3
"""Benchmark e amostragem do Simulated Annealing."""

import argparse
import csv
import os
import random
import statistics
import sys
import time
from typing import Dict, List, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.problem import MaintenanceProblem
from src.evaluator import Evaluator
from src.algorithms.simulated_annealing import simulated_annealing

REFERENCE_OPTIMA = {
    "manutencao_pequena.txt": 1875.0,
    "manutencao_media.txt": 16332.0,
    "manutencao_grande.txt": 60417.0,
}

DEFAULT_CONFIGS = {
    "pequena": {
        "file": "manutencao_pequena.txt",
        "budget": 50000,
        "runs": 30,
    },
    "media": {
        "file": "manutencao_media.txt",
        "budget": 150000,
        "runs": 30,
    },
    "grande": {
        "file": "manutencao_grande.txt",
        "budget": 300000,
        "runs": 10,
    },
}

QUICK_CONFIGS = {
    "pequena": {
        "file": "manutencao_pequena.txt",
        "budget": 20000,
        "runs": 5,
    },
    "media": {
        "file": "manutencao_media.txt",
        "budget": 40000,
        "runs": 3,
    },
    "grande": {
        "file": "manutencao_grande.txt",
        "budget": 60000,
        "runs": 2,
    },
}


def run_sampling_for_instance(
    filepath: str,
    budget: int,
    runs: int,
    base_seed: int = 1000,
    verbose: bool = False,
) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Instancia nao encontrada: {filepath}")

    problem = MaintenanceProblem(filepath)
    filename = os.path.basename(filepath)
    ref_optimum = REFERENCE_OPTIMA.get(filename, None)

    costs = []
    times = []
    initial_rates = []
    final_rates = []
    individual_runs = []

    print(f"\n>> Amostragem em '{filename}' (n={problem.n}, Orcamento={budget:,}, Runs={runs})...")

    for i in range(runs):
        seed = base_seed + i
        rng = random.Random(seed)
        ev = Evaluator(problem, max_evaluations=budget)

        t_start = time.perf_counter()
        best_pi, best_cost, stats = simulated_annealing(ev, rng=rng)
        elapsed = time.perf_counter() - t_start

        costs.append(best_cost)
        times.append(elapsed)
        init_rate = stats.initial_acceptance_rate
        fin_rate = stats.final_acceptance_rate
        initial_rates.append(init_rate)
        final_rates.append(fin_rate)

        gap = None
        if ref_optimum is not None:
            gap = ((best_cost - ref_optimum) / ref_optimum) * 100.0

        individual_runs.append({
            "run": i + 1,
            "seed": seed,
            "instance": filename,
            "n": problem.n,
            "budget": budget,
            "best_cost": best_cost,
            "elapsed_seconds": elapsed,
            "initial_acceptance_rate": init_rate,
            "final_acceptance_rate": fin_rate,
            "worsening_proposed": stats.worsening_proposed,
            "worsening_accepted": stats.worsening_accepted,
            "improvements": stats.improvements,
            "rejections": stats.rejections,
            "total_steps": stats.total_steps,
            "history": stats.acceptance_history,
        })

        if verbose:
            gap_str = f" | Gap: {gap:+.2f}%" if gap is not None else ""
            print(f"   [Run {i+1:02d}/{runs:02d}] Seed {seed} -> Custo: {best_cost:.1f}{gap_str} | Tempo: {elapsed:.3f}s | Aceit. Ini: {init_rate:.1%} -> Fim: {fin_rate:.2%}")

    mean_cost = statistics.mean(costs)
    std_cost = statistics.stdev(costs) if len(costs) > 1 else 0.0
    median_cost = statistics.median(costs)
    min_cost = min(costs)
    max_cost = max(costs)
    mean_time = statistics.mean(times)
    mean_init_rate = statistics.mean(initial_rates)
    mean_fin_rate = statistics.mean(final_rates)

    relative_gap = None
    if ref_optimum is not None:
        relative_gap = ((min_cost - ref_optimum) / ref_optimum) * 100.0

    return {
        "filename": filename,
        "n": problem.n,
        "budget": budget,
        "runs": runs,
        "ref_optimum": ref_optimum,
        "min_cost": min_cost,
        "max_cost": max_cost,
        "mean_cost": mean_cost,
        "std_cost": std_cost,
        "median_cost": median_cost,
        "mean_time": mean_time,
        "mean_init_rate": mean_init_rate,
        "mean_fin_rate": mean_fin_rate,
        "relative_gap": relative_gap,
        "raw_runs": individual_runs,
    }


def print_summary_table(results: List[Dict[str, Any]]):
    header = (
        f"{'Instancia':<24} | {'n':>3} | {'Orcamento':>9} | {'Runs':>4} | "
        f"{'Melhor h':>10} | {'Media +- Desvio':>20} | {'Mediana':>10} | "
        f"{'Gap Min':>8} | {'Tempo Med':>9} | {'Aceit. Ini':>10} | {'Aceit. Fim':>10}"
    )
    separator = "-" * len(header)
    print("\n" + separator)
    print("RESUMO CONSOLIDADO DA AMOSTRAGEM — SIMULATED ANNEALING")
    print(separator)
    print(header)
    print(separator)

    for r in results:
        gap_str = f"{r['relative_gap']:>7.2f}%" if r["relative_gap"] is not None else f"{'—':>8}"
        mean_std = f"{r['mean_cost']:.1f} +- {r['std_cost']:.1f}"
        print(
            f"{r['filename']:<24} | {r['n']:>3} | {r['budget']:>9,} | {r['runs']:>4} | "
            f"{r['min_cost']:>10.1f} | {mean_std:>20} | {r['median_cost']:>10.1f} | "
            f"{gap_str} | {r['mean_time']:>8.3f}s | {r['mean_init_rate']:>9.1%} | {r['mean_fin_rate']:>9.2%}"
        )
    print(separator + "\n")


def save_results_to_csv(results: List[Dict[str, Any]], csv_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    rows = []
    for r in results:
        for run in r["raw_runs"]:
            row = dict(run)
            del row["history"]
            rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dados salvos em: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark e amostragem do Simulated Annealing."
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        choices=["pequena", "media", "grande", "todas"],
        default=["todas"],
        help="Instancias a avaliar.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Amostragem rapida.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Protocolo oficial completo.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Quantidade de repeticoes por instancia.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Orcamento de avaliacoes.",
    )
    parser.add_argument(
        "--custom-file",
        type=str,
        default=None,
        help="Arquivo de instancia customizado.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=os.path.join("data", "sa_sampling_results.csv"),
        help="Arquivo CSV para salvar os resultados.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Exibe detalhes por run.",
    )

    args = parser.parse_args()

    configs = QUICK_CONFIGS if args.quick else DEFAULT_CONFIGS

    targets = args.instances
    if "todas" in targets:
        targets = ["pequena", "media", "grande"]

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    results = []

    if args.custom_file:
        budget = args.budget or 50000
        runs = args.runs or 5
        res = run_sampling_for_instance(
            args.custom_file, budget=budget, runs=runs, verbose=args.verbose
        )
        results.append(res)
    else:
        for t in targets:
            cfg = configs[t]
            filepath = os.path.join(data_dir, cfg["file"])
            budget = args.budget or cfg["budget"]
            runs = args.runs or cfg["runs"]

            res = run_sampling_for_instance(
                filepath=filepath,
                budget=budget,
                runs=runs,
                verbose=args.verbose,
            )
            results.append(res)

    print_summary_table(results)

    if args.csv:
        save_results_to_csv(results, args.csv)


if __name__ == "__main__":
    main()
