#!/usr/bin/env python3
"""
Framework Principal de Amostragem Experimental (Protocolo Oficial Seção 3 e 5).
Executa as 9 combinações (3 algoritmos x 3 instâncias) + experimento adicional do GA.
"""

import argparse
import csv
import os
import random
import statistics
import sys
import time
from typing import Dict, List, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.problem import MaintenanceProblem
from src.evaluator import Evaluator
from src.operators import get_swap_neighbors
from src.algorithms.greedy_descent import greedy_descent_restart
from src.algorithms.simulated_annealing import simulated_annealing
from src.algorithms.genetic_algorithm import genetic_algorithm, GAConfig

REFERENCE_OPTIMA = {
    "pequena": 1875.0,
    "media": 16332.0,
    "grande": 60417.0,
}

INSTANCE_FILES = {
    "pequena": "manutencao_pequena.txt",
    "media": "manutencao_media.txt",
    "grande": "manutencao_grande.txt",
}

FULL_PROTOCOL = {
    "pequena": {"budget": 50000, "runs": 30},
    "media": {"budget": 150000, "runs": 30},
    "grande": {"budget": 300000, "runs": 10},
}

QUICK_PROTOCOL = {
    "pequena": {"budget": 10000, "runs": 5},
    "media": {"budget": 25000, "runs": 3},
    "grande": {"budget": 40000, "runs": 2},
}


def sample_trajectory_at_checkpoints(
    history: List[tuple], total_budget: int, n_points: int = 50
) -> List[float]:
    """Interpola o melhor custo encontrado em pontos regulares de checagem ao longo do orçamento."""
    checkpoints = [int(total_budget * (k / n_points)) for k in range(1, n_points + 1)]
    result = []

    hist_idx = 0
    current_best = float("inf")

    for cp in checkpoints:
        while hist_idx < len(history) and history[hist_idx][0] <= cp:
            current_best = history[hist_idx][1]
            hist_idx += 1
        result.append(current_best)

    return result


def run_single_experiment(
    algo_name: str,
    problem: MaintenanceProblem,
    budget: int,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    ev = Evaluator(problem, max_evaluations=budget)

    t_start = time.perf_counter()

    ga_mean_traj = None
    ga_div_traj = None
    sa_accept_traj = None

    if algo_name == "greedy":
        best_pi, best_cost, stats = greedy_descent_restart(
            ev, get_swap_neighbors, variant="best", rng=rng
        )
        extra = {
            "completed_descents": stats.completed_descents,
            "distinct_local_optima": stats.distinct_local_optima,
            "improvement_steps": stats.improvement_steps,
        }
    elif algo_name == "sa":
        best_pi, best_cost, stats = simulated_annealing(ev, rng=rng)
        extra = {
            "initial_rate": stats.initial_acceptance_rate,
            "final_rate": stats.final_acceptance_rate,
            "global_rate": stats.global_acceptance_rate,
            "worsening_proposed": stats.worsening_proposed,
            "worsening_accepted": stats.worsening_accepted,
        }
        sa_accept_traj = [
            (h[0], h[2]) for h in stats.acceptance_history
        ]
    elif algo_name == "ga":
        cfg = GAConfig(population_size=60, crossover_rate=0.85, mutation_rate=0.15)
        best_pi, best_cost, stats = genetic_algorithm(ev, config=cfg, rng=rng)
        extra = {
            "generations": stats.total_generations,
            "final_diversity": stats.final_diversity,
            "initial_diversity": stats.initial_diversity,
        }
        ga_mean_traj = [(h[0], h[3]) for h in stats.history]
        ga_div_traj = [(h[0], h[4]) for h in stats.history]
    elif algo_name == "ga_nomx":
        cfg = GAConfig(population_size=60, crossover_rate=0.0, mutation_rate=0.25)
        best_pi, best_cost, stats = genetic_algorithm(ev, config=cfg, rng=rng)
        extra = {
            "generations": stats.total_generations,
            "final_diversity": stats.final_diversity,
            "initial_diversity": stats.initial_diversity,
        }
        ga_mean_traj = [(h[0], h[3]) for h in stats.history]
        ga_div_traj = [(h[0], h[4]) for h in stats.history]
    else:
        raise ValueError(f"Algoritmo desconhecido: {algo_name}")

    elapsed = time.perf_counter() - t_start

    eval_trajectory = sample_trajectory_at_checkpoints(ev.history, budget, n_points=50)

    return {
        "best_pi": best_pi,
        "best_cost": best_cost,
        "elapsed": elapsed,
        "evaluations": ev.evaluations_count,
        "trajectory": eval_trajectory,
        "ga_mean_traj": ga_mean_traj,
        "ga_div_traj": ga_div_traj,
        "sa_accept_traj": sa_accept_traj,
        "extra": extra,
    }


def run_benchmark_matrix(
    instances: List[str],
    algorithms: List[str],
    is_quick: bool = False,
    base_seed: int = 2026,
    verbose: bool = True,
) -> Dict[str, Any]:
    protocol = QUICK_PROTOCOL if is_quick else FULL_PROTOCOL

    all_run_records = []
    trajectories_by_combo = {}
    extra_telemetry = {}

    print("\n" + "=" * 80)
    print(f"INICIANDO MATRIZ EXPERIMENTAL — MODO: {'QUICK' if is_quick else 'OFICIAL (FULL)'}")
    print("=" * 80)

    for inst in instances:
        cfg = protocol[inst]
        budget = cfg["budget"]
        runs = cfg["runs"]
        filename = INSTANCE_FILES[inst]
        filepath = os.path.join("data", filename)
        if not os.path.exists(filepath):
            filepath = os.path.join("..", "data", filename)

        problem = MaintenanceProblem(filepath)
        ref_h = REFERENCE_OPTIMA[inst]

        print(f"\n>>> INSTÂNCIA: {inst.upper()} (n={problem.n}, Orçamento={budget:,}, Repetições={runs})")

        for algo in algorithms:
            combo_key = f"{inst}_{algo}"
            trajectories_by_combo[combo_key] = []
            combo_costs = []
            combo_times = []

            print(f"  * Algoritmo: {algo.upper()} ...", end="", flush=True)
            t_combo_start = time.perf_counter()

            for run_id in range(1, runs + 1):
                seed = base_seed + run_id * 100 + len(algo)
                res = run_single_experiment(algo, problem, budget, seed)

                cost = res["best_cost"]
                elapsed = res["elapsed"]
                combo_costs.append(cost)
                combo_times.append(elapsed)
                trajectories_by_combo[combo_key].append(res["trajectory"])

                gap = ((cost - ref_h) / ref_h) * 100.0 if inst == "pequena" else None

                record = {
                    "instance": inst,
                    "n": problem.n,
                    "algorithm": algo,
                    "run_id": run_id,
                    "seed": seed,
                    "budget": budget,
                    "best_cost": cost,
                    "gap_percent": gap,
                    "elapsed_seconds": elapsed,
                }
                record.update(res["extra"])
                all_run_records.append(record)

                if res["ga_mean_traj"]:
                    extra_telemetry.setdefault(f"{combo_key}_ga_mean", []).append(res["ga_mean_traj"])
                if res["ga_div_traj"]:
                    extra_telemetry.setdefault(f"{combo_key}_ga_div", []).append(res["ga_div_traj"])
                if res["sa_accept_traj"]:
                    extra_telemetry.setdefault(f"{combo_key}_sa_accept", []).append(res["sa_accept_traj"])

            t_combo_total = time.perf_counter() - t_combo_start
            m_cost = statistics.mean(combo_costs)
            s_cost = statistics.stdev(combo_costs) if len(combo_costs) > 1 else 0.0
            med_cost = statistics.median(combo_costs)
            min_cost = min(combo_costs)
            m_time = statistics.mean(combo_times)

            gap_str = f" | Gap(h*): {((min_cost - ref_h)/ref_h)*100:+.2f}%" if inst == "pequena" else ""
            print(
                f" Concluído em {t_combo_total:.2f}s | "
                f"Mín: {min_cost:.1f} | Média: {m_cost:.1f} ± {s_cost:.1f} | Mediana: {med_cost:.1f}{gap_str} | Tempo médio: {m_time:.3f}s"
            )

    return {
        "records": all_run_records,
        "trajectories": trajectories_by_combo,
        "telemetry": extra_telemetry,
    }


def save_results_to_csv(records: List[Dict[str, Any]], filepath: str):
    if not records:
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames_set = set()
    for r in records:
        fieldnames_set.update(r.keys())

    priority = [
        "instance", "n", "algorithm", "run_id", "seed", "budget", "best_cost", "gap_percent", "elapsed_seconds"
    ]
    remaining = sorted([k for k in fieldnames_set if k not in priority])
    fieldnames = [k for k in priority if k in fieldnames_set] + remaining

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\n[OK] Resultados detalhados salvos em: {filepath}")


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description="Runner Oficial de Experimentos TP2")
    parser.add_argument("--quick", action="store_true", help="Executa em modo rápido (smoke test)")
    parser.add_argument("--full", action="store_true", help="Executa o protocolo oficial completo do edital")
    parser.add_argument("--instances", nargs="+", default=["pequena", "media", "grande"], help="Instâncias a rodar")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["greedy", "sa", "ga", "ga_nomx"],
        help="Algoritmos a rodar: greedy, sa, ga, ga_nomx",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Semente base")
    parser.add_argument("--csv", type=str, default="data/resultados_oficiais_todos.csv", help="Arquivo CSV de saída")
    parser.add_argument(
        "--trajectories-json",
        type=str,
        default="data/convergencia_trajetorias.json",
        help="Arquivo JSON para salvar curvas e telemetria",
    )
    args = parser.parse_args()

    is_quick = args.quick or (not args.full)
    results = run_benchmark_matrix(
        instances=args.instances,
        algorithms=args.algorithms,
        is_quick=is_quick,
        base_seed=args.seed,
    )
    save_results_to_csv(results["records"], args.csv)

    with open(args.trajectories_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "trajectories": results["trajectories"],
                "telemetry": results["telemetry"],
            },
            f,
            indent=2,
        )
    print(f"[OK] Trajetórias salvas em: {args.trajectories_json}")
