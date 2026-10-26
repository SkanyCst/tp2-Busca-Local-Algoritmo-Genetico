import os
import random
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.problem import MaintenanceProblem
from src.evaluator import Evaluator
from src.algorithms.genetic_algorithm import genetic_algorithm, GAConfig


def write_instance(tmp_path, n, seed=0):
    r = random.Random(seed)
    lines = [str(n), f"{r.randint(0, 100)} {r.randint(0, 100)}"]
    for _ in range(n):
        lines.append(
            f"{r.randint(0, 100)} {r.randint(0, 100)} {r.randint(1, 9)} "
            f"{r.randint(10, 10 * n)} {r.randint(1, 9)}"
        )
    path = tmp_path / f"ga_inst_{n}_{seed}.txt"
    path.write_text("\n".join(lines))
    return str(path)


def test_ga_respeita_orcamento_estrito(tmp_path):
    budget = 1000
    problem = MaintenanceProblem(write_instance(tmp_path, n=12, seed=42))
    ev = Evaluator(problem, max_evaluations=budget)
    best_pi, best_cost, stats = genetic_algorithm(ev, rng=random.Random(123))

    assert ev.evaluations_count == budget
    assert len(best_pi) == 12
    assert sorted(best_pi) == list(range(1, 13))
    assert problem.h(best_pi) == best_cost == ev.best_cost


def test_ga_reprodutibilidade_com_mesma_semente(tmp_path):
    inst = write_instance(tmp_path, n=10, seed=7)
    
    ev1 = Evaluator(MaintenanceProblem(inst), max_evaluations=500)
    pi1, cost1, _ = genetic_algorithm(ev1, rng=random.Random(42))

    ev2 = Evaluator(MaintenanceProblem(inst), max_evaluations=500)
    pi2, cost2, _ = genetic_algorithm(ev2, rng=random.Random(42))

    assert cost1 == cost2
    assert pi1 == pi2


def test_ga_telemetria_e_diversidade(tmp_path):
    problem = MaintenanceProblem(write_instance(tmp_path, n=12, seed=99))
    ev = Evaluator(problem, max_evaluations=600)
    cfg = GAConfig(population_size=30)
    best_pi, best_cost, stats = genetic_algorithm(ev, config=cfg, rng=random.Random(1))

    assert stats.total_generations > 0
    assert len(stats.history) > 0

    # Verifica que em cada geração registrada:
    # 0 <= diversity <= 1.0 e best_cost <= mean_cost
    for evals, gen, b_cost, m_cost, div in stats.history:
        assert 0.0 <= div <= 1.0
        assert b_cost <= m_cost + 1e-6
        assert evals <= 600


def test_ga_figura1_encontra_boa_solucao(tmp_path):
    path = tmp_path / "f1.txt"
    path.write_text("4\n0 0\n3 4 2 10 3\n6 0 3 14 1\n0 8 1 25 2\n6 8 4 20 5\n")
    ev = Evaluator(MaintenanceProblem(str(path)), max_evaluations=500)
    best_pi, best_cost, _ = genetic_algorithm(
        ev, config=GAConfig(population_size=20), rng=random.Random(0)
    )
    assert best_cost == 22.0
    assert best_pi == [1, 4, 3, 2]
