"""Testes do greedy descent com random restart.
Rodar:  python -m pytest tests/test_greedy.py -v
"""
import os
import pathlib
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.problem import MaintenanceProblem
from src.evaluator import Evaluator
from src.operators import get_swap_neighbors, get_insertion_neighbors, get_two_opt_neighbors
from src.algorithms.greedy_descent import greedy_descent_restart

NEIGHBORHOODS = {
    "swap": get_swap_neighbors,
    "insertion": get_insertion_neighbors,
    "2opt": get_two_opt_neighbors,
}
VARIANTS = ["best", "first"]


def write_instance(tmp_path, n, seed=0):
    r = random.Random(seed)
    lines = [str(n), f"{r.randint(0, 100)} {r.randint(0, 100)}"]
    for _ in range(n):
        lines.append(f"{r.randint(0, 100)} {r.randint(0, 100)} {r.randint(1, 9)} "
                     f"{r.randint(10, 10 * n)} {r.randint(1, 9)}")
    path = tmp_path / f"i{n}_{seed}.txt"
    path.write_text("\n".join(lines))
    return str(path)


def run(tmp_path, n, budget, hood, variant, seed=0):
    problem = MaintenanceProblem(write_instance(tmp_path, n, seed=1))
    ev = Evaluator(problem, max_evaluations=budget)
    result = greedy_descent_restart(ev, NEIGHBORHOODS[hood], variant, random.Random(seed))
    return problem, ev, result


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("hood", list(NEIGHBORHOODS))
def test_usa_exatamente_o_orcamento(tmp_path, hood, variant):
    _, ev, _ = run(tmp_path, n=12, budget=2000, hood=hood, variant=variant)
    assert ev.evaluations_count == 2000


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("hood", list(NEIGHBORHOODS))
def test_resultado_consistente(tmp_path, hood, variant):
    problem, ev, (best_pi, best_cost, stats) = run(tmp_path, 12, 3000, hood, variant)
    assert sorted(best_pi) == list(range(1, 13))
    assert problem.h(best_pi) == best_cost == ev.best_cost
    if stats.local_optima_costs:
        assert best_cost <= min(stats.local_optima_costs)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("hood", list(NEIGHBORHOODS))
def test_otimos_locais_registrados_sao_realmente_locais(tmp_path, hood, variant):
    """Nenhum vizinho de um ótimo local registrado pode ter custo menor."""
    problem, _, (_, _, stats) = run(tmp_path, 8, 6000, hood, variant)
    assert stats.distinct_local_optima > 0
    for pi_tuple in stats.local_optima:
        pi = list(pi_tuple)
        cost = problem.h(pi)
        assert all(problem.h(nb) >= cost for nb, _ in NEIGHBORHOODS[hood](pi))


@pytest.mark.parametrize("variant", VARIANTS)
def test_contadores_de_estatistica(tmp_path, variant):
    _, _, (_, _, stats) = run(tmp_path, 10, 4000, "swap", variant)
    assert stats.completed_descents == len(stats.local_optima_costs) > 0
    assert stats.distinct_local_optima <= stats.completed_descents
    assert stats.improvement_steps > 0


@pytest.mark.parametrize("variant", VARIANTS)
def test_reprodutivel_com_mesma_semente(tmp_path, variant):
    a = run(tmp_path, 12, 2000, "insertion", variant, seed=5)[2]
    b = run(tmp_path, 12, 2000, "insertion", variant, seed=5)[2]
    assert a[1] == b[1] and a[0] == b[0]


@pytest.mark.parametrize("variant", VARIANTS)
def test_encontra_otimo_da_figura1(tmp_path, variant):
    path = tmp_path / "f1.txt"
    path.write_text("4\n0 0\n3 4 2 10 3\n6 0 3 14 1\n0 8 1 25 2\n6 8 4 20 5\n")
    ev = Evaluator(MaintenanceProblem(str(path)), max_evaluations=300)
    best_pi, best_cost, _ = greedy_descent_restart(ev, get_swap_neighbors, variant, random.Random(0))
    assert best_cost == 22 and best_pi == [1, 4, 3, 2]


def test_variante_invalida(tmp_path):
    ev = Evaluator(MaintenanceProblem(write_instance(tmp_path, 5)), max_evaluations=10)
    with pytest.raises(ValueError):
        greedy_descent_restart(ev, get_swap_neighbors, "qualquer")


def test_arquitetura_algoritmo_nao_acessa_o_problema():
    """Algoritmos só usam o Evaluator (contagem justa de avaliações).
    Analisa o código via AST, então docstrings e comentários não geram falso positivo."""
    import ast
    pasta = pathlib.Path(__file__).resolve().parent.parent / "src" / "algorithms"
    attrs_proibidos = {"h", "h_incremental", "_problem"}
    for arquivo in pasta.glob("*.py"):
        arvore = ast.parse(arquivo.read_text(encoding="utf-8"))
        for no in ast.walk(arvore):
            if isinstance(no, ast.ImportFrom):
                assert no.module != "src.problem", f"{arquivo.name} importa src.problem"
                assert all(a.name != "MaintenanceProblem" for a in no.names)
            elif isinstance(no, ast.Import):
                assert all(a.name != "src.problem" for a in no.names)
            elif isinstance(no, ast.Attribute):
                assert no.attr not in attrs_proibidos, \
                    f"{arquivo.name}:{no.lineno} acessa .{no.attr} (use o Evaluator)"