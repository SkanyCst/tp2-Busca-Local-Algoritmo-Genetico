"""
Testes da avaliação incremental, vizinhanças, Evaluator e operadores do GA.

Rodar na raiz do projeto (precisa de:  pip install pytest):
    python -m pytest tests/test_incremental.py -v
ou, sem pytest instalado, só para ver se tudo passa:
    python -m tests.test_incremental
"""
import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.problem import MaintenanceProblem
from src.evaluator import Evaluator, BudgetExhaustedException
from src.operators import (
    get_swap_neighbors,
    get_insertion_neighbors,
    get_two_opt_neighbors,
    get_random_swap_successor,
    order_crossover_ox,
    swap_mutation,
)

NEIGHBORHOODS = {
    "swap": get_swap_neighbors,
    "insertion": get_insertion_neighbors,
    "2opt": get_two_opt_neighbors,
}

# Tamanho esperado de cada vizinhança, em função de n
EXPECTED_SIZE = {
    "swap": lambda n: n * (n - 1) // 2,
    "insertion": lambda n: (n - 1) ** 2,
    "2opt": lambda n: n * (n - 1) // 2,
}

# Instância da Figura 1. O PDF só traz um desenho esquemático; estas coordenadas
# foram reconstruídas a partir da tabela (t01=5, t14=5, t42=8, t23=10, t43=6).
FIGURA1 = """4
0 0
3 4 2 10 3
6 0 3 14 1
0 8 1 25 2
6 8 4 20 5
"""


# ---------------------------------------------------------------- helpers
def write_instance(tmp_path, n, seed=0, name=None):
    """Escreve uma instância sintética no formato dos arquivos do trabalho."""
    r = random.Random(seed)
    lines = [str(n), f"{r.randint(0, 100)} {r.randint(0, 100)}"]
    for _ in range(n):
        lines.append(
            f"{r.randint(0, 100)} {r.randint(0, 100)} {r.randint(1, 9)} "
            f"{r.randint(10, 10 * n)} {r.randint(1, 9)}"
        )
    path = tmp_path / (name or f"inst_{n}_{seed}.txt")
    path.write_text("\n".join(lines))
    return str(path)


def assert_incremental_matches_full(problem, pi, start, neighbor):
    """O custo incremental de `neighbor` tem que ser igual ao custo do zero."""
    assert neighbor[:start] == pi[:start], "start_index mente: o vizinho já mudou antes dele"
    expected = problem.h(neighbor)
    got = problem.h_incremental(neighbor, problem.h(pi), start)
    assert got == expected


# ------------------------------------------------------------- Figura 1
def test_figura1(tmp_path):
    path = tmp_path / "figura1.txt"
    path.write_text(FIGURA1)
    problem = MaintenanceProblem(str(path))
    assert problem.h([1, 4, 2, 3]) == 39   # rota do guloso por proximidade
    assert problem.h([1, 4, 3, 2]) == 22   # ótimo da instância


def test_figura1_otimo_por_forca_bruta(tmp_path):
    from itertools import permutations
    path = tmp_path / "figura1.txt"
    path.write_text(FIGURA1)
    problem = MaintenanceProblem(str(path))
    best = min(permutations(range(1, 5)), key=lambda p: problem.h(list(p)))
    assert list(best) == [1, 4, 3, 2]
    assert problem.h(list(best)) == 22


# ------------------------------------------- incremental == completo
@pytest.mark.parametrize("name", list(NEIGHBORHOODS))
@pytest.mark.parametrize("n", [2, 3, 8, 25])
def test_incremental_igual_completo_em_toda_a_vizinhanca(tmp_path, name, n):
    """Para TODOS os vizinhos de uma permutação aleatória."""
    problem = MaintenanceProblem(write_instance(tmp_path, n, seed=n))
    rng = random.Random(n)
    pi = list(range(1, n + 1))
    rng.shuffle(pi)

    for neighbor, start in NEIGHBORHOODS[name](pi):
        assert sorted(neighbor) == list(range(1, n + 1))   # ainda é permutação
        assert_incremental_matches_full(problem, pi, start, neighbor)


@pytest.mark.parametrize("name", list(NEIGHBORHOODS))
def test_incremental_em_varias_permutacoes_e_instancias(tmp_path, name):
    for seed in range(5):
        problem = MaintenanceProblem(write_instance(tmp_path, 15, seed=seed))
        rng = random.Random(100 + seed)
        for _ in range(5):
            pi = list(range(1, 16))
            rng.shuffle(pi)
            for neighbor, start in NEIGHBORHOODS[name](pi):
                assert_incremental_matches_full(problem, pi, start, neighbor)


def test_incremental_com_start_zero_e_ultimo_indice(tmp_path):
    problem = MaintenanceProblem(write_instance(tmp_path, 10))
    pi = list(range(1, 11))
    assert problem.h_incremental(pi, problem.h(pi), 0) == problem.h(pi)
    # mudar só os dois últimos ativos: prefixo de 8 reaproveitado
    neighbor = pi[:8] + [10, 9]
    assert problem.h_incremental(neighbor, problem.h(pi), 8) == problem.h(neighbor)


def test_incremental_encadeado_durante_uma_descida(tmp_path):
    """Simula o uso real: vai aceitando vizinhos e o custo guardado nunca deriva."""
    problem = MaintenanceProblem(write_instance(tmp_path, 20, seed=7))
    rng = random.Random(0)
    pi = list(range(1, 21))
    rng.shuffle(pi)
    cost = problem.h(pi)
    for _ in range(200):
        neighbor, start = get_random_swap_successor(pi)
        new_cost = problem.h_incremental(neighbor, cost, start)
        assert new_cost == problem.h(neighbor)
        if new_cost <= cost:
            pi, cost = neighbor, new_cost


# ------------------------------------------------------- vizinhanças
@pytest.mark.parametrize("name", list(NEIGHBORHOODS))
@pytest.mark.parametrize("n", [3, 8, 25])
def test_tamanho_e_ausencia_de_repeticao(name, n):
    pi = list(range(1, n + 1))
    neighbors = [tuple(nb) for nb, _ in NEIGHBORHOODS[name](pi)]
    assert len(neighbors) == EXPECTED_SIZE[name](n)
    assert len(set(neighbors)) == len(neighbors)     # nenhum vizinho repetido
    assert tuple(pi) not in neighbors                # nenhum igual à própria solução


def test_tamanhos_para_n150():
    assert EXPECTED_SIZE["swap"](150) == 11175       # valor citado no enunciado
    assert EXPECTED_SIZE["insertion"](150) == 22201
    assert EXPECTED_SIZE["2opt"](150) == 11175


def test_sucessor_aleatorio_valido(tmp_path):
    problem = MaintenanceProblem(write_instance(tmp_path, 12))
    pi = list(range(1, 13))
    todos = {tuple(nb) for nb, _ in get_swap_neighbors(pi)}
    for _ in range(300):
        neighbor, start = get_random_swap_successor(pi)
        assert tuple(neighbor) in todos
        assert_incremental_matches_full(problem, pi, start, neighbor)


# ---------------------------------------------------------- Evaluator
def test_evaluator_conta_completo_e_incremental_igualmente(tmp_path):
    problem = MaintenanceProblem(write_instance(tmp_path, 10))
    ev = Evaluator(problem, max_evaluations=100)
    pi = list(range(1, 11))
    ev.evaluate(pi)
    neighbor, start = get_random_swap_successor(pi)
    ev.evaluate_incremental(neighbor, problem.h(pi), start)
    assert ev.evaluations_count == 2     # 1 + 1


def test_evaluator_para_exatamente_no_orcamento(tmp_path):
    problem = MaintenanceProblem(write_instance(tmp_path, 10))
    ev = Evaluator(problem, max_evaluations=50)
    pi = list(range(1, 11))
    cost = problem.h(pi)
    with pytest.raises(BudgetExhaustedException):
        while True:
            neighbor, start = get_random_swap_successor(pi)
            ev.evaluate_incremental(neighbor, cost, start)
    assert ev.evaluations_count == 50


def test_evaluator_best_e_history(tmp_path):
    problem = MaintenanceProblem(write_instance(tmp_path, 15, seed=2))
    ev = Evaluator(problem, max_evaluations=2000)
    rng = random.Random(1)
    pi = list(range(1, 16))
    rng.shuffle(pi)
    cost = ev.evaluate(pi)
    try:
        while True:
            neighbor, start = get_random_swap_successor(pi)
            new_cost = ev.evaluate_incremental(neighbor, cost, start)
            if new_cost <= cost:
                pi, cost = neighbor, new_cost
    except BudgetExhaustedException:
        pass
    assert problem.h(ev.best_pi) == ev.best_cost       # best_pi corresponde a best_cost
    costs = [c for _, c in ev.history]
    assert costs == sorted(costs, reverse=True)        # o recorde só melhora
    assert ev.history[-1][1] == ev.best_cost


def test_evaluator_reset(tmp_path):
    problem = MaintenanceProblem(write_instance(tmp_path, 10))
    ev = Evaluator(problem, max_evaluations=10)
    ev.evaluate(list(range(1, 11)))
    ev.reset()
    assert ev.evaluations_count == 0 and ev.best_pi is None and ev.history == []


# ------------------------------------------------------ operadores do GA
def test_ox_e_mutacao_geram_permutacoes_validas():
    random.seed(9)
    base = list(range(1, 41))
    for _ in range(300):
        a, b = random.sample(base, 40), random.sample(base, 40)
        assert sorted(order_crossover_ox(a, b)) == base
        assert sorted(swap_mutation(a, pm=1.0)) == base


def test_mutacao_com_pm_zero_nao_altera():
    pi = list(range(1, 11))
    assert swap_mutation(pi, pm=0.0) == pi


def test_ox_preserva_trecho_do_pai1():
    """O filho deve conter, em bloco contíguo e na mesma posição, um trecho do pai 1."""
    random.seed(3)
    a = list(range(1, 21))
    b = a[::-1]
    for _ in range(100):
        child = order_crossover_ox(a, b)
        coincidencias = [k for k in range(20) if child[k] == a[k]]
        assert len(coincidencias) >= 2     # sempre herda pelo menos 2 genes de a (|i..j| >= 2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))