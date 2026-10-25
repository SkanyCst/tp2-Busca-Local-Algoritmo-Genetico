import os
import random
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.problem import MaintenanceProblem
from src.evaluator import Evaluator
from src.operators import (
    get_random_swap_successor,
    get_random_insertion_successor,
    get_random_two_opt_successor,
)
from src.algorithms.simulated_annealing import (
    simulated_annealing,
    CoolingSchedule,
    get_default_schedule,
)


class TestSimulatedAnnealing(unittest.TestCase):
    def setUp(self):
        self.fig1_path = os.path.join("data", "figura1.txt")
        self.pequena_path = os.path.join("data", "manutencao_pequena.txt")

    def test_usa_exatamente_o_orcamento(self):
        prob = MaintenanceProblem(self.pequena_path)
        budget = 500
        ev = Evaluator(prob, max_evaluations=budget)
        best_pi, best_cost, stats = simulated_annealing(
            ev, rng=random.Random(42)
        )
        self.assertEqual(ev.evaluations_count, budget)
        self.assertEqual(stats.total_steps, budget - 1)

    def test_retorna_permutacao_valida(self):
        prob = MaintenanceProblem(self.pequena_path)
        ev = Evaluator(prob, max_evaluations=200)
        best_pi, best_cost, _ = simulated_annealing(ev, rng=random.Random(7))
        self.assertEqual(sorted(best_pi), list(range(1, prob.n + 1)))
        self.assertEqual(prob.h(best_pi), best_cost)
        self.assertEqual(best_cost, ev.best_cost)

    def test_reprodutibilidade_com_mesma_semente(self):
        prob = MaintenanceProblem(self.pequena_path)
        ev1 = Evaluator(prob, max_evaluations=300)
        res1 = simulated_annealing(ev1, rng=random.Random(123))

        ev2 = Evaluator(prob, max_evaluations=300)
        res2 = simulated_annealing(ev2, rng=random.Random(123))

        self.assertEqual(res1[0], res2[0])
        self.assertEqual(res1[1], res2[1])

    def test_temperatura_extrema_alta_comporta_como_random_walk(self):
        prob = MaintenanceProblem(self.pequena_path)
        ev = Evaluator(prob, max_evaluations=1000)
        schedule = CoolingSchedule(t0=1e9, tf=1e9, alpha=1.0)
        _, _, stats = simulated_annealing(
            ev, schedule=schedule, rng=random.Random(42)
        )
        self.assertGreater(stats.worsening_proposed, 0)
        self.assertGreater(stats.global_acceptance_rate, 0.95)
        self.assertEqual(stats.rejections, 0)

    def test_temperatura_extrema_baixa_comporta_como_greedy(self):
        prob = MaintenanceProblem(self.pequena_path)
        ev = Evaluator(prob, max_evaluations=1000)
        schedule = CoolingSchedule(t0=1e-9, tf=1e-9, alpha=1.0)
        _, _, stats = simulated_annealing(
            ev, schedule=schedule, rng=random.Random(42)
        )
        self.assertGreater(stats.worsening_proposed, 0)
        self.assertEqual(stats.worsening_accepted, 0)
        self.assertEqual(stats.global_acceptance_rate, 0.0)

    def test_transicao_temperatura_bem_calibrada(self):
        prob = MaintenanceProblem(self.pequena_path)
        budget = 3000
        ev = Evaluator(prob, max_evaluations=budget)
        schedule = get_default_schedule(prob.n)
        _, _, stats = simulated_annealing(
            ev, schedule=schedule, window_size=50, rng=random.Random(42)
        )
        self.assertGreater(stats.initial_acceptance_rate, 0.65)
        self.assertLess(stats.final_acceptance_rate, 0.05)

    def test_encontra_otimo_da_figura1(self):
        prob = MaintenanceProblem(self.fig1_path)
        ev = Evaluator(prob, max_evaluations=500)
        best_pi, best_cost, _ = simulated_annealing(
            ev,
            schedule=CoolingSchedule(t0=50.0, tf=0.01),
            rng=random.Random(42),
        )
        self.assertEqual(best_cost, 22.0)
        self.assertEqual(best_pi, [1, 4, 3, 2])

    def test_suporte_a_diferentes_operadores_de_vizinhanca(self):
        prob = MaintenanceProblem(self.pequena_path)
        for name, op in [
            ("swap", get_random_swap_successor),
            ("insertion", get_random_insertion_successor),
            ("2opt", get_random_two_opt_successor),
        ]:
            ev = Evaluator(prob, max_evaluations=150)
            best_pi, best_cost, stats = simulated_annealing(
                ev, successor_fn=op, rng=random.Random(10)
            )
            self.assertEqual(len(best_pi), prob.n, f"Falha no operador {name}")
            self.assertEqual(ev.evaluations_count, 150)


if __name__ == "__main__":
    unittest.main()
