import inspect
import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from src.evaluator import Evaluator, BudgetExhaustedException
from src.operators import get_random_swap_successor

SuccessorFn = Callable[[List[int]], Tuple[List[int], int]]


@dataclass(frozen=True)
class CoolingSchedule:
    t0: float
    tf: float
    alpha: Optional[float] = None

    def get_alpha(self, total_budget: int) -> float:
        if self.alpha is not None:
            return self.alpha
        if total_budget <= 1 or self.t0 <= 0 or self.tf <= 0:
            return 1.0
        return (self.tf / self.t0) ** (1.0 / total_budget)


def get_default_schedule(n: int) -> CoolingSchedule:
    # Calibrado com base na magnitude media de delta_h de cada instancia
    if n <= 15:
        return CoolingSchedule(t0=4000.0, tf=0.1)
    if n <= 60:
        return CoolingSchedule(t0=25000.0, tf=0.5)
    return CoolingSchedule(t0=70000.0, tf=1.0)


@dataclass
class SAStats:
    total_steps: int = 0
    improvements: int = 0
    worsening_proposed: int = 0
    worsening_accepted: int = 0
    rejections: int = 0
    acceptance_history: List[Tuple[int, float, float, float, float]] = field(
        default_factory=list
    )

    @property
    def global_acceptance_rate(self) -> float:
        if self.worsening_proposed == 0:
            return 0.0
        return self.worsening_accepted / self.worsening_proposed

    @property
    def initial_acceptance_rate(self) -> float:
        if not self.acceptance_history:
            return 0.0
        return self.acceptance_history[0][2]

    @property
    def final_acceptance_rate(self) -> float:
        if not self.acceptance_history:
            return 0.0
        return self.acceptance_history[-1][2]


def simulated_annealing(
    evaluator: Evaluator,
    schedule: Optional[CoolingSchedule] = None,
    successor_fn: Optional[SuccessorFn] = None,
    initial_pi: Optional[List[int]] = None,
    window_size: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[List[int], float, SAStats]:
    """Simulated Annealing com criterio de aceitacao de Gibbs/Metropolis."""
    random_gen = rng if rng is not None else random.Random()
    n = evaluator.get_problem_size()
    stats = SAStats()

    sched = schedule if schedule is not None else get_default_schedule(n)
    alpha = sched.get_alpha(evaluator.max_evaluations)
    current_temp = sched.t0

    if successor_fn is not None:
        sig = inspect.signature(successor_fn)
        if "rng" in sig.parameters:
            gen_successor = lambda p: successor_fn(p, rng=random_gen)
        else:
            gen_successor = successor_fn
    else:
        gen_successor = lambda p: get_random_swap_successor(p, rng=random_gen)

    window = window_size if window_size is not None else max(50, evaluator.max_evaluations // 200)
    window_worse_proposed = 0
    window_worse_accepted = 0

    try:
        if initial_pi is not None:
            pi = list(initial_pi)
        else:
            pi = list(range(1, n + 1))
            random_gen.shuffle(pi)

        cost = evaluator.evaluate(pi)

        while True:
            neighbor, start_idx = gen_successor(pi)
            neighbor_cost = evaluator.evaluate_incremental(neighbor, cost, start_idx)
            delta_cost = neighbor_cost - cost

            accepted = False
            if delta_cost <= 0.0:
                accepted = True
                stats.improvements += 1
            else:
                stats.worsening_proposed += 1
                window_worse_proposed += 1

                p = math.exp(-delta_cost / current_temp) if current_temp > 1e-12 else 0.0
                if random_gen.random() < p:
                    accepted = True
                    stats.worsening_accepted += 1
                    window_worse_accepted += 1
                else:
                    stats.rejections += 1

            if accepted:
                pi, cost = neighbor, neighbor_cost

            current_temp = max(1e-12, current_temp * alpha)
            stats.total_steps += 1

            if stats.total_steps % window == 0:
                rate = (
                    (window_worse_accepted / window_worse_proposed)
                    if window_worse_proposed > 0
                    else 0.0
                )
                stats.acceptance_history.append((
                    evaluator.evaluations_count,
                    current_temp,
                    rate,
                    cost,
                    evaluator.best_cost,
                ))
                window_worse_proposed = 0
                window_worse_accepted = 0

    except BudgetExhaustedException:
        pass

    if window_worse_proposed > 0:
        rate = window_worse_accepted / window_worse_proposed
        stats.acceptance_history.append((
            evaluator.evaluations_count,
            current_temp,
            rate,
            cost,
            evaluator.best_cost,
        ))

    return evaluator.best_pi, evaluator.best_cost, stats
