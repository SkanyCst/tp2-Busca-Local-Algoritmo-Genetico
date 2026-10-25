import random
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, Set, Tuple
 
from src.evaluator import Evaluator, BudgetExhaustedException
 
Neighborhood = Callable[[List[int]], Iterator[Tuple[List[int], int]]]
 
 
@dataclass
class GreedyStats:
    improvement_steps: int = 0                                       # movimentos de melhora efetivados
    local_optima_costs: List[float] = field(default_factory=list)    # um por descida concluída
    local_optima: Set[tuple] = field(default_factory=set)            # ótimos locais distintos
 
    @property
    def completed_descents(self) -> int:
        return len(self.local_optima_costs)
 
    @property
    def distinct_local_optima(self) -> int:
        return len(self.local_optima)
 
 
def greedy_descent_restart(
    evaluator: Evaluator,
    neighborhood: Neighborhood,
    variant: str = "best",
    rng: Optional[random.Random] = None,
):
    """Roda descidas com random restart até o orçamento do evaluator acabar.

    Devolve (best_pi, best_cost, stats).
    """
    if variant not in ("best", "first"):
        raise ValueError("variant deve ser 'best' ou 'first'")
    rng = rng or random.Random()
    n = evaluator.get_problem_size()
    stats = GreedyStats()
 
    try:
        while True:
            pi = list(range(1, n + 1))
            rng.shuffle(pi)
            cost = evaluator.evaluate(pi)
            pi, cost = _descend(evaluator, neighborhood, variant, pi, cost, stats)
            # só chegamos aqui se a vizinhança inteira foi varrida sem melhora
            stats.local_optima_costs.append(cost)
            stats.local_optima.add(tuple(pi))
    except BudgetExhaustedException:
        pass
 
    return evaluator.best_pi, evaluator.best_cost, stats
 
 
def _descend(evaluator, neighborhood, variant, pi, cost, stats):
    """Desce até um ótimo local. Propaga BudgetExhaustedException se o orçamento acabar."""
    while True:
        move = _find_move(evaluator, neighborhood, variant, pi, cost)
        if move is None:
            return pi, cost               # ótimo local
        pi, cost = move
        stats.improvement_steps += 1
 
 
def _find_move(evaluator, neighborhood, variant, pi, cost):
    """Devolve (novo_pi, novo_custo) ou None se nenhum vizinho melhora."""
    best_neighbor, best_cost = None, cost
    for neighbor, start in neighborhood(pi):
        neighbor_cost = evaluator.evaluate_incremental(neighbor, cost, start)
        if neighbor_cost < best_cost:                  # melhora estrita (evita ciclar em platôs)
            best_neighbor, best_cost = neighbor, neighbor_cost
            if variant == "first":
                break
    if best_neighbor is None:
        return None
    return best_neighbor, best_cost