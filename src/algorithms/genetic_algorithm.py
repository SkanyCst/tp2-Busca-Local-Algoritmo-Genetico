import random
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.evaluator import Evaluator, BudgetExhaustedException
from src.operators import order_crossover_ox, swap_mutation


@dataclass
class GAConfig:
    population_size: int = 60
    crossover_rate: float = 0.85
    mutation_rate: float = 0.15
    tournament_size: int = 3
    elitism_count: int = 1


@dataclass
class GAStats:
    total_generations: int = 0
    history: List[Tuple[int, int, float, float, float]] = field(
        default_factory=list
    )  # (evaluations, generation, best_cost, mean_cost, diversity_ratio)


def _tournament_selection(
    population: List[List[int]],
    fitnesses: List[float],
    k: int,
    rng: random.Random,
) -> List[int]:
    """Torneio de tamanho k (minimização); retorna cópia do vencedor."""
    n_pop = len(population)
    selected_indices = rng.sample(range(n_pop), min(k, n_pop))
    best_idx = min(selected_indices, key=lambda idx: fitnesses[idx])
    return population[best_idx][:]


def _compute_diversity(population: List[List[int]]) -> float:
    """Calcula a fração de cromossomos únicos na população (entre 0.0 e 1.0)."""
    if not population:
        return 0.0
    unique_count = len({tuple(ind) for ind in population})
    return unique_count / len(population)


def genetic_algorithm(
    evaluator: Evaluator,
    config: Optional[GAConfig] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[List[int], float, GAStats]:
    """Algoritmo Genético com OX, swap mutation e elitismo, limitado pelo orçamento do Evaluator."""
    r = rng if rng is not None else random.Random()
    cfg = config if config is not None else GAConfig()
    n = evaluator.get_problem_size()
    stats = GAStats()

    pop_size = cfg.population_size
    population: List[List[int]] = []
    fitnesses: List[float] = []

    try:
        base_ind = list(range(1, n + 1))
        for _ in range(pop_size):
            ind = base_ind[:]
            r.shuffle(ind)
            cost = evaluator.evaluate(ind)
            population.append(ind)
            fitnesses.append(cost)
    except BudgetExhaustedException:
        if fitnesses:
            mean_f = statistics.mean(fitnesses)
            div = _compute_diversity(population)
            stats.history.append((evaluator.evaluations_count, 0, evaluator.best_cost, mean_f, div))
        return evaluator.best_pi, evaluator.best_cost, stats

    mean_f = statistics.mean(fitnesses)
    div = _compute_diversity(population)
    stats.history.append((evaluator.evaluations_count, 0, evaluator.best_cost, mean_f, div))

    generation = 0

    try:
        while True:
            generation += 1

            ranked_pop = sorted(zip(population, fitnesses), key=lambda x: x[1])
            new_population: List[List[int]] = []
            new_fitnesses: List[float] = []

            # elites são copiados sem reavaliação
            n_elites = min(cfg.elitism_count, pop_size)
            for i in range(n_elites):
                elite_ind, elite_cost = ranked_pop[i]
                new_population.append(elite_ind[:])
                new_fitnesses.append(elite_cost)

            while len(new_population) < pop_size:
                parent1 = _tournament_selection(population, fitnesses, cfg.tournament_size, r)
                parent2 = _tournament_selection(population, fitnesses, cfg.tournament_size, r)

                if r.random() < cfg.crossover_rate:
                    child = order_crossover_ox(parent1, parent2, rng=r)
                else:
                    child = parent1[:]

                child = swap_mutation(child, cfg.mutation_rate, rng=r)

                child_cost = evaluator.evaluate(child)
                new_population.append(child)
                new_fitnesses.append(child_cost)

            population = new_population
            fitnesses = new_fitnesses

            mean_f = statistics.mean(fitnesses)
            div = _compute_diversity(population)
            stats.history.append((evaluator.evaluations_count, generation, evaluator.best_cost, mean_f, div))
            stats.total_generations = generation

    except BudgetExhaustedException:
        pass

    stats.total_generations = generation
    return evaluator.best_pi, evaluator.best_cost, stats
