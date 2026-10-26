import random
from typing import List, Generator, Tuple, Optional


def get_swap_neighbors(pi: List[int]) -> Generator[Tuple[List[int], int], None, None]:
    """Gera vizinhos por Swap e retorna (vizinho, primeiro_indice_alterado)."""
    n = len(pi)
    for i in range(n):
        for j in range(i + 1, n):
            neighbor = pi[:]
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            yield neighbor, i

def get_insertion_neighbors(pi: List[int]) -> Generator[Tuple[List[int], int], None, None]:
    """Gera vizinhos por Insertion, omitindo j == i - 1 (equivalente a j == i)."""
    n = len(pi)
    for i in range(n):
        for j in range(n):
            if i == j or j == i - 1:
                continue
            neighbor = pi[:]
            elem = neighbor.pop(i)
            neighbor.insert(j, elem)
            first_changed = min(i, j)
            yield neighbor, first_changed

def get_two_opt_neighbors(pi: List[int]) -> Generator[Tuple[List[int], int], None, None]:
    """Gera vizinhos por 2-opt (inversão do segmento i..j)."""
    n = len(pi)
    for i in range(n):
        for j in range(i + 1, n):
            neighbor = pi[:]
            neighbor[i:j+1] = reversed(neighbor[i:j+1])
            yield neighbor, i

def get_random_swap_successor(
    pi: List[int], rng: random.Random = None
) -> Tuple[List[int], int]:
    """Sorteia um vizinho por Swap e retorna (vizinho, primeiro_indice_alterado)."""
    r = rng if rng is not None else random
    n = len(pi)
    i, j = sorted(r.sample(range(n), 2))
    neighbor = pi[:]
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor, i

def get_random_insertion_successor(
    pi: List[int], rng: random.Random = None
) -> Tuple[List[int], int]:
    """Sorteia um vizinho por Insertion e retorna (vizinho, primeiro_indice_alterado)."""
    r = rng if rng is not None else random
    n = len(pi)
    i = r.randrange(n)
    valid_j = [j for j in range(n) if j != i and j != i - 1]
    j = r.choice(valid_j)
    neighbor = pi[:]
    elem = neighbor.pop(i)
    neighbor.insert(j, elem)
    return neighbor, min(i, j)

def get_random_two_opt_successor(
    pi: List[int], rng: random.Random = None
) -> Tuple[List[int], int]:
    """Sorteia um vizinho por 2-opt (inversão) e retorna (vizinho, primeiro_indice_alterado)."""
    r = rng if rng is not None else random
    n = len(pi)
    i, j = sorted(r.sample(range(n), 2))
    neighbor = pi[:]
    neighbor[i:j+1] = reversed(neighbor[i:j+1])
    return neighbor, i

def order_crossover_ox(
    parent1: List[int], parent2: List[int], rng: Optional[random.Random] = None
) -> List[int]:
    """Order Crossover: copia [i..j] de parent1 e preenche o resto, circularmente a partir de j+1, na ordem de parent2."""
    r = rng if rng is not None else random
    n = len(parent1)
    if n <= 1:
        return parent1[:]
    i, j = sorted(r.sample(range(n), 2))

    child = [None] * n
    child[i:j+1] = parent1[i:j+1]
    in_child = set(parent1[i:j+1])
    
    p2_idx = (j + 1) % n
    child_idx = (j + 1) % n
    
    for _ in range(n):
        val = parent2[p2_idx]
        if val not in in_child:
            child[child_idx] = val
            child_idx = (child_idx + 1) % n
        p2_idx = (p2_idx + 1) % n

    return child

def swap_mutation(
    pi: List[int], pm: float, rng: Optional[random.Random] = None
) -> List[int]:
    """Aplica mutação swap com probabilidade pm."""
    r = rng if rng is not None else random
    child = pi[:]
    if r.random() < pm:
        n = len(child)
        if n >= 2:
            i, j = r.sample(range(n), 2)
            child[i], child[j] = child[j], child[i]
    return child