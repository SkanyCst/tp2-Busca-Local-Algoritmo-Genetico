import random
from typing import List, Generator, Tuple

# --- VIZINHANÇAS COM ÍNDICE DE MUDANÇA PARA AVALIAÇÃO INCREMENTAL ---

def get_swap_neighbors(pi: List[int]) -> Generator[Tuple[List[int], int], None, None]:
    """Gera vizinhos por Swap e retorna (vizinho, primeiro_indice_alterado)."""
    n = len(pi)
    for i in range(n):
        for j in range(i + 1, n):
            neighbor = pi[:]
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            yield neighbor, i

def get_insertion_neighbors(pi: List[int]) -> Generator[Tuple[List[int], int], None, None]:
    """
    Gera vizinhos por Insertion sem duplicações redundantes.
    Remove a redundância de j == i - 1.
    """
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

# --- SUCESSORES ALEATÓRIOS (Para SA e Greedy Random) ---

def get_random_swap_successor(pi: List[int]) -> Tuple[List[int], int]:
    n = len(pi)
    i, j = sorted(random.sample(range(n), 2))
    neighbor = pi[:]
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor, i

# --- OPERADORES GENÉTICOS (Para o Genetic Algorithm) ---

def order_crossover_ox(parent1: List[int], parent2: List[int]) -> List[int]:
    """Order Crossover (OX): Preserva a validade de permutações sem duplicar ativos."""
    n = len(parent1)
    i, j = sorted(random.sample(range(n), 2))
    
    child = [None] * n
    child[i:j+1] = parent1[i:j+1]
    
    p2_idx = 0
    for k in range(n):
        if child[k] is None:
            while parent2[p2_idx] in child:
                p2_idx += 1
            child[k] = parent2[p2_idx]
            
    return child

def swap_mutation(pi: List[int], pm: float) -> List[int]:
    """Aplica mutação swap com probabilidade pm."""
    child = pi[:]
    if random.random() < pm:
        n = len(child)
        i, j = random.sample(range(n), 2)
        child[i], child[j] = child[j], child[i]
    return child