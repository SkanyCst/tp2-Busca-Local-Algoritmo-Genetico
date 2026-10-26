import math
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Asset:
    id: int
    x: float
    y: float
    p: float  # Duração em horas
    d: float  # Prazo regulatório em horas
    w: float  # Peso de criticidade

class MaintenanceProblem:
    def __init__(self, filepath: str):
        self.assets: List[Asset] = []
        self.base_coords: Tuple[float, float] = (0.0, 0.0)
        self.n: int = 0
        self._load_instance(filepath)
        self.dist_matrix: List[List[int]] = self._compute_dist_matrix()
        self._base_pi: List[int] = []
        self._base_prefix_c: List[float] = []
        self._base_prefix_cost: List[float] = []

    def _load_instance(self, filepath: str):
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        self.n = int(lines[0])
        base_x, base_y = map(float, lines[1].split())
        self.base_coords = (base_x, base_y)
        
        for idx in range(1, self.n + 1):
            parts = list(map(float, lines[1 + idx].split()))
            self.assets.append(Asset(
                id=idx,
                x=parts[0],
                y=parts[1],
                p=parts[2],
                d=parts[3],
                w=parts[4]
            ))

    def _compute_dist_matrix(self) -> List[List[int]]:
        """Matriz de tempos t_ij arredondados. Índice 0 é a base; 1..N são os ativos."""
        all_points = [self.base_coords] + [(a.x, a.y) for a in self.assets]
        total_nodes = len(all_points)
        matrix = [[0] * total_nodes for _ in range(total_nodes)]
        
        for i in range(total_nodes):
            for j in range(total_nodes):
                if i != j:
                    dx = all_points[i][0] - all_points[j][0]
                    dy = all_points[i][1] - all_points[j][1]
                    matrix[i][j] = round(math.sqrt(dx**2 + dy**2))
        return matrix

    def h(self, pi: List[int]) -> float:
        """Atraso ponderado total da permutação pi; guarda prefixos para h_incremental."""
        c_current = 0.0
        prev_node = 0
        total_penalty = 0.0

        prefix_c = []
        prefix_cost = []

        for asset_id in pi:
            asset = self.assets[asset_id - 1]
            travel_time = self.dist_matrix[prev_node][asset_id]
            
            c_current += travel_time + asset.p
            lateness = max(0.0, c_current - asset.d)
            total_penalty += asset.w * lateness
            
            prev_node = asset_id
            prefix_c.append(c_current)
            prefix_cost.append(total_penalty)

        self._base_pi = list(pi)
        self._base_prefix_c = prefix_c
        self._base_prefix_cost = prefix_cost

        return total_penalty

    def h_incremental(self, pi: List[int], prev_cost: float, start_index: int) -> float:
        """Recalcula h apenas a partir de start_index, reaproveitando o prefixo em cache."""
        if start_index == 0:
            return self.h(pi)

        if (
            len(self._base_pi) == len(pi)
            and self._base_prefix_c
            and self._base_pi[:start_index] == pi[:start_index]
        ):
            c_current = self._base_prefix_c[start_index - 1]
            prefix_penalty = self._base_prefix_cost[start_index - 1]
            prev_node = pi[start_index - 1]
        else:
            # cache não corresponde a pi: recalcula o prefixo
            c_current = 0.0
            prev_node = 0
            prefix_penalty = 0.0
            for idx in range(start_index):
                asset_id = pi[idx]
                asset = self.assets[asset_id - 1]
                c_current += self.dist_matrix[prev_node][asset_id] + asset.p
                prefix_penalty += asset.w * max(0.0, c_current - asset.d)
                prev_node = asset_id

        suffix_penalty = 0.0
        for idx in range(start_index, len(pi)):
            asset_id = pi[idx]
            asset = self.assets[asset_id - 1]
            travel_time = self.dist_matrix[prev_node][asset_id]
            c_current += travel_time + asset.p
            suffix_penalty += asset.w * max(0.0, c_current - asset.d)
            prev_node = asset_id

        return prefix_penalty + suffix_penalty