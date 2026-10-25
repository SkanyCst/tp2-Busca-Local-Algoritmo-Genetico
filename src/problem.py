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
        """
        Calcula a matriz de tempos t_ij com arredondamento para o inteiro mais próximo.
        Índice 0 é a Base. Índices 1..N são os ativos.
        """
        all_points = [self.base_coords] + [(a.x, a.y) for a in self.assets]
        total_nodes = len(all_points)
        matrix = [[0] * total_nodes for _ in range(total_nodes)]
        
        for i in range(total_nodes):
            for j in range(total_nodes):
                if i != j:
                    dx = all_points[i][0] - all_points[j][0]
                    dy = all_points[i][1] - all_points[j][1]
                    matrix[i][j] = round(math.sqrt(dx**2 + dy**2)) # Equação (1)
        return matrix

    def h(self, pi: List[int]) -> float:
        """
        Função de Avaliação (Fitness): Atraso Ponderado Total.
        pi é uma permutação dos ativos (ex: [1, 2, ..., n])
        """
        c_current = 0.0
        prev_node = 0  # Começa na Base (índice 0)
        total_penalty = 0.0

        for asset_id in pi:
            asset = self.assets[asset_id - 1]
            travel_time = self.dist_matrix[prev_node][asset_id]
            
            c_current += travel_time + asset.p
            lateness = max(0.0, c_current - asset.d)
            total_penalty += asset.w * lateness
            
            prev_node = asset_id

        return total_penalty

    def h_incremental(self, pi: List[int], prev_cost: float, start_index: int) -> float:
        """
        Avaliação Incremental: Como h é acumulativo (não decrescente ao longo do prefixo),
        só precisamos recalcular os tempos de conclusão e penalidades do índice start_index em diante.
        """
        # Se a mudança ocorreu na primeira posição, avalia do zero
        if start_index == 0:
            return self.h(pi)

        # 1. Recupera o tempo de conclusão acumulado C até o índice anterior ao alterado
        c_current = 0.0
        prev_node = 0  # Base
        
        # Acumula C até (start_index - 1)
        for idx in range(start_index):
            asset_id = pi[idx]
            travel_time = self.dist_matrix[prev_node][asset_id]
            c_current += travel_time + self.assets[asset_id - 1].p
            prev_node = asset_id

        # 2. Recalcula o custo parcial do prefixo intacto
        prefix_penalty = 0.0
        p_node = 0
        c_temp = 0.0
        for idx in range(start_index):
            a_id = pi[idx]
            c_temp += self.dist_matrix[p_node][a_id] + self.assets[a_id - 1].p
            prefix_penalty += self.assets[a_id - 1].w * max(0.0, c_temp - self.assets[a_id - 1].d)
            p_node = a_id

        # 3. Recalcula o sufixo alterado a partir de start_index
        suffix_penalty = 0.0
        for idx in range(start_index, len(pi)):
            asset_id = pi[idx]
            asset = self.assets[asset_id - 1]
            travel_time = self.dist_matrix[prev_node][asset_id]
            c_current += travel_time + asset.p
            suffix_penalty += asset.w * max(0.0, c_current - asset.d)
            prev_node = asset_id

        return prefix_penalty + suffix_penalty