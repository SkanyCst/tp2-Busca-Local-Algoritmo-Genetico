class BudgetExhaustedException(Exception):
    """Exceção lançada quando o orçamento de avaliações é esgotado."""
    pass

class Evaluator:
    def __init__(self, problem, max_evaluations: int):
        self._problem = problem
        self.max_evaluations = max_evaluations
        self.evaluations_count = 0
        self.best_cost = float("inf")
        self.best_pi = None
        self.history = []

    def evaluate(self, pi) -> float:
        """Avalia a solução completa h(pi) incrementando o contador global."""
        if self.evaluations_count >= self.max_evaluations:
            raise BudgetExhaustedException("Orçamento máximo de avaliações atingido.")
        
        self.evaluations_count += 1
        cost = self._problem.h(pi)
        
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_pi = list(pi)
            self.history.append((self.evaluations_count, cost))
            
        return cost

    def evaluate_incremental(self, pi, prev_cost: float, delta_k: int) -> float:
        """
        Avalia o custo incremental de uma alteração no sufixo a partir do índice delta_k.
        Contabiliza exatamente 1 avaliação no orçamento global.
        """
        if self.evaluations_count >= self.max_evaluations:
            raise BudgetExhaustedException("Orçamento máximo de avaliações atingido.")
        
        self.evaluations_count += 1
        cost = self._problem.h_incremental(pi, prev_cost, delta_k)
        
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_pi = list(pi)
            self.history.append((self.evaluations_count, cost))
            
        return cost

    def get_problem_size(self) -> int:
        """Retorna n para conveniência dos operadores sem expor o problem diretamente."""
        return self._problem.n

    def reset(self):
        """Reinicia o estado do Evaluator para novas execuções independentes."""
        self.evaluations_count = 0
        self.best_cost = float("inf")
        self.best_pi = None
        self.history = []