# TP2: Busca Local e Algoritmo Genético
**Fundamentos de Inteligência Artificial**  
Departamento de Ciência da Computação (DCC) — Universidade Federal de Minas Gerais (UFMG)

Este repositório contém a modelagem, implementação e análise experimental de heurísticas de busca local e metaheurísticas para o **Problema de Sequenciamento de Manutenção Preventiva de Ativos** com penalidade por atraso regulatório ponderado (*Total Weighted Tardiness*).

---

## Estrutura do Repositório

```text
├── data/                               # Arquivos de instâncias do problema
│   ├── figura1.txt                     # Exemplo ilustrativo do edital (n=4)
│   ├── manutencao_pequena.txt          # Instância pequena (n=12, ótimo conhecido h*=1875)
│   ├── manutencao_media.txt            # Instância média (n=50, referência h=16332)
│   └── manutencao_grande.txt           # Instância grande (n=150, referência h=60417)
├── docs/                               # Relatórios técnicos e fontes LaTeX
│   ├── relatorio_tp2.tex               # Relatório final de entrega do TP (formato oficial)
│   ├── relatorio_simulated_annealing.tex # Relatório técnico dedicado ao Simulated Annealing
│   └── relatorio_simulated_annealing.md  # Relatório técnico em Markdown
├── scripts/
│   └── benchmark_sa.py                 # Framework de amostragem, benchmark e exportação CSV
├── src/
│   ├── algorithms/
│   │   ├── greedy_descent.py           # Algoritmo A: Descida gulosa com Random Restart
│   │   ├── simulated_annealing.py      # Algoritmo B: Têmpera simulada com decaimento térmico
│   │   └── genetic_algorithm.py        # Algoritmo C: Algoritmo Genético populacional
│   ├── evaluator.py                    # Gerenciador estrito de orçamento de avaliações
│   ├── operators.py                    # Vizinhanças (Swap, Insertion, 2-opt) e operadores genéticos
│   └── problem.py                      # Modelagem do problema e avaliação incremental
└── tests/
    ├── test_figura1.py                 # Validação da Figura 1 do edital
    ├── test_verification.py            # Validação formal T1 (permutação identidade e budget)
    ├── test_simulated_annealing.py     # Testes unitários do Simulated Annealing
    └── test_sampling.py                # Teste do framework de amostragem
```

---

## Requisitos de Ambiente

O projeto foi construído utilizando prioritariamente a **biblioteca padrão do Python** (`math`, `random`, `statistics`, `dataclasses`, `unittest`, `argparse`), sem exigir instalação de bibliotecas externas pesadas para rodar o núcleo dos algoritmos.

* **Python:** Versão 3.10 ou superior.
* **Opcional (para suíte avançada de testes):**
  ```bash
  pip install pytest
  ```

---

## Instruções para Reprodução

### 1. Validação da Função de Avaliação e Orçamento (Tarefa T1)
Valida os casos obrigatórios do edital: permutação identidade ($h=4.366,0$) e valores da Figura 1 ($h_{\text{guloso}}=39,0$, $h^*=22,0$):
```bash
python3 tests/test_verification.py
python3 tests/test_figura1.py
```

### 2. Testes Unitários do Simulated Annealing
Executa a validação do respeito estrito ao orçamento, critério de Gibbs, decaimento térmico e consistência das soluções:
```bash
python3 tests/test_simulated_annealing.py
```

### 3. Testes do Framework de Amostragem
Verifica o comportamento do benchmark em diferentes arquivos e ordens de magnitude:
```bash
python3 tests/test_sampling.py
```

### 4. Amostragem Experimental e Benchmark (Geração de Dados e CSV)
O script `scripts/benchmark_sa.py` permite coletar métricas empíricas (Média $\pm$ Desvio, Mediana, Mínimo, Gap relativo ao ótimo, tempos de execução e taxas de aceitação):

* **Amostragem Rápida (*Smoke Test* em todas as instâncias):**
  ```bash
  python3 scripts/benchmark_sa.py --quick --verbose
  ```

* **Amostragem em Instância Específica (ex: 30 runs na pequena):**
  ```bash
  python3 scripts/benchmark_sa.py --instances pequena --runs 30
  ```

* **Protocolo Oficial Completo do Edital (30x pequena, 30x média, 10x grande):**
  ```bash
  python3 scripts/benchmark_sa.py --full --csv data/resultados_oficiais_sa.csv
  ```

* **Testar Instância Customizada:**
  ```bash
  python3 scripts/benchmark_sa.py --custom-file data/figura1.txt --budget 1000 --runs 5
  ```

---

## Relatórios Disponíveis

* **Relatório Oficial de Entrega:** [`docs/relatorio_tp2.tex`](docs/relatorio_tp2.tex) — documento consolidado do grupo com a modelagem, T1, comparação das vizinhanças (T2), resultados do Greedy Descent e Simulated Annealing, discussões teóricas e espaço reservado para o GA.
* **Relatório Técnico do Simulated Annealing:** [`docs/relatorio_simulated_annealing.tex`](docs/relatorio_simulated_annealing.tex) e [`docs/relatorio_simulated_annealing.md`](docs/relatorio_simulated_annealing.md) — documento aprofundado sobre a calibração de hiperparâmetros, taxas de aceitação de piora e métricas do Algoritmo B.
