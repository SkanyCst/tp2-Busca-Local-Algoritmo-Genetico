# TP2: Busca Local e Algoritmo Genético
**Fundamentos de Inteligência Artificial**  
Departamento de Ciência da Computação (DCC) — Universidade Federal de Minas Gerais (UFMG)

Este repositório contém a modelagem formal, implementação computacional e análise experimental de heurísticas de busca local (**Greedy Descent com Random Restart** e **Simulated Annealing**) e metaheurísticas populacionais (**Algoritmo Genético com Order Crossover**) para o **Problema de Sequenciamento de Manutenção Preventiva de Ativos** sujeito a penalidades por atraso regulatório ponderado (*Total Weighted Tardiness*).

---

## Estrutura do Repositório

```text
├── data/                                 # Instâncias e resultados experimentais
│   ├── figura1.txt                       # Exemplo ilustrativo do edital (n=4)
│   ├── manutencao_pequena.txt            # Instância pequena (n=12, ótimo conhecido h*=1875)
│   ├── manutencao_media.txt              # Instância média (n=50, referência h=16332)
│   ├── manutencao_grande.txt             # Instância grande (n=150, referência h=60417)
│   ├── t2_neighborhoods_comparison.csv   # Comparação das 3 vizinhanças (T2)
│   ├── resultados_oficiais_todos.csv     # Matriz completa dos experimentos oficiais (3x3)
│   ├── tabela_principal_resultados.csv   # Tabela consolidada com estatísticas e tempos
│   └── convergencia_trajetorias.json     # Históricos e telemetria para geração de curvas
├── docs/                                 # Relatórios técnicos e gráficos gerados
│   ├── relatorio_tp2.tex                 # Relatório oficial de entrega em LaTeX
│   ├── relatorio_tp2.md                  # Relatório completo formatado em Markdown
│   └── figures/                          # Gráficos em alta resolução
│       ├── boxplots_comparativos.png     # Boxplots dos algoritmos nas 3 instâncias
│       ├── curvas_convergencia.png       # Curvas de convergência e média da população do GA
│       ├── sa_taxa_aceitacao.png         # Telemetria do SA (transição de fase)
│       └── ga_evolucao_diversidade.png   # Evolução da diversidade do GA
├── scripts/
│   ├── benchmark_neighborhoods_t2.py     # Comparação empírica de vizinhanças e ganho incremental (T2)
│   ├── run_experiments.py                # Runner oficial dos 3 algoritmos + experimento adicional do GA
│   └── generate_plots_and_stats.py       # Gerador de gráficos, boxplots e testes de Mann-Whitney U
├── src/
│   ├── algorithms/
│   │   ├── greedy_descent.py             # Algoritmo A: Descida gulosa com Random Restart
│   │   ├── simulated_annealing.py        # Algoritmo B: Têmpera simulada com decaimento térmico
│   │   └── genetic_algorithm.py          # Algoritmo C: Algoritmo Genético com OX e Elitismo
│   ├── evaluator.py                      # Gerenciador estrito de orçamento de avaliações
│   ├── operators.py                      # Vizinhanças (Swap, Insertion, 2-opt) e operadores genéticos
│   └── problem.py                        # Modelagem do problema e avaliação incremental O(n-k)
└── tests/
    ├── test_figura1.py                   # Validação da Figura 1 do edital
    ├── test_verification.py              # Validação formal T1 (permutação identidade e budget)
    ├── test_greedy.py                    # Testes unitários e verificação de arquitetura AST
    ├── test_simulated_annealing.py       # Testes unitários do Simulated Annealing
    ├── test_genetic_algorithm.py         # Testes unitários do Algoritmo Genético
    └── test_incremental.py              # Validação de concordância incremental vs completa
```

---

## Requisitos de Ambiente

O projeto utiliza primariamente a biblioteca padrão do Python, complementada por bibliotecas científicas padrão para amostragem estatística e geração de figuras.

* **Python:** Versão 3.10 ou superior (testado em Python 3.12).
* **Dependências:**
  ```bash
  pip install pytest numpy scipy pandas matplotlib seaborn
  ```

---

## Instruções para Reprodução

### 1. Execução de Todos os Testes Unitários
Valida a função de avaliação (T1), concordância matemática entre avaliação incremental e completa, e integridade dos 3 algoritmos:
```bash
python tests/test_verification.py
pytest tests -v
```

### 2. Comparação de Vizinhanças e Avaliação Incremental (Tarefa T2)
Executa a comparação empírica entre **Swap**, **Insertion** e **2-opt** sob o mesmo orçamento e mede o speedup de tempo de parede da avaliação incremental:
```bash
python scripts/benchmark_neighborhoods_t2.py --budget 20000
```
*Gera a tabela comparativa e salva em `data/t2_neighborhoods_comparison.csv`.*

### 3. Protocolo Experimental Completo (Tarefa T3 e Experimento Adicional do GA)
Para executar o protocolo oficial com os orçamentos e repetições do edital (30x pequena, 30x média, 10x grande):

* **Execução Oficial Completa:**
  ```bash
  python scripts/run_experiments.py --full
  ```

* **Smoke Test Rápido (para verificação ágil):**
  ```bash
  python scripts/run_experiments.py --quick
  ```

*Os dados detalhados de cada execução são salvos em `data/resultados_oficiais_todos.csv` e `data/convergencia_trajetorias.json`.*

### 4. Geração Automática de Todas as Figuras, Tabelas e Testes Estatísticos
Processa os dados coletados e produz:
1. **Tabela Principal (Tabela 1 do relatório)** com média $\pm$ desvio, mediana, melhor $h$, gap relativo ao ótimo e tempo de execução;
2. **Boxplots Comparativos** (`docs/figures/boxplots_comparativos.png`);
3. **Curvas de Convergência** com a média da população do GA (`docs/figures/curvas_convergencia.png`);
4. **Telemetria do SA** (`docs/figures/sa_taxa_aceitacao.png`);
5. **Evolução da Diversidade do GA** (`docs/figures/ga_evolucao_diversidade.png`);
6. **Testes Não-Paramétricos de Mann-Whitney U** com $p$-valores para todas as comparações par a par e o experimento adicional do GA.

Comando:
```bash
python scripts/generate_plots_and_stats.py
```

---

## Relatórios Disponíveis

* **Relatório Oficial de Entrega:** [`docs/relatorio_tp2.tex`](docs/relatorio_tp2.tex) — fonte LaTeX pronto para compilação ou submissão ao Overleaf.
* **Relatório Completo em Markdown:** [`docs/relatorio_tp2.md`](docs/relatorio_tp2.md) — documento com todas as tabelas reais consolidadas, equações renderizadas e figuras embutidas para leitura e revisão direta.
