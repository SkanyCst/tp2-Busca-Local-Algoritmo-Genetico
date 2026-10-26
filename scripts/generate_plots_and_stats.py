#!/usr/bin/env python3
"""
Gera todos os gráficos, boxplots, curvas de convergência e testes estatísticos (Mann-Whitney)
exigidos pelas Seções 3, 4 e 5 do edital do TP2.
"""

import json
import os
import sys
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

FIGURES_DIR = os.path.join("docs", "figures")
DATA_CSV = os.path.join("data", "resultados_oficiais_todos.csv")
TRAJECTORIES_JSON = os.path.join("data", "convergencia_trajetorias.json")
T2_CSV = os.path.join("data", "t2_neighborhoods_comparison.csv")

ALGO_COLORS = {
    "greedy": "#D9534F",    # Vermelho telha
    "sa": "#0275D8",        # Azul
    "ga": "#5CB85C",        # Verde
    "ga_nomx": "#F0AD4E",   # Amarelo/Laranja
}

ALGO_NAMES = {
    "greedy": "Greedy Descent",
    "sa": "Simulated Annealing",
    "ga": "Genetic Algorithm (OX)",
    "ga_nomx": "GA sem Crossover (Mutação)",
}


def setup_matplotlib():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.autolayout": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })
    os.makedirs(FIGURES_DIR, exist_ok=True)


def generate_boxplots(df: pd.DataFrame):
    """Gera boxplots comparativos para cada instância comparando os 3 algoritmos principais."""
    main_algos = ["greedy", "sa", "ga"]
    instances = ["pequena", "media", "grande"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, inst in enumerate(instances):
        ax = axes[idx]
        sub = df[(df["instance"] == inst) & (df["algorithm"].isin(main_algos))]

        data_to_plot = [sub[sub["algorithm"] == a]["best_cost"].values for a in main_algos]
        labels = [ALGO_NAMES[a] for a in main_algos]

        bp = ax.boxplot(
            data_to_plot,
            tick_labels=labels,
            patch_artist=True,
            showmeans=True,
            meanline=True,
        )

        for patch, algo in zip(bp["boxes"], main_algos):
            patch.set_facecolor(ALGO_COLORS[algo])
            patch.set_alpha(0.6)

        ax.set_title(f"Instância {inst.capitalize()} (n={sub['n'].iloc[0]})")
        ax.set_ylabel("Custo Total Ponderado h(π)")
        ax.grid(True)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Comparação de Desempenho (Boxplots): Greedy vs SA vs GA", fontsize=14, fontweight="bold")
    out_path = os.path.join(FIGURES_DIR, "boxplots_comparativos.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Boxplots salvos em: {out_path}")


def generate_convergence_curves(traj_data: Dict[str, Any]):
    """Gera as curvas de convergência (mediana de h vs avaliações) com média da população do GA."""
    trajectories = traj_data.get("trajectories", {})
    instances = ["pequena", "media", "grande"]
    main_algos = ["greedy", "sa", "ga"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, inst in enumerate(instances):
        ax = axes[idx]
        max_budget = 0

        for algo in main_algos:
            combo_key = f"{inst}_{algo}"
            runs = trajectories.get(combo_key, [])
            if not runs:
                continue

            arr = np.array(runs)
            median_curve = np.median(arr, axis=0)
            n_pts = len(median_curve)

            if inst == "pequena":
                budget = 50000
            elif inst == "media":
                budget = 150000
            else:
                budget = 300000

            x_evals = np.linspace(budget / n_pts, budget, n_pts)

            ax.plot(
                x_evals,
                median_curve,
                label=f"{ALGO_NAMES[algo]} (Melhor)",
                color=ALGO_COLORS[algo],
                linewidth=2.0,
            )

        ga_mean_key = f"{inst}_ga_ga_mean"
        ga_means_runs = traj_data.get("telemetry", {}).get(ga_mean_key, [])
        if ga_means_runs:
            all_pts = []
            for r in ga_means_runs:
                all_pts.extend(r)
            if all_pts:
                all_pts = sorted(all_pts, key=lambda x: x[0])
                x_vals = [p[0] for p in all_pts]
                y_vals = [p[1] for p in all_pts]
                if len(x_vals) > 5:
                    x_interp = np.linspace(min(x_vals), max(x_vals), 50)
                    y_interp = np.interp(x_interp, x_vals, y_vals)
                    ax.plot(
                        x_interp,
                        y_interp,
                        label="GA (Média População)",
                        color="#2CA02C",
                        linestyle="--",
                        linewidth=1.8,
                    )

        ax.set_title(f"Convergência — {inst.capitalize()}")
        ax.set_xlabel("Número de Avaliações")
        ax.set_ylabel("Custo h(π)")
        ax.grid(True)
        ax.legend()

    fig.suptitle("Curvas de Convergência (Mediana das Execuções): Melhor h vs Orçamento", fontsize=14, fontweight="bold")
    out_path = os.path.join(FIGURES_DIR, "curvas_convergencia.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Curvas de convergência salvas em: {out_path}")


def generate_sa_acceptance_curve(traj_data: Dict[str, Any]):
    """Gera gráfico da taxa de aceitação de movimentos de piora no Simulated Annealing."""
    telemetry = traj_data.get("telemetry", {})
    instances = ["pequena", "media", "grande"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for inst in instances:
        key = f"{inst}_sa_sa_accept"
        runs = telemetry.get(key, [])
        if not runs:
            continue

        points = []
        for r in runs:
            points.extend(r)
        points.sort(key=lambda x: x[0])

        x_pts = [p[0] for p in points]
        y_pts = [p[1] * 100 for p in points]

        max_x = max(x_pts) if x_pts else 1
        x_norm = [p / max_x * 100 for p in x_pts]

        if len(y_pts) > 10:
            window = max(5, len(y_pts) // 25)
            y_smooth = pd.Series(y_pts).rolling(window, min_periods=1).mean()
            ax.plot(x_norm, y_smooth, label=f"Instância {inst.capitalize()}", linewidth=2.0)

    ax.set_title("Simulated Annealing: Taxa de Aceitação de Pioras ao Longo do Orçamento")
    ax.set_xlabel("Progresso da Execução (% do Orçamento Gasto)")
    ax.set_ylabel("Taxa de Aceitação de Pioras (%)")
    ax.set_ylim(-2, 102)
    ax.grid(True)
    ax.legend()

    out_path = os.path.join(FIGURES_DIR, "sa_taxa_aceitacao.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Gráfico de aceitação SA salvo em: {out_path}")


def generate_ga_diversity_curve(traj_data: Dict[str, Any]):
    """Gera gráfico da evolução da diversidade populacional no Genetic Algorithm."""
    telemetry = traj_data.get("telemetry", {})
    instances = ["pequena", "media", "grande"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for inst in instances:
        key = f"{inst}_ga_ga_div"
        runs = telemetry.get(key, [])
        if not runs:
            continue

        points = []
        for r in runs:
            points.extend(r)
        points.sort(key=lambda x: x[0])

        x_pts = [p[0] for p in points]
        y_pts = [p[1] * 100 for p in points]

        max_x = max(x_pts) if x_pts else 1
        x_norm = [p / max_x * 100 for p in x_pts]

        if len(y_pts) > 10:
            window = max(5, len(y_pts) // 25)
            y_smooth = pd.Series(y_pts).rolling(window, min_periods=1).mean()
            ax.plot(x_norm, y_smooth, label=f"Instância {inst.capitalize()}", linewidth=2.0)

    ax.set_title("Algoritmo Genético: Evolução da Diversidade Populacional")
    ax.set_xlabel("Progresso da Execução (% do Orçamento Gasto)")
    ax.set_ylabel("Cromossomos Únicos na População (%)")
    ax.set_ylim(-2, 105)
    ax.grid(True)
    ax.legend()

    out_path = os.path.join(FIGURES_DIR, "ga_evolucao_diversidade.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Gráfico de diversidade GA salvo em: {out_path}")


def compute_statistical_tests(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Calcula testes não-paramétricos de Mann-Whitney U para comparações par a par."""
    instances = ["pequena", "media", "grande"]
    comparisons = [
        ("greedy", "sa"),
        ("sa", "ga"),
        ("greedy", "ga"),
        ("ga", "ga_nomx"),  # Experimento adicional do GA!
    ]

    results = []
    print("\n" + "=" * 80)
    print("TESTES ESTATÍSTICOS NÃO-PARAMÉTRICOS (MANN-WHITNEY U)")
    print("=" * 80)

    for inst in instances:
        sub = df[df["instance"] == inst]
        print(f"\n>> Instância: {inst.upper()}")

        for a1, a2 in comparisons:
            vals1 = sub[sub["algorithm"] == a1]["best_cost"].dropna().values
            vals2 = sub[sub["algorithm"] == a2]["best_cost"].dropna().values

            if len(vals1) == 0 or len(vals2) == 0:
                continue

            stat, pval = stats.mannwhitneyu(vals1, vals2, alternative="two-sided")
            mean1, mean2 = np.mean(vals1), np.mean(vals2)
            med1, med2 = np.median(vals1), np.median(vals2)

            is_significant = pval < 0.05
            sig_str = "SIM (p < 0.05)" if is_significant else "NÃO (p >= 0.05)"

            row = {
                "instance": inst,
                "algo1": a1,
                "algo2": a2,
                "mean1": mean1,
                "mean2": mean2,
                "median1": med1,
                "median2": med2,
                "u_stat": stat,
                "p_value": pval,
                "significant": is_significant,
            }
            results.append(row)

            print(
                f"   * {a1.upper().ljust(7)} vs {a2.upper().ljust(7)}: "
                f"Mediana = {med1:.1f} vs {med2:.1f} | U = {stat:6.1f} | p-valor = {pval:.4e} -> Dif. Significativa: {sig_str}"
            )

    return results


def generate_main_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """Gera a tabela principal do relatório com as 9 combinações obrigatórias."""
    main_algos = ["greedy", "sa", "ga"]
    instances = ["pequena", "media", "grande"]

    rows = []
    for inst in instances:
        sub_inst = df[df["instance"] == inst]
        ref = 1875.0 if inst == "pequena" else None

        for algo in main_algos:
            sub = sub_inst[sub_inst["algorithm"] == algo]
            if len(sub) == 0:
                continue

            costs = sub["best_cost"].values
            times = sub["elapsed_seconds"].values

            mean_c = np.mean(costs)
            std_c = np.std(costs, ddof=1) if len(costs) > 1 else 0.0
            med_c = np.median(costs)
            min_c = np.min(costs)
            mean_t = np.mean(times)
            evals = sub["budget"].iloc[0]

            gap_str = f"{((min_c - ref) / ref) * 100:+.2f}%" if ref is not None else "—"

            rows.append({
                "Instância": inst.capitalize(),
                "n": sub["n"].iloc[0],
                "Algoritmo": ALGO_NAMES[algo],
                "Orçamento": f"{evals:,}",
                "Execuções": len(costs),
                "Melhor h": f"{min_c:,.1f}",
                "Média ± Desvio": f"{mean_c:,.1f} ± {std_c:,.1f}",
                "Mediana": f"{med_c:,.1f}",
                "Gap (h*)": gap_str,
                "Tempo Médio (s)": f"{mean_t:.3f}s",
            })

    table_df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("TABELA PRINCIPAL DE RESULTADOS (3 INSTÂNCIAS x 3 ALGORITMOS)")
    print("=" * 80)
    print(table_df.to_string(index=False))

    out_csv = os.path.join("data", "tabela_principal_resultados.csv")
    table_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n[OK] Tabela salva em: {out_csv}")
    return table_df


if __name__ == "__main__":
    setup_matplotlib()

    if not os.path.exists(DATA_CSV):
        print(f"[ERRO] Arquivo {DATA_CSV} não encontrado! Execute run_experiments.py primeiro.")
        sys.exit(1)

    df_runs = pd.read_csv(DATA_CSV)

    with open(TRAJECTORIES_JSON, "r", encoding="utf-8") as f:
        traj_data = json.load(f)

    generate_boxplots(df_runs)
    generate_convergence_curves(traj_data)
    generate_sa_acceptance_curve(traj_data)
    generate_ga_diversity_curve(traj_data)
    compute_statistical_tests(df_runs)
    generate_main_results_table(df_runs)
    print("\n[SUCESSO] Todos os gráficos e estatísticas foram gerados com sucesso!")
