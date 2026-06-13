# ============================================================
# Lights Out and Away — F1 ML Project
# src/visualizations.py
#
# Generates all 26+ F1-themed visualizations for the project.
# Every plot uses the official F1 color scheme with dark navy
# backgrounds, F1 red accents, and team-specific colors.
# All plots saved as high-res PNGs (300 DPI) to outputs/plots/.
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── F1 Theme ─────────────────────────────────────────────────

F1_STYLE = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e8002d",
    "axes.labelcolor": "#ffffff",
    "axes.titlecolor": "#ffffff",
    "xtick.color": "#ffffff",
    "ytick.color": "#ffffff",
    "grid.color": "#2a2a4a",
    "grid.alpha": 0.5,
    "text.color": "#ffffff",
    "legend.facecolor": "#1a1a2e",
    "legend.edgecolor": "#e8002d",
}

TEAM_COLORS = {
    "Red Bull Racing": "#3671C6",
    "Ferrari": "#E8002D",
    "Mercedes": "#27F4D2",
    "McLaren": "#FF8000",
    "Aston Martin": "#229971",
    "Alpine": "#FF87BC",
    "Williams": "#64C4FF",
    "RB": "#6692FF",
    "AlphaTauri": "#6692FF",
    "Kick Sauber": "#52E252",
    "Alfa Romeo": "#52E252",
    "Haas F1 Team": "#B6BABD",
}

TYRE_COLORS = {
    "SOFT": "#E8002D",
    "MEDIUM": "#FFF200",
    "HARD": "#FFFFFF",
    "INTERMEDIATE": "#39B54A",
    "WET": "#0067FF",
}

F1_RED = "#e8002d"
F1_BG = "#1a1a2e"
F1_DARK = "#16213e"
F1_WHITE = "#ffffff"
F1_ORANGE = "#ff7c43"
F1_GRID = "#2a2a4a"


# ── Helper Functions ─────────────────────────────────────────

def apply_f1_style():
    """Apply the F1 theme to all matplotlib plots."""
    plt.rcParams.update(F1_STYLE)


def add_f1_branding(fig, ax, title, subtitle="Lights Out and Away — F1 ML Project"):
    """
    Add F1 branding to a plot:
    - Bold white title with red underline effect
    - Subtitle text
    - Copyright watermark bottom-right
    """
    ax.set_title(title, fontsize=16, fontweight="bold", color=F1_WHITE, pad=20)

    # Red underline under title
    ax.annotate(
        "", xy=(0.0, 1.02), xycoords="axes fraction",
        xytext=(1.0, 1.02), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-", color=F1_RED, lw=2),
    )

    # Subtitle
    fig.text(
        0.5, 0.94, subtitle,
        ha="center", va="top", fontsize=9, color=F1_ORANGE,
        fontstyle="italic", alpha=0.8,
    )

    # Copyright watermark
    fig.text(
        0.98, 0.01, "© Lights Out and Away — AI/ML Project",
        ha="right", va="bottom", fontsize=7, color=F1_WHITE, alpha=0.4,
    )


def save_plot(fig, filename):
    """Save plot to outputs/plots/ at 300 DPI."""
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"      ✅ Saved: {filename}")


def get_team_color(team_name):
    """Get the F1 team color, with fallback to white."""
    for key, color in TEAM_COLORS.items():
        if key.lower() in str(team_name).lower() or str(team_name).lower() in key.lower():
            return color
    return "#AAAAAA"


# ════════════════════════════════════════════════════════════
# DATA OVERVIEW PLOTS (1–5)
# ════════════════════════════════════════════════════════════

def plot_race_calendar_heatmap(race_df):
    """
    Plot 1: Race calendar heatmap — all 24 races across 3 seasons,
    colored by number of data points collected.
    """
    apply_f1_style()

    pivot = race_df.groupby(["Year", "Race"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(18, 6), facecolor=F1_BG)
    sns.heatmap(
        pivot, annot=True, fmt="d", cmap="YlOrRd",
        linewidths=0.5, linecolor=F1_GRID, cbar_kws={"label": "Data Points"},
        ax=ax,
    )
    ax.set_facecolor(F1_DARK)
    ax.set_xlabel("Race", fontsize=12)
    ax.set_ylabel("Season", fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0)

    add_f1_branding(fig, ax, "F1 Race Calendar — Data Collection Overview")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "01_race_calendar_heatmap.png")


def plot_driver_radar_chart(race_df):
    """
    Plot 2: Driver performance radar chart — top 10 drivers,
    metrics: wins, podiums, fastest laps, avg position.
    """
    apply_f1_style()

    # Calculate driver stats
    driver_stats = race_df.groupby("Driver").agg(
        Wins=("Position", lambda x: (x == 1).sum()),
        Podiums=("Position", lambda x: (x <= 3).sum()),
        Points=("Points", "sum"),
        AvgPos=("Position", "mean"),
        Races=("Position", "count"),
    ).reset_index()

    # Top 10 by points
    top10 = driver_stats.nlargest(10, "Points")

    # Normalize metrics to 0-1 scale for radar
    categories = ["Wins", "Podiums", "Points", "Races"]
    num_cats = len(categories)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True), facecolor=F1_BG)
    ax.set_facecolor(F1_DARK)

    angles = np.linspace(0, 2 * np.pi, num_cats, endpoint=False).tolist()
    angles += angles[:1]

    colors = [F1_RED, F1_ORANGE, "#27F4D2", "#3671C6", "#FF8000",
              "#229971", "#FF87BC", "#64C4FF", "#6692FF", "#52E252"]

    for idx, (_, row) in enumerate(top10.iterrows()):
        values = []
        for cat in categories:
            max_val = top10[cat].max()
            values.append(row[cat] / max_val if max_val > 0 else 0)
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2, color=colors[idx % len(colors)],
                label=row["Driver"], markersize=4)
        ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, color=F1_WHITE)
    ax.set_yticklabels([])
    ax.spines["polar"].set_color(F1_RED)
    ax.grid(color=F1_GRID, alpha=0.3)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              fontsize=9, facecolor=F1_BG, edgecolor=F1_RED, labelcolor=F1_WHITE)

    ax.set_title("Top 10 Driver Performance Radar", fontsize=16,
                 fontweight="bold", color=F1_WHITE, pad=30)
    fig.text(0.5, 0.94, "Lights Out and Away — F1 ML Project",
             ha="center", fontsize=9, color=F1_ORANGE, fontstyle="italic", alpha=0.8)
    fig.text(0.98, 0.01, "© Lights Out and Away — AI/ML Project",
             ha="right", fontsize=7, color=F1_WHITE, alpha=0.4)

    fig.tight_layout(rect=[0, 0.02, 0.9, 0.92])
    save_plot(fig, "02_driver_radar_chart.png")


def plot_team_performance(race_df):
    """
    Plot 3: Team performance bar chart — constructor points
    colored by official team colors.
    """
    apply_f1_style()

    team_points = race_df.groupby("Team")["Points"].sum().sort_values(ascending=True)
    colors = [get_team_color(team) for team in team_points.index]

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=F1_BG)
    bars = ax.barh(range(len(team_points)), team_points.values, color=colors, edgecolor=F1_WHITE, linewidth=0.5)
    ax.set_yticks(range(len(team_points)))
    ax.set_yticklabels(team_points.index, fontsize=10)
    ax.set_xlabel("Total Points (2022–2024)", fontsize=12)
    ax.set_facecolor(F1_DARK)

    # Add value labels
    for bar, val in zip(bars, team_points.values):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", fontsize=9, color=F1_WHITE, fontweight="bold")

    add_f1_branding(fig, ax, "Constructor Championship Points (2022–2024)")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "03_team_performance.png")


def plot_circuit_difficulty(race_df):
    """
    Plot 4: Circuit difficulty chart — average position variance per circuit.
    """
    apply_f1_style()

    circuit_var = race_df.groupby("Race")["Position"].std().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=F1_BG)

    colors_gradient = plt.cm.YlOrRd(np.linspace(0.3, 1.0, len(circuit_var)))
    bars = ax.bar(range(len(circuit_var)), circuit_var.values, color=colors_gradient,
                  edgecolor=F1_WHITE, linewidth=0.3)
    ax.set_xticks(range(len(circuit_var)))
    ax.set_xticklabels(circuit_var.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Position Std Dev (Higher = More Unpredictable)", fontsize=10)
    ax.set_facecolor(F1_DARK)

    add_f1_branding(fig, ax, "Circuit Difficulty — Position Variance by Track")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "04_circuit_difficulty.png")


def plot_tyre_usage_pie(race_df, tyre_df):
    """
    Plot 5: Tyre compound usage pie chart per season.
    """
    apply_f1_style()

    if tyre_df is not None and not tyre_df.empty:
        compound_data = tyre_df
    elif "StartingCompound" in race_df.columns:
        compound_data = race_df.rename(columns={"StartingCompound": "Compound"})
    else:
        print("      ⚠ No tyre data available for pie chart")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=F1_BG)

    for idx, year in enumerate([2022, 2023, 2024]):
        ax = axes[idx]
        ax.set_facecolor(F1_DARK)

        year_data = compound_data[compound_data["Year"] == year] if "Year" in compound_data.columns else compound_data
        if year_data.empty:
            ax.text(0.5, 0.5, f"No data\n{year}", ha="center", va="center",
                    fontsize=14, color=F1_WHITE, transform=ax.transAxes)
            continue

        counts = year_data["Compound"].value_counts()
        colors = [TYRE_COLORS.get(c, "#AAAAAA") for c in counts.index]

        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index, colors=colors,
            autopct="%1.1f%%", startangle=90, pctdistance=0.85,
            textprops={"fontsize": 9, "color": F1_WHITE},
        )
        for text in autotexts:
            text.set_color("#000000")
            text.set_fontweight("bold")

        ax.set_title(f"Season {year}", fontsize=13, color=F1_WHITE, fontweight="bold")

    fig.suptitle("Tyre Compound Usage Per Season", fontsize=16,
                 fontweight="bold", color=F1_WHITE, y=0.98)
    fig.text(0.5, 0.93, "Lights Out and Away — F1 ML Project",
             ha="center", fontsize=9, color=F1_ORANGE, fontstyle="italic", alpha=0.8)
    fig.text(0.98, 0.01, "© Lights Out and Away — AI/ML Project",
             ha="right", fontsize=7, color=F1_WHITE, alpha=0.4)

    fig.tight_layout(rect=[0, 0.02, 1, 0.91])
    save_plot(fig, "05_tyre_usage_pie.png")


# ════════════════════════════════════════════════════════════
# PROBLEM 1 — WINNER PREDICTION PLOTS (6–13)
# ════════════════════════════════════════════════════════════

def plot_feature_correlation_heatmap(data, title_suffix="Race Winner"):
    """
    Plot 6: Feature correlation heatmap — F1 themed.
    """
    apply_f1_style()

    numeric_data = data.select_dtypes(include=[np.number])
    corr = numeric_data.corr()

    fig, ax = plt.subplots(figsize=(12, 10), facecolor=F1_BG)

    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(0, 240, s=100, l=50, as_cmap=True)

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
        center=0, square=True, linewidths=0.5, linecolor=F1_GRID,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        ax=ax, vmin=-1, vmax=1,
    )
    ax.set_facecolor(F1_DARK)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    add_f1_branding(fig, ax, f"Feature Correlation Heatmap — {title_suffix}")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "06_feature_correlation_heatmap.png")


def plot_grid_vs_finish(race_df):
    """
    Plot 7: Grid position vs finishing position scatter plot —
    colored by team colors, sized by championship points.
    """
    apply_f1_style()

    fig, ax = plt.subplots(figsize=(12, 10), facecolor=F1_BG)

    teams = race_df["Team"].unique()
    for team in teams:
        team_data = race_df[race_df["Team"] == team]
        color = get_team_color(team)
        sizes = team_data["Points"].fillna(1) * 3 + 10

        ax.scatter(
            team_data["GridPosition"], team_data["Position"],
            c=color, s=sizes, alpha=0.6, edgecolors=F1_WHITE,
            linewidth=0.3, label=team, zorder=3,
        )

    # Add diagonal reference line (grid = finish)
    ax.plot([0, 25], [0, 25], "--", color=F1_RED, alpha=0.7, linewidth=2,
            label="Grid = Finish", zorder=2)

    ax.set_xlabel("Grid Position (Qualifying)", fontsize=12)
    ax.set_ylabel("Finishing Position", fontsize=12)
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 22)
    ax.set_facecolor(F1_DARK)
    ax.legend(loc="upper left", fontsize=7, ncol=2,
              facecolor=F1_BG, edgecolor=F1_RED, labelcolor=F1_WHITE)
    ax.invert_yaxis()

    add_f1_branding(fig, ax, "Grid Position vs Finishing Position")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "07_grid_vs_finish.png")


def plot_lr_actual_vs_predicted_winner(y_test, y_pred):
    """
    Plot 8: Linear Regression — actual vs predicted finishing position
    with regression line.
    """
    apply_f1_style()

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=F1_BG)

    ax.scatter(y_test, y_pred, c=F1_ORANGE, s=40, alpha=0.6,
               edgecolors=F1_WHITE, linewidth=0.3, zorder=3)

    # Perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "--", color=F1_RED,
            linewidth=2, label="Perfect Prediction", zorder=2)

    # Regression line through predictions
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min_val, max_val, 100)
    ax.plot(x_line, p(x_line), "-", color="#27F4D2", linewidth=2,
            label=f"Regression Line (y={z[0]:.2f}x+{z[1]:.2f})", zorder=2)

    ax.set_xlabel("Actual Finishing Position", fontsize=12)
    ax.set_ylabel("Predicted Finishing Position", fontsize=12)
    ax.set_facecolor(F1_DARK)
    ax.legend(facecolor=F1_BG, edgecolor=F1_RED, labelcolor=F1_WHITE, fontsize=10)

    add_f1_branding(fig, ax, "Linear Regression — Actual vs Predicted Position")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "08_lr_actual_vs_predicted_winner.png")


def plot_knn_confusion_matrix(y_test, y_pred, top_n=10):
    """
    Plot 9: KNN — confusion matrix heatmap — F1 red theme.
    Focus on top N positions for readability.
    """
    apply_f1_style()

    # Limit to top N positions for readability
    labels = sorted(set(y_test) | set(y_pred))
    if len(labels) > top_n:
        labels = sorted(labels)[:top_n]

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=F1_BG)

    cmap = sns.color_palette([F1_DARK, "#4a1030", "#8a1030", F1_RED, "#ff4060"], as_cmap=True)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor=F1_GRID,
        cbar_kws={"label": "Count"}, ax=ax,
    )
    ax.set_xlabel("Predicted Position", fontsize=12)
    ax.set_ylabel("Actual Position", fontsize=12)
    ax.set_facecolor(F1_DARK)

    add_f1_branding(fig, ax, f"KNN Confusion Matrix (Top {top_n} Positions)")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "09_knn_confusion_matrix.png")


def plot_knn_accuracy_vs_k(k_values, cv_scores_mean, cv_scores_std, best_k):
    """
    Plot 10: KNN — accuracy vs K value line chart — find optimal K.
    """
    apply_f1_style()

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=F1_BG)

    ax.plot(k_values, cv_scores_mean, "-o", color=F1_RED, linewidth=2,
            markersize=8, markerfacecolor=F1_ORANGE, markeredgecolor=F1_WHITE,
            markeredgewidth=1, zorder=3)

    ax.fill_between(
        k_values,
        np.array(cv_scores_mean) - np.array(cv_scores_std),
        np.array(cv_scores_mean) + np.array(cv_scores_std),
        alpha=0.2, color=F1_RED,
    )

    # Highlight best K
    best_idx = list(k_values).index(best_k) if best_k in list(k_values) else 0
    ax.axvline(x=best_k, color=F1_ORANGE, linestyle="--", linewidth=1.5,
               label=f"Best K = {best_k}", zorder=2)
    ax.scatter([best_k], [cv_scores_mean[best_idx]], c="#27F4D2", s=200,
               zorder=4, edgecolors=F1_WHITE, linewidth=2)

    ax.set_xlabel("K (Number of Neighbors)", fontsize=12)
    ax.set_ylabel("Cross-Validation Score", fontsize=12)
    ax.set_facecolor(F1_DARK)
    ax.set_xticks(k_values)
    ax.legend(facecolor=F1_BG, edgecolor=F1_RED, labelcolor=F1_WHITE, fontsize=11)

    add_f1_branding(fig, ax, "KNN — Finding Optimal K Value")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "10_knn_accuracy_vs_k.png")


def plot_algorithm_comparison_winner(metrics_dict):
    """
    Plot 11: Algorithm comparison bar chart — LR vs KNN
    on all metrics side by side.

    metrics_dict: {
        'Linear Regression': {'MSE': ..., 'RMSE': ..., ...},
        'KNN': {'MSE': ..., 'RMSE': ..., ...}
    }
    """
    apply_f1_style()

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=F1_BG)

    algorithms = list(metrics_dict.keys())
    metrics = list(list(metrics_dict.values())[0].keys())
    x = np.arange(len(metrics))
    width = 0.35

    colors = [F1_RED, F1_ORANGE]

    for i, algo in enumerate(algorithms):
        values = [metrics_dict[algo].get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=algo,
                      color=colors[i % len(colors)], edgecolor=F1_WHITE, linewidth=0.5)
        # Value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                    color=F1_WHITE, fontweight="bold")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_facecolor(F1_DARK)
    ax.legend(facecolor=F1_BG, edgecolor=F1_RED, labelcolor=F1_WHITE, fontsize=11)

    add_f1_branding(fig, ax, "Algorithm Comparison — Race Winner Prediction")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "11_algorithm_comparison_winner.png")


def plot_top_predicted_winners(predictions_df):
    """
    Plot 12: Top 10 predicted winners probability chart —
    horizontal bar chart.
    """
    apply_f1_style()

    # Get top 10 drivers by frequency of P1 predictions
    if "PredictedPosition" in predictions_df.columns:
        winners = predictions_df[predictions_df["PredictedPosition"] == 1]
        if winners.empty:
            winners = predictions_df.nsmallest(10, "PredictedPosition")
        win_counts = winners["Driver"].value_counts().head(10)
    else:
        win_counts = predictions_df["Driver"].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=F1_BG)

    colors_gradient = plt.cm.Reds(np.linspace(0.4, 1.0, len(win_counts)))[::-1]
    bars = ax.barh(range(len(win_counts)), win_counts.values,
                   color=colors_gradient, edgecolor=F1_WHITE, linewidth=0.5)

    ax.set_yticks(range(len(win_counts)))
    ax.set_yticklabels(win_counts.index, fontsize=11)
    ax.set_xlabel("Predicted Wins", fontsize=12)
    ax.set_facecolor(F1_DARK)
    ax.invert_yaxis()

    for bar, val in zip(bars, win_counts.values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val}", va="center", fontsize=10, color=F1_WHITE, fontweight="bold")

    add_f1_branding(fig, ax, "Top 10 Predicted Race Winners (KNN)")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "12_top_predicted_winners.png")


def plot_driver_win_probability_heatmap(race_df):
    """
    Plot 13: Driver win probability by circuit — heatmap grid.
    """
    apply_f1_style()

    # Calculate win rate per driver per circuit
    wins = race_df.copy()
    wins["IsWinner"] = (wins["Position"] == 1).astype(int)

    pivot = wins.groupby(["Driver", "Race"])["IsWinner"].mean().unstack(fill_value=0)

    # Keep only drivers with at least one win
    drivers_with_wins = pivot.index[pivot.max(axis=1) > 0]
    if len(drivers_with_wins) == 0:
        drivers_with_wins = pivot.index[:10]
    pivot = pivot.loc[drivers_with_wins]

    fig, ax = plt.subplots(figsize=(18, 8), facecolor=F1_BG)

    sns.heatmap(
        pivot, cmap="YlOrRd", annot=True, fmt=".1f",
        linewidths=0.5, linecolor=F1_GRID,
        cbar_kws={"label": "Win Probability"}, ax=ax,
    )
    ax.set_facecolor(F1_DARK)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    add_f1_branding(fig, ax, "Driver Win Probability by Circuit")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "13_driver_win_probability_heatmap.png")


# ════════════════════════════════════════════════════════════
# PROBLEM 2 — PIT STOP PREDICTION PLOTS (14–22)
# ════════════════════════════════════════════════════════════

def plot_pitstop_violin_by_team(pitstop_df):
    """
    Plot 14: Pit stop duration distribution per team — violin plot.
    """
    apply_f1_style()

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=F1_BG)

    teams = pitstop_df["Team"].unique()
    team_order = pitstop_df.groupby("Team")["PitStopDuration"].median().sort_values().index

    palette = {team: get_team_color(team) for team in teams}

    sns.violinplot(
        data=pitstop_df, x="Team", y="PitStopDuration",
        order=team_order, palette=palette, inner="box",
        linewidth=0.5, ax=ax,
    )

    ax.set_xlabel("Team", fontsize=12)
    ax.set_ylabel("Pit Stop Duration (seconds)", fontsize=12)
    ax.set_facecolor(F1_DARK)
    ax.tick_params(axis="x", rotation=45, labelsize=9)

    add_f1_branding(fig, ax, "Pit Stop Duration Distribution by Team")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "14_pitstop_violin_by_team.png")


def plot_pitstop_vs_lap(pitstop_df):
    """
    Plot 15: Pit stop duration vs lap number scatter —
    colored by tyre compound.
    """
    apply_f1_style()

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=F1_BG)

    for compound in pitstop_df["TyreCompound"].unique():
        mask = pitstop_df["TyreCompound"] == compound
        color = TYRE_COLORS.get(compound, "#AAAAAA")
        ax.scatter(
            pitstop_df.loc[mask, "LapNumber"],
            pitstop_df.loc[mask, "PitStopDuration"],
            c=color, s=20, alpha=0.5, label=compound,
            edgecolors=F1_WHITE, linewidth=0.2, zorder=3,
        )

    ax.set_xlabel("Lap Number", fontsize=12)
    ax.set_ylabel("Pit Stop Duration (seconds)", fontsize=12)
    ax.set_facecolor(F1_DARK)
    ax.legend(facecolor=F1_BG, edgecolor=F1_RED, labelcolor=F1_WHITE, fontsize=10,
              title="Tyre Compound", title_fontsize=11)

    add_f1_branding(fig, ax, "Pit Stop Duration vs Lap Number")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "15_pitstop_vs_lap.png")


def plot_lr_actual_vs_predicted_pitstop(y_test, y_pred):
    """
    Plot 16: Linear Regression — actual vs predicted pit stop duration
    with confidence interval bands.
    """
    apply_f1_style()

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=F1_BG)

    ax.scatter(y_test, y_pred, c=F1_ORANGE, s=30, alpha=0.5,
               edgecolors=F1_WHITE, linewidth=0.2, zorder=3)

    # Perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "--", color=F1_RED,
            linewidth=2, label="Perfect Prediction", zorder=2)

    # Confidence interval bands (±1 std of residuals)
    residuals = np.array(y_test) - np.array(y_pred)
    std_res = np.std(residuals)
    x_line = np.linspace(min_val, max_val, 100)
    ax.fill_between(x_line, x_line - std_res, x_line + std_res,
                    alpha=0.15, color=F1_RED, label=f"±1 Std Dev ({std_res:.2f}s)")

    ax.set_xlabel("Actual Pit Stop Duration (s)", fontsize=12)
    ax.set_ylabel("Predicted Pit Stop Duration (s)", fontsize=12)
    ax.set_facecolor(F1_DARK)
    ax.legend(facecolor=F1_BG, edgecolor=F1_RED, labelcolor=F1_WHITE, fontsize=10)

    add_f1_branding(fig, ax, "Linear Regression — Pit Stop Duration Prediction")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "16_lr_actual_vs_predicted_pitstop.png")


def plot_knn_actual_vs_predicted_pitstop(y_test, y_pred):
    """
    Plot 17: KNN — actual vs predicted pit stop duration.
    """
    apply_f1_style()

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=F1_BG)

    ax.scatter(y_test, y_pred, c="#27F4D2", s=30, alpha=0.5,
               edgecolors=F1_WHITE, linewidth=0.2, zorder=3)

    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "--", color=F1_RED,
            linewidth=2, label="Perfect Prediction", zorder=2)

    ax.set_xlabel("Actual Pit Stop Duration (s)", fontsize=12)
    ax.set_ylabel("Predicted Pit Stop Duration (s)", fontsize=12)
    ax.set_facecolor(F1_DARK)
    ax.legend(facecolor=F1_BG, edgecolor=F1_RED, labelcolor=F1_WHITE, fontsize=10)

    add_f1_branding(fig, ax, "KNN Regression — Pit Stop Duration Prediction")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "17_knn_actual_vs_predicted_pitstop.png")


def plot_residual_comparison(y_test_lr, y_pred_lr, y_test_knn, y_pred_knn, problem_name="Pit Stop"):
    """
    Plot 18: Residual plot for both algorithms side by side.
    """
    apply_f1_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor=F1_BG)

    # Linear Regression residuals
    residuals_lr = np.array(y_test_lr) - np.array(y_pred_lr)
    ax1.scatter(y_pred_lr, residuals_lr, c=F1_RED, s=20, alpha=0.5,
                edgecolors=F1_WHITE, linewidth=0.2)
    ax1.axhline(y=0, color=F1_ORANGE, linestyle="--", linewidth=2)
    ax1.set_xlabel("Predicted Value", fontsize=11)
    ax1.set_ylabel("Residual", fontsize=11)
    ax1.set_title("Linear Regression Residuals", fontsize=13, color=F1_WHITE, fontweight="bold")
    ax1.set_facecolor(F1_DARK)

    # KNN residuals
    residuals_knn = np.array(y_test_knn) - np.array(y_pred_knn)
    ax2.scatter(y_pred_knn, residuals_knn, c="#27F4D2", s=20, alpha=0.5,
                edgecolors=F1_WHITE, linewidth=0.2)
    ax2.axhline(y=0, color=F1_ORANGE, linestyle="--", linewidth=2)
    ax2.set_xlabel("Predicted Value", fontsize=11)
    ax2.set_ylabel("Residual", fontsize=11)
    ax2.set_title("KNN Residuals", fontsize=13, color=F1_WHITE, fontweight="bold")
    ax2.set_facecolor(F1_DARK)

    fig.suptitle(f"Residual Analysis — {problem_name} Prediction", fontsize=16,
                 fontweight="bold", color=F1_WHITE, y=0.99)
    fig.text(0.5, 0.94, "Lights Out and Away — F1 ML Project",
             ha="center", fontsize=9, color=F1_ORANGE, fontstyle="italic", alpha=0.8)
    fig.text(0.98, 0.01, "© Lights Out and Away — AI/ML Project",
             ha="right", fontsize=7, color=F1_WHITE, alpha=0.4)

    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "18_residual_comparison_pitstop.png")


def plot_algorithm_comparison_pitstop(metrics_dict):
    """
    Plot 19: Algorithm comparison bar chart — LR vs KNN on RMSE, MAE, R².
    """
    apply_f1_style()

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=F1_BG)

    algorithms = list(metrics_dict.keys())
    metrics = list(list(metrics_dict.values())[0].keys())
    x = np.arange(len(metrics))
    width = 0.35

    colors = [F1_RED, "#27F4D2"]

    for i, algo in enumerate(algorithms):
        values = [metrics_dict[algo].get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=algo,
                      color=colors[i % len(colors)], edgecolor=F1_WHITE, linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                    color=F1_WHITE, fontweight="bold")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_facecolor(F1_DARK)
    ax.legend(facecolor=F1_BG, edgecolor=F1_RED, labelcolor=F1_WHITE, fontsize=11)

    add_f1_branding(fig, ax, "Algorithm Comparison — Pit Stop Duration Prediction")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "19_algorithm_comparison_pitstop.png")


def plot_pitstop_by_circuit(pitstop_df):
    """
    Plot 20: Pit stop duration by circuit boxplot — ranked fastest to slowest.
    """
    apply_f1_style()

    circuit_order = pitstop_df.groupby("Race")["PitStopDuration"].median().sort_values().index

    fig, ax = plt.subplots(figsize=(16, 7), facecolor=F1_BG)

    bp = ax.boxplot(
        [pitstop_df[pitstop_df["Race"] == r]["PitStopDuration"].values for r in circuit_order],
        positions=range(len(circuit_order)),
        patch_artist=True,
        widths=0.6,
    )

    colors_gradient = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(circuit_order)))
    for patch, color in zip(bp["boxes"], colors_gradient):
        patch.set_facecolor(color)
        patch.set_edgecolor(F1_WHITE)
        patch.set_alpha(0.8)
    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_color(F1_WHITE)
    for flier in bp["fliers"]:
        flier.set_markeredgecolor(F1_RED)
        flier.set_markerfacecolor(F1_RED)
        flier.set_markersize(3)

    ax.set_xticks(range(len(circuit_order)))
    ax.set_xticklabels(circuit_order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Pit Stop Duration (seconds)", fontsize=12)
    ax.set_facecolor(F1_DARK)

    add_f1_branding(fig, ax, "Pit Stop Duration by Circuit (Fastest → Slowest)")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "20_pitstop_by_circuit.png")


def plot_tyre_compound_vs_pitstop(pitstop_df):
    """
    Plot 21: Tyre compound vs pit stop duration violin plot.
    """
    apply_f1_style()

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=F1_BG)

    compounds = pitstop_df["TyreCompound"].unique()
    palette = {c: TYRE_COLORS.get(c, "#AAAAAA") for c in compounds}

    sns.violinplot(
        data=pitstop_df, x="TyreCompound", y="PitStopDuration",
        palette=palette, inner="box", linewidth=0.5, ax=ax,
    )

    ax.set_xlabel("Tyre Compound", fontsize=12)
    ax.set_ylabel("Pit Stop Duration (seconds)", fontsize=12)
    ax.set_facecolor(F1_DARK)

    add_f1_branding(fig, ax, "Tyre Compound vs Pit Stop Duration")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "21_tyre_compound_vs_pitstop.png")


def plot_team_pitstop_ranking(pitstop_df):
    """
    Plot 22: Team pit stop performance ranking — bar chart.
    """
    apply_f1_style()

    team_median = pitstop_df.groupby("Team")["PitStopDuration"].median().sort_values()
    colors = [get_team_color(team) for team in team_median.index]

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=F1_BG)

    bars = ax.barh(range(len(team_median)), team_median.values,
                   color=colors, edgecolor=F1_WHITE, linewidth=0.5)
    ax.set_yticks(range(len(team_median)))
    ax.set_yticklabels(team_median.index, fontsize=10)
    ax.set_xlabel("Median Pit Stop Duration (seconds)", fontsize=12)
    ax.set_facecolor(F1_DARK)

    for i, (bar, val) in enumerate(zip(bars, team_median.values)):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}s", va="center", fontsize=9, color=F1_WHITE, fontweight="bold")
        # Add rank medal for top 3
        medal = ["🥇", "🥈", "🥉"]
        if i < 3:
            ax.text(0.1, bar.get_y() + bar.get_height() / 2,
                    medal[i], va="center", fontsize=14)

    add_f1_branding(fig, ax, "Team Pit Stop Performance Ranking")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "22_team_pitstop_ranking.png")


# ════════════════════════════════════════════════════════════
# ADVANCED PLOTS (23–26)
# ════════════════════════════════════════════════════════════

def plot_learning_curves(train_sizes, train_scores, test_scores, algorithm_names):
    """
    Plot 23: Learning curve — training size vs accuracy for multiple algorithms.

    Args:
        train_sizes: list of training set sizes
        train_scores: dict of {algo_name: list_of_scores}
        test_scores: dict of {algo_name: list_of_scores}
        algorithm_names: list of algorithm names
    """
    apply_f1_style()

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=F1_BG)

    colors = [F1_RED, F1_ORANGE, "#27F4D2", "#3671C6"]

    for i, algo in enumerate(algorithm_names):
        color = colors[i % len(colors)]

        if algo in train_scores:
            ax.plot(train_sizes, train_scores[algo], "-o", color=color,
                    linewidth=2, markersize=6, label=f"{algo} (Train)",
                    markerfacecolor=color, markeredgecolor=F1_WHITE, alpha=0.8)
        if algo in test_scores:
            ax.plot(train_sizes, test_scores[algo], "--s", color=color,
                    linewidth=2, markersize=6, label=f"{algo} (Test)",
                    markerfacecolor="none", markeredgecolor=color, alpha=0.8)

    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_facecolor(F1_DARK)
    ax.legend(facecolor=F1_BG, edgecolor=F1_RED, labelcolor=F1_WHITE, fontsize=9)

    add_f1_branding(fig, ax, "Learning Curves — Model Performance vs Training Size")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "23_learning_curves.png")


def plot_cv_score_distribution(cv_results):
    """
    Plot 24: Cross validation score distribution — box plots.

    Args:
        cv_results: dict of {model_name: array_of_cv_scores}
    """
    apply_f1_style()

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=F1_BG)

    data = list(cv_results.values())
    labels = list(cv_results.keys())

    bp = ax.boxplot(data, patch_artist=True, labels=labels, widths=0.5)

    colors = [F1_RED, F1_ORANGE, "#27F4D2", "#3671C6"]
    for i, (patch, color) in enumerate(zip(bp["boxes"], colors)):
        patch.set_facecolor(color)
        patch.set_edgecolor(F1_WHITE)
        patch.set_alpha(0.8)
    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_color(F1_WHITE)
    for flier in bp["fliers"]:
        flier.set_markeredgecolor(F1_WHITE)
        flier.set_markersize(4)

    ax.set_ylabel("Cross-Validation Score", fontsize=12)
    ax.set_facecolor(F1_DARK)
    ax.tick_params(axis="x", labelsize=10)

    add_f1_branding(fig, ax, "Cross-Validation Score Distribution")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "24_cv_score_distribution.png")


def plot_feature_importance(feature_names, importances, title_suffix=""):
    """
    Plot 25: Feature importance ranking — horizontal bar chart.
    """
    apply_f1_style()

    # Sort by importance
    sorted_idx = np.argsort(importances)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = [importances[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=F1_BG)

    colors_gradient = plt.cm.Reds(np.linspace(0.3, 1.0, len(sorted_features)))
    bars = ax.barh(range(len(sorted_features)), sorted_importances,
                   color=colors_gradient, edgecolor=F1_WHITE, linewidth=0.5)

    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=10)
    ax.set_xlabel("Feature Importance (Coefficient Magnitude)", fontsize=12)
    ax.set_facecolor(F1_DARK)

    for bar, val in zip(bars, sorted_importances):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8, color=F1_WHITE)

    add_f1_branding(fig, ax, f"Feature Importance Ranking {title_suffix}")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "25_feature_importance.png")


def plot_season_comparison(season_metrics):
    """
    Plot 26: Season comparison — how model accuracy changes with data years.

    Args:
        season_metrics: dict of {year_label: {metric: value}}
    """
    apply_f1_style()

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=F1_BG)

    labels = list(season_metrics.keys())
    metrics = list(list(season_metrics.values())[0].keys())

    x = np.arange(len(labels))
    width = 0.2
    colors = [F1_RED, F1_ORANGE, "#27F4D2", "#3671C6"]

    for i, metric in enumerate(metrics):
        values = [season_metrics[lbl].get(metric, 0) for lbl in labels]
        bars = ax.bar(x + i * width, values, width, label=metric,
                      color=colors[i % len(colors)], edgecolor=F1_WHITE, linewidth=0.5)

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_facecolor(F1_DARK)
    ax.legend(facecolor=F1_BG, edgecolor=F1_RED, labelcolor=F1_WHITE, fontsize=10)

    add_f1_branding(fig, ax, "Model Accuracy by Training Data Size")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "26_season_comparison.png")


# ════════════════════════════════════════════════════════════
# PREPROCESSING PHASE PLOTS
# ════════════════════════════════════════════════════════════

def plot_missing_values_heatmap(df, title, filename):
    """Plot missing values heatmap for before/after cleaning comparison."""
    apply_f1_style()

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=F1_BG)

    missing = df.isnull().astype(int)
    if missing.sum().sum() == 0:
        # If no missing values, show a message
        ax.text(0.5, 0.5, "No Missing Values ✅",
                ha="center", va="center", fontsize=24,
                color="#27F4D2", fontweight="bold", transform=ax.transAxes)
        ax.set_facecolor(F1_DARK)
    else:
        sns.heatmap(missing, cbar=True, cmap=["#16213e", F1_RED],
                    yticklabels=False, ax=ax)
        ax.set_facecolor(F1_DARK)

    add_f1_branding(fig, ax, title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, filename)


def plot_outlier_boxplot(df_before, df_after, column, filename):
    """Plot box plot before and after removing outliers."""
    apply_f1_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=F1_BG)

    bp1 = ax1.boxplot(df_before[column].dropna().values, patch_artist=True, widths=0.5)
    bp1["boxes"][0].set_facecolor(F1_RED)
    bp1["boxes"][0].set_edgecolor(F1_WHITE)
    for el in ["whiskers", "caps", "medians"]:
        for item in bp1[el]:
            item.set_color(F1_WHITE)
    ax1.set_title("Before Cleaning", fontsize=13, color=F1_WHITE, fontweight="bold")
    ax1.set_ylabel(column, fontsize=11)
    ax1.set_facecolor(F1_DARK)

    bp2 = ax2.boxplot(df_after[column].dropna().values, patch_artist=True, widths=0.5)
    bp2["boxes"][0].set_facecolor("#27F4D2")
    bp2["boxes"][0].set_edgecolor(F1_WHITE)
    for el in ["whiskers", "caps", "medians"]:
        for item in bp2[el]:
            item.set_color(F1_WHITE)
    ax2.set_title("After Cleaning", fontsize=13, color=F1_WHITE, fontweight="bold")
    ax2.set_ylabel(column, fontsize=11)
    ax2.set_facecolor(F1_DARK)

    fig.suptitle(f"Outlier Detection — {column}", fontsize=16,
                 fontweight="bold", color=F1_WHITE, y=0.99)
    fig.text(0.5, 0.94, "Lights Out and Away — F1 ML Project",
             ha="center", fontsize=9, color=F1_ORANGE, fontstyle="italic", alpha=0.8)
    fig.text(0.98, 0.01, "© Lights Out and Away — AI/ML Project",
             ha="right", fontsize=7, color=F1_WHITE, alpha=0.4)

    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, filename)


def plot_train_test_split_pie(train_size, test_size):
    """Plot train/test split visualization as pie chart."""
    apply_f1_style()

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=F1_BG)

    sizes = [train_size, test_size]
    labels = [f"Training Set\n({train_size} samples)", f"Test Set\n({test_size} samples)"]
    colors = [F1_RED, F1_ORANGE]
    explode = (0.05, 0.05)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct="%1.1f%%", startangle=90, pctdistance=0.85,
        textprops={"fontsize": 11, "color": F1_WHITE},
    )
    for text in autotexts:
        text.set_fontweight("bold")
        text.set_fontsize(13)
        text.set_color(F1_WHITE)

    ax.set_facecolor(F1_DARK)

    fig.suptitle("Train / Test Split", fontsize=16,
                 fontweight="bold", color=F1_WHITE, y=0.96)
    fig.text(0.5, 0.92, "Lights Out and Away — F1 ML Project",
             ha="center", fontsize=9, color=F1_ORANGE, fontstyle="italic", alpha=0.8)
    fig.text(0.98, 0.01, "© Lights Out and Away — AI/ML Project",
             ha="right", fontsize=7, color=F1_WHITE, alpha=0.4)

    fig.tight_layout(rect=[0, 0.02, 1, 0.9])
    save_plot(fig, "train_test_split_pie.png")


def plot_records_per_race(race_df):
    """Bar chart — number of records per race per season."""
    apply_f1_style()

    counts = race_df.groupby(["Year", "Race"]).size().reset_index(name="Count")
    pivot = counts.pivot(index="Race", columns="Year", values="Count").fillna(0)

    fig, ax = plt.subplots(figsize=(16, 7), facecolor=F1_BG)

    x = np.arange(len(pivot.index))
    width = 0.25
    year_colors = [F1_RED, F1_ORANGE, "#27F4D2"]

    for i, year in enumerate(sorted(pivot.columns)):
        ax.bar(x + i * width, pivot[year].values, width, label=str(int(year)),
               color=year_colors[i % len(year_colors)], edgecolor=F1_WHITE, linewidth=0.3)

    ax.set_xticks(x + width)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of Records", fontsize=12)
    ax.set_facecolor(F1_DARK)
    ax.legend(facecolor=F1_BG, edgecolor=F1_RED, labelcolor=F1_WHITE, fontsize=10,
              title="Season", title_fontsize=11)

    add_f1_branding(fig, ax, "Records Collected Per Race Per Season")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_plot(fig, "records_per_race.png")


# ════════════════════════════════════════════════════════════
# MASTER VISUALIZATION FUNCTION
# ════════════════════════════════════════════════════════════

def generate_all_visualizations(results):
    """
    Generate all visualizations for the project.

    Args:
        results: dict containing all data and model results needed for plots.
            Expected keys:
            - race_full: full race DataFrame
            - pitstop_full: full pit stop DataFrame
            - tyre_df: tyre data DataFrame
            - race_p1: Problem 1 processed data
            - pitstop_p2: Problem 2 processed data
            - winner_results: dict with winner prediction results
            - pitstop_results: dict with pit stop prediction results
    """
    print("\n   📊 Generating F1-themed visualizations...")
    print(f"   📁 Saving to: {PLOTS_DIR}\n")

    race_full = results.get("race_full", pd.DataFrame())
    pitstop_full = results.get("pitstop_full", pd.DataFrame())
    tyre_df = results.get("tyre_df", pd.DataFrame())
    race_p1 = results.get("race_p1", pd.DataFrame())
    pitstop_p2 = results.get("pitstop_p2", pd.DataFrame())
    winner_res = results.get("winner_results", {})
    pitstop_res = results.get("pitstop_results", {})

    # ── Data Overview Plots (1–5) ────────────────────────────

    print("   📈 Data Overview Plots:")
    if not race_full.empty:
        plot_race_calendar_heatmap(race_full)
        plot_driver_radar_chart(race_full)
        plot_team_performance(race_full)
        plot_circuit_difficulty(race_full)
        plot_records_per_race(race_full)

    if not race_full.empty:
        plot_tyre_usage_pie(race_full, tyre_df if not tyre_df.empty else None)

    # ── Problem 1 Plots (6–13) ──────────────────────────────

    print("\n   🏆 Problem 1 — Winner Prediction Plots:")
    if not race_p1.empty:
        plot_feature_correlation_heatmap(race_p1, "Race Winner")

    if not race_full.empty:
        plot_grid_vs_finish(race_full)

    if winner_res:
        if "y_test_lr" in winner_res and "y_pred_lr" in winner_res:
            plot_lr_actual_vs_predicted_winner(winner_res["y_test_lr"], winner_res["y_pred_lr"])

        if "y_test_knn" in winner_res and "y_pred_knn" in winner_res:
            plot_knn_confusion_matrix(winner_res["y_test_knn"], winner_res["y_pred_knn"])

        if "k_values" in winner_res and "cv_scores_mean" in winner_res:
            plot_knn_accuracy_vs_k(
                winner_res["k_values"],
                winner_res["cv_scores_mean"],
                winner_res.get("cv_scores_std", [0] * len(winner_res["k_values"])),
                winner_res.get("best_k", 5),
            )

        if "metrics_comparison" in winner_res:
            plot_algorithm_comparison_winner(winner_res["metrics_comparison"])

        if "predictions_df" in winner_res:
            plot_top_predicted_winners(winner_res["predictions_df"])

    if not race_full.empty:
        plot_driver_win_probability_heatmap(race_full)

    # ── Problem 2 Plots (14–22) ─────────────────────────────

    print("\n   ⏱️  Problem 2 — Pit Stop Prediction Plots:")
    if not pitstop_full.empty:
        plot_pitstop_violin_by_team(pitstop_full)
        plot_pitstop_vs_lap(pitstop_full)
        plot_pitstop_by_circuit(pitstop_full)
        plot_tyre_compound_vs_pitstop(pitstop_full)
        plot_team_pitstop_ranking(pitstop_full)

    if pitstop_res:
        if "y_test_lr" in pitstop_res and "y_pred_lr" in pitstop_res:
            plot_lr_actual_vs_predicted_pitstop(pitstop_res["y_test_lr"], pitstop_res["y_pred_lr"])

        if "y_test_knn" in pitstop_res and "y_pred_knn" in pitstop_res:
            plot_knn_actual_vs_predicted_pitstop(pitstop_res["y_test_knn"], pitstop_res["y_pred_knn"])

        if all(k in pitstop_res for k in ["y_test_lr", "y_pred_lr", "y_test_knn", "y_pred_knn"]):
            plot_residual_comparison(
                pitstop_res["y_test_lr"], pitstop_res["y_pred_lr"],
                pitstop_res["y_test_knn"], pitstop_res["y_pred_knn"],
                "Pit Stop",
            )

        if "metrics_comparison" in pitstop_res:
            plot_algorithm_comparison_pitstop(pitstop_res["metrics_comparison"])

    # ── Advanced Plots (23–26) ──────────────────────────────

    print("\n   🔬 Advanced Plots:")
    if "learning_curves" in results:
        lc = results["learning_curves"]
        plot_learning_curves(
            lc["train_sizes"], lc["train_scores"],
            lc["test_scores"], lc["algorithm_names"],
        )

    if "cv_results" in results:
        plot_cv_score_distribution(results["cv_results"])

    if "feature_importance" in results:
        fi = results["feature_importance"]
        plot_feature_importance(fi["feature_names"], fi["importances"], fi.get("suffix", ""))

    if "season_metrics" in results:
        plot_season_comparison(results["season_metrics"])

    # ── Preprocessing Plots ─────────────────────────────────

    print("\n   🧹 Preprocessing Plots:")
    if not race_full.empty:
        plot_missing_values_heatmap(
            race_full, "Missing Values — Race Data (After Cleaning)",
            "missing_values_after.png",
        )

    if not pitstop_full.empty:
        plot_train_test_split_pie(
            int(len(pitstop_full) * 0.8),
            int(len(pitstop_full) * 0.2),
        )

    print(f"\n   ✅ All visualizations saved to {PLOTS_DIR}")


# ── Standalone execution ─────────────────────────────────────

if __name__ == "__main__":
    print("\n📊 LIGHTS OUT AND AWAY — Generating Visualizations\n")
    print("   This module is designed to be called from main.py")
    print("   with model results passed as arguments.")
    print("   Run main.py to generate all visualizations.")
