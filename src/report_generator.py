# ============================================================
# Lights Out and Away — F1 ML Project
# src/report_generator.py
#
# Generates a professional PDF report using FPDF2 with F1 theming.
# The report includes all visualizations, metrics tables, and
# written analysis for both ML problems.
# ============================================================

import os
import warnings
import glob
from fpdf import FPDF

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


# ── F1 Colors (RGB tuples) ──────────────────────────────────

F1_BG = (26, 26, 46)        # #1a1a2e
F1_DARK = (22, 33, 62)      # #16213e
F1_RED = (232, 0, 45)       # #e8002d
F1_WHITE = (255, 255, 255)  # #ffffff
F1_ORANGE = (255, 124, 67)  # #ff7c43
F1_GRAY = (180, 180, 180)


class F1Report(FPDF):
    """Custom FPDF class with F1 theming for the project report."""

    def __init__(self):
        """Initialize the F1 themed PDF report."""
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        self.chapter_num = 0

    def header(self):
        """Add F1 themed header to every page (except cover)."""
        if self.page_no() > 1:
            self.set_fill_color(*F1_BG)
            self.rect(0, 0, 210, 15, "F")
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(*F1_RED)
            self.set_y(5)
            self.cell(0, 5, "LIGHTS OUT AND AWAY | F1 ML PROJECT", align="C")
            # Red line under header
            self.set_draw_color(*F1_RED)
            self.set_line_width(0.5)
            self.line(10, 15, 200, 15)

    def footer(self):
        """Add F1 themed footer with page number."""
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*F1_GRAY)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def f1_page_background(self):
        """Set F1 dark background for the page."""
        self.set_fill_color(*F1_BG)
        self.rect(0, 0, 210, 297, "F")

    def chapter_title(self, title):
        """Add a styled chapter title with F1 red accent."""
        self.chapter_num += 1
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(*F1_WHITE)
        self.ln(10)
        self.cell(0, 12, f"Chapter {self.chapter_num}: {title}", new_x="LMARGIN", new_y="NEXT")
        # Red underline
        self.set_draw_color(*F1_RED)
        self.set_line_width(1)
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.ln(8)

    def section_title(self, title):
        """Add a section subtitle."""
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*F1_ORANGE)
        self.ln(4)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def body_text(self, text):
        """Add body text in white on dark background."""
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*F1_WHITE)
        self.multi_cell(0, 6, text)
        self.ln(3)

    def add_plot(self, image_path, caption="", width=170):
        """Add a plot image with optional caption."""
        if not os.path.exists(image_path):
            self.body_text(f"[Plot not found: {os.path.basename(image_path)}]")
            return

        # Check if enough space on page
        if self.get_y() > 180:
            self.add_page()
            self.f1_page_background()

        try:
            self.image(image_path, x=20, w=width)
        except Exception as e:
            self.body_text(f"[Error loading plot: {e}]")
            return

        if caption:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*F1_GRAY)
            self.cell(0, 5, caption, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def metrics_table(self, headers, rows):
        """Add a styled metrics comparison table."""
        self.set_font("Helvetica", "B", 9)

        # Calculate column widths
        num_cols = len(headers)
        col_width = 180 / num_cols

        # Header row
        self.set_fill_color(*F1_RED)
        self.set_text_color(*F1_WHITE)
        for header in headers:
            self.cell(col_width, 8, str(header), border=1, align="C", fill=True)
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", 9)
        self.set_fill_color(*F1_DARK)
        for row in rows:
            self.set_text_color(*F1_WHITE)
            for cell in row:
                self.cell(col_width, 7, str(cell), border=1, align="C", fill=True)
            self.ln()
        self.ln(5)


def generate_report(winner_results=None, pitstop_results=None):
    """
    Generate the complete PDF report with all chapters.

    Args:
        winner_results: dict with winner prediction results
        pitstop_results: dict with pit stop prediction results
    """
    print("\n   📄 Generating PDF Report...")

    pdf = F1Report()
    pdf.alias_nb_pages()

    # ══════════════════════════════════════════════════════════
    # COVER PAGE
    # ══════════════════════════════════════════════════════════

    pdf.add_page()
    pdf.f1_page_background()

    # F1 Car ASCII art
    pdf.set_font("Courier", "B", 8)
    pdf.set_text_color(*F1_RED)
    pdf.ln(20)

    car_art = [
        "                          _______________",
        "                   __===~~~~~            ~~~~===__",
        "              _=~~                                ~~=_",
        "           _=~          FORMULA 1 ML PROJECT         ~=_",
        "         =~                                             ~=",
        "       =~       _____    ________________________    ____~=",
        "     =(_____===~     ~~~~                    ~~~~===~    _)=",
        "        |                                              |",
        "        `---____________________________________---'",
        "              |  ||| || ||  || || |||  |  |||",
        "              ^  ||| || ||  || || |||  |  |||",
    ]

    for line in car_art:
        pdf.cell(0, 4, line, align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(15)

    # Title
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(*F1_WHITE)
    pdf.cell(0, 15, "LIGHTS OUT AND AWAY", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*F1_RED)
    pdf.cell(0, 10, "Predicting F1 Race Winners and Pit Stop Durations", align="C",
             new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(*F1_ORANGE)
    pdf.cell(0, 8, "Using Supervised Machine Learning", align="C",
             new_x="LMARGIN", new_y="NEXT")

    pdf.ln(15)

    # Line separator
    pdf.set_draw_color(*F1_RED)
    pdf.set_line_width(1)
    pdf.line(40, pdf.get_y(), 170, pdf.get_y())
    pdf.ln(10)

    # Course details
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*F1_WHITE)
    pdf.cell(0, 7, "Course: Fundamentals of AI/ML", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Algorithms: Linear Regression & K-Nearest Neighbors",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Data Source: FastF1 Python Library (2022-2024 Seasons)",
             align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(10)

    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*F1_GRAY)
    pdf.cell(0, 7, "Technologies: Python | Scikit-learn | Matplotlib | Pandas | FastF1",
             align="C", new_x="LMARGIN", new_y="NEXT")

    # ══════════════════════════════════════════════════════════
    # TABLE OF CONTENTS
    # ══════════════════════════════════════════════════════════

    pdf.add_page()
    pdf.f1_page_background()
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*F1_WHITE)
    pdf.cell(0, 12, "Table of Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*F1_RED)
    pdf.line(10, pdf.get_y() + 2, 200, pdf.get_y() + 2)
    pdf.ln(10)

    toc_items = [
        ("Chapter 1", "Introduction and Problem Statement"),
        ("Chapter 2", "Dataset Description"),
        ("Chapter 3", "Data Preprocessing"),
        ("Chapter 4", "Exploratory Data Analysis"),
        ("Chapter 5", "Problem 1 — Race Winner Prediction"),
        ("Chapter 6", "Problem 2 — Pit Stop Duration Prediction"),
        ("Chapter 7", "Algorithm Comparison"),
        ("Chapter 8", "Conclusion and Future Work"),
        ("", "References"),
    ]

    for num, title in toc_items:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*F1_RED)
        pdf.cell(30, 8, num)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(*F1_WHITE)
        dots = "." * (50 - len(title))
        pdf.cell(0, 8, f"{title} {dots}", new_x="LMARGIN", new_y="NEXT")

    # ══════════════════════════════════════════════════════════
    # CHAPTER 1 — INTRODUCTION
    # ══════════════════════════════════════════════════════════

    pdf.add_page()
    pdf.f1_page_background()
    pdf.chapter_title("Introduction and Problem Statement")

    pdf.body_text(
        "Formula 1 is one of the most data-intensive sports in the world. Every race "
        "generates millions of data points from telemetry sensors, timing systems, and "
        "weather stations. This project leverages real F1 data to build predictive "
        "machine learning models using supervised learning algorithms."
    )

    pdf.body_text(
        "We address two prediction problems: (1) predicting the race winner based on "
        "qualifying results, team performance, weather conditions, and circuit "
        "characteristics, and (2) predicting pit stop durations based on team, tyre "
        "compound, lap number, and environmental conditions."
    )

    pdf.section_title("Motivation")
    pdf.body_text(
        "The ability to predict race outcomes is valuable for teams, strategists, "
        "commentators, and fans. Pit stop strategy is one of the most critical factors "
        "in modern F1, where fractions of a second can determine the race outcome. "
        "By applying Linear Regression and K-Nearest Neighbors to these problems, "
        "we can compare the effectiveness of different ML approaches on real-world "
        "sports data."
    )

    pdf.section_title("Objectives")
    pdf.body_text(
        "1. Collect comprehensive F1 data from 2022, 2023, and 2024 seasons using "
        "the FastF1 Python library.\n"
        "2. Clean and preprocess the data for machine learning.\n"
        "3. Train and evaluate Linear Regression and KNN models on both problems.\n"
        "4. Compare algorithm performance using standard ML metrics.\n"
        "5. Visualize results with professional, F1-themed plots.\n"
        "6. Determine which algorithm performs better for each prediction task."
    )

    # ══════════════════════════════════════════════════════════
    # CHAPTER 2 — DATASET DESCRIPTION
    # ══════════════════════════════════════════════════════════

    pdf.add_page()
    pdf.f1_page_background()
    pdf.chapter_title("Dataset Description")

    pdf.body_text(
        "All data was collected using the FastF1 Python library (v3.4.0), which "
        "provides programmatic access to official F1 timing data. We collected data "
        "from 24 races across three seasons (2022, 2023, 2024), totaling up to 72 "
        "race sessions."
    )

    pdf.section_title("Data Sources")
    pdf.body_text(
        "Race Results: Final positions, points scored, fastest lap times.\n"
        "Qualifying Data: Grid positions for each driver.\n"
        "Pit Stop Data: Duration in seconds, lap number, tyre compound.\n"
        "Weather Data: Air temperature, track temperature, humidity, rainfall, wind speed.\n"
        "Circuit Info: Circuit name, country, number of laps.\n"
        "Tyre Data: Compound used per stint, stint length."
    )

    pdf.section_title("Seasons Covered")
    pdf.body_text(
        "2022 Season: 22 races (no Emilia Romagna was cancelled)\n"
        "2023 Season: 22 races\n"
        "2024 Season: 24 races including new additions like Las Vegas and China"
    )

    # Add data collection plots
    pdf.section_title("Data Collection Visualizations")

    plot_files = [
        ("01_race_calendar_heatmap.png", "Figure 2.1: Race Calendar Heatmap"),
        ("records_per_race.png", "Figure 2.2: Records Collected Per Race"),
    ]

    for filename, caption in plot_files:
        path = os.path.join(PLOTS_DIR, filename)
        pdf.add_plot(path, caption)

    # ══════════════════════════════════════════════════════════
    # CHAPTER 3 — DATA PREPROCESSING
    # ══════════════════════════════════════════════════════════

    pdf.add_page()
    pdf.f1_page_background()
    pdf.chapter_title("Data Preprocessing")

    pdf.body_text(
        "Raw data from FastF1 requires significant preprocessing before it can be "
        "used for machine learning. Our preprocessing pipeline includes the following "
        "steps:"
    )

    pdf.section_title("Cleaning Steps")
    pdf.body_text(
        "1. Handling Missing Values: Dropped rows with missing position or grid data. "
        "Filled missing weather values with session averages.\n"
        "2. Outlier Removal: Pit stops longer than 60 seconds (penalties, red flags) "
        "and shorter than 1.5 seconds (data errors) were removed.\n"
        "3. Type Conversion: Converted position and lap numbers to integers, "
        "timedeltas to seconds.\n"
        "4. Pit Lane Start Handling: Grid position of 0 (pit lane start) set to 20."
    )

    pdf.section_title("Feature Engineering")
    pdf.body_text(
        "IsWinner: Binary flag (1 if Position == 1, else 0).\n"
        "AvgPitStops: Average number of pit stops per driver per race.\n"
        "TyreAge: Number of laps on current tyre compound at pit stop.\n"
        "StartingCompound: Tyre compound used at race start.\n"
        "RaceYear: Year of the race as a feature."
    )

    pdf.section_title("Feature Encoding")
    pdf.body_text(
        "Categorical features (Team, Driver, Circuit, Tyre Compound) were encoded "
        "using sklearn LabelEncoder. This converts text labels to numeric values "
        "while maintaining the ability to decode back to original values."
    )

    pdf.section_title("Train/Test Split")
    pdf.body_text(
        "Data was split 80/20 for training and testing using sklearn train_test_split "
        "with random_state=42 for reproducibility. Features were then scaled using "
        "StandardScaler to normalize all features to zero mean and unit variance."
    )

    # Preprocessing plots
    plot_files = [
        ("missing_values_after.png", "Figure 3.1: Missing Values After Cleaning"),
        ("06_feature_correlation_heatmap.png", "Figure 3.2: Feature Correlation Heatmap"),
        ("train_test_split_pie.png", "Figure 3.3: Train/Test Split Distribution"),
    ]

    for filename, caption in plot_files:
        path = os.path.join(PLOTS_DIR, filename)
        pdf.add_plot(path, caption)

    # ══════════════════════════════════════════════════════════
    # CHAPTER 4 — EXPLORATORY DATA ANALYSIS
    # ══════════════════════════════════════════════════════════

    pdf.add_page()
    pdf.f1_page_background()
    pdf.chapter_title("Exploratory Data Analysis")

    pdf.body_text(
        "Before building our models, we performed exploratory data analysis to "
        "understand the patterns and distributions in the F1 dataset. Key findings "
        "are presented through the following visualizations."
    )

    eda_plots = [
        ("02_driver_radar_chart.png", "Figure 4.1: Top 10 Driver Performance Radar"),
        ("03_team_performance.png", "Figure 4.2: Constructor Championship Points"),
        ("04_circuit_difficulty.png", "Figure 4.3: Circuit Difficulty by Position Variance"),
        ("05_tyre_usage_pie.png", "Figure 4.4: Tyre Compound Usage Per Season"),
    ]

    for filename, caption in eda_plots:
        path = os.path.join(PLOTS_DIR, filename)
        pdf.add_plot(path, caption)

    # ══════════════════════════════════════════════════════════
    # CHAPTER 5 — WINNER PREDICTION
    # ══════════════════════════════════════════════════════════

    pdf.add_page()
    pdf.f1_page_background()
    pdf.chapter_title("Problem 1 — Race Winner Prediction")

    pdf.section_title("Problem Definition")
    pdf.body_text(
        "Predict the finishing position of each driver in a race. The target variable "
        "is Position (1 = winner, 2 = second place, etc.). We use features such as "
        "grid position from qualifying, team, driver, circuit, weather conditions, "
        "average pit stops, and starting tyre compound."
    )

    pdf.section_title("Linear Regression")
    pdf.body_text(
        "Linear Regression treats finishing position as a continuous variable and "
        "finds the best linear relationship between features and the target. While "
        "positions are discrete, this approach captures the general trend."
    )

    pdf.section_title("KNN Classifier")
    pdf.body_text(
        "K-Nearest Neighbors classifies each race entry into a finishing position "
        "by finding the K most similar past race entries and using majority voting. "
        "We tested K values from 1 to 20 and selected the optimal K using 5-fold "
        "cross-validation."
    )

    # Add metrics table if available
    if winner_results and "lr_metrics" in winner_results:
        pdf.section_title("Results")
        lr = winner_results["lr_metrics"]
        knn = winner_results["knn_metrics"]
        best_k = winner_results.get("best_k", "?")

        pdf.metrics_table(
            ["Metric", "Linear Regression", f"KNN (K={best_k})"],
            [
                ["MSE", f"{lr['MSE']:.4f}", f"{knn['MSE']:.4f}"],
                ["RMSE", f"{lr['RMSE']:.4f}", f"{knn['RMSE']:.4f}"],
                ["MAE", f"{lr['MAE']:.4f}", f"{knn['MAE']:.4f}"],
                ["R\u00b2 Score", f"{lr['R2']:.4f}", f"{knn['R2']:.4f}"],
                ["Accuracy", "N/A", f"{knn['Accuracy']:.4f}"],
                ["Precision", "N/A", f"{knn['Precision']:.4f}"],
                ["Recall", "N/A", f"{knn['Recall']:.4f}"],
                ["F1 Score", "N/A", f"{knn['F1']:.4f}"],
            ],
        )

    # Winner prediction plots
    winner_plots = [
        ("07_grid_vs_finish.png", "Figure 5.1: Grid vs Finishing Position"),
        ("08_lr_actual_vs_predicted_winner.png", "Figure 5.2: LR Actual vs Predicted"),
        ("09_knn_confusion_matrix.png", "Figure 5.3: KNN Confusion Matrix"),
        ("10_knn_accuracy_vs_k.png", "Figure 5.4: KNN Accuracy vs K Value"),
        ("11_algorithm_comparison_winner.png", "Figure 5.5: Algorithm Comparison"),
        ("12_top_predicted_winners.png", "Figure 5.6: Top Predicted Winners"),
        ("13_driver_win_probability_heatmap.png", "Figure 5.7: Win Probability Heatmap"),
    ]

    for filename, caption in winner_plots:
        path = os.path.join(PLOTS_DIR, filename)
        pdf.add_plot(path, caption)

    # ══════════════════════════════════════════════════════════
    # CHAPTER 6 — PIT STOP PREDICTION
    # ══════════════════════════════════════════════════════════

    pdf.add_page()
    pdf.f1_page_background()
    pdf.chapter_title("Problem 2 — Pit Stop Duration Prediction")

    pdf.section_title("Problem Definition")
    pdf.body_text(
        "Predict the duration of a pit stop in seconds. Pit stop duration is affected "
        "by the team's crew speed, tyre compound being fitted, lap number (fatigue "
        "effects), weather conditions, and circuit characteristics."
    )

    pdf.section_title("Linear Regression")
    pdf.body_text(
        "Linear Regression models the relationship between pit stop features and "
        "duration as a linear function. The model learns coefficients for each "
        "feature that best predict the continuous duration value."
    )

    pdf.section_title("KNN Regressor")
    pdf.body_text(
        "KNN Regression predicts pit stop duration by averaging the durations of "
        "the K nearest neighbors in feature space. This captures non-linear "
        "relationships that Linear Regression may miss."
    )

    # Add metrics table if available
    if pitstop_results and "lr_metrics" in pitstop_results:
        pdf.section_title("Results")
        lr = pitstop_results["lr_metrics"]
        knn = pitstop_results["knn_metrics"]
        best_k = pitstop_results.get("best_k", "?")

        pdf.metrics_table(
            ["Metric", "Linear Regression", f"KNN (K={best_k})"],
            [
                ["MSE", f"{lr['MSE']:.4f}", f"{knn['MSE']:.4f}"],
                ["RMSE", f"{lr['RMSE']:.4f}", f"{knn['RMSE']:.4f}"],
                ["MAE", f"{lr['MAE']:.4f}", f"{knn['MAE']:.4f}"],
                ["R\u00b2 Score", f"{lr['R2']:.4f}", f"{knn['R2']:.4f}"],
                ["CV RMSE", f"{lr['CV_RMSE_mean']:.4f}", f"{knn['CV_RMSE_mean']:.4f}"],
                ["Residual Mean", f"{lr['Residual_Mean']:.4f}", f"{knn['Residual_Mean']:.4f}"],
                ["Residual Std", f"{lr['Residual_Std']:.4f}", f"{knn['Residual_Std']:.4f}"],
            ],
        )

    # Pit stop plots
    pitstop_plots = [
        ("14_pitstop_violin_by_team.png", "Figure 6.1: Pit Stop Duration by Team"),
        ("15_pitstop_vs_lap.png", "Figure 6.2: Pit Stop Duration vs Lap Number"),
        ("16_lr_actual_vs_predicted_pitstop.png", "Figure 6.3: LR Prediction"),
        ("17_knn_actual_vs_predicted_pitstop.png", "Figure 6.4: KNN Prediction"),
        ("18_residual_comparison_pitstop.png", "Figure 6.5: Residual Analysis"),
        ("19_algorithm_comparison_pitstop.png", "Figure 6.6: Algorithm Comparison"),
        ("20_pitstop_by_circuit.png", "Figure 6.7: Pit Stop by Circuit"),
        ("21_tyre_compound_vs_pitstop.png", "Figure 6.8: Tyre vs Pit Stop"),
        ("22_team_pitstop_ranking.png", "Figure 6.9: Team Ranking"),
    ]

    for filename, caption in pitstop_plots:
        path = os.path.join(PLOTS_DIR, filename)
        pdf.add_plot(path, caption)

    # ══════════════════════════════════════════════════════════
    # CHAPTER 7 — ALGORITHM COMPARISON
    # ══════════════════════════════════════════════════════════

    pdf.add_page()
    pdf.f1_page_background()
    pdf.chapter_title("Algorithm Comparison")

    pdf.body_text(
        "This chapter provides a comprehensive comparison of Linear Regression and "
        "K-Nearest Neighbors across both prediction problems. We analyze the "
        "strengths and weaknesses of each approach."
    )

    pdf.section_title("Key Findings")

    if winner_results and pitstop_results:
        w_lr = winner_results.get("lr_metrics", {})
        w_knn = winner_results.get("knn_metrics", {})
        p_lr = pitstop_results.get("lr_metrics", {})
        p_knn = pitstop_results.get("knn_metrics", {})

        pdf.body_text(
            f"Winner Prediction:\n"
            f"  Linear Regression RMSE: {w_lr.get('RMSE', 'N/A')}\n"
            f"  KNN RMSE: {w_knn.get('RMSE', 'N/A')}\n\n"
            f"Pit Stop Prediction:\n"
            f"  Linear Regression RMSE: {p_lr.get('RMSE', 'N/A')}\n"
            f"  KNN RMSE: {p_knn.get('RMSE', 'N/A')}"
        )
    else:
        pdf.body_text(
            "Detailed metrics comparison is available in the plots below."
        )

    pdf.section_title("Linear Regression: Strengths and Weaknesses")
    pdf.body_text(
        "Strengths: Fast training, interpretable coefficients, works well when "
        "the relationship between features and target is approximately linear.\n"
        "Weaknesses: Cannot capture non-linear patterns, sensitive to outliers, "
        "assumes feature independence."
    )

    pdf.section_title("KNN: Strengths and Weaknesses")
    pdf.body_text(
        "Strengths: Non-parametric, captures non-linear relationships, simple "
        "to understand and implement.\n"
        "Weaknesses: Slower prediction time, sensitive to feature scaling and "
        "the curse of dimensionality, requires careful K selection."
    )

    # Advanced plots
    advanced_plots = [
        ("23_learning_curves.png", "Figure 7.1: Learning Curves"),
        ("24_cv_score_distribution.png", "Figure 7.2: Cross-Validation Distributions"),
        ("25_feature_importance.png", "Figure 7.3: Feature Importance"),
        ("26_season_comparison.png", "Figure 7.4: Season Comparison"),
    ]

    for filename, caption in advanced_plots:
        path = os.path.join(PLOTS_DIR, filename)
        pdf.add_plot(path, caption)

    # ══════════════════════════════════════════════════════════
    # CHAPTER 8 — CONCLUSION
    # ══════════════════════════════════════════════════════════

    pdf.add_page()
    pdf.f1_page_background()
    pdf.chapter_title("Conclusion and Future Work")

    pdf.section_title("Conclusion")
    pdf.body_text(
        "This project successfully demonstrated the application of supervised "
        "machine learning algorithms to real-world Formula 1 data. By collecting "
        "comprehensive data from three F1 seasons using the FastF1 library, we "
        "built and compared Linear Regression and KNN models for two distinct "
        "prediction problems."
    )

    pdf.body_text(
        "Key takeaways:\n"
        "1. Grid position is the strongest predictor of finishing position, "
        "confirming the importance of qualifying in F1.\n"
        "2. Team and driver encode significant performance information that "
        "improves predictions.\n"
        "3. Pit stop durations are influenced by team capability, with top teams "
        "consistently achieving faster pit stops.\n"
        "4. Both algorithms have their strengths, with KNN often capturing "
        "non-linear patterns that Linear Regression misses."
    )

    pdf.section_title("Future Work")
    pdf.body_text(
        "1. Ensemble Methods: Combine LR and KNN with Random Forest or Gradient "
        "Boosting for improved accuracy.\n"
        "2. Time Series Analysis: Incorporate temporal patterns and momentum.\n"
        "3. Deep Learning: Apply neural networks for more complex feature "
        "interactions.\n"
        "4. Real-time Prediction: Build a live dashboard during race weekends.\n"
        "5. Strategy Optimization: Extend pit stop prediction to full race "
        "strategy optimization.\n"
        "6. Additional Features: Include tyre degradation models, fuel load, "
        "and safety car probability."
    )

    # ══════════════════════════════════════════════════════════
    # REFERENCES
    # ══════════════════════════════════════════════════════════

    pdf.add_page()
    pdf.f1_page_background()
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*F1_WHITE)
    pdf.cell(0, 12, "References", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*F1_RED)
    pdf.line(10, pdf.get_y() + 2, 200, pdf.get_y() + 2)
    pdf.ln(10)

    references = [
        "[1] FastF1 Python Library — https://docs.fastf1.dev/",
        "[2] Scikit-learn: Machine Learning in Python — https://scikit-learn.org/",
        "[3] Matplotlib: Visualization with Python — https://matplotlib.org/",
        "[4] Pandas: Python Data Analysis Library — https://pandas.pydata.org/",
        "[5] Formula 1 Official Website — https://www.formula1.com/",
        "[6] Seaborn: Statistical Data Visualization — https://seaborn.pydata.org/",
        "[7] FPDF2: PDF Generation Library — https://py-pdf.github.io/fpdf2/",
        "[8] NumPy: Scientific Computing — https://numpy.org/",
        "[9] Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of "
        "Statistical Learning. Springer.",
        "[10] James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An "
        "Introduction to Statistical Learning. Springer.",
    ]

    for ref in references:
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*F1_WHITE)
        pdf.multi_cell(0, 6, ref)
        pdf.ln(2)

    # ── Save PDF ─────────────────────────────────────────────

    output_path = os.path.join(REPORTS_DIR, "lights_out_and_away_report.pdf")
    pdf.output(output_path)

    print(f"   ✅ Report saved: {output_path}")
    print(f"   📄 Total pages: {pdf.page_no()}")

    return output_path


# ── Standalone execution ─────────────────────────────────────

if __name__ == "__main__":
    print("\n📄 LIGHTS OUT AND AWAY — Generating Report\n")
    generate_report()
    print("\n✅ Report generation complete!")
