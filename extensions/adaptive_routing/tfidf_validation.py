"""Validate TF-IDF structural-complexity signals against model performance."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


METRIC_COLUMNS = (
    "direct_em",
    "tiser_em",
    "direct_f1",
    "tiser_f1",
)
SIGNAL_COLUMNS = (
    "tfidf_concentration",
    "tfidf_effective_evidence_count",
)
REQUIRED_COLUMNS = METRIC_COLUMNS + SIGNAL_COLUMNS
QUARTILE_LABELS = (
    "Q1 - Lowest",
    "Q2",
    "Q3",
    "Q4 - Highest",
)
PERCENTAGE_COLUMNS = (
    "direct_em",
    "tiser_em",
    "em_gain",
    "direct_f1",
    "tiser_f1",
    "f1_gain",
    "both_correct_rate",
    "tiser_rescue_rate",
)
CORRELATION_TARGETS = (
    ("Direct EM", "direct_em"),
    ("Direct F1", "direct_f1"),
    ("TISER EM gain", "em_gain"),
    ("TISER F1 gain", "f1_gain"),
)
SIGNAL_TITLES = {
    "tfidf_concentration": "TF-IDF concentration",
    "tfidf_effective_evidence_count": "TF-IDF effective evidence count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate TF-IDF structural-complexity signals."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_validation_data(path: Path) -> pd.DataFrame:
    dataframe = pd.read_json(path, lines=True)
    missing_columns = set(REQUIRED_COLUMNS).difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"Input JSONL is missing required columns: {missing}")
    if len(dataframe) < 4:
        raise ValueError("At least four examples are required to create quartiles.")

    for column in REQUIRED_COLUMNS:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="raise").astype(
            float
        )
    if dataframe.loc[:, REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Required metric and TF-IDF columns must not contain missing values.")

    dataframe["em_gain"] = dataframe["tiser_em"] - dataframe["direct_em"]
    dataframe["f1_gain"] = dataframe["tiser_f1"] - dataframe["direct_f1"]
    dataframe["both_correct"] = (
        (dataframe["direct_em"] == 1.0) & (dataframe["tiser_em"] == 1.0)
    ).astype(int)
    dataframe["tiser_rescue"] = (
        (dataframe["direct_em"] == 0.0) & (dataframe["tiser_em"] == 1.0)
    ).astype(int)
    return dataframe


def get_complexity_score(
    dataframe: pd.DataFrame,
    signal: str,
) -> pd.Series:
    raw_feature = dataframe[signal]
    if signal == "tfidf_concentration":
        return -raw_feature
    return raw_feature


def assign_complexity_quartiles(complexity_score: pd.Series) -> pd.Series:
    # Ranking keeps tied values in complexity order while ensuring qcut can
    # still produce four approximately equal groups.
    ordered_ranks = complexity_score.rank(method="first")
    return pd.qcut(ordered_ranks, q=4, labels=QUARTILE_LABELS)


def calculate_quartile_summary(
    dataframe: pd.DataFrame,
    signal: str,
    quartiles: pd.Series,
) -> pd.DataFrame:
    analysis = dataframe.copy()
    analysis["quartile"] = quartiles

    summary = (
        analysis.groupby("quartile", observed=False)
        .agg(
            examples=(signal, "size"),
            raw_feature_mean=(signal, "mean"),
            raw_feature_median=(signal, "median"),
            direct_em=("direct_em", "mean"),
            tiser_em=("tiser_em", "mean"),
            em_gain=("em_gain", "mean"),
            direct_f1=("direct_f1", "mean"),
            tiser_f1=("tiser_f1", "mean"),
            f1_gain=("f1_gain", "mean"),
            both_correct_rate=("both_correct", "mean"),
            tiser_rescue_rate=("tiser_rescue", "mean"),
        )
        .reset_index()
    )
    summary.loc[:, list(PERCENTAGE_COLUMNS)] *= 100.0
    return summary


def calculate_correlations(
    dataframe: pd.DataFrame,
    signal: str,
    complexity_score: pd.Series,
) -> list[dict[str, str | float]]:
    correlations = []
    for target_label, target_column in CORRELATION_TARGETS:
        result = spearmanr(
            complexity_score,
            dataframe[target_column],
            nan_policy="omit",
        )
        correlations.append(
            {
                "signal": signal,
                "target": target_label,
                "spearman_rho": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    return correlations


def print_quartile_table(signal: str, summary: pd.DataFrame) -> None:
    print(f"\n{SIGNAL_TITLES[signal]} quartiles")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def print_correlations(
    signal: str,
    correlations: list[dict[str, str | float]],
) -> None:
    print(f"\n{SIGNAL_TITLES[signal]} correlations")
    for correlation in correlations:
        print(
            f"{correlation['target']}: "
            f"rho={correlation['spearman_rho']:.6f}, "
            f"p-value={correlation['p_value']:.6g}"
        )


def print_lowest_highest_comparison(
    signal: str,
    summary: pd.DataFrame,
) -> None:
    indexed_summary = summary.set_index("quartile")
    lowest = indexed_summary.loc["Q1 - Lowest"]
    highest = indexed_summary.loc["Q4 - Highest"]

    comparisons = (
        ("Direct EM", "direct_em"),
        ("TISER EM", "tiser_em"),
        ("TISER EM advantage", "em_gain"),
        ("Both-correct rate", "both_correct_rate"),
        ("TISER-rescue rate", "tiser_rescue_rate"),
    )
    print(f"\n{SIGNAL_TITLES[signal]}: Q1 vs Q4")
    for label, column in comparisons:
        print(f"{label}: {lowest[column]:.2f}% -> {highest[column]:.2f}%")


def main() -> None:
    args = parse_args()
    dataframe = load_validation_data(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded examples: {len(dataframe)}")

    correlation_rows = []
    output_paths = []
    for signal in SIGNAL_COLUMNS:
        complexity_score = get_complexity_score(dataframe, signal)
        quartiles = assign_complexity_quartiles(complexity_score)
        summary = calculate_quartile_summary(dataframe, signal, quartiles)
        correlations = calculate_correlations(
            dataframe,
            signal,
            complexity_score,
        )

        output_path = args.output_dir / f"{signal}_validation.csv"
        summary.to_csv(output_path, index=False)
        output_paths.append(output_path)
        correlation_rows.extend(correlations)

        print_quartile_table(signal, summary)
        print_correlations(signal, correlations)
        print_lowest_highest_comparison(signal, summary)

    correlation_path = args.output_dir / "tfidf_validation_correlations.csv"
    pd.DataFrame(
        correlation_rows,
        columns=("signal", "target", "spearman_rho", "p_value"),
    ).to_csv(correlation_path, index=False)

    print("\nSaved outputs:")
    for output_path in output_paths:
        print(output_path)
    print(correlation_path)


if __name__ == "__main__":
    main()
