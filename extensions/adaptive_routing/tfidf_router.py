"""Evaluate threshold-based routing with TF-IDF evidence complexity."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROUTING_SIGNAL = "tfidf_effective_evidence_count"
THRESHOLDS = (3, 4, 5, 6, 7, 8, 9, 10, 12, 15)
REQUIRED_COLUMNS = (
    "direct_em",
    "tiser_em",
    "direct_f1",
    "tiser_f1",
    "direct_generated_tokens",
    "tiser_generated_tokens",
    ROUTING_SIGNAL,
)
OUTPUT_COLUMNS = (
    "threshold",
    "routed_em",
    "routed_f1",
    "avg_generated_tokens",
    "total_generated_tokens",
    "direct_count",
    "direct_pct",
    "tiser_count",
    "tiser_pct",
    "token_saving_vs_tiser_pct",
    "both_correct_to_direct",
    "both_correct_coverage_pct",
    "rescued_sent_to_direct",
    "rescue_loss_pct",
    "rescue_preserved_pct",
    "tiser_gain_retained_pct",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TF-IDF adaptive-routing thresholds."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_routing_data(path: Path) -> pd.DataFrame:
    dataframe = pd.read_json(path, lines=True)
    missing_columns = set(REQUIRED_COLUMNS).difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"Input JSONL is missing required columns: {missing}")
    if dataframe.empty:
        raise ValueError("Input JSONL contains no examples.")

    for column in REQUIRED_COLUMNS:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="raise").astype(
            float
        )
    if dataframe.loc[:, list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required routing columns must not contain missing values.")
    return dataframe


def percentage(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator * 100.0)


def calculate_baseline(dataframe: pd.DataFrame, path_name: str) -> dict[str, float]:
    return {
        "em": float(dataframe[f"{path_name}_em"].mean() * 100.0),
        "f1": float(dataframe[f"{path_name}_f1"].mean() * 100.0),
        "average_generated_tokens": float(
            dataframe[f"{path_name}_generated_tokens"].mean()
        ),
        "total_generated_tokens": float(
            dataframe[f"{path_name}_generated_tokens"].sum()
        ),
    }


def evaluate_thresholds(
    dataframe: pd.DataFrame,
    direct_baseline: dict[str, float],
    tiser_baseline: dict[str, float],
    both_correct: pd.Series,
    tiser_rescue: pd.Series,
) -> pd.DataFrame:
    total_examples = len(dataframe)
    total_both_correct = int(both_correct.sum())
    total_tiser_rescues = int(tiser_rescue.sum())
    tiser_gain_over_direct = tiser_baseline["em"] - direct_baseline["em"]
    rows = []

    for threshold in THRESHOLDS:
        route_to_direct = dataframe[ROUTING_SIGNAL] < threshold
        route_to_tiser = ~route_to_direct

        routed_em_values = np.where(
            route_to_direct,
            dataframe["direct_em"],
            dataframe["tiser_em"],
        )
        routed_f1_values = np.where(
            route_to_direct,
            dataframe["direct_f1"],
            dataframe["tiser_f1"],
        )
        routed_token_values = np.where(
            route_to_direct,
            dataframe["direct_generated_tokens"],
            dataframe["tiser_generated_tokens"],
        )

        routed_em = float(np.mean(routed_em_values) * 100.0)
        routed_f1 = float(np.mean(routed_f1_values) * 100.0)
        total_generated_tokens = float(np.sum(routed_token_values))
        direct_count = int(route_to_direct.sum())
        tiser_count = int(route_to_tiser.sum())
        both_correct_to_direct = int((both_correct & route_to_direct).sum())
        rescued_sent_to_direct = int((tiser_rescue & route_to_direct).sum())

        rescue_loss_pct = percentage(
            rescued_sent_to_direct,
            total_tiser_rescues,
        )
        rescue_preserved_pct = (
            100.0 - rescue_loss_pct
            if not np.isnan(rescue_loss_pct)
            else float("nan")
        )
        gain_retained_pct = (
            (routed_em - direct_baseline["em"]) / tiser_gain_over_direct * 100.0
            if tiser_gain_over_direct != 0.0
            else float("nan")
        )
        token_saving_pct = (
            (1.0 - total_generated_tokens / tiser_baseline["total_generated_tokens"])
            * 100.0
            if tiser_baseline["total_generated_tokens"] != 0.0
            else float("nan")
        )

        rows.append(
            {
                "threshold": threshold,
                "routed_em": routed_em,
                "routed_f1": routed_f1,
                "avg_generated_tokens": float(np.mean(routed_token_values)),
                "total_generated_tokens": total_generated_tokens,
                "direct_count": direct_count,
                "direct_pct": percentage(direct_count, total_examples),
                "tiser_count": tiser_count,
                "tiser_pct": percentage(tiser_count, total_examples),
                "token_saving_vs_tiser_pct": token_saving_pct,
                "both_correct_to_direct": both_correct_to_direct,
                "both_correct_coverage_pct": percentage(
                    both_correct_to_direct,
                    total_both_correct,
                ),
                "rescued_sent_to_direct": rescued_sent_to_direct,
                "rescue_loss_pct": rescue_loss_pct,
                "rescue_preserved_pct": rescue_preserved_pct,
                "tiser_gain_retained_pct": gain_retained_pct,
            }
        )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def print_baseline(name: str, baseline: dict[str, float]) -> None:
    print(f"{name}:")
    print(f"  EM: {baseline['em']:.2f}%")
    print(f"  F1: {baseline['f1']:.2f}%")
    print(
        "  Average generated tokens: "
        f"{baseline['average_generated_tokens']:.2f}"
    )


def print_selected_operating_point(sweep: pd.DataFrame) -> None:
    selected = sweep.loc[sweep["threshold"] == 6].iloc[0]
    print("\nSelected operating point: threshold 6")
    print(f"Routed EM: {selected['routed_em']:.2f}%")
    print(f"Routed F1: {selected['routed_f1']:.2f}%")
    print(f"Average generated tokens: {selected['avg_generated_tokens']:.2f}")
    print(f"Sent to Direct: {selected['direct_pct']:.2f}%")
    print(f"Sent to TISER: {selected['tiser_pct']:.2f}%")
    print(
        "Token saving vs Always TISER: "
        f"{selected['token_saving_vs_tiser_pct']:.2f}%"
    )
    print(
        "Both-correct coverage: "
        f"{selected['both_correct_coverage_pct']:.2f}%"
    )
    print(
        "TISER rescues preserved: "
        f"{selected['rescue_preserved_pct']:.2f}%"
    )
    print(
        "TISER gain retained: "
        f"{selected['tiser_gain_retained_pct']:.2f}%"
    )


def main() -> None:
    args = parse_args()
    dataframe = load_routing_data(args.input)
    direct_baseline = calculate_baseline(dataframe, "direct")
    tiser_baseline = calculate_baseline(dataframe, "tiser")

    both_correct = (dataframe["direct_em"] == 1.0) & (
        dataframe["tiser_em"] == 1.0
    )
    tiser_rescue = (dataframe["direct_em"] == 0.0) & (
        dataframe["tiser_em"] == 1.0
    )

    sweep = evaluate_thresholds(
        dataframe,
        direct_baseline,
        tiser_baseline,
        both_correct,
        tiser_rescue,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.output, index=False)

    print(f"Loaded examples: {len(dataframe)}\n")
    print_baseline("Always Direct", direct_baseline)
    print()
    print_baseline("Always TISER", tiser_baseline)
    print(f"\nBoth-correct examples: {int(both_correct.sum())}")
    print(f"TISER-rescue examples: {int(tiser_rescue.sum())}")
    print("\nThreshold sweep")
    print(sweep.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print_selected_operating_point(sweep)
    print(f"\nSaved threshold sweep to: {args.output}")


if __name__ == "__main__":
    main()
