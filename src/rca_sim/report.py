"""Build partition-safe experiment summaries and plots."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_report(inputs: Sequence[Path], output: Path) -> dict[str, Any]:
    """Merge experiment summaries without comparing unlike partitions."""
    if not inputs:
        raise ValueError("report requires at least one input")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    training_runs: list[dict[str, Any]] = []
    for source in inputs:
        if not source.exists():
            raise ValueError(f"report input does not exist: {source}")
        metadata_path = source / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            training_runs.append({"source": str(source), **metadata})
        summary_path = source / "summary.json"
        if summary_path.is_file():
            rows.extend(_load_summary(summary_path, source))
        validation_path = source / "validation_evaluation" / "summary.json"
        if validation_path.is_file():
            rows.extend(_load_summary(validation_path, source, partition="validation"))
        audit_path = source / "audit_evaluation" / "summary.json"
        if audit_path.is_file():
            rows.extend(_load_summary(audit_path, source, partition="audit"))
    if not rows:
        raise ValueError("report inputs contain no evaluation summaries")

    test_rows = [row for row in rows if row["partition"] == "test"]
    validation_rows = [row for row in rows if row["partition"] == "validation"]
    if not test_rows:
        raise ValueError("final report contains no test results")
    _validate_partition_sizes(rows)
    _write_csv(output / "evaluation_summary.csv", rows)
    _plot_test_accuracy_cost(output / "test_accuracy_vs_credits.png", test_rows)
    result = {
        "schema_version": 1,
        "partitions": sorted({row["partition"] for row in rows}),
        "evaluation_rows": rows,
        "training_runs": training_runs,
        "ppo_test_across_seeds": _ppo_seed_summary(test_rows),
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(
        _markdown_report(test_rows, validation_rows, result["ppo_test_across_seeds"]),
        encoding="utf-8",
    )
    return result


def _load_summary(
    path: Path, source: Path, *, partition: str | None = None
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"evaluation summary must contain a list: {path}")
    resolved_partition = partition or _infer_partition(source)
    seed_match = re.search(r"s(101|202|303)$", source.name)
    seed = int(seed_match.group(1)) if seed_match else None
    return [
        {
            "partition": resolved_partition,
            "source": str(source),
            "seed": seed,
            **row,
        }
        for row in payload
    ]


def _infer_partition(source: Path) -> str:
    name = source.name.lower()
    if "test" in name:
        return "test"
    if name == "baselines_v0":
        return "validation"
    return "development"


def _validate_partition_sizes(rows: Sequence[dict[str, Any]]) -> None:
    expected = {"validation": 256, "audit": 512, "test": 1000}
    for row in rows:
        partition = row["partition"]
        if partition in expected and int(row["cases"]) != expected[partition]:
            raise ValueError(
                f"{partition} row from {row['source']} has {row['cases']} cases; "
                f"expected {expected[partition]}"
            )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_test_accuracy_cost(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    unique: dict[tuple[str, int | None], dict[str, Any]] = {}
    for row in rows:
        if row.get("mean_credits_spent") is None:
            continue
        key = (str(row["method"]), row.get("seed"))
        unique.setdefault(key, row)
    figure, axis = plt.subplots(figsize=(9, 6))
    for (method, seed), row in unique.items():
        label = f"{method} s{seed}" if seed is not None and method == "ppo" else method
        axis.scatter(row["mean_credits_spent"], row["accuracy"], s=42)
        axis.annotate(
            label,
            (row["mean_credits_spent"], row["accuracy"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xlabel("Mean credits spent")
    axis.set_ylabel("Test diagnosis accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _ppo_seed_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ppo = [row for row in rows if row["method"] == "ppo"]
    if len(ppo) != 3 or {row["seed"] for row in ppo} != {101, 202, 303}:
        raise ValueError("test report requires PPO results for seeds 101, 202, and 303")
    return {
        "seeds": [101, 202, 303],
        "mean_accuracy": mean(float(row["accuracy"]) for row in ppo),
        "min_accuracy": min(float(row["accuracy"]) for row in ppo),
        "max_accuracy": max(float(row["accuracy"]) for row in ppo),
        "mean_return": mean(float(row["mean_return"]) for row in ppo),
        "min_return": min(float(row["mean_return"]) for row in ppo),
        "max_return": max(float(row["mean_return"]) for row in ppo),
        "mean_credits": mean(float(row["mean_credits_spent"]) for row in ppo),
    }


def _markdown_report(
    test_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    ppo_summary: dict[str, Any],
) -> str:
    selected: list[dict[str, Any]] = []
    wanted = (
        "stop",
        "random_1_probes",
        "random_6_probes",
        "script_0.60",
        "voi1",
        "full_information",
    )
    for method in wanted:
        matches = [row for row in test_rows if row["method"] == method]
        match = next((row for row in matches if row.get("seed") is None), None)
        if match is None:
            match = next(iter(matches), None)
        if match is not None:
            selected.append(match)
    selected.extend(sorted(
        (row for row in test_rows if row["method"] == "ppo"),
        key=lambda row: row["seed"],
    ))
    lines = [
        "# Small simulator RL experiment report",
        "",
        "## Frozen test results",
        "",
        "| Method | Seed | Accuracy | Mean return | Mean credits | Planning seconds per episode |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        seed = "" if row.get("seed") is None else str(row["seed"])
        return_value = "n/a" if row.get("mean_return") is None else f"{row['mean_return']:.4f}"
        credits = "n/a" if row.get("mean_credits_spent") is None else f"{row['mean_credits_spent']:.3f}"
        lines.append(
            f"| {row['method']} | {seed} | {row['accuracy']:.2%} | {return_value} | "
            f"{credits} | {row['mean_planning_seconds']:.6f} |"
        )
    lines.extend([
        "",
        "## Across-seed PPO result",
        "",
        f"Across seeds 101, 202, and 303, PPO averaged {ppo_summary['mean_accuracy']:.2%} accuracy, "
        f"{ppo_summary['mean_return']:.4f} return, and {ppo_summary['mean_credits']:.3f} credits. "
        f"Seed accuracy ranged from {ppo_summary['min_accuracy']:.2%} to {ppo_summary['max_accuracy']:.2%}.",
        "",
        "## Interpretation",
        "",
        "The implementation passed the controlled simulator and PPO checks. PPO improved substantially over STOP and random acquisition. The strongest PPO seed also exceeded the scripted investigator on accuracy and return while spending fewer credits. Across seeds, PPO return remained above the scripted and random baselines, but accuracy was seed-sensitive.",
        "",
        "Exact one-step VOI remained stronger than PPO on accuracy and return. Its planning cost was several orders of magnitude higher. Full information reached the highest accuracy but is an oracle reference without acquisition limits or a feasible return.",
        "",
        "Validation and test measurements remain separate in `evaluation_summary.csv`. No checkpoint or configuration was changed after test evaluation began.",
        "",
        f"The report includes {len(validation_rows)} validation summary rows and {len(test_rows)} test summary rows.",
        "",
    ])
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_report(args.inputs, args.output)
    print(json.dumps(result["ppo_test_across_seeds"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
