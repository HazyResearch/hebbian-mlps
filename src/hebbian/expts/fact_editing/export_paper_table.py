"""Export fact-editing paper tables from a best-results CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence


METHOD_ORDER = ["gd_construction", "alpha_edit", "memit", "rome"]
METHOD_DISPLAY = {
    "gd_construction": r"\textsc{MLP Swapping}",
    "alpha_edit": "AlphaEdit",
    "memit": "MEMIT",
    "rome": "ROME",
}
DEFAULT_CAPTION = (
    "Updated fact-editing results on the h512 normalized-token MSE BinaryMoE "
    "base. \\(r_{\\mathrm{NF}}\\) is the non-fact-token perplexity ratio, "
    "equivalently \\(\\exp(\\Delta\\mathrm{CE}_{\\mathrm{NF}})\\)."
)
DEFAULT_LABEL = "tab:fact-editing-updated-h512"


def _to_float(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing required column {key!r} in row: {row}")
    return float(value)


def _method_rank(method: str) -> tuple[int, str]:
    if method in METHOD_ORDER:
        return (METHOD_ORDER.index(method), method)
    return (len(METHOD_ORDER), method)


def _format_percent(value: float) -> str:
    return rf"\({value:.2f}\%\)"


def _format_metric(value: float) -> str:
    return f"{value:.3f}"


def load_best_rows(input_csv: Path) -> list[dict[str, str]]:
    with open(input_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {input_csv}")
    return rows


def select_rows(rows: Sequence[dict[str, str]], methods: set[str] | None = None) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        method = row.get("method") or row.get("type")
        if method is None:
            raise ValueError(f"Row is missing method/type: {row}")
        if methods is not None and method not in methods:
            continue
        selected.append(row)
    if not selected:
        raise ValueError("No rows matched the requested methods.")
    return sorted(
        selected,
        key=lambda row: (
            _method_rank(str(row.get("method") or row.get("type"))),
            _to_float(row, "percent_edited"),
        ),
    )


def render_table(
    rows: Sequence[dict[str, str]],
    *,
    caption: str = DEFAULT_CAPTION,
    label: str = DEFAULT_LABEL,
    table_env: bool = True,
) -> str:
    lines: list[str] = []
    if table_env:
        lines.extend(
            [
                r"\begin{table}[h]",
                r"\centering",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{4pt}",
            ]
        )
    lines.extend(
        [
            r"\begin{tabular}{llccccc}",
            r"\toprule",
            r"Method & Edited facts & Efficacy & Paraphrase & Specificity & Score & \(r_{\mathrm{NF}}\) \\",
            r"\midrule",
        ]
    )
    for row in rows:
        method = str(row.get("method") or row.get("type"))
        display = METHOD_DISPLAY.get(method, method)
        fields = [
            display,
            _format_percent(_to_float(row, "percent_edited")),
            _format_metric(_to_float(row, "efficacy")),
            _format_metric(_to_float(row, "paraphrase")),
            _format_metric(_to_float(row, "specificity")),
            _format_metric(_to_float(row, "score")),
            _format_metric(_to_float(row, "non_fact_ppl_ratio")),
        ]
        lines.append(" & ".join(fields) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if table_env:
        lines.extend([rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", help="Path to a fact-editing best-results CSV.")
    parser.add_argument("--output-tex", required=True, help="Where to write the LaTeX table.")
    parser.add_argument("--methods", default="gd_construction,alpha_edit,memit,rome")
    parser.add_argument("--caption", default=DEFAULT_CAPTION)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--fragment", action="store_true", help="Write only the tabular fragment, not a table environment.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_csv = Path(args.input_csv)
    output_tex = Path(args.output_tex)
    methods = (
        {method.strip() for method in args.methods.split(",") if method.strip()}
        if args.methods
        else None
    )
    rows = select_rows(load_best_rows(input_csv), methods=methods)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(
        render_table(rows, caption=args.caption, label=args.label, table_env=not args.fragment),
        encoding="utf-8",
    )
    print(output_tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
