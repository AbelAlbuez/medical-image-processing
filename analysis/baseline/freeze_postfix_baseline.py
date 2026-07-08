from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "output" / "tablas" / "metricas_ET.csv"
BASELINE = ROOT / "analysis" / "baseline"
FIXTURE = BASELINE / "regression_fixture.json"


def main() -> None:
    df = pd.read_csv(METRICS)
    df.to_csv(BASELINE / "metricas_ET_baseline.csv", index=False)

    means = (
        df.groupby("metodo", as_index=False)["dice_ET"]
        .mean()
        .rename(columns={"dice_ET": "mean_dice"})
        .sort_values("mean_dice", ascending=False)
    )
    means.to_csv(BASELINE / "baseline_means.csv", index=False)

    old_fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cases = {}
    for case_id, old_case in old_fixture["cases"].items():
        sub = df[df["case_id"] == case_id].sort_values("metodo")
        cases[case_id] = {
            "type": old_case.get("type", ""),
            "dice_ET": {
                row.metodo: float(row.dice_ET)
                for row in sub.itertuples(index=False)
            },
        }

    fixture = {
        "tolerance": old_fixture.get("tolerance", 0.001),
        "cases": cases,
    }
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {BASELINE / 'metricas_ET_baseline.csv'}")
    print(f"wrote {BASELINE / 'baseline_means.csv'}")
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    main()
