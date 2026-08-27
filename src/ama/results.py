"""Result schemas and CSV writers.

Two levels are recorded for every policy: one summary row per method, and one
row per test example. The per-example rows are what support paired
comparisons between methods, since every method sees the same examples with
the same sampled costs.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

SUMMARY_FIELDS = [
    "value_fn",
    "cost_setting",
    "name",
    "loss",
    "cost",
    "accuracy",
    "skipped",
    "reward",
    "efficiency",
    "auc",
    "seconds",
]


@dataclass
class SampleResult:
    """One test example under one acquisition method."""

    name: str
    sample_idx: int
    label: int
    pred: int
    pred_proba: float
    loss: float
    correct: float
    baseline_correct: float
    cost: float
    reward: float
    n_acquired: int
    skipped: float
    selected_modes: str
    num_stages: int | None


SAMPLE_FIELDS = ["value_fn", "cost_setting"] + [f.name for f in fields(SampleResult)]


class ResultWriter:
    """Writes summary and per-sample CSVs for one (value_fn, cost setting) pair.

    Rows are flushed as they are written so a long sweep can be inspected, or
    salvaged, while it is still running.
    """

    def __init__(self, summary_path, sample_path, value_fn: str, cost_setting: str):
        self.value_fn = value_fn
        self.cost_setting = cost_setting
        self.summary_path = Path(summary_path)
        self.sample_path = Path(sample_path)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_path.parent.mkdir(parents=True, exist_ok=True)
        self._files = []

    def __enter__(self) -> "ResultWriter":
        self._summary_file = open(self.summary_path, "w", newline="", buffering=1)
        self._sample_file = open(self.sample_path, "w", newline="", buffering=1)
        self._files = [self._summary_file, self._sample_file]
        self._summary = csv.DictWriter(self._summary_file, fieldnames=SUMMARY_FIELDS)
        self._samples = csv.DictWriter(self._sample_file, fieldnames=SAMPLE_FIELDS)
        self._summary.writeheader()
        self._samples.writeheader()
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._files:
            handle.close()

    def _tagged(self, row: dict) -> dict:
        return {"value_fn": self.value_fn, "cost_setting": self.cost_setting, **row}

    def write_summary(self, name: str, summary: dict) -> None:
        row = self._tagged({"name": name, **summary})
        self._summary.writerow({k: row.get(k) for k in SUMMARY_FIELDS})
        self._summary_file.flush()

    def write_samples(self, rows) -> None:
        for row in rows:
            self._samples.writerow(
                self._tagged(asdict(row) if isinstance(row, SampleResult) else row)
            )
        self._sample_file.flush()
