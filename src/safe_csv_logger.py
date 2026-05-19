import csv
import os
from typing import Any, Optional

from lightning_fabric.loggers.csv_logs import _ExperimentWriter, rank_zero_experiment
from pytorch_lightning.loggers import CSVLogger as PLCSVLogger


class _SafeExperimentWriter(_ExperimentWriter):
    """CSV writer that tolerates metrics appearing in different phases.

    Lightning's default CSV writer can fail when train/val/test log different
    key sets before the next flush. We keep the same on-disk format but ensure
    the header always covers the union of existing and pending metric keys.
    """

    @staticmethod
    def _sanitize_row(row: dict[Any, Any]) -> dict[str, Any]:
        clean = {}
        for key, value in row.items():
            if key is None:
                continue
            clean[str(key)] = value
        return clean

    def log_hparams(self, params: dict[Any, Any]) -> None:
        params = self._sanitize_row(params)
        hparams_path = os.path.join(self.log_dir, "hparams.yaml")
        with self._fs.open(hparams_path, "w") as file:
            for key in sorted(params.keys()):
                file.write(f"{key}: {params[key]}\n")

    def save(self) -> None:
        if not self.metrics:
            return

        pending_metrics = [self._sanitize_row(metric) for metric in self.metrics]
        pending_keys = set().union(*(metric.keys() for metric in pending_metrics)) if pending_metrics else set()
        known_keys = set(self.metrics_keys)
        file_exists = self._fs.isfile(self.metrics_file_path)

        existing_metrics = []
        existing_header = []
        if file_exists:
            with self._fs.open(self.metrics_file_path, "r", newline="") as file:
                reader = csv.DictReader(file)
                existing_header = list(reader.fieldnames or [])
                existing_metrics = [self._sanitize_row(row) for row in reader]

        all_keys = sorted(set(existing_header) | known_keys | pending_keys)
        self.metrics_keys = all_keys

        rewrite_required = file_exists and set(existing_header) != set(all_keys)
        if rewrite_required:
            with self._fs.open(self.metrics_file_path, "w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=all_keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(existing_metrics)
            append_mode = "a"
        else:
            append_mode = "a" if file_exists else "w"

        with self._fs.open(self.metrics_file_path, mode=append_mode, newline="") as file:
            writer = csv.DictWriter(file, fieldnames=all_keys, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerows(pending_metrics)

        self.metrics = []


class SafeCSVLogger(PLCSVLogger):
    @property
    @rank_zero_experiment
    def experiment(self) -> _SafeExperimentWriter:
        if self._experiment is not None:
            return self._experiment

        os.makedirs(self._root_dir, exist_ok=True)
        self._experiment = _SafeExperimentWriter(log_dir=self.log_dir)
        return self._experiment
