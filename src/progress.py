import logging
import math
import time
from typing import Optional

from pytorch_lightning.callbacks import Callback


def format_duration(total_seconds: Optional[float]) -> str:
    if total_seconds is None:
        return "unknown"
    if not math.isfinite(total_seconds) or total_seconds < 0:
        return "unknown"

    rounded = int(total_seconds + 0.5)
    hours, rem = divmod(rounded, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class DetailedProgressLogger(Callback):
    def __init__(
        self,
        train_log_every_n_steps: int = 20,
        val_log_every_n_batches: int = 1,
        test_log_every_n_batches: int = 1,
    ):
        super().__init__()
        self.train_log_every_n_steps = max(int(train_log_every_n_steps), 0)
        self.val_log_every_n_batches = max(int(val_log_every_n_batches), 0)
        self.test_log_every_n_batches = max(int(test_log_every_n_batches), 0)
        self._epoch_start_time: Optional[float] = None
        self._val_epoch_start_time: Optional[float] = None
        self._test_epoch_start_time: Optional[float] = None
        self._val_total_batches: Optional[int] = None
        self._test_total_batches: Optional[int] = None
        self._completed_epoch_seconds = []

    @staticmethod
    def _safe_total_batches(num_batches) -> Optional[int]:
        if isinstance(num_batches, (list, tuple)):
            total = 0
            for item in num_batches:
                parsed = DetailedProgressLogger._safe_total_batches(item)
                if parsed is not None:
                    total += parsed
            return total if total > 0 else None

        try:
            value = float(num_batches)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(value) or value <= 0:
            return None
        return int(value)

    @staticmethod
    def _should_log(current_batch: int, total_batches: Optional[int], every_n: int) -> bool:
        if every_n <= 0:
            return total_batches is not None and current_batch >= total_batches
        return (
            current_batch == 1
            or current_batch % every_n == 0
            or (total_batches is not None and current_batch >= total_batches)
        )

    @staticmethod
    def _estimate_eta(elapsed: float, current_batch: int, total_batches: Optional[int]) -> tuple[Optional[float], Optional[float]]:
        if total_batches is None or current_batch <= 0:
            return None, None

        avg_batch_seconds = elapsed / current_batch
        remaining_batches = max(total_batches - current_batch, 0)
        return avg_batch_seconds, remaining_batches * avg_batch_seconds

    def on_fit_start(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return

        total_train_batches = self._safe_total_batches(trainer.num_training_batches)
        logging.info(
            "[Progress] Training started: epochs=%s, train_batches_per_epoch=%s",
            trainer.max_epochs,
            total_train_batches if total_train_batches is not None else "unknown",
        )

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return

        self._epoch_start_time = time.time()
        logging.info(
            "[Progress][Epoch] %d/%d started",
            trainer.current_epoch + 1,
            trainer.max_epochs,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not trainer.is_global_zero or self._epoch_start_time is None:
            return

        total_batches = self._safe_total_batches(trainer.num_training_batches)
        current_batch = batch_idx + 1
        if not self._should_log(current_batch, total_batches, self.train_log_every_n_steps):
            return

        elapsed = time.time() - self._epoch_start_time
        avg_batch_seconds, epoch_eta = self._estimate_eta(elapsed, current_batch, total_batches)

        avg_epoch_seconds = None
        if self._completed_epoch_seconds:
            avg_epoch_seconds = sum(self._completed_epoch_seconds) / len(self._completed_epoch_seconds)

        epoch_seconds_estimate = None
        if total_batches is not None and avg_batch_seconds is not None:
            epoch_seconds_estimate = total_batches * avg_batch_seconds

        if epoch_seconds_estimate is not None:
            if avg_epoch_seconds is None:
                avg_epoch_seconds = epoch_seconds_estimate
            else:
                avg_epoch_seconds = (
                    avg_epoch_seconds * len(self._completed_epoch_seconds) + epoch_seconds_estimate
                ) / (len(self._completed_epoch_seconds) + 1)

        total_eta = None
        if epoch_eta is not None and avg_epoch_seconds is not None:
            remaining_full_epochs = max(trainer.max_epochs - trainer.current_epoch - 1, 0)
            total_eta = epoch_eta + remaining_full_epochs * avg_epoch_seconds

        logging.info(
            "[Progress][Train] epoch %d/%d batch %d/%s elapsed=%s avg_batch=%ss eta_epoch=%s eta_total=%s",
            trainer.current_epoch + 1,
            trainer.max_epochs,
            current_batch,
            total_batches if total_batches is not None else "?",
            format_duration(elapsed),
            f"{avg_batch_seconds:.2f}" if avg_batch_seconds is not None else "unknown",
            format_duration(epoch_eta),
            format_duration(total_eta),
        )

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero or self._epoch_start_time is None:
            return

        epoch_seconds = time.time() - self._epoch_start_time
        self._completed_epoch_seconds.append(epoch_seconds)

        avg_epoch_seconds = sum(self._completed_epoch_seconds) / len(self._completed_epoch_seconds)
        remaining_epochs = max(trainer.max_epochs - trainer.current_epoch - 1, 0)
        total_eta = remaining_epochs * avg_epoch_seconds

        logging.info(
            "[Progress][Epoch] %d/%d finished in %s, est_remaining=%s",
            trainer.current_epoch + 1,
            trainer.max_epochs,
            format_duration(epoch_seconds),
            format_duration(total_eta),
        )

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return

        self._val_epoch_start_time = time.time()
        self._val_total_batches = self._safe_total_batches(trainer.num_val_batches)

        sampling_flag = ""
        should_sample_fn = getattr(pl_module, "should_sample_validation_epoch", None)
        if callable(should_sample_fn):
            try:
                if should_sample_fn():
                    max_batches = int(getattr(pl_module.cfg.general, "val_sampling_max_batches", 0))
                    if max_batches > 0:
                        sampling_flag = f", sampling_validation=on(max_batches={max_batches})"
                    else:
                        sampling_flag = ", sampling_validation=on(all_batches)"
            except Exception:
                sampling_flag = ""

        logging.info(
            "[Progress][Val] epoch %d/%d started, batches=%s%s",
            trainer.current_epoch + 1,
            trainer.max_epochs,
            self._val_total_batches if self._val_total_batches is not None else "unknown",
            sampling_flag,
        )

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0) -> None:
        if not trainer.is_global_zero or self.val_log_every_n_batches <= 0 or self._val_epoch_start_time is None:
            return

        current_batch = batch_idx + 1
        total_batches = self._val_total_batches
        if not self._should_log(current_batch, total_batches, self.val_log_every_n_batches):
            return

        elapsed = time.time() - self._val_epoch_start_time
        avg_batch_seconds, eta = self._estimate_eta(elapsed, current_batch, total_batches)

        logging.info(
            "[Progress][Val] epoch %d/%d batch %d/%s elapsed=%s avg_batch=%ss eta_val=%s",
            trainer.current_epoch + 1,
            trainer.max_epochs,
            current_batch,
            total_batches if total_batches is not None else "?",
            format_duration(elapsed),
            f"{avg_batch_seconds:.2f}" if avg_batch_seconds is not None else "unknown",
            format_duration(eta),
        )

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero or self._val_epoch_start_time is None:
            return

        val_seconds = time.time() - self._val_epoch_start_time
        logging.info(
            "[Progress][Val] epoch %d/%d finished in %s",
            trainer.current_epoch + 1,
            trainer.max_epochs,
            format_duration(val_seconds),
        )

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return

        self._test_epoch_start_time = time.time()
        self._test_total_batches = self._safe_total_batches(trainer.num_test_batches)
        logging.info(
            "[Progress][Test] started, batches=%s",
            self._test_total_batches if self._test_total_batches is not None else "unknown",
        )

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        if not trainer.is_global_zero or self.test_log_every_n_batches <= 0 or self._test_epoch_start_time is None:
            return

        current_batch = batch_idx + 1
        total_batches = self._test_total_batches
        if not self._should_log(current_batch, total_batches, self.test_log_every_n_batches):
            return

        elapsed = time.time() - self._test_epoch_start_time
        avg_batch_seconds, eta = self._estimate_eta(elapsed, current_batch, total_batches)

        logging.info(
            "[Progress][Test] batch %d/%s elapsed=%s avg_batch=%ss eta_test=%s",
            current_batch,
            total_batches if total_batches is not None else "?",
            format_duration(elapsed),
            f"{avg_batch_seconds:.2f}" if avg_batch_seconds is not None else "unknown",
            format_duration(eta),
        )

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero or self._test_epoch_start_time is None:
            return

        test_seconds = time.time() - self._test_epoch_start_time
        logging.info("[Progress][Test] finished in %s", format_duration(test_seconds))
