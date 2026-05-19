from dataclasses import dataclass
from typing import Iterable, Tuple

import torch

from src import utils


_METHOD_ALIASES = {
    "ancestral": "ancestral",
    "exact": "exact_long_range",
    "exact_long_range": "exact_long_range",
    "elrt": "exact_long_range",
}

_SCHEDULE_ALIASES = {
    "full": "full",
    "stride": "stride",
    "num_steps": "num_steps",
    "custom": "custom",
    "custom_timesteps": "custom",
}


@dataclass(frozen=True)
class InferenceSamplingPlan:
    method: str
    schedule: str
    visited_steps: Tuple[int, ...]
    step_pairs: Tuple[Tuple[int, int], ...]

    @property
    def num_transitions(self) -> int:
        return len(self.step_pairs)

    @property
    def max_jump(self) -> int:
        if not self.step_pairs:
            return 0
        return max(t_int - s_int for s_int, t_int in self.step_pairs)

    @property
    def uses_exact_long_range(self) -> bool:
        return self.method == "exact_long_range" and self.max_jump > 1

    def describe(self) -> str:
        return (
            f"method={self.method}, schedule={self.schedule}, "
            f"transitions={self.num_transitions}, max_jump={self.max_jump}, "
            f"path={list(self.visited_steps)}"
        )


def build_inference_sampling_plan(general_cfg, total_steps: int) -> InferenceSamplingPlan:
    method = _normalize_method(getattr(general_cfg, "inference_sampling_method", "ancestral"))
    schedule = _normalize_schedule(getattr(general_cfg, "inference_sampling_schedule", "full"))

    if schedule == "full":
        visited_steps = tuple(range(int(total_steps), -1, -1))
    elif schedule == "stride":
        step_stride = int(getattr(general_cfg, "inference_sampling_step_stride", 1))
        if step_stride <= 0:
            raise ValueError("general.inference_sampling_step_stride must be positive")
        raw_steps = list(range(int(total_steps), -1, -step_stride))
        if raw_steps[-1] != 0:
            raw_steps.append(0)
        visited_steps = _normalize_visited_steps(raw_steps, total_steps)
    elif schedule == "num_steps":
        requested_steps = getattr(general_cfg, "inference_sampling_num_steps", None)
        if requested_steps is None:
            raise ValueError(
                "general.inference_sampling_num_steps must be set when "
                "general.inference_sampling_schedule='num_steps'"
            )
        requested_steps = int(requested_steps)
        if requested_steps <= 0:
            raise ValueError("general.inference_sampling_num_steps must be positive")
        if requested_steps >= int(total_steps):
            visited_steps = tuple(range(int(total_steps), -1, -1))
        else:
            raw_steps = [
                int(round(float(total_steps) - (float(idx) * float(total_steps) / float(requested_steps))))
                for idx in range(requested_steps + 1)
            ]
            visited_steps = _normalize_visited_steps(raw_steps, total_steps)
    else:
        custom_timesteps = getattr(general_cfg, "inference_sampling_custom_timesteps", None)
        if custom_timesteps is None:
            raise ValueError(
                "general.inference_sampling_custom_timesteps must be set when "
                "general.inference_sampling_schedule='custom'"
            )
        visited_steps = _normalize_visited_steps(custom_timesteps, total_steps)

    step_pairs = tuple(
        (visited_steps[idx + 1], visited_steps[idx])
        for idx in range(len(visited_steps) - 1)
    )

    if method == "ancestral" and any((t_int - s_int) != 1 for s_int, t_int in step_pairs):
        raise ValueError(
            "Ancestral sampling only supports consecutive reverse steps. "
            "Set general.inference_sampling_method='exact_long_range' to enable jump sampling."
        )

    return InferenceSamplingPlan(
        method=method,
        schedule=schedule,
        visited_steps=visited_steps,
        step_pairs=step_pairs,
    )


def get_sampling_transition_cache_entry(cache, noise_schedule, transition_model, step_int, device, dtype):
    device_key = (device.type, device.index if device.index is not None else -1, str(dtype))
    device_cache = cache.setdefault(device_key, {})
    step_int = int(step_int)

    if step_int not in device_cache:
        t_int = torch.tensor([[step_int]], device=device, dtype=torch.long)
        alpha_bar_t = noise_schedule.get_alpha_bar(t_int=t_int).to(dtype=dtype)
        qt_bar = transition_model.get_Qt_bar(alpha_bar_t, device)
        device_cache[step_int] = {
            "alpha_bar": alpha_bar_t,
            "Qt_bar": utils.PlaceHolder(
                X=qt_bar.X.to(dtype=dtype),
                E=qt_bar.E.to(dtype=dtype),
                y=qt_bar.y.to(dtype=dtype),
            ),
        }

    return device_cache[step_int]


def expand_cached_transition(transition, batch_size: int):
    return utils.PlaceHolder(
        X=transition.X.expand(batch_size, -1, -1),
        E=transition.E.expand(batch_size, -1, -1),
        y=transition.y.expand(batch_size, -1, -1),
    )


def get_reverse_sampling_transitions(
    cache,
    noise_schedule,
    transition_model,
    s_int: int,
    t_int: int,
    device,
    dtype,
    batch_size: int,
):
    if t_int <= s_int:
        raise ValueError(f"Expected t_int > s_int, got s_int={s_int}, t_int={t_int}")

    s_entry = get_sampling_transition_cache_entry(cache, noise_schedule, transition_model, s_int, device, dtype)
    t_entry = get_sampling_transition_cache_entry(cache, noise_schedule, transition_model, t_int, device, dtype)

    alpha_s_bar = s_entry["alpha_bar"].expand(batch_size, -1)
    alpha_t_bar = t_entry["alpha_bar"].expand(batch_size, -1)
    alpha_t_given_s = alpha_t_bar / torch.clamp(alpha_s_bar, min=1e-12)
    alpha_t_given_s = torch.clamp(alpha_t_given_s, min=0.0, max=1.0)

    qt = transition_model.get_Qt_from_alpha(alpha_t_given_s, device)
    qt = utils.PlaceHolder(
        X=qt.X.to(dtype=dtype),
        E=qt.E.to(dtype=dtype),
        y=qt.y.to(dtype=dtype),
    )

    return {
        "alpha_s_bar": alpha_s_bar,
        "alpha_t_bar": alpha_t_bar,
        "Qt": qt,
        "Qsb": expand_cached_transition(s_entry["Qt_bar"], batch_size),
        "Qtb": expand_cached_transition(t_entry["Qt_bar"], batch_size),
    }


def _normalize_method(method) -> str:
    normalized = str(method).strip().lower()
    if normalized not in _METHOD_ALIASES:
        raise ValueError(
            "Unknown general.inference_sampling_method "
            f"'{method}'. Expected one of: {sorted(_METHOD_ALIASES)}"
        )
    return _METHOD_ALIASES[normalized]


def _normalize_schedule(schedule) -> str:
    normalized = str(schedule).strip().lower()
    if normalized not in _SCHEDULE_ALIASES:
        raise ValueError(
            "Unknown general.inference_sampling_schedule "
            f"'{schedule}'. Expected one of: {sorted(_SCHEDULE_ALIASES)}"
        )
    return _SCHEDULE_ALIASES[normalized]


def _normalize_visited_steps(raw_steps: Iterable[int], total_steps: int) -> Tuple[int, ...]:
    if raw_steps is None:
        raise ValueError("Sampling timesteps must not be None")

    normalized = set()
    for raw_step in raw_steps:
        step_int = int(raw_step)
        if step_int < 0 or step_int > int(total_steps):
            raise ValueError(
                f"Sampling timestep {step_int} is out of range for diffusion_steps={total_steps}"
            )
        normalized.add(step_int)

    normalized.add(int(total_steps))
    normalized.add(0)

    visited_steps = tuple(sorted(normalized, reverse=True))
    if len(visited_steps) < 2:
        raise ValueError("Sampling schedule must include at least two distinct steps")
    return visited_steps
