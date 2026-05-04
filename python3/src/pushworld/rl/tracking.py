# Copyright 2026
#
# Optional experiment tracking integrations.

from dataclasses import dataclass
import os
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class CometTrackingConfig:
    enabled: bool = False
    project_name: str = "pushworld-ppo"
    workspace: Optional[str] = None
    experiment_name: Optional[str] = None
    tags: tuple = ()
    log_artifacts: bool = True


class NoOpTracker:
    """Tracker interface used when Comet logging is disabled."""

    enabled = False

    def log_parameters(self, parameters: Dict[str, object]) -> None:
        del parameters

    def log_metrics(
        self,
        metrics: Dict[str, object],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:
        del metrics, step, prefix

    def log_asset(self, path: str, name: Optional[str] = None) -> None:
        del path, name

    def end(self) -> None:
        pass


class CometTracker:
    """Small wrapper around Comet's Python SDK."""

    enabled = True

    def __init__(self, experiment) -> None:
        self._experiment = experiment

    def log_parameters(self, parameters: Dict[str, object]) -> None:
        self._experiment.log_parameters(parameters)

    def log_metrics(
        self,
        metrics: Dict[str, object],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:
        numeric_metrics = _numeric_metrics(metrics)
        if numeric_metrics:
            self._experiment.log_metrics(numeric_metrics, step=step, prefix=prefix)

    def log_asset(self, path: str, name: Optional[str] = None) -> None:
        if not os.path.exists(path):
            return
        if hasattr(self._experiment, "log_asset"):
            try:
                kwargs = {"file_data": path}
                if name is not None:
                    kwargs["file_name"] = name
                self._experiment.log_asset(**kwargs)
            except TypeError:
                if name is None:
                    self._experiment.log_asset(path)
                else:
                    self._experiment.log_asset(path, file_name=name)

    def end(self) -> None:
        self._experiment.end()


def create_comet_tracker(config: CometTrackingConfig):
    """Creates a Comet tracker or a no-op tracker when disabled."""
    if not config.enabled:
        return NoOpTracker()

    try:
        import comet_ml
    except ImportError as exc:
        raise ImportError(
            "Comet tracking is enabled but `comet_ml` is not installed. "
            "Install it with `pip install -r requirements_rl.txt`."
        ) from exc

    start_kwargs = {"project_name": config.project_name}
    if config.workspace:
        start_kwargs["workspace"] = config.workspace
    experiment = comet_ml.start(**start_kwargs)

    if config.experiment_name and hasattr(experiment, "set_name"):
        experiment.set_name(config.experiment_name)

    tags = [tag for tag in config.tags if tag]
    if tags:
        if hasattr(experiment, "add_tags"):
            experiment.add_tags(tags)
        elif hasattr(experiment, "add_tag"):
            for tag in tags:
                experiment.add_tag(tag)

    return CometTracker(experiment)


def parse_tags(value: str) -> tuple:
    if not value:
        return ()
    return tuple(tag.strip() for tag in value.split(",") if tag.strip())


def _numeric_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    numeric = {}
    for key, value in metrics.items():
        if value == "" or value is None:
            continue
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        if isinstance(value, bool):
            value = float(value)
        if isinstance(value, (int, float)) and np.isfinite(value):
            numeric[key] = float(value)
    return numeric
