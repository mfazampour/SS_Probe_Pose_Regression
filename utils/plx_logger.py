from polyaxon_client.tracking import Experiment
from pytorch_lightning.loggers.base import LightningLoggerBase
#from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities import rank_zero_only
from typing import Dict, Optional, Union
import argparse
import numbers
from argparse import Namespace
from pathlib import Path


class PolyaxonLogger(LightningLoggerBase):
    """Docstring for PolyaxonLogger. """

    def __init__(self, hparams):
        """TODO: to be defined. """
        super().__init__()
        self.hparams = hparams
        self._experiment = Experiment()
        self.output_path = Path(self.experiment.get_outputs_path())


    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Record metrics.
        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
                  Polyaxon currently does not support assigning a specific step.
        """

        self.experiment.log_metrics(step=step, **metrics)


    @property
    def experiment(self):
        """Return the experiment object associated with this logger"""
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: argparse.Namespace):
        """Record hyperparameters.

        Args:
            params: argparse.Namespace containing the hyperparameters
        """
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        params = self._sanitize_params(params)
        self.experiment.log_params(**params)

    @property
    def name(self) -> str:
        """Return the experiment name."""
        if self._experiment.get_experiment_info() is not None:
            return self._experiment.get_experiment_info()['project_name']

    @property
    def version(self) -> Union[int, str]:
        """Return the experiment version."""
        if self._experiment.get_experiment_info() is not None:
            return self._experiment.experiment_id
