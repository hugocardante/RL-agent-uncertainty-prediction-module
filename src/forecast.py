# NOTE: This class is central to the whole process
# Classes that inherit from it must implement the perform forecast_forecast method,
# so that it can be called during the pipeline

import datetime
from abc import ABC, abstractmethod
from typing import Any

from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation

from utils.data_structures import SimpleObservation


class Forecaster(ABC):
    """
    Abstract base class for forecasters

    All forecasters must implement the perform_forecast method
    """

    def __init__(self, model: Any, **kwargs):
        """
        Initialize the forecaster.

        Args:
            model: The prediction model (e.g. a model that comes from a pickle)
            **kwargs: If we want to pass some params in the future
        """
        self.model = model

    @abstractmethod
    def perform_forecast(
        self,
        obs: BaseObservation,
        observations_array: list[SimpleObservation],
        agent: BaseAgent,
        **kwargs,  # if we need to pass other arguments..
    ) -> tuple[
        dict[datetime.datetime, list[float]],
        set[datetime.datetime],
        dict[datetime.datetime, SimpleObservation],
    ]:
        """
        Performs the forecast of the config.HORIZON next timesteps,
        simulating the agent action to obtain forecasted rho values.

        Args:
            obs: Current Grid2Op observation
            observations_array: Array of past SimpleObservation objects
            agent: RL agent that can make actions on the grid

        Returns:
            Tuple with:
                - agent_forecast_dict: Mapping of datetime to forecasted rho values
                - forecast_timestamps: Set of forecast datetime timestamps
                - obs_dict: Mapping of datetime to SimpleObservation objects
        """
        pass
