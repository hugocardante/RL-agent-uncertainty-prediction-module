import importlib
import inspect
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import numpy.typing as npt

import config
from utils.cache import DataManager

if TYPE_CHECKING:
    from utils.data_structures import Trajectory


def get_enabled_models() -> list[str]:
    """
    Gets the list of conformal prediction models that are enabled in the config file

    Returns:
        list of model name strings marked as enabled in config.MODELS
    """
    return [name for name, model_config in config.MODELS.items() if model_config[0]]


class ModelWrapper:
    """
    Wrapper for conformal prediction model(s).

    Two training modes:
    - Ensemble mode: Trains N models (one per transmission line in which one single line is subjected to adversarial attacks)
    - Single mode: Trains a single model (can have attacks or not depending on config.CALIBRATION_LINES_ATTACKED)
    """

    def __init__(
        self,
        model_name: str,
        n_lines: int,
        alpha: float,
        data_manager: DataManager,
    ):
        self.model_name: str = model_name
        self.n_lines: int = n_lines
        self.alpha: float = alpha
        self.data_manager: DataManager = data_manager
        self.is_ensemble: bool = config.ENSEMBLE_MODE

        # In single mode we store one model, and in ensemble multiple models (one per line)
        self.single_model: Any = None  # For single mode
        self.line_models: dict[int, Any] = {}  # For ensemble mode

        self._module_name: str = config.MODELS[model_name][1]
        self._compute_fn: Callable[..., Any]
        self._compute_sig: inspect.Signature
        self._compute_fn, self._compute_sig = self._get_compute_function()

    def _get_compute_function(self) -> tuple[Callable[..., Any], inspect.Signature]:
        """
        Gets the compute function and signature for this model

        Returns:
            Tuple of (compute_function, function signature)
        """
        module = importlib.import_module("conformalized_models." + self._module_name)
        compute_model_fn = getattr(module, "compute_model")
        sig = inspect.signature(compute_model_fn)
        return compute_model_fn, sig

    def fit(self) -> None:
        """
        Small wrapper to call ensemble / single fit function
        """
        if self.is_ensemble:
            self._fit_ensemble_mode()
        else:
            self._fit_single_mode()

    def _fit_single_mode(self) -> None:
        """
        Trains a single model in which some of the lines (possibly none)
        were subjected to attack during calibration (see config.CALIBRATION_LINES_ATTACKED)
        """
        print(f"Training {self.model_name} in single mode")

        attacked_lines = self.data_manager.get_attacked_lines()

        calib_data = self._load_calibration_data(attacked_lines)
        if calib_data is None:
            return

        features_array, forecast_array, actual_array = calib_data
        total_samples = len(actual_array)

        model_kwargs = self._prepare_model_kwargs(total_samples)

        prepared_data = (features_array, forecast_array, actual_array)
        model = self._compute_fn(
            prepared_data, self.n_lines, self.alpha, **model_kwargs
        )

        self.single_model = model

        del features_array, forecast_array, actual_array, calib_data

    def _fit_ensemble_mode(self) -> None:
        """
        Trains N models, in which N is the number of power lines
        During calibration, data from N scenarios is collected, in which
        a single line is attacked in each scenario
        """
        attacked_lines = self.data_manager.get_attacked_lines()
        print(
            f"Training {self.model_name} in ensemble mode ({len(attacked_lines)} lines)"
        )

        # we iterate through all lines, because all lines are attacked
        # individually in ensemble mode
        for line_number in attacked_lines:
            print(f"  fitting the ensemble for line: {line_number}")

            calib_data = self._load_calibration_data([line_number])
            if calib_data is None:
                continue

            features_array, forecast_array, actual_array = calib_data
            total_samples = len(actual_array)

            model_kwargs = self._prepare_model_kwargs(total_samples)

            model = self._compute_fn(
                calib_data, self.n_lines, self.alpha, **model_kwargs
            )

            self.line_models[line_number] = model

            del features_array, forecast_array, actual_array, calib_data

        print("Ensemble training complete")

    def _load_calibration_data(
        self, attacked_lines: list[int]
    ) -> (
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
        | None
    ):
        """
        Load calibration data for a given set of attacked lines.

        Args:
            attacked_lines: List of integer line numbers that were attacked during calibration

        Returns:
            A tuple (features_array, forecast_array, actual_array) or None if no data
        """

        from training import prepare_calibration_data

        # TrajectoryManagers for calibration
        calib_managers = self.data_manager.load_all_episodes(attacked_lines)

        if not calib_managers:
            print(f"No calibration data for attacked lines {attacked_lines}")
            return None

        features_array, forecast_array, actual_array = prepare_calibration_data(
            calib_managers, self.n_lines, self.model_name
        )

        if len(actual_array) == 0:
            print("No valid samples (after calling prepare_calibration_data)")
            return None

        # This has to be true
        assert len(actual_array) == len(forecast_array) == features_array.shape[0]

        return features_array, forecast_array, actual_array

    def _prepare_model_kwargs(self, total_samples: int) -> dict[str, int]:
        """
        Prepare model-specific hyperparameters

        Args:
            total_samples: Integer number of total samples during calibraiton

        Returns:
            Dictionary mapping {str -> int} (can be changed if we need another type of parameter)
                    for now only using it for "k" (kNN) and "gamma" for ACI models
        """
        model_kwargs = {}

        # K for the kNN models
        if "k" in self._compute_sig.parameters:
            k_override = getattr(config, "KNN_NEIGHBORS_OVERRIDE", None)
            model_kwargs["k"] = (
                k_override
                if k_override is not None
                else int(config.KNN_PERCENTAGE * total_samples)
            )

        # gamma for the models that use the ACI update rule
        if "gamma" in self._compute_sig.parameters:
            model_kwargs["gamma"] = config.GAMMA

        return model_kwargs

    def predict(
        self, trajectory: "Trajectory"
    ) -> tuple[list[list[float]], list[list[float]]]:
        """
        Generates predictions for a trajectory

        - Single mode: Uses the one trained model
        - Ensemble mode: Uses worst-case bounds across all line models

        Args:
            trajectory: A Trajectory object for which we provide the forecasted intervals

        Returns:
            A tuple (lower_bounds, upper_bounds) in which upper and lower are
            lists of lists of floats, the forecasted intervals for
            each forecast horizon (multiple lines)
        """
        if self.is_ensemble:
            return self._predict_ensemble(trajectory)
        else:
            return self._predict_single(trajectory)

    def _predict_single(
        self, trajectory: "Trajectory"
    ) -> tuple[list[list[float]], list[list[float]]]:
        """
        Uses the single trained model for prediction

        Args:
            trajectory: Trajectory for which we wish to predict

        Returns:
            Tuple of (lower_bounds, upper_bounds), each a list of up to 12 predictions
        """
        if self.single_model is None:
            raise ValueError(f"Single model not trained for {self.model_name}")

        return self.single_model.predict(trajectory)

    def _predict_ensemble(
        self, trajectory: "Trajectory"
    ) -> tuple[list[list[float]], list[list[float]]]:
        """
        Uses ensemble of line models and takes worst-case bounds
        by taking the worst lower bound and the worst upper bound
        from all the models in the ensemble

        Args:
            trajectory: Trajectory for which we wish to predict

        Returns:
            Tuple of (lower_bounds, upper_bounds), the worst case scenario from all the models for each of the horizons
        """
        all_lower: list[list[float]] = []
        all_upper = []

        for model in self.line_models.values():
            lower, upper = model.predict(trajectory)
            all_lower.append(lower)
            all_upper.append(upper)

        if not all_lower:
            raise ValueError(f"No models available in ensemble for {self.model_name}")

        lower_array = np.array(all_lower)
        upper_array = np.array(all_upper)

        # NOTE: For the Ensemble strategy, we take the worst case scenario (lowest lower bound, highest upper bound)
        # it would be interesting if in the future, we do something more involved, or we explore another metric
        # such as the median..
        worst_lower = np.min(lower_array, axis=0)
        worst_upper = np.max(upper_array, axis=0)

        return worst_lower.tolist(), worst_upper.tolist()

    def update_alphas(
        self,
        actual_values: list[float],
        predicted_lower: list[float],
        predicted_upper: list[float],
    ) -> None:
        """
        Updates significance levels for adaptive conformal prediction models
        Only applicable to models that implement the update_alphas method (e.g., ACI-based models).

        Args:
            actual_values: List of actual observed rho values
            predicted_lower: List of lower bound predictions
            predicted_upper: List of upper bound predictions
        """
        if self.is_ensemble:
            self._update_alphas_ensemble(
                actual_values, predicted_lower, predicted_upper
            )
        else:
            self._update_alphas_single(actual_values, predicted_lower, predicted_upper)

    def _update_alphas_single(
        self,
        actual_values: list[float],
        predicted_lower: list[float],
        predicted_upper: list[float],
    ) -> None:
        """
        Updates alphas for single model if it supports adaptive updates (ACI update rule)

        Args:
            actual_values: List of actual observed rho values
            predicted_lower: List of lower bound predictions
            predicted_upper: List of upper bound predictions
        """
        if self.single_model is None:
            return

        if hasattr(self.single_model, "update_alphas"):
            self.single_model.update_alphas(
                actual_values, predicted_lower, predicted_upper
            )

    def _update_alphas_ensemble(
        self,
        actual_values: list[float],
        predicted_lower: list[float],
        predicted_upper: list[float],
    ):
        """
        Update alphas for all ensemble models that support adaptive updates (ACI update rule)

        Args:
            actual_values: List of actual observed rho values
            predicted_lower: List of lower bound predictions
            predicted_upper: List of upper bound predictions
        """
        for model in self.line_models.values():
            if hasattr(model, "update_alphas"):
                model.update_alphas(actual_values, predicted_lower, predicted_upper)
