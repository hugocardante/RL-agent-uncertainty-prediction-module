# TODO: Maybe implement an Ensemble mode for STL.
# One idea could be that each model of the ensemble predicts if it is
# safe or unsafe, and then we consider it unsafe if any mode considers it unsafe (worst case)
# Otherwise we could use something like the majority of models

import importlib
import inspect
from typing import TYPE_CHECKING, Any, Callable

import config
from utils.cache import DataManager, extract_line_number

if TYPE_CHECKING:
    from utils.data_structures import Trajectory


def get_enabled_stl_rules() -> list[str]:
    """
    Get the list of STL rules that are enabled in the config file

    Returns:
        List of rule name strings marked as enabled in config.STL_RULES
    """

    # TODO: When ensemble mode is implemented for STL, take this out!
    if config.ENSEMBLE_MODE:
        print(
            "[IMPORATNT] Ensemble mode is not implemented for STL, so making get_enabled_stl_rules return empty list..."
        )
        return []

    return [name for name, rule_config in config.STL_RULE.items() if rule_config[0]]


class STLWrapper:
    """
    Wrapper for STL (Signal Temporal Logic) rule models
    """

    def __init__(
        self,
        rule_name: str,
        n_lines: int,
        alpha: float,
        data_manager: DataManager,
    ):
        """
        Initialize STL verifier for a specific rule type.

        Args:
            rule_name: String name of the STL rule
            n_lines: Integer number of power lines in the grid
            alpha: Float significance level
            data_manager: DataManager object containing calibration data
        """
        self.rule_name: str = rule_name
        self.n_lines: int = n_lines
        self.alpha: float = alpha
        self.data_manager: DataManager = data_manager
        self.rule_model: Any = None  # This is the rule, like Vanilla, kNN, etc..
        self._module_name: str = config.STL_RULE[rule_name][1]
        self._compute_stl_fn: Callable[..., Any]
        self._compute_sig: inspect.Signature
        self._compute_stl_fn, self._compute_sig = self._get_compute_function()

    def _get_compute_function(self) -> tuple[Callable[..., Any], inspect.Signature]:
        """
        Gets the compute function and signature for this stl rule

        Returns:
            Tuple of (compute_function, function signature)
        """
        module = importlib.import_module("stl_rules." + self._module_name)
        compute_model_fn = getattr(module, "compute_stl")
        sig = inspect.signature(compute_model_fn)
        return compute_model_fn, sig

    def fit(self) -> None:
        """
        Train this STL rule using calibration data
        """
        if config.ENSEMBLE_MODE:
            self._fit_ensemble_mode()
        else:
            self._fit_single_mode()

    def _fit_single_mode(self) -> None:
        """
        Train a single STL rule using data from attacked lines
        """
        print(f"Training {self.rule_name}...")

        all_trajectories = self._load_calibration_trajectories()
        num_traj = len(all_trajectories)

        stl_kwargs = self._prepare_stl_kwargs(num_traj)

        if not all_trajectories:
            print(f"    {self.rule_name}: no trajectories available")
            return

        self.rule_model = self._compute_stl_fn(
            all_trajectories, self.n_lines, self.alpha, **stl_kwargs
        )

        if self.rule_model is not None:
            print(f"{self.rule_name}: trained successfully")
        else:
            print(f"{self.rule_name}: failed to train")

    def _fit_ensemble_mode(self) -> None:
        # TODO: This is not implemented yet. Check the note at the top
        # implement when I have time.
        # When this is implemented, erase the if in the TODO in the get_enabled_stl_rules() function
        return

    def _prepare_stl_kwargs(self, num_traj: int) -> dict[str, int]:
        """
        Prepare stl-specific hyperparameters

        Args:
            num_traj: Number of total trajectories

        Returns:
            Dictionary mapping {str -> int} (can be changed if we need another type of parameter)
                    for now only using it for "k" (kNN rule)
        """
        model_kwargs: dict[str, int] = {}

        # K for the kNN models
        if "k" in self._compute_sig.parameters:
            k_override = getattr(config, "KNN_NEIGHBORS_OVERRIDE", None)
            model_kwargs["k"] = (
                k_override
                if k_override is not None
                else int(config.KNN_PERCENTAGE * num_traj)
            )

        return model_kwargs

    def _load_calibration_trajectories(self) -> list["Trajectory"]:
        """
        Load and prepare calibration trajectories for the attacked lines
        It only makes sense to load complete trajectories, so that we can know
        if they are safe or not

        Returns:
            List of Trajectory objects from all calibration episodes
        """
        # uses the same calibration data as the onformal models
        attacked_lines = [
            extract_line_number(line) for line in config.CALIBRATION_LINES_ATTACKED
        ]
        calib_episodes = self.data_manager.load_all_episodes(attacked_lines)

        if not calib_episodes:
            return []

        all_completed_traj: list[Trajectory] = []
        for tm in calib_episodes:
            all_completed_traj.extend(tm.get_completed_trajectories())

        return all_completed_traj

    def predict(self, trajectory: "Trajectory") -> dict[int, bool] | None:
        """
        Verify a trajectory against this STL rule for all lines

        Args:
            trajectory: Trajectory object to verify

        Returns:
            Dictionary mapping line_idx to boolean safety values
            or None if not calibrated
        """
        if self.rule_model is None:
            return None

        return self.rule_model.predict(trajectory)
