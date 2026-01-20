# TODO: In the future, maybe cache the STL rules as well
# at the moment, we always have to train them

import os
import pickle
from typing import TYPE_CHECKING

import config
from utils.global_utils import ensure_dir, get_dir_name

if TYPE_CHECKING:
    from utils.data_structures import TrajectoryManager
    from utils.model_wrapper import ModelWrapper


def lines_to_bitmask(line_numbers: list[int]) -> int:
    """
    Converts a list of line numbers to a bitmask

    Args:
        line_numbers: List of integer line indices

    Returns:
        Integer bitmask with bits set for each line number
    """
    # A special case for when there are no lines attacked
    # (we do not return 0 because that would mean line 0 being attacked)
    if not line_numbers:
        return -1

    bitmask = 0
    for line_num in line_numbers:
        bitmask |= 1 << line_num
    return bitmask


def extract_line_number(line_name: str) -> int:
    """
    Extracts the line number from a line name string

    Args:
        line_name: String in format "X_Y_Z" where Z is the line number

    Returns:
        Integer line number
    """
    return int(line_name.split("_")[-1])


class DataManager:
    """
    Data manager for tracking and caching calibrated line episodes and trained models
    """

    def __init__(self):
        self.cache_dir = config.CALIBRATION_CACHE_DIR
        self.models_cache_dir = config.MODELS_CACHE_DIR

        ensure_dir(self.cache_dir)
        ensure_dir(self.models_cache_dir)
        self._attacked_lines: list[int] = []

    def set_attacked_lines(self, attacked_lines: list[int]) -> None:
        """
        Sets the list of line numbers that were attacked during calibration

        Args:
            attacked_lines: list of integer line numbers that were attacked
        """
        self._attacked_lines = attacked_lines

    def get_attacked_lines(self) -> list[int]:
        """
        Returns the line numbers that were attacked during calibration

        Returns:
            list of integer line numbers that were attacked
        """
        return self._attacked_lines

    @staticmethod
    def _generate_calibration_filename() -> str:
        """
        generates unique cache filename based on config parameters

        Returns:
            string filename that works as a hash for a certain configuration, so we don't need to re-run cached episodes
            that produce exactly the same results (deterministic behaviour in Grid2Op)
        """
        mode = "ensemble" if config.ENSEMBLE_MODE else "single"

        # line numbers from config.LINES_ATTACKED
        # e.g., ["1_3_3", "1_4_4", "3_6_15"] -> lines 3, 4, 15
        attacked_lines = [
            extract_line_number(line) for line in config.CALIBRATION_LINES_ATTACKED
        ]

        # The idea here (because of filename length) is instead of encoding it into a string,
        # we create a bitmask that is a single integer but in each bit position represents
        # if a particular line was attacked or not
        bitmask = lines_to_bitmask(attacked_lines)

        filename = (
            f"calib_{mode}_"
            f"chronic{config.BASE_CHRONIC}_"
            f"horizon{config.HORIZON}_"
            f"warmup{config.WARMUP_STEPS}_"
            f"steps{config.STEPS_TO_RUN}_"
            f"lines{bitmask}_"
            f"seed{config.BASE_ENV_SEED}_"
            f"oac{config.OPPONENT_ATTACK_COOLDOWN}_"
            f"oad{config.OPPONENT_ATTACK_DURATION}_"
            f"obpt{config.OPPONENT_BUDGET_PER_TS}_"
            f"oib{config.OPPONENT_INIT_BUDGET}_"
            f"en{config.ENV_NAME}_"
            f"pfoa{config.PERFORM_FORECAST_ON_ACTION}_"
            f"an{config.AGENT_NAME}_"
            f"fn{config.FORECASTER_CLASS}"
        )
        return filename

    def _get_episode_cache_path(
        self, attacked_lines: list[int], episode_idx: int
    ) -> str:
        """
        Generate complete cache file path for a specific episode

        Args:
            attacked_lines: List of integer line numbers that were attacked in this episode
            episode_idx: Integer episode index

        Returns:
            String path to the cache file for this episode
        """
        filename = self._generate_calibration_filename()
        bitmask = lines_to_bitmask(attacked_lines)
        mypath = f"{filename}_attacked{bitmask}_episode_{episode_idx}.pkl"
        return os.path.join(self.cache_dir, mypath)

    def episode_exists(self, attacked_lines: list[int], episode_idx: int) -> bool:
        """
        Check if a specific calibration episode exists in the cache

        Args:
            attacked_lines: List of integer line numbers that were attacked
            episode_idx: Integer episode index

        Returns:
            Boolean (True if the episode cache file exists, else False)
        """
        cache_path = self._get_episode_cache_path(attacked_lines, episode_idx)
        return os.path.exists(cache_path)

    def load_episode(
        self, attacked_lines: list[int], episode_idx: int
    ) -> "TrajectoryManager | None":
        """
        Loads a previously cached calibration episode

        Args:
            attacked_lines: List of integer line numbers that were attacked
            episode_idx: Integer episode index

        Returns:
            The cached TrajectoryManager
        """
        cache_path = self._get_episode_cache_path(attacked_lines, episode_idx)

        with open(cache_path, "rb") as f:
            return pickle.load(f)

    def save_episode(
        self,
        attacked_lines: list[int],
        episode_idx: int,
        episode_content: "TrajectoryManager",
    ):
        """
        Save calibration episode so we dont need to compute it again (deterministic)

        Args:
            attacked_lines: List of integer line numbers that were attacked
            episode_idx: Integer episode index
            episode_content: TrajectoryManager to save
        """
        cache_path = self._get_episode_cache_path(attacked_lines, episode_idx)

        ensure_dir(get_dir_name(cache_path))
        with open(cache_path, "wb") as f:
            pickle.dump(episode_content, f)
        print(f"Saved episode {episode_idx} for attacked lines {attacked_lines}")

    def load_all_episodes(self, attacked_lines: list[int]) -> list["TrajectoryManager"]:
        """
        Loads all episodes for a specific set of attacked lines

        Args:
            attacked_lines: List of integer line numbers that were attacked

        Returns:
            List of TrajectoryManager objects for all cached episodes
        """
        episode_list = []
        for episode_idx in range(config.CALIB_EPISODES):
            if self.episode_exists(attacked_lines, episode_idx):
                episode_data = self.load_episode(attacked_lines, episode_idx)
                if episode_data is not None:
                    episode_list.append(episode_data)
        return episode_list

    def _generate_model_filename(
        self, model_name: str, n_episodes: int, alpha: float
    ) -> str:
        """
        Generates unique cache filename for a trained model (works as a hash)
        Uses same base filename as calibration episodes and also the model info and the used alpha

        Args:
            model_name: String name of the conformal prediction model
            n_episodes: Integer number of calibration episodes used for training
            alpha: Float significance level

        Returns:
            String filename with all relevant configuration parameters
        """
        # we use the same base filename as calibration episodes
        base_filename = self._generate_calibration_filename()

        # just to clean so it uses underscores
        model_name = model_name.replace(" ", "_").replace("/", "_")

        # format alpha to avoid floating point issues (e.g., 0.1 -> "0.1", 0.01 -> "0.01", 2.0 -> "2")
        # removing trailing zeros and the dot if it is an integer
        alpha_str = f"{alpha:.4f}".rstrip("0").rstrip(".")

        # model-specific info: model name + number of episodes + alpha
        filename = (
            f"{base_filename}_model_{model_name}_episodes{n_episodes}_alpha{alpha_str}"
        )

        return filename

    def _get_model_cache_path(
        self, model_name: str, n_episodes: int, alpha: float
    ) -> str:
        """
        Generate complete cache file path for a trained model
        Models are saved in config.MODELS_CACHE_DIR directory (different from the calibration_cache directory)

        Args:
            model_name: String name of the conformal prediction model
            n_episodes: Integer number of calibration episodes used for training
            alpha: float significance level

        Returns:
            String path to the cache file for this model
        """
        filename = self._generate_model_filename(model_name, n_episodes, alpha)
        return os.path.join(self.models_cache_dir, f"{filename}.pkl")

    def model_exists(self, model_name: str, n_episodes: int, alpha: float) -> bool:
        """
        Check if a trained model exists in cache

        Args:
            model_name: String name of the conformal prediction model
            n_episodes: Integer number of calibration episodes used for training
            alpha: float significance level

        Returns:
            Boolean indicating if the model exists in the cache
        """
        cache_path = self._get_model_cache_path(model_name, n_episodes, alpha)
        return os.path.exists(cache_path)

    def load_model(
        self, model_name: str, n_episodes: int, alpha: float
    ) -> "ModelWrapper | None":
        """
        Load a previously cached trained model

        Args:
            model_name: String name of the conformal prediction model
            n_episodes: Integer number of calibration episodes used for training
            alpha: float significance level

        Returns:
            ModelWrapper object if successful, None if loading fails
        """
        cache_path = self._get_model_cache_path(model_name, n_episodes, alpha)

        with open(cache_path, "rb") as f:
            model = pickle.load(f)
        print(f"Loaded cached model: {model_name}")
        return model

    def save_model(
        self, model_name: str, n_episodes: int, model: "ModelWrapper", alpha: float
    ):
        """
        Save trained model to cache for future reuse

        Args:
            model_name: String name of the conformal prediction model
            n_episodes: Integer number of calibration episodes used for training
            model: ModelWrapper object to save
            alpha: float significance level
        """
        cache_path = self._get_model_cache_path(model_name, n_episodes, alpha)

        ensure_dir(get_dir_name(cache_path))
        with open(cache_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved model {model_name} with alpha={alpha} to cache")
