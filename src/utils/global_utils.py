import os
import sys
import warnings
from pathlib import Path

# NOTE: This is a bit ugly but works,
# check if i find another way to structure
# (add project root to path)
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

first_root = Path(__file__).parent.parent.resolve()
if str(first_root) not in sys.path:
    sys.path.insert(0, str(first_root))

main_root = Path(__file__).parent.parent.parent.resolve()
if str(main_root) not in sys.path:
    sys.path.insert(0, str(main_root))

from functools import cache
from typing import Tuple

from utils.environment import create_environment


@cache
def get_env_details() -> Tuple[int, list[str]]:
    """
    Creates a temporary environment to obtain the number of lines and their names.
    The results are cached, so that the temporary environment is only created once.

    Returns:
        tuple: (n_lines, line_names) representing the number of lines (int), and the line names List[str], respectively
    """
    temp_env, _ = create_environment(42, [])
    n_lines = len(temp_env.name_line)
    line_names = temp_env.name_line
    temp_env.close()

    return n_lines, line_names


def get_forecaster_model():
    """
    To instantiate one of the forecasters
    Returns:
        The forecaster object
    """
    import importlib

    import config

    module = importlib.import_module(config.FORECASTER_MODULE)
    forecaster_class = getattr(module, config.FORECASTER_CLASS)
    forecaster = forecaster_class(config.FORECASTER_PATH)

    return forecaster


def ensure_dir(path: str, folder: str | None = None) -> str:
    """
    Ensure directory exists and return its path.
    If a folder is provided, the path is joined with folder

    Args:
        path: dir path to create
        folder: a possible folder to put at the end of the path

    Returns:
        the same path (string) after ensuring it exists
    """
    if folder:
        path = os.path.join(path, folder)

    os.makedirs(path, exist_ok=True)
    return path


def get_dir_name(file_path: str) -> str:
    """
    to return the path of the file
    if path of the file is "/a/b/c", it returns "/a/b"

    Args:
        file_path: string representing the path

    Returns:
        string dirname
    """
    return os.path.dirname(file_path)


def ignore_warnings() -> None:
    """
    We choose to ignore these warnings because they come from the agent / forecaster
    and are not problematic, but are annoying
    """
    warnings_to_ignore = [
        "X does not have valid feature names, but HistGradientBoostingRegressor was fitted with feature names",
        "The initializer Orthogonal is unseeded and being called multiple times",
    ]

    for warning in warnings_to_ignore:
        warnings.filterwarnings("ignore", warning)
