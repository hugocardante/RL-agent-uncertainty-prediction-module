import pandas as pd

from plotting.plotting_config import MODELS_TO_PLOT, RULES_TO_PLOT


def load_csv(path: str | None) -> pd.DataFrame | None:
    """
    Loads csv file, returning None if missing or empty

    Args:
        path: path to the csv file

    Returns:
        DataFrame if successful, None otherwise
    """
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
        if df.empty or len(df.columns) == 0:
            return None
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def _get_names_from_config(
    config: dict[str, str | int | float], prefix: str, suffix: str = "_name"
) -> list[str]:
    """
    Extracts names from the config csv

    The config csv stores names with pattern like "line_0_name", "model_1_name", etc.
    This function finds all keys matching the prefix and suffix pattern.

    Args:
        config: dictionary with config parameters
        prefix: key prefix to match (e.g., "line_", "model_", "stl_rule_")
        suffix: key suffix to match, defaults to "_name"

    Returns:
        List of extracted name values
    """
    if not config:
        return []

    names = []
    # iterate over the keys and when we find the patter
    # e.g., line_0_name or stl_rule_1_name, we take the value
    for key, value in config.items():
        if key.startswith(prefix) and key.endswith(suffix):
            names.append(value)

    return names


def get_model_names(config: dict[str, str | int | float]) -> list[str]:
    """
    Extracts model names from config and applies filtering

    Args:
        config: dictionary with config parameters

    Returns:
        List of model names, filtered by MODELS_TO_PLOT if set (see MODELS_TO_PLOT)
    """
    names = _get_names_from_config(config, "model")

    if MODELS_TO_PLOT is not None:
        names = [n for n in names if n in MODELS_TO_PLOT]

    return names


def get_stl_rule_names(config: dict[str, str | int | float]) -> list[str]:
    """
    Extracts STL rule names from config and applies filtering

    Args:
        config: dictionary with config parameters

    Returns:
        List of STL rule names, filtered by RULES_TO_PLOT if set (see RULES_TO_PLOT)
    """
    names = _get_names_from_config(config, "stl_rule")

    if RULES_TO_PLOT is not None:
        names = [n for n in names if n in RULES_TO_PLOT]

    return names


def load_config(config_csv: str | None) -> dict[str, str | int | float]:
    """
    Loads configuration parameters from csv

    Args:
        config_csv: path to the config csv file

    Returns:
        Dictionary mapping parameter names to values
    """
    if config_csv is None:
        return {}
    df = load_csv(config_csv)
    if df is None:
        return {}
    # convert to dict in the form {parameter -> value}
    return df.set_index("parameter")["value"].to_dict()


def load_comparison_data(csv_path: str | None) -> pd.DataFrame | pd.Series:
    """
    Loads comparison data and filters by MODELS_TO_PLOT if set

    Args:
        csv_path: path to the comparison csv file

    Returns:
        DataFrame with comparison data, or empty DataFrame if loading fails
    """
    if csv_path is None:
        return pd.DataFrame()

    df = load_csv(csv_path)
    if df is None:
        return pd.DataFrame()

    if MODELS_TO_PLOT is not None:
        df = df[df["model_name"].isin(MODELS_TO_PLOT)]

    return df


def load_timeseries_data(csv_path: str) -> pd.DataFrame | None:
    """
    Loads time series data from timeseires csv

    Args:
        csv_path: path to the timeseries csv file

    Returns:
        DataFrame with timeseries data, or None if loading fails
    """
    df = load_csv(csv_path)
    if df is None:
        return None

    return df


def get_line_data(
    timeseries_df: pd.DataFrame, line_idx: int
) -> pd.DataFrame | pd.Series:
    """
    Gets timeseries data for a specific line

    Args:
        timeseries_df: dataframe with all timeseries data
        line_idx: index of the power line to extract

    Returns:
        DataFrame filtered for the specified line, with datetime column added
    """
    df = timeseries_df[timeseries_df.line_index == line_idx].copy()
    # convert timestamp string to datetime for plotting
    if not df.empty and "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"])
    return df


def get_line_names(config: dict[str, str | int | float]) -> list[str]:
    """
    Gets line names from config csv

    Args:
        config: dictionary with config parameters

    Returns:
        List of power line names in the config
    """
    return _get_names_from_config(config, "line")


def get_model_stats(
    comparison_df: pd.DataFrame, model_name: str, line_idx: int
) -> dict[str, float | None] | None:
    """
    Gets coverage/width stats for a model on a specific line

    Args:
        comparison_df: dataframe with comparison metrics
        model_name: name of the model
        line_idx: index of the power line

    Returns:
        Dictionary with coverage, width, and action_inf_coverage, or None if not found
    """
    if comparison_df.empty:
        return None

    # filter for the specific model and line combination
    mask = (comparison_df.model_name == model_name) & (
        comparison_df.line_index == line_idx
    )
    rows = comparison_df[mask]

    if rows.empty:
        return None

    # the dataframe only has a row but we still have to
    # extract it to use it like a series
    row = rows.iloc[0]
    return {
        "coverage": row["coverage"],
        "width": row["width"],
        "action_inf_coverage": row.get("action_inf_coverage"),
    }
