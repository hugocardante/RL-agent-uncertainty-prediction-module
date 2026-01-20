# Utils

Utility modules for data structures, caching, environment setup, and parallelization.

## Modules

### `data_structures.py`

The main data structures used in this framework.

#### `SimpleObservation`

A simpler representation of a Grid2Op observation. Stores only the essential data needed for conformal prediction (rho, load, generation, time info) instead of the full Grid2Op observation object.
This can be edited if other parameters of the observation are needed.

#### `TimestepData`

This is a structure for all the data related to a single timestep.
This object lives inside of a Trajectory object.
Holds forecasted values, real values (when available), and conformal predictions (upper and lower bound for this timestep).

#### `Trajectory`

Represents a forecast event containing up to 12 timesteps (one per horizon step). Each trajectory stores:

- Forecasted rho values and observations
- Real rho values (filled in as simulation progresses)
- Conformal prediction intervals per model
- STL safety results per rule

Several methods regarding trajectories are implemented such as:

- `generate_conformal_predictions()` - generates prediction intervals using the trained models
- `generate_stl_predictions()` - evaluates STL rules on the trajectory
- `is_conf_predicted_safe()` / `is_stl_predicted_safe()` - check if trajectory was predicted safe
- `is_traj_actually_safe()` - check if trajectory was actually safe (ground truth)

#### `TrajectoryManager`

Manages active and completed trajectories for an entire episode.

For example, it:

- Adds new forecast events
- Updates trajectories with real data as simulation progresses
- Filters warmup timesteps
- Updates adaptive alphas (for ACI-based models)

### `cache.py`

This file implements the DataManager. It is responsible for saving calibration episode data
and trained models when the parameters are the same, to avoid retraining for every simulation.

#### `DataManager`

Handles saving and loading of:

- **Calibration episodes** - `TrajectoryManager` objects from calibration runs, cached to avoid recomputation since Grid2Op simulations are deterministic with the same parameters
- **Trained models** - `ModelWrapper` objects, cached per model/alpha/episode configuration

Cache filenames encode all relevant parameters (mode, chronic, horizon, warmup, seed, etc.) to ensure that we dont cache the wrong
data.

### `environment.py`

Grid2Op environment setup and utilities.

- `create_environment()` - creates and configures the Grid2Op environment with opponent attacks, loads the agent and forecaster
- `is_action_empty()` - checks if an agent action is a "do nothing" action
- `get_warmup_cutoff_time()` - calculates when warmup period ends
- `needs_new_forecast()` - determines if a new forecast should be generated

### `model_wrapper.py`

Wrapper for conformal prediction models.

#### `ModelWrapper`

Provides an interface for training and using conformal models. Supports two modes:

- **Single mode** - one model trained on all data
- **Ensemble mode** - one model per power line, each trained with adversarial attacks on that specific line

The wrapper handles:

- Loading calibration data from cache
- Calling the appropriate `compute_model()` function from `conformalized_models/`
- Generating predictions (worst-case bounds in ensemble mode)
- Updating adaptive alphas for ACI-based models

### `stl_wrapper.py`

Wrapper for STL (Signal Temporal Logic) rules.

#### `STLWrapper`

Similar to `ModelWrapper` but for STL rules. Loads the appropriate rule module from `stl_rules/` and provides:

- `fit()` - trains the rule using calibration trajectories
- `predict()` - evaluates a trajectory and returns safety results per line

### `parallel.py`

Parallel execution utilities using `ProcessPoolExecutor`.
Since initializing envs in Grid2Op with the same parameters is determinisic
we can use this to run parellelize some parts of the code.

#### Worker functions

- `calibration_worker()` - runs a single calibration episode
- `model_training_worker()` - trains a single conformal model
- `test_episode_worker()` - runs a single test episode

#### `run_parallel_jobs()`

Function that executes any of the workers in parallel

### `global_utils.py`

For now holds only the function:

- `get_env_details()` - returns the number of lines and line names from the environment (cached to avoid recreating the environment multiple times)

Might be removed in the future

### `classification.py`

This is experimental. It might not make complete sense
to evaluate results based on Conformal Prediction through the lens
of classification metrics.

The idea is to treat safety prediction as a binary classification problem:

- **Positive** = predicted/actually unsafe (rho exceeds threshold)
- **Negative** = predicted/actually safe

So that we can evaluate and compare the conformal models and the stl rules (see [plotting/README.md](../plotting/README.md) `classification_plots`)
It is composed of:

Shared utilities for classification analysis:

#### `ClassificationMetrics`

Container class that holds confusion matrix counts and derived metrics:

- `tp`, `fp`, `tn`, `fn` - confusion matrix values
- `fa` - false alarm rate: ``FP / (TP + FP)``
- `miss` - overlooked violations: ``FN / (TP + FN)``
- `f1` - F1 score: ``(2 * TP) / (2 * TP + FP + FN)``

The `update()` method takes a prediction and actual value, and increments the right counter.
This centralizes the classification logic so both STL and conformal modules use the same definitions.

#### Helper functions

- `aggregate_metrics()` - combines metrics across multiple episodes by summing confusion matrix counts
- `export_classification()` - exports metrics to csv

Classification analysis for conformal prediction models.

A trajectory is considered **predicted unsafe** by a conformal model if any timestep's predicted interval upper bound exceeds `config.RHO_SAFETY_THRESHOLD` for that line.

- `analyze_episode_conformal_safety()` - computes metrics for one episode across all models and lines

Classification analysis for STL rules.

A trajectory is considered **predicted unsafe** by an STL rule based on the rule's verification result (stored in `trajectory.stl_safety_results`).

- `analyze_episode_stl_safety()` - computes metrics for one episode across all rules and lines
