# ALPHA = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
ALPHA = 0.1


# Run in ensemble mode (one model per line, with that line subjected to adversarial attacks)
ENSEMBLE_MODE = False


# The base chronic is important. This value represents
# from when the simulation starts.
# If the forecaster was trained on the first k episdoes
# then we must set BASE_CHRONIC = k to ensure data independence
BASE_CHRONIC = 900
# number of episodes from which we collect the
# calibration dataset. For example, if BASE_CHRONIC = 900 and
# CALIB_EPISODES = 30, then we collect data from episode 901 to 930
CALIB_EPISODES = 15
# number of test episodes. They execute after the calibration, for example
# if BASE_CHRONIC = 900, and CALIB_EPISODES = 30 and TEST_EPISODES = 30,
# then the test episodes are 931 to 960
TEST_EPISODES = 1
STEPS_TO_RUN = 8064


# The directory in which the results will be placed
OUTPUT_DIR = "RESULTS/ERASE_ME"


# These four parameters are for cache only
# with the first two we can ignore cached calibration episodes/models
# in case we change something in the code.
# It will save the new calibration episodes/models and replace the ones
# that were there. it only overwrites if it completes the episode/model training
# The other two parameters set the path where to save the cache
IGNORE_CACHE_CALIBRATION = True
IGNORE_CACHE_MODELS = True
CALIBRATION_CACHE_DIR = "CACHE/calibration_cache"
MODELS_CACHE_DIR = "CACHE/models_cache"


# If this parameter is set to True, the plotting utils
# are automatically ran after the conformal simulation
# and the results are put in the same folders
AUTO_GEN_PLOTS = True


# currently using KNN neighbours = KNN_PERCENTAGE * n_calibration_samples
# we can override this value below (replace the None)
KNN_NEIGHBORS_OVERRIDE = None
KNN_PERCENTAGE = 0.1
# The batch size to use in kNN (see the kNN methods)
# increasing this number makes the kNN method a little
# faster to train, but it also increases memory
# 500 seems to be a nice value
BATCH_SIZE = 500


# This value is for the methods that use the ACI update rule
# and it controls how abruptly the adaptive alphas change
# in practise this value must not be very high, otherwise
# it becomes too unstable
GAMMA = 0.005


# These are the different conformal models we can use
# to provide intervals of uncertainty around point forecasts
# Format: "model_name": [enabled, file_name (without .py)]
MODELS = {
    "vanilla": (True, "vanilla"),
    "knn_norm": (True, "knn_norm"),
    "aci": (True, "aci"),
    "knn_norm_with_aci": (True, "knn_norm_with_aci"),
}


# These are the STL files for the rule psi := G[1,12](rho ≤ threshold)
# Format: "rule_name": (enabled, module_name (without .py))
STL_RULE = {
    "vanilla_rule": (True, "vanilla_rule"),
    "knn_norm_rule": (True, "knn_norm_rule"),
}
# the rho value above which a line is considered unsafe
RHO_SAFETY_THRESHOLD = 0.95


# Opponent configuration
# (check https://grid2op.readthedocs.io/en/latest/user/opponent.html)
OPPONENT_ATTACK_COOLDOWN = 12 * 24  # cooldown of a day
OPPONENT_ATTACK_DURATION = 12 * 4  # 4 hours
OPPONENT_BUDGET_PER_TS = 0.5  # opponent_attack_duration
OPPONENT_INIT_BUDGET = 0.0
CALIBRATION_LINES_ATTACKED = [
    "1_3_3",
    "1_4_4",
    "3_6_15",
    "9_10_12",
    "11_12_13",
    "12_13_14",
]
TESTING_LINES_ATTACKED = [
    "1_3_3",
    "1_4_4",
    "3_6_15",
    "9_10_12",
    "11_12_13",
    "12_13_14",
]


# Environment, agent and predictor config
# It is important to set a different agent name
# and different forecaster class every time it changes
# to ensure that the cache works

ENV_NAME = "l2rpn_case14_sandbox"
AGENT_NAME = "CurriculumAgentTest_14"
MODEL_PATH = "../curriculum_14/"
FORECASTER_CLASS = "HBGB_14"
FORECASTER_PATH = "../HBGB_14.pkl"
FORECASTER_MODULE = "forecasters.hbgb_14"


# ENV_NAME = "l2rpn_icaps_2021_small"
# AGENT_NAME = "CurriculumAgentTest_36"
# MODEL_PATH = "../curriculum_36/"
# FORECASTER_CLASS = "HBGB_36"
# FORECASTER_PATH = "../HBGB_36.pkl"
# FORECASTER_MODULE = "forecasters.hbgb_36"


# ENV_NAME = ""
# AGENT_NAME = "CurriculumAgentTest_118"
# MODEL_PATH = "../curriculum_118/"
# FORECASTER_CLASS = "HBGB_118"
# FORECASTER_PATH = "../HBGB_118.pkl"
# FORECASTER_MODULE = "forecasters.hbgb_118"


# Environment details
# changing the environment requires that both the agent
# and forecaster are also changed. For the forecaster,
# the perform_forecast fucntion must be implemented
BASE_ENV_SEED = 2000 + 42


# When collecting the real values after a forecast, if we detect an action
# we immediatly start another forecast.
# This is important because we want to mark the following timesteps as action influenced
# If set to false, every trajectory will always have 12 timesteps
PERFORM_FORECAST_ON_ACTION = True


# The HORIZON represents for how long we forecast
# for example, HORIZON=12 represents one hour of forecasted time
# (t+1, ..., t+12) at 5 min interval resolution
HORIZON = 12


# Depending on the forecasting agent, it may make sense to ignore
# the first WARMUP_STEPS in each episode
WARMUP_STEPS = 288


# This parameter depends on the forecaster used.
# for example, if a forecaster only uses the most recent 2016 observations, we try
# to limit the size of the observations array to 2020 (a little extra (sentinels))
# Default is setting this value to None
MAX_OBSERVATIONS = 2020


# Below are some assertions that must be ensured
# it's important that these values are between 1 and 12
assert 1 <= HORIZON <= 12
# we cannot use more than 8064 in this environment
STEPS_TO_RUN = min(STEPS_TO_RUN + WARMUP_STEPS, 8064)


import multiprocessing as mp

# Parallel parameters for calibration, model training and testing
MAX_CPUS = mp.cpu_count()
# If we want to run single-core, just set MAX_WORKERS_* to 1
MAX_WORKERS_CALIBRATION = min(11, MAX_CPUS)  # collecting data
MAX_WORKERS_MODELS = min(4, MAX_CPUS)  # training models
MAX_WORKERS_TESTING = min(4, MAX_CPUS)  # testing time


# Minimum number of complete timesteps required for episode to be included in aggregation
MIN_DATAPOINTS_FOR_AGGREGATION = 50
