import datetime

import joblib
import numpy as np
from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation

import config
from forecast import Forecaster
from utils.data_structures import SimpleObservation


class HBGB_14(Forecaster):
    """
    forecaster using Histogram-Based Gradient Boosting (HBGB) for the 14 network

    This forecaster uses historical load/generation data along with
    temporal features to predict future grid states (load/gen) and then
    runs a powerflow to get the forecasted values of rho
    """

    def __init__(self, model_path: str, **kwargs):
        """
        Initialize the HBGB forecaster.

        Args:
            model_path: Path to the trained HBGB model file (uses config.FORECASTER_PATH if None)
            **kwargs: Additional parameters (unused for HBGB)
        """

        # load the model from the pkl
        with open(model_path, "rb") as f:
            model = joblib.load(f)

        super().__init__(model, **kwargs)

    def perform_forecast(
        self,
        obs: BaseObservation,
        observations_array: list[SimpleObservation],
        agent: BaseAgent,
    ) -> tuple[
        dict[datetime.datetime, list[float]],
        set[datetime.datetime],
        dict[datetime.datetime, SimpleObservation],
    ]:
        """
        Performs forecast using HBGB model
        """
        agent_forecast_dict = {}
        forecast_timestamps = set()
        obs_dict = {}

        load_p = []
        load_q = []
        gen_p = []
        timestamps = []
        step = 1

        # predictions for each future timestep (for load/gen..)
        for minutes_to_add in range(5, 5 + 5 * config.HORIZON, 5):
            new_obs = obs.copy()
            new_timestamp = new_obs.get_time_stamp() + datetime.timedelta(
                minutes=minutes_to_add
            )
            new_obs = self._extract_timestamp_info(new_obs, new_timestamp)

            data = self._get_features(observations_array, new_obs, step).reshape(1, -1)
            y_pred = self.model.predict(data)[0]

            load_p.append(np.array(np.round(y_pred[0:11], 3)))
            load_q.append(np.array(np.round(y_pred[11:22], 3)))
            gen_p_aux = np.array(self._sort_vals_aux(np.round(y_pred[22:], 3)))
            gen_p.append(gen_p_aux)
            timestamps.append(new_timestamp)
            step += 1

        load_p = np.array(load_p)
        load_q = np.array(load_q)
        gen_p = np.array(gen_p)
        timestamps = np.array(timestamps)

        forecast_obs = obs.copy()
        forecast_done = False
        i = 0

        assert len(timestamps) == config.HORIZON
        while not forecast_done:
            # inject in the grid
            forecast_obs._forecasted_inj = [
                (
                    forecast_obs.get_time_stamp(),
                    {
                        "injection": {
                            "load_p": forecast_obs.load_p,
                            "load_q": forecast_obs.load_q,
                            "prod_p": forecast_obs.gen_p,
                            "prod_v": forecast_obs.gen_v,
                        }
                    },
                ),
                (
                    timestamps[i],
                    {
                        "injection": {
                            "load_p": load_p[i],
                            "load_q": load_q[i],
                            "prod_p": gen_p[i],
                            "prod_v": forecast_obs.gen_v,
                        }
                    },
                ),
            ]

            # we store as SimpleObservation
            obs_dict[timestamps[i]] = SimpleObservation(forecast_obs)
            forecast_timestamps.add(timestamps[i])

            # apply agent action to this forecasted state
            agent_action = agent.act(forecast_obs, reward=0, done=False)
            forecasted_obs_with_agent_action, _, forecast_done, _ = (
                forecast_obs.simulate(agent_action)
            )

            # store the forecasted rho values
            agent_forecast_dict[timestamps[i]] = (
                forecasted_obs_with_agent_action.rho.copy().tolist()
            )

            forecast_obs = forecasted_obs_with_agent_action

            i += 1
            if forecast_done or i == config.HORIZON:
                break

        return agent_forecast_dict, forecast_timestamps, obs_dict

    @staticmethod
    def _convert_to_cos_sin(value, max_value):
        value_cos = np.cos(2 * np.pi * value / max_value)
        value_sin = np.sin(2 * np.pi * value / max_value)
        return value_cos, value_sin

    @staticmethod
    def _sort_loads_power(features, element_week, element_day, element_hour):
        for i in range(11):
            features.append(element_week[i])
            features.append(element_day[i])
            features.append(element_hour[i])
        return features

    @staticmethod
    def _aux_sorts(values_gens):
        aux = [0] * 6
        aux[0] = values_gens[5]
        aux[1] = values_gens[0]
        aux[2] = values_gens[1]
        aux[3] = values_gens[2]
        aux[4] = values_gens[3]
        aux[5] = values_gens[4]
        return aux

    def _sort_gens_power(self, features, element_week, element_day, element_hour):
        element_week = self._aux_sorts(element_week)
        element_day = self._aux_sorts(element_day)
        element_hour = self._aux_sorts(element_hour)

        for i in range(6):
            features.append(element_week[i])
            features.append(element_day[i])
            features.append(element_hour[i])

        return features

    @staticmethod
    def _get_past_value_load_p(observations_array, step, steps):
        past_index = len(observations_array) + step - steps
        if 0 <= past_index < len(observations_array):
            return observations_array[past_index].load_p
        else:
            return [None] * 11

    @staticmethod
    def _get_past_value_load_q(observations_array, step, steps):
        past_index = len(observations_array) + step - steps
        if 0 <= past_index < len(observations_array):
            return observations_array[past_index].load_q
        else:
            return [None] * 11

    @staticmethod
    def _get_past_value_gen_p(observations_array, step, steps):
        past_index = len(observations_array) + step - steps
        if 0 <= past_index < len(observations_array):
            return observations_array[past_index].gen_p
        else:
            return [None] * 6

    def _get_features(self, observations_array, obs, step):
        features = []

        day = obs.day
        hour = obs.hour_of_day
        minute = obs.minute_of_hour
        day_of_week = obs.day_of_week

        load_p_hour_before = self._get_past_value_load_p(observations_array, step, 12)
        load_p_day_before = self._get_past_value_load_p(observations_array, step, 288)
        load_p_week_before = self._get_past_value_load_p(observations_array, step, 2016)

        load_q_hour_before = self._get_past_value_load_q(observations_array, step, 12)
        load_q_day_before = self._get_past_value_load_q(observations_array, step, 288)
        load_q_week_before = self._get_past_value_load_q(observations_array, step, 2016)

        gen_p_hour_before = self._get_past_value_gen_p(observations_array, step, 12)
        gen_p_day_before = self._get_past_value_gen_p(observations_array, step, 288)
        gen_p_week_before = self._get_past_value_gen_p(observations_array, step, 2016)

        features = self._sort_loads_power(
            features, load_p_week_before, load_p_day_before, load_p_hour_before
        )
        features = self._sort_loads_power(
            features, load_q_week_before, load_q_day_before, load_q_hour_before
        )
        features = self._sort_gens_power(
            features, gen_p_week_before, gen_p_day_before, gen_p_hour_before
        )

        hour_cos, hour_sin = self._convert_to_cos_sin(hour, 23)
        minute_cos, minute_sin = self._convert_to_cos_sin(minute, 59)
        day_of_week_cos, day_of_week_sin = self._convert_to_cos_sin(day_of_week, 6)

        features.append(day)
        features.append(hour_cos)
        features.append(hour_sin)
        features.append(minute_cos)
        features.append(minute_sin)
        features.append(day_of_week_cos)
        features.append(day_of_week_sin)

        return np.array(features)

    @staticmethod
    def _sort_vals_aux(gen_p):
        aux = [0] * 6
        aux[0] = gen_p[1]
        aux[1] = gen_p[2]
        aux[2] = gen_p[3]
        aux[3] = gen_p[4]
        aux[4] = gen_p[5]
        aux[5] = gen_p[0]
        return aux

    @staticmethod
    def _extract_timestamp_info(obs, dt):
        obs.year = dt.year
        obs.month = dt.month
        obs.day = dt.day
        obs.hour_of_day = dt.hour
        obs.minute_of_hour = dt.minute
        obs.day_of_week = dt.weekday()
        return obs
