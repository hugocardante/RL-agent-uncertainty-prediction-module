import datetime

import joblib
import numpy as np
from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation

import config
from forecast import Forecaster
from utils.data_structures import SimpleObservation


class HBGB_36(Forecaster):
    """
    forecaster using Histogram-Based Gradient Boosting (HBGB) for the 36 network

    This forecaster uses historical load/generation data along with
    temporal features to predict future grid states (load/gen) and then
    runs a powerflow to get the forecasted values of rho
    """

    def __init__(self, model_path: str, **kwargs):
        """
        Initialize the HBGB forecaster.

        Args:
            model_path: Path to the trained HBGB model file (uses config.FORECASTER_PATH if None)
            **kwargs: Additional parameters (we dont use it, but other forecasters might)
        """
        model = joblib.load(model_path)
        self.no_loads = 37
        self.no_gens = 22
        super().__init__(model, **kwargs)

    def perform_forecast(
        self,
        obs: BaseObservation,
        observations_array: list[SimpleObservation],
        agent: BaseAgent,
        **kwargs,
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

        for i in range(1, config.HORIZON + 1):
            features, new_timestamp = self._get_features(observations_array, obs, i)
            y_pred = self.model.predict(features)[0]

            load_p.append(y_pred[: self.no_loads])
            load_q.append(y_pred[self.no_loads : self.no_loads * 2])
            gen_p.append(y_pred[self.no_loads * 2 :])
            timestamps.append(new_timestamp)

        forecast_obs = obs.copy()
        forecast_done = False
        i = 0

        while not forecast_done:
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

            obs_dict[timestamps[i]] = SimpleObservation(forecast_obs)
            forecast_timestamps.add(timestamps[i])

            agent_action = agent.act(forecast_obs, reward=0, done=False)
            forecasted_obs_with_agent_action, _, forecast_done, _ = (
                forecast_obs.simulate(agent_action, time_step=1)
            )

            agent_forecast_dict[timestamps[i]] = (
                forecasted_obs_with_agent_action.rho.copy().tolist()
            )

            forecasted_obs_with_agent_action = self._create_clean_obs(
                forecasted_obs_with_agent_action, forecast_obs
            )

            forecast_obs = forecasted_obs_with_agent_action
            i += 1
            if forecast_done or i == config.HORIZON:
                break

        return agent_forecast_dict, forecast_timestamps, obs_dict

    def _get_features(self, observations_array, obs, step):

        def get_past(steps):
            past_index = len(observations_array) + step - steps
            if 0 <= past_index < len(observations_array):
                return observations_array[past_index]
            return None

        def extract_features(past_obs, attr, size):
            if past_obs is not None:
                return getattr(past_obs, attr)
            return np.array([None] * size)

        past_obs_hour = get_past(12)
        past_obs_day = get_past(288)
        past_obs_week = get_past(2016)

        feature_vec = np.concatenate(
            [
                extract_features(past_obs_week, "load_p", self.no_loads),
                extract_features(past_obs_day, "load_p", self.no_loads),
                extract_features(past_obs_hour, "load_p", self.no_loads),
                extract_features(past_obs_week, "load_q", self.no_loads),
                extract_features(past_obs_day, "load_q", self.no_loads),
                extract_features(past_obs_hour, "load_q", self.no_loads),
                extract_features(past_obs_week, "gen_p", self.no_gens),
                extract_features(past_obs_day, "gen_p", self.no_gens),
                extract_features(past_obs_hour, "gen_p", self.no_gens),
            ]
        )

        temporal_features = []
        new_timestamp = 0

        if step == 0:
            hour_cos, hour_sin = self._convert_to_cos_sin(obs.hour_of_day, 23)
            minute_cos, minute_sin = self._convert_to_cos_sin(obs.minute_of_hour, 59)
            dow_cos, dow_sin = self._convert_to_cos_sin(obs.day_of_week, 6)
            temporal_features = [
                obs.day,
                hour_cos,
                hour_sin,
                minute_cos,
                minute_sin,
                dow_cos,
                dow_sin,
            ]
        else:
            atual_timestamp = obs.get_time_stamp()
            new_timestamp = atual_timestamp + datetime.timedelta(minutes=step * 5)
            day = new_timestamp.day
            hour_of_day = new_timestamp.hour
            minute_of_hour = new_timestamp.minute
            day_of_week = new_timestamp.weekday()
            hour_cos, hour_sin = self._convert_to_cos_sin(hour_of_day, 23)
            minute_cos, minute_sin = self._convert_to_cos_sin(minute_of_hour, 59)
            dow_cos, dow_sin = self._convert_to_cos_sin(day_of_week, 6)
            temporal_features = [
                day,
                hour_cos,
                hour_sin,
                minute_cos,
                minute_sin,
                dow_cos,
                dow_sin,
            ]

        return (
            np.concatenate([feature_vec, temporal_features]).reshape(1, -1),
            new_timestamp,
        )

    @staticmethod
    def _convert_to_cos_sin(value, max_value):
        value_cos = np.cos(2 * np.pi * value / max_value)
        value_sin = np.sin(2 * np.pi * value / max_value)
        return value_cos, value_sin

    # FIX: Isto é experimental, um hack vá. Confirmar com a margarida
    @staticmethod
    def _create_clean_obs(simulated_obs, base_obs):
        clean_obs = base_obs.copy()

        # a ideia é copiar os atributos que já existiam um a um
        for attr in base_obs.attr_list_vect:
            setattr(clean_obs, attr, getattr(simulated_obs, attr))
        return clean_obs
