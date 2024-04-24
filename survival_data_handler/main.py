# Copyright 2024 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn import metrics

from survival_data_handler.utils import process_survival_function, \
    compute_derivative, residual_life, shift_from_interp
from survival_data_handler import plotting


class TimeCurve(pd.DataFrame):
    def __init__(self, *args, unit='D', n_unit=1., ):
        super().__init__(*args)
        self.n_unit = n_unit
        self.unit = pd.to_timedelta(n_unit, unit=unit).total_seconds()
        self.__checks()

    def __checks(self):
        if not all(self.columns[i] <= self.columns[i + 1] for i in range(len(self.columns) - 1)):
            raise ValueError("Columns must be sorted")
        # if not isinstance(self.columns[0], pd.Timedelta):
        #     raise ValueError("Columns must symbolize time index")

    def __interpolator(self):
        self.__interpolation = {
            c: interp1d(self.columns.astype(int),
                        self.loc[c], fill_value="extrapolate") for
            c in self.index}

    @property
    def interpolation(self) -> dict:
        if not hasattr(self, "__interpolation"):
            self.__interpolator()
        return self.__interpolation

    def derivative(self):
        return TimeCurve(compute_derivative(self, self.unit))

    def log(self):
        return TimeCurve(np.log(self))

    def exp(self):
        return TimeCurve(np.exp(self))

    def __div__(self, other):
        return TimeCurve(self / other, self.unit, self.n_unit)

    def __add__(self, other):
        return TimeCurve(self + other, self.unit, self.n_unit)

    def __sub__(self, other):
        return TimeCurve(self - other, self.unit, self.n_unit)

    def __neg__(self):
        return TimeCurve(- self, self.unit, self.n_unit)

# ======================================================================================================================
# AESTHETICS
cmap = sns.color_palette('rainbow', as_cmap=True)
event_censored_color = cmap(0.1)
event_observed_color = cmap(0.95)


class Lifespan:
    __index = "origin_index"

    # ==============================================
    #           CLASS METHOD
    # ==============================================

    def __init__(self,
                 ageing_curves: pd.DataFrame,
                 index: iter,
                 age: iter,
                 birth: iter,
                 window: tuple,
                 default_precision="float"):

        self.__curves = ageing_curves.__deepcopy__()
        self.df_idx = pd.DataFrame(
            dict(age=age,
                 birth=birth,
                 index=index))

        self.df_idx.index.name = self.__index
        self.index = index
        self.age = age
        self.window = window
        self.__p = default_precision

        self.__check_args()

        self.survival_estimation = SurvivalEstimation(
            self.curves, unit='D',
            n_unit=365.25)

        self._df_idx_unique = self.df_idx.drop_duplicates(
            subset=["index", "birth"])
        self.starting_dates = self._df_idx_unique["birth"]
        c = ["birth", "index"]

        self._corresp = pd.merge(
            self._df_idx_unique[c].reset_index(), self.df_idx[c].reset_index(),
            on=["birth", "index"],
            suffixes=("_u", ""))

        self._corresp = self._corresp.drop(columns=c)
        self.confusion_matrix_threshold = np.nan
        self.__computed = []
        self.__supervised = False

    def __check_args(self):

        if hasattr(self.curves.columns, "dt"):
            ValueError("curves index must have dt property")

    @property
    def computed_historical_data(self):
        return self.__computed

    @property
    def supervised(self):
        return self.__supervised

    def add_supervision(self, event, durations, entry=None):
        self.event = event
        self.durations = durations
        self.entry = entry
        self.__supervised = True

    def _decompose_unique(self, data, get_supervision=False):
        ret = pd.merge(self._corresp, data, right_index=True,
                       left_on='origin_index_u').drop(
            columns=["origin_index_u"]).set_index("origin_index")

        if not get_supervision:
            return ret, None, None, None
        else:
            index = [i for i in ret.index if i in self.event.index]
            entry = self.entry if self.entry is None else self.entry.loc[index]
            return ret.loc[index], self.event.loc[index], self.durations.loc[
                index], entry

    def _shift_helper(self, data: typing.Union[pd.DataFrame, pd.Series]):
        data = pd.merge(self._df_idx_unique["index"].reset_index(), data,
                        right_index=True,
                        left_on='index')
        data = data.set_index(self.__index).drop("index", axis=1)
        ret = shift_from_interp(
            data=data,
            starting_dates=self.starting_dates,
            period=pd.to_timedelta(30, unit="d"),
            date_initial=self.window[0], date_final=self.window[1],
            dtypes=self.__p
        )
        return ret

    # ==============================================
    #           PROPERTIES
    # ==============================================
    @property
    def survival_function(self) -> pd.DataFrame:
        if "survival_function" not in self.__computed:
            self.__compute_survival()
        return self.__survival_function

    @property
    def residual_expected_life(self) -> pd.DataFrame:
        if "residual_expected_life" not in self.__computed:
            self.__compute_expected_residual_life()
        return self.__residual_expected_life

    # ==============================================
    #           ENRICH REPRESENTATION
    # ==============================================
    def __compute_expected_residual_life(self) -> None:
        data = pd.Series(self.survival_estimation.residual_life_interp)
        data.name = "interp"
        data = self._shift_helper(data)
        self.__residual_expected_life = self._decompose_unique(data)[0].astype(
            self.__p)
        self.__computed.append("residual_lifespan")

    def __compute_survival(self):
        data = pd.Series(self.survival_estimation.survival_interp)
        data.name = "interp"
        data = self._shift_helper(data)
        self.__survival_function = self._decompose_unique(data)[0].astype(
            self.__p)
        self.__survival_function[self.__survival_function > 1] = np.nan
        self.__survival_function[self.__survival_function < 0] = 0
        self.__computed.append("survival_function")

    def residual_survival(self, t0) -> pd.DataFrame:

        t0 = max(t0, self.survival_function.columns[0])
        s0 = get(self.survival_function, t0).astype("float16").values
        df = self.survival_function.divide(
            s0, axis=0).copy()
        df = df[[c for c in df.columns if c > t0]]
        return df

    def compute_times(self, p=0.95, on="survival_function"):

        cond_low = (self.__getattribute__(on) > p).sum(axis=1)
        dates = self.__getattribute__(on).columns.to_numpy()
        df_time = self._df_idx_unique.__deepcopy__()
        df_time.loc[cond_low.index, "time"] = dates[
            np.where(cond_low.values >= len(dates), len(dates) - 1,
                     cond_low.values)]
        # remove past positive sample
        df_time, _, _, _ = self._decompose_unique(df_time)
        return df_time["time"]

    def remove_the_dead(self, data: pd.DataFrame, error="coerce"):
        try:
            data = data.loc[self.event.index[~self.event.astype(bool)]]
        except KeyError as e:
            print(e)

            if error == "raise":
                raise KeyError(e)
            elif error == "coerce":
                index = [i for i in self.event.index[~self.event.astype(bool)]
                         if i in data.index]
                data = data.loc[index]

        return data

    def percentile_life(self, q: float = 0.5) -> pd.DataFrame:
        """
        Compute the percentile lifetime corresponding to the proportion of
        the population (in terms of curves) that remains
        """
        min_ = pd.DataFrame(self.survival_function > q)
        cs = min_.cumsum(axis=1)
        df = pd.DataFrame(
            np.array(cs.columns.to_list())[cs.iloc[:, -1].values - 1],
            columns=["percentile_life_%s" % q]
        )
        del min_, cs
        return df

    def compute_confusion_matrix(self, on: str, threshold: float, score=True):
        from sklearn import metrics
        assert on in self.computed_historical_data
        data = self.__getattribute__(on)
        if score:
            data = 1 - data

        time_start, time_stop = self.durations.min(), self.durations.max()
        self.confusion_matrix_threshold = threshold
        predicted_events = pd.DataFrame(data > threshold)[
            [c for c in data.columns if time_start <= c <= time_stop]]

        real_life_events = predicted_events.__deepcopy__()
        nan = data.isna()[
            [c for c in data.columns if time_start <= c <= time_stop]]

        df_matrix = pd.DataFrame(index=real_life_events.columns,
                                 columns=["TN", "FP", "FN", "TP"])
        for c in real_life_events.columns:
            real_life_events[c] = (
                    self.event.astype(bool) & (self.durations < c))
            args = real_life_events[c][~nan[c]].astype(bool), \
                predicted_events[c][~nan[c]].astype(bool)

            df_matrix.loc[c] = metrics.confusion_matrix(*args).ravel()

        tn, fp, fn, tp = df_matrix["TN"], df_matrix["FP"], df_matrix["FN"], \
            df_matrix["TP"]

        df_matrix["P"] = p = tp + fn
        df_matrix["N"] = n = tn + fp
        df_matrix["Total"] = p + n
        df_matrix["accuracy"] = (tp + tn) / (p + n)

        if all(tp + fp > 0) and all(tp + fn > 0):
            df_matrix["precision"] = precision = np.divide(tp, tp + fp)

            df_matrix["recall"] = recall = np.divide(tp, tp + fn)

            if all(recall > 0) and all(precision > 0):
                df_matrix["f1-score"] = 2 * recall * precision / (
                        recall + precision)
                df_matrix["f2-score"] = (1 + 2) ** 2 * recall * precision / (
                        2 ** 2 * precision + recall)

        return df_matrix

    # ==============================================
    #           PLOTS CURVES
    # ==============================================
    def plot_curves(self):
        plt.figure(dpi=200, figsize=(7, 5))
        self.curves.plot(legend=False)

    def plot_curves_residual_life(self, **kwargs):
        self.survival_estimation.plot_residual_life(**kwargs)
        plt.ylabel("Expected residual lifespan")
        plt.xlabel("Time [Y]")

    def plot_average_tagged(self, data: pd.DataFrame,
                            event_type=None,
                            plot_test_window=False, plot_type="dist"):
        corresp, event, duration, entry = self._decompose_unique(
            data,
            get_supervision=True)
        if event_type == "censored":
            sample = corresp[~event.astype(bool)].astype("float32")
            color = event_censored_color
        elif event_type == "observed":
            sample = corresp[event.astype(bool)].astype("float32")
            color = event_observed_color

        else:
            sample = corresp.astype("float32")
            color = event_censored_color

        if plot_type == "dist":
            percentiles = np.linspace(0, 1, 21)
            sample_des = sample.describe(percentiles=percentiles)

            def ts(__p):
                return str(int(np.round(__p, 3) * 100)) + "%"

            for p in percentiles:
                label = None if p != percentiles[0] else f"{event_type} event"
                plt.fill_between(
                    sample_des.columns,
                    sample_des.loc[ts(p)],
                    sample_des.loc[ts(1 - p)], color=color,
                    alpha=0.1, label=label)
        else:
            sample_des = sample.describe()

            plt.plot(sample_des.columns, sample_des.loc["50%"],
                     color=event_censored_color,
                     label=f"{event_type} event")

            plt.fill_between(
                sample_des.columns,
                sample_des.loc["25%"],
                sample_des.loc["75%"], color=color, alpha=0.2)

        if plot_test_window:
            t0 = duration[event.astype(bool)].min()
            t1 = duration[event.astype(bool)].max()

            plt.axvline(t0, lw=1.5, ls="--", color="k")
            plt.axvline(t1, lw=1.5, ls='--', color="k", label="Test window")
        plt.legend()

    def plot_dist_facet_grid(self, data, n=4):
        bins = np.linspace(0, 1, 50)
        data, event, duration, entry = self._decompose_unique(
            data,
            get_supervision=True)
        s = data.shape[1]
        dates = data.columns.to_numpy()[
            np.array([(s * i) // n for i in range(1, n)])]

        df = data[dates].unstack().reset_index()
        data["event"] = event
        df = pd.merge(df, data.reset_index()[["origin_index", "event"]],
                      on="origin_index")
        df = df.rename({"level_0": "dates", 0: "survival"}, axis=1)
        df["dates"] = df["dates"].dt.year

        g = sns.FacetGrid(df, row="dates", hue="event", aspect=5, height=2,
                          palette="RdBu_r")
        g.map(sns.histplot, "survival",
              fill=False, alpha=1, linewidth=2,
              bins=bins,
              element="step",
              # cumulative=True,
              stat="density", common_norm=False
              )
        plt.legend()

    def plot_tagged_sample(
            self, data: pd.DataFrame, n_sample=None,
            n_sample_pos=None, n_sample_neg=None,
    ):

        assert self.__supervised, "No supervision provided"
        corresp, event, duration, entry = self._decompose_unique(data,
                                                                 get_supervision=True)
        if n_sample is None and n_sample_neg is None and n_sample_pos is None:
            sample = corresp
        elif n_sample_pos is None and n_sample_neg is None:
            sample = corresp.sample(n_sample)
        elif n_sample is None:
            pos_sample = corresp[event.astype(bool)].sample(n_sample_pos)
            neg_sample = corresp[~event.astype(bool)].sample(n_sample_neg)
            sample = pd.concat((neg_sample, pos_sample), axis=0)
        else:
            raise ValueError("Error in argument specifications")
        index = sample.index

        plotting.tagged_curves(
            temporal_curves=sample,
            label=event.loc[index],
            time_event=duration.loc[index],
            event_censored_color=event_censored_color,
            event_observed_color=event_observed_color,
            add_marker=True,
        )
        return index

    def plotly_auc_vs_score(self, temporal_scores: pd.DataFrame,
                            temporal_metric: pd.Series, **kwargs):
        corresp, event, duration, entry = self._decompose_unique(
            temporal_scores, get_supervision=True)
        time_start, time_stop = duration.min(), duration.max()
        corresp = corresp[
            [c for c in corresp.columns if time_start <= c <= time_stop]]
        return plotting.auc_vs_score_plotly(temporal_scores=corresp,
                                            event_observed=event,
                                            censoring_time=duration,
                                            entry_time=entry,
                                            temporal_metric=temporal_metric,
                                            **kwargs)

    @staticmethod
    def plotly_roc_curve(roc: dict, colormap="magma_r", template="plotly_white",
                         **kwargs):
        return plotting.plot_roc_curve_plotly(roc, colormap=colormap,
                                              template=template, **kwargs)

    # ==============================================
    #           METRIC ASSESSMENT
    # ==============================================
    def assess_metric(self, data, method="roc-cd",
                      metric=metrics.roc_auc_score):

        corresp, event, duration, _ = self._decompose_unique(data,
                                                             get_supervision=True)

        time_start, time_stop = duration.min(), duration.max()
        corresp = corresp[
            [c for c in corresp.columns if time_start <= c <= time_stop]]
        return metric.time_dependent_helper(
            corresp,
            event_observed=event,
            censoring_time=duration,
            method=method,
            function=metric
        )

    # ==============================================
    #           Properties
    # ==============================================
    @property
    def curves(self) -> pd.DataFrame:
        return self.__curves


def get(data: pd.DataFrame, date):
    search = np.searchsorted(data.columns.to_list(), date)
    return data.iloc[:, search]


class SurvivalEstimation:
    """
    Instantiate class with data as survival curves. Columns must be timedelta objects
    """

    def __init__(self,
                 survival_curves: pd.DataFrame,
                 unit='D', n_unit=1.,
                 process_input_data=True):

        survival_curves = survival_curves.__deepcopy__()
        if process_input_data:
            survival_curves = process_survival_function(survival_curves)

        self.survival = TimeCurve(survival_curves, unit='D', n_unit=1.)

        self.unit = pd.to_timedelta(n_unit, unit=unit).total_seconds()
        self.__check_args()

        self.times = self.survival.columns.to_numpy()

        self.derivative = self.survival.derivative()

        self.hazard: TimeCurve = - self.derivative / self.survival
        self.cumulative_hazard: TimeCurve = - self.survival.log()

        self.__survival = self.survival.__deepcopy__()
        self.__survival.columns = pd.to_timedelta(
            self.__survival.columns).total_seconds() / self.unit

        self.residual_life = TimeCurve(residual_life(self.__survival), unit='D', n_unit=1.)
        self.residual_life.columns = self.survival.columns
        self.residual_life[
            self.residual_life > 1e5] = self.residual_life.columns.max()

        self.residual_survival = None
        # CREATE INTERPOLATION TODO : remove
        self.hazard_interp = self.hazard.interpolation
        self.cumulative_hazard_interp = self.cumulative_hazard.interpolation
        self.residual_life_interp = self.residual_life.interpolation

    def plot_residual_life(self, sample=None, mean_behaviour=True):
        if not mean_behaviour:
            if sample is not None:
                residual_life_ = self.residual_life.sample(sample)
            else:
                residual_life_ = self.residual_life
            residual_life_.columns = pd.to_timedelta(
                self.survival.index).total_seconds() / self.unit
            color = plt.get_cmap("magma_r")(
                residual_life_.iloc[:, 0] / residual_life_.iloc[:, 0].max())

            residual_life_.T.plot(legend=False, ax=plt.gca(), color=color,
                                  lw=0.15, alpha=1)
            plt.grid()
            bounds = residual_life_.columns.min(), residual_life_.columns.max()
            plt.plot(bounds, bounds[::-1], c="k", alpha=0.5, lw=2, ls="--")
        else:
            des = self.residual_life.astype("float32").describe(
                [0.05, .25, .5, .75, 0.95])
            des.columns = pd.to_timedelta(
                self.survival.columns).total_seconds() / self.unit

            des = des.drop(["count", "mean", "std"])
            for i, d in enumerate(des.index):
                c = plt.get_cmap("crest")(i / (len(des) - 1))
                des.loc[d].plot(color=c, ax=plt.gca())
                if i > 0:
                    plt.fill_between(des.columns.to_list(), des.iloc[i],
                                     des.iloc[i - 1], alpha=0.2, facecolor=c)

            plt.grid()
            plt.legend()
            plt.ylabel("Expected residual lifespan")
            plt.xlabel("Time")

    def __check_args(self):
        if hasattr(self.survival.columns, "dt"):
            ValueError("curves index must have dt property")
