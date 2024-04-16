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
                 risks: tuple = (0.96, 0.98),
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

        self.risk_level = risks
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
    #           ENRICH REPRESENTATION
    # ==============================================

    def compute_expected_residual_life(self) -> None:
        data = pd.Series(self.survival_estimation.residual_life_interp)
        data.name = "interp"
        data = self._shift_helper(data)
        self.residual_lifespan = self._decompose_unique(data)[0].astype(
            self.__p)
        self.__computed.append("residual_lifespan")

    def compute_survival(self):
        data = pd.Series(self.survival_estimation.survival_interp)
        data.name = "interp"
        data = self._shift_helper(data)
        self.survival_function = self._decompose_unique(data)[0].astype(
            self.__p)
        self.survival_function[self.survival_function > 1] = np.nan
        self.survival_function[self.survival_function < 0] = 0
        self.__computed.append("survival_function")

    def compute_residual_survival(self, t0) -> None:
        assert "survival_function" in self.computed_historical_data

        t0 = max(t0, self.survival_function.columns[0])
        s0 = get(self.survival_function, t0).astype("float16").values
        self.residual_survival = self.survival_function.divide(s0,
                                                               axis=0).copy()
        self.residual_survival = self.residual_survival[
            [c for c in self.residual_survival.columns if c > t0]]
        self.__computed.append("residual_survival")

    def compute_times(self, on="residual_survival"):
        assert on in self.computed_historical_data
        low, upp = self.risk_level
        cond_low = (self.__getattribute__(on) > low).sum(axis=1)
        cond_upp = (self.__getattribute__(on) > upp).sum(axis=1)
        dates = self.__getattribute__(on).columns.to_numpy()

        df_time = self._df_idx_unique.__deepcopy__()

        df_time.loc[cond_upp.index, "surveillance_time"] = dates[
            np.where(cond_upp.values >= len(dates), len(dates) - 1,
                     cond_upp.values)]

        df_time.loc[cond_low.index, "critical_time"] = dates[
            np.where(cond_low.values >= len(dates), len(dates) - 1,
                     cond_low.values)]
        # remove past positive sample
        self.df_time, _, _, _ = self._decompose_unique(df_time)
        return self.df_time

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

    def compute_percentile_life(self, q: float = 0.5) -> pd.DataFrame:
        """
        Compute the percentile lifetime corresponding to the proportion of
        the population (in terms of curves) that remains
        """
        min_ = pd.DataFrame(self.survival_function > q)
        cs = min_.cumsum(axis=1)
        self.percentile_life = pd.DataFrame(
            np.array(cs.columns.to_list())[cs.iloc[:, -1].values - 1],
            columns=["percentile_life_%s" % q]
        )
        del min_, cs
        return self.percentile_life

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

    def plot_cumulated_times(self):
        times = self.df_time
        df_plot = times[["temps_surveillance",
                         "temps_critique"]].unstack().reset_index().sort_values(
            'level_0')
        df_plot.columns = ["type", "0", "temps"]

        fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
        histplot = sns.histplot(df_plot, x="temps", hue="type", element="step",
                                fill=False, cumulative=True, bins=50,
                                ax=ax)
        ax.set_xlabel("Date [Y]")
        ax.set_ylabel("Length of rail [Km]")

        plt.tight_layout()
        ylb = [
            format(int(int(elt.get_text()) * 108 / 1000), ",").replace(",", " ")
            for elt in
            histplot.get_yticklabels()]
        ax.set_yticklabels(ylb)
        plt.grid()

    @staticmethod
    def plot_sample(data, n_sample):
        plt.plot(data)

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
        self.survival = survival_curves

        self.times = self.survival.index.to_numpy()
        self.unit = pd.to_timedelta(n_unit, unit=unit).total_seconds()

        self.derivative = compute_derivative(self.survival, self.unit)
        self.hazard = - self.derivative / self.survival
        self.cumulative_hazard = - np.log(self.survival)

        self.__survival = self.survival.__deepcopy__()
        self.__survival.columns = pd.to_timedelta(
            self.__survival.columns).total_seconds() / self.unit
        self.residual_life = residual_life(self.__survival)
        self.residual_life.columns = self.survival.columns
        self.residual_life[
            self.residual_life > 1e5] = self.residual_life.columns.max()
        self.residual_survival = None
        # CREATE INTERPOLATION
        self.hazard_interp = {
            c: interp1d(self.hazard.index.astype(int),
                        self.hazard[c], fill_value="extrapolate") for c in
            self.hazard.columns
        }
        self.survival_interp = {
            c: interp1d(self.survival.index.astype(int),
                        self.survival[c], fill_value="extrapolate") for c in
            self.survival.columns
        }
        self.cumulative_hazard_interp = {
            c: interp1d(self.cumulative_hazard.index.astype(int),
                        self.cumulative_hazard[c], fill_value="extrapolate") for
            c in self.cumulative_hazard.columns
        }
        self.residual_life_interp = {
            c: interp1d(self.residual_life.T.index.astype(int),
                        self.residual_life.T[c], fill_value="extrapolate") for
            c in self.residual_life.T.columns
        }

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

class Curves:
    pass