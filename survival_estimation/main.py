from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numba import njit, jit
from scipy.signal import savgol_filter
from sklearn import metrics

# ======================================================================================================================
# AESTHETICS
cmap = sns.color_palette('rainbow', as_cmap=True)
event_censored_color = cmap(0.1)
event_observed_color = cmap(0.95)


# ==============================================================================
#                          FOR A NEW PACKAGE
# ==============================================================================
def shift_origin(data: pd.DataFrame,
                 starting_dates: pd.Series,
                 period,
                 date_final,
                 date_initial,
                 dtypes="float16"
                 ):
    """
    Given a temporal dataframe, this function shift
    the origin
    """
    range_ = np.arange(
        starting_dates.min(),
        date_final + period,
        step=period)
    data = data[data.columns.sort_values()]

    idx_end, idx_sta = [np.searchsorted(range_, d)
                        for d in (date_final, date_initial)]

    df_data_date = pd.DataFrame(
        index=data.index, columns=range_[idx_sta:idx_end], dtype=dtypes)
    columns_save = df_data_date.columns

    s = np.sort(starting_dates.dropna().unique()).astype(int)
    s_dates = starting_dates.values.astype(int)
    columns = df_data_date.columns.astype(int).to_numpy()
    df_data_date.columns = columns
    datetime = np.array([int(v.asm8) for v in data.columns.values])
    values = data.values
    ret = find_date(s, s_dates, columns, datetime, values)

    for t0, (loc, select_col, v) in ret.items():
        df_data_date.loc[
            loc, select_col] = v
    df_data_date.columns = columns_save
    return df_data_date


@njit
def find_date(s, s_dates, columns, datetime, values):
    ret = {}
    for d in s:
        loc = s_dates == d
        idx_start = np.argmin(np.abs((columns - d)))
        t0 = (columns - d)[0]
        select_column_input = datetime > t0
        idx_stop = idx_start + sum(select_column_input) - 1
        if idx_start == -1:
            continue

        select_col = columns[idx_start:idx_stop]
        n_cols = len(select_col)
        v = values[loc][:, select_column_input][:, :n_cols]

        ret[t0] = loc, select_col, v
    return ret


class Lifespan:

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
                 smoothed: bool = False,
                 default_precision="float"):

        self.df_curves = ageing_curves
        self.df_idx = pd.DataFrame(
            dict(age=age,
                 birth=birth,
                 index=index))
        self._index = "origin_index"
        self.df_idx.index.name = self._index
        self.index = index
        self.age = age
        self.window = window
        self.__p = default_precision
        if smoothed:
            df_curves = SurvivalEstimation.smooth(self.df_curves.T,
                                                  freq="20D").astype(self.__p)
        else:
            df_curves = self.df_curves.T
        self.survival_estimation = SurvivalEstimation(
            df_curves, unit='D',
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

    def _shift_helper(self, data: pd.DataFrame):
        data = pd.merge(self._df_idx_unique["index"].reset_index(), data,
                        right_index=True,
                        left_on='index')
        data = data.set_index(self._index).drop("index", axis=1)
        ret = shift_origin(
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
        r = self.survival_estimation.residual_life
        data = self._shift_helper(r)
        self.residual_lifespan = self._decompose_unique(data)[0].astype(
            self.__p)
        self.__computed.append("residual_lifespan")

    def compute_survival(self):
        data = self.survival_estimation.survival.T
        data = self._shift_helper(data)
        self.survival_function = self._decompose_unique(data)[0].astype(
            self.__p)
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
        try:
            if all(tp + fp > 0):
                df_matrix["precision"] = precision = tp / (tp + fp)
            if all(tp + np > 0):
                df_matrix["recall"] = recall = tp / (tp + fn)
            df_matrix["accuracy"] = (tp + tn) / (p + n)

            if all(tp + np > 0) and all(tp + fp > 0):
                df_matrix["f1-score"] = 2 * recall * precision / (
                            recall + precision)
                df_matrix["f2-score"] = (1 + 2) ** 2 * recall * precision / (
                        2 ** 2 * precision + recall)
        except:
            pass
        return df_matrix

    # ==============================================
    #           PLOTS CURVES
    # ==============================================
    def plot_curves(self):
        plt.figure(dpi=200, figsize=(7, 5))
        self.df_curves.T.plot(legend=False)

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

    def plot_average_tagged(self, data: pd.DataFrame):
        corresp, event, duration, entry = self._decompose_unique(
            data,
            get_supervision=True)
        pos_sample = corresp[event.astype(bool)].astype("float32")
        neg_sample = corresp[~event.astype(bool)].astype("float32")

        pos_sample_m = pos_sample.mean(skipna=True)
        neg_sample_m = neg_sample.mean(skipna=True)
        pos_sample_s = pos_sample.std(skipna=True)
        neg_sample_s = neg_sample.std(skipna=True)

        plt.fill_between(
            neg_sample_m.index,
            neg_sample_m - neg_sample_s,
            neg_sample_m + neg_sample_s, color=event_censored_color, alpha=0.5)
        plt.plot(neg_sample_m.index, neg_sample_m, color=event_censored_color,
                 label="Censored event")

        plt.fill_between(
            pos_sample_m.index,
            pos_sample_m - pos_sample_s,
            pos_sample_m + pos_sample_s, color=event_observed_color, alpha=0.5)
        plt.plot(pos_sample_m.index, pos_sample_m, color=event_observed_color,
                 label="Observed event")

        t0 = duration[event.astype(bool)].min()
        t1 = duration[event.astype(bool)].max()

        plt.axvline(t0, lw=1.5, ls="--", color="k")
        plt.axvline(t1, lw=1.5, ls='--', color="k", label="Test window")
        plt.legend()

    def plot_tagged_sample(
            self, data: pd.DataFrame, n_sample=None,
            n_sample_pos=None, n_sample_neg=None,
    ):
        from survival_trees import plotting
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
        from survival_trees import plotting
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
        from survival_trees import plotting
        return plotting.plot_roc_curve_plotly(roc, colormap=colormap,
                                              template=template, **kwargs)

    # ==============================================
    #           METRIC ASSESSMENT
    # ==============================================
    def assess_metric(self, data, method="roc-cd",
                      metric=metrics.roc_auc_score):
        from survival_trees import metric as m
        corresp, event, duration, _ = self._decompose_unique(data,
                                                             get_supervision=True)

        time_start, time_stop = duration.min(), duration.max()
        corresp = corresp[
            [c for c in corresp.columns if time_start <= c <= time_stop]]
        return m.time_dependent_helper(
            corresp,
            event_observed=event,
            censoring_time=duration,
            method=method,
            function=metric
        )


def get(data: pd.DataFrame, date):
    search = np.searchsorted(data.columns.to_list(), date)
    return data.iloc[:, search]


@jit(parallel=True)
def cut(x):
    y = x.copy()
    for i in range(0, y.shape[1] - 1):
        y[:, i] = np.nanmin(y[:, :i + 1], axis=1)
    return y


class SurvivalEstimation:
    """
    Instantiate class with data as survival curves. Columns must be timedelta objects
    """

    @classmethod
    def compute_derivative(cls, data: pd.DataFrame, times,
                           unit) -> pd.DataFrame:
        return data.diff().divide(
            pd.Series(times, index=times).dt.total_seconds().diff() / unit,
            axis=0
        )

    @classmethod
    def smooth(cls, data: pd.DataFrame, freq) -> pd.DataFrame:
        ret = data.resample(freq).interpolate(method='linear', order=3)
        ret = pd.DataFrame(
            savgol_filter(ret, window_length=20, polyorder=3, axis=0),
            index=ret.index,
            columns=ret.columns)
        return ret

    @classmethod
    def process_survival(cls, data: pd.DataFrame):
        data[data.columns[0]] = 1
        data[data <= 0] = 0

        # target non decreasing lines
        condition = data.diff(axis=1) > 0
        crit_lines = condition.sum(axis=1) > 0
        data_correct = data.loc[crit_lines]

        # apply correction
        data_correct = cut(data_correct.astype("float32").values).astype(
            "float16")
        data.loc[crit_lines] = data_correct

    def __init__(self,
                 survival_curves: pd.DataFrame,
                 unit='D', n_unit=1.,
                 process_input_data=True):
        self.survival = survival_curves.__deepcopy__()
        if process_input_data:
            data_temp = self.survival.T
            self.process_survival(data_temp)
            self.survival = data_temp.T
        self.times = self.survival.index.to_numpy()
        self.unit = pd.to_timedelta(n_unit, unit=unit).total_seconds()
        self.derivative = self.compute_derivative(self.survival, self.times,
                                                  self.unit)
        self.hazard = - self.derivative / self.survival
        self.cumulative_hazard = - np.log(self.survival)

        self.__survival = self.survival.T.__deepcopy__()
        self.__survival.columns = pd.to_timedelta(
            self.__survival.columns).total_seconds() / self.unit
        self.residual_life = residual_life(self.__survival)
        self.residual_life.columns = self.survival.index
        self.residual_life[
            self.residual_life > 1e5] = self.residual_life.columns.max()
        self.residual_survival = None

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
                self.survival.index).total_seconds() / self.unit

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


def score(t_true: iter, t_pred: iter, observed: iter, score_function: callable):
    times = np.sort(np.unique(t_true))
    assert t_true.__len__() == t_pred.__len__(), TypeError(
        "The number of observation in prediction and ground truth does not correspond")
    ret = pd.Series(index=times)
    for t in times:
        y_true = (t_true > t) & observed
        y_pred = t_pred.loc[t]
        ret.loc[t] = score_function(y_true, y_pred)
    return ret


class Search:

    def __init__(self, fun):
        self.fun = fun

    @njit
    def __call__(self, support):
        fun = self.fun
        if len(support) == 1:
            return support[0]
        m = len(support) // 2
        left = fun(support[0])
        right = fun(support[-1])
        middle = fun(support[m])
        if left >= middle >= right:
            return self(support[:m])
        elif left <= middle <= right:
            return self(support[m:])
        else:
            return max(self(fun, support[m:]), self(support[:m]))


def test_is_survival_curves(data: pd.DataFrame):
    assertion_1 = (data.diff(axis=1) > 0).sum().sum() == 0
    assertion_2 = np.sum((data > 1) | (data < 0)).sum() == 0
    assert assertion_1, "Survival function should be decreasing"
    assert assertion_2, "Survival function should be in [0, 1]"


def find_threshold(y_true: np.ndarray, y_pred: np.ndarray,
                   target_metric: Union[Callable, str],
                   score: bool):
    """
    Given true and predicted values, compute decision rule's threshold which
    optimizes the target metric
    :param y_true:
    :param y_pred:
    :param target_metric:
    :param score:
    :return:
    """
    n = 100000
    if isinstance(target_metric, str):
        target_metric = getattr(metrics, target_metric)
    threshold = np.sort(np.unique(y_pred))

    if len(y_pred) > n:
        choice = np.random.choice(range(0, len(y_pred)), size=n)
    else:
        choice = range(len(y_pred))

    sign = 1 if score else -1

    def compute_metric(th):
        return sign * target_metric(
            np.array(y_true)[choice],
            np.array(y_pred)[choice] > th)

    return Search(compute_metric)(threshold)


def date_to_year(date):
    return date.value / 1e9 / 60 / 60 / 24 / 365.25 + 1970


def residual_life(survival_estimate: pd.DataFrame,
                  precision="float32") -> pd.DataFrame:
    """
    Compute mean residual lifespan according to
    doi:10.1016/j.jspi.2004.06.012
    :param survival_estimate: Matrix of the estimate of the survival function
    :return:
    """
    deltas = np.diff(survival_estimate.columns.values)

    # days = deltas.astype('timedelta64[D]').astype(int) / 365.25
    dt = np.ones((len(survival_estimate), 1), dtype=precision) * deltas.reshape(
        1, -1)

    surv_int = survival_estimate.__deepcopy__()

    s_left = survival_estimate.iloc[:, 1:].values
    s_right = survival_estimate.iloc[:, :-1].values

    surv_int.iloc[:, :-1] = (s_left + s_right) / 2
    surv_int.iloc[:, :-1] *= dt

    surv_int = surv_int[np.sort(surv_int.columns)[::-1]].cumsum(axis=1)
    ret = (surv_int / survival_estimate).astype(precision)
    ret[survival_estimate == 0] = 0
    del surv_int, dt, deltas
    return ret - survival_estimate.columns[0]
