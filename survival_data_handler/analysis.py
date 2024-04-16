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


from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics

from survival_data_handler import Lifespan, test_is_survival_curves, \
    event_censored_color


def complete(
        life: Lifespan,
        fig_path,
        t0,
        starting_date
):
    life.compute_survival()
    life.compute_residual_survival(t0=t0)
    life.compute_expected_residual_life()
    life.compute_percentile_life(0.9)
    life.survival_function = life.survival_function.astype("float32").T.fillna(
        method="pad").T.astype("float16")

    # Test if everything is ok
    test_is_survival_curves(data=life.survival_function)
    test_is_survival_curves(data=life.survival_estimation.survival.T)

    # 1) plot curve sample
    plt.figure(figsize=(8, 6), dpi=250)
    life.plot_tagged_sample(
        data=life.survival_function,
        n_sample_neg=100,
        n_sample_pos=10)
    plt.xlabel("Year [Y]")
    plt.ylabel("Nelson Aalen estimator")
    plt.savefig(f"{fig_path}figure_survival_sample")

    plt.figure(figsize=(8, 6), dpi=250)
    life.plot_tagged_sample(
        data=life.survival_function,
        n_sample_neg=100,
        n_sample_pos=20)
    plt.xlabel("Year [Y]")
    ax = plt.gca()
    ax.set_yscale('function', functions=(partial(np.power, 10.0), np.log10))
    ax.set_ylim((0, 1))
    plt.ylabel("Nelson Aalen estimator")
    plt.savefig(f"{fig_path}figure_survival_sample_exp")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=250)
    df, event, duration, entry = life._decompose_unique(
        life.survival_function, get_supervision=True)
    pos_sample = df[event.astype(bool)].astype("float32").mean().rolling(10).mean()
    neg_sample = df[~event.astype(bool)].astype("float32").mean().rolling(10).mean()
    ax.plot(pos_sample.index, pos_sample.values, color="r")
    ax.plot(neg_sample.index, neg_sample.values, color=event_censored_color)

    plt.xlabel("Year [Y]")
    plt.ylabel("Nelson Aalen estimator")
    plt.savefig(f"{fig_path}figure_mean_survival_behaviour")

    fig, (ax1, ax2) = plt.subplots(figsize=(12, 6), dpi=250, ncols=2)
    plt.sca(ax2)
    life.plot_tagged_sample(
        life.survival_function,
        n_sample_neg=100,
        n_sample_pos=10)
    plt.xlabel("Date [Y]")
    plt.sca(ax1)
    df_plot = life.survival_estimation.survival.T.sample(110).T
    df_plot.index = df_plot.index.days / 365.25
    df_plot.plot(color=event_censored_color, ax=ax1, legend=False, lw=1)
    plt.ylabel("Nelson Aalen estimator")
    plt.xlabel("Year [Y]")
    plt.savefig(f"{fig_path}figure_survival_sample_time_shift")

    fig, (ax1, ax2) = plt.subplots(figsize=(8, 6), dpi=250, ncols=2,
                                   sharex=True, sharey=True)

    plt.sca(ax1)
    life.plot_average_tagged(life.survival_function, event_type="censored")
    plt.xlabel("Year [Y]")
    plt.ylabel("Nelson Aalen estimator")
    plt.sca(ax2)
    life.plot_average_tagged(life.survival_function, event_type="observed")
    plt.xlabel("Year [Y]")
    plt.savefig(f"{fig_path}figure_mean_survival")

    # 2) Expected residual life
    plt.figure(figsize=(8, 6), dpi=250)
    life.plot_curves_residual_life(mean_behaviour=False, sample=None)
    plt.grid(True)
    plt.savefig(f"{fig_path}figure_residual_life")

    plt.figure(figsize=(8, 6), dpi=250)
    life.residual_lifespan.sample(30).T.plot(legend=False,
                                             color=event_censored_color,
                                             lw=1, ax=plt.gca())
    plt.grid(True)
    plt.xlabel("Date [Y]")
    plt.ylabel("Age [Y]")
    plt.savefig(f"{fig_path}figure_residual_life_sample")

    life.plot_dist_facet_grid(life.survival_function)
    plt.savefig(f"{fig_path}figure_time_dependent_hist")

    # 3) ROC CURVE
    roc = life.assess_metric(data=1 - life.survival_function,
                             metric=metrics.roc_curve)
    roc_under_sample = {}
    for k, (fpr, tpr, ths) in roc.items():
        if k.date() > starting_date:
            idx = np.linspace(0, len(fpr) - 1, 100, dtype=int)
            roc_under_sample[k.date()] = fpr[idx], tpr[idx], ths[idx]

    fig = life.plotly_roc_curve(
        roc_under_sample,
        colormap="rocket",
        marker_size=8)
    fig.update_layout(
        autosize=False,
        width=630,
        height=500)
    fig.write_html(f"{fig_path}figure_roc_curve.html")
    fig.write_image(f"{fig_path}figure_roc_curve.png", scale=5)

    # 5) AUC
    plt.figure(dpi=200)
    auc = pd.Series(life.assess_metric(data=1 - life.survival_function))
    auc.plot()
    plt.grid(True)
    plt.savefig(f"{fig_path}figure_auc")
    auc.to_csv(f"{fig_path}data_auc.csv")

    fig = life.plotly_auc_vs_score(
        temporal_scores=life.survival_function.sample(500), temporal_metric=auc)
    fig.update_layout(
        autosize=False,
        width=800,
        height=800, )
    fig.write_html(f"{fig_path}figure_auc_vs_score.html")
    fig.write_image(f"{fig_path}figure_auc_vs_score.png", scale=5)

    # 4) confusion matrix
    matrix = life.compute_confusion_matrix(on="survival_function",
                                           threshold=0.90)

    matrix.to_csv(f"{fig_path}data_confusion_matrix.csv")

    # 4.5) metrics
    plt.figure(figsize=(8, 6), dpi=250)
    for s in matrix.drop(
            columns=["TN", "FP", "FN", "TP", "P", "N", "Total"]).columns:
        plt.plot(matrix[s], label=s)
    plt.grid(True)
    plt.ylabel("Metrics")
    plt.xlabel("Date [Y]")
    plt.legend()
    plt.savefig(f"{fig_path}figures_metrics")

    # =========================================================================
    #                       Plot results
    # =========================================================================

    q_life, event, duration, _ = life._decompose_unique(life.percentile_life,
                                                        get_supervision=True)

    q_life["event"] = event
    q_life["observed event"] = q_life["event"].values.astype(bool)

    plt.figure(dpi=150)

    sns.kdeplot(q_life, x=q_life.columns[0],
                common_norm=False, label="Distribution for all individuals",
                palette="RdBu_r")
    time_start, time_stop = life.durations.min(), life.durations.max()
    sns.kdeplot(q_life[q_life["observed event"]], x=q_life.columns[0],
                common_norm=False,
                color="red", label=f"For observed event between"
                                   f" \n{time_start.date()} and {time_stop.date()}")

    plt.gca().set_xlabel("Date")
    plt.legend()
    plt.savefig(f"{fig_path}figure_percentile_life.png")
