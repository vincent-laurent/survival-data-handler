import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics

from survival_estimation import Lifespan, test_is_survival_curves, \
    event_censored_color


def complete(
        life: Lifespan,
        fig_path,
        t0,
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

    plt.figure(figsize=(8, 6), dpi=250)
    life.plot_average_tagged(life.survival_function)
    plt.savefig(f"{fig_path}figure_mean_survival")

    # 2) Expected residual life
    plt.figure(figsize=(8, 6), dpi=250)
    life.plot_curves_residual_life(mean_behaviour=False, sample=None)
    plt.grid(True)
    plt.savefig(f"{fig_path}figure_residual_life")

    plt.figure(figsize=(8, 6), dpi=250)
    life.plot_curves_residual_life(mean_behaviour=False, sample=10)
    plt.grid(True)
    plt.savefig(f"{fig_path}figure_residual_life_sample")

    plt.figure(figsize=(8, 6), dpi=250)
    life.residual_lifespan.sample(30).T.plot(legend=False,
                                             color=event_censored_color,
                                             lw=1)
    plt.grid(True)
    plt.xlabel("Date [Y]")
    plt.ylabel("Age [Y]")
    plt.savefig(f"{fig_path}figure_residual_life_sample")

    # 3) confusion matrix
    matrix = life.compute_confusion_matrix(on="survival_function",
                                           threshold=0.90)

    matrix.to_csv(f"{fig_path}data_confusion_matrix.csv")

    # 3.5) metrics
    plt.figure(figsize=(8, 6), dpi=250)
    for s in ["precision", "recall", "accuracy", "f1-score", "f2-score"]:
        plt.plot(matrix[s], label=s)
    plt.grid(True)
    plt.ylabel("Metrics")
    plt.xlabel("Date [Y]")
    plt.legend()
    plt.savefig(f"{fig_path}figures_metrics")

    # 4) ROC CURVE
    roc = life.assess_metric(data=1 - life.survival_function,
                             metric=metrics.roc_curve)
    roc_under_sample = {}
    for k, (fpr, tpr, ths) in roc.items():
        idx = range(0, len(fpr), 50)
        roc_under_sample[k.date()] = fpr[idx], tpr[idx], ths[idx]
    fig = life.plotly_roc_curve(roc_under_sample, colormap="magma_r",
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
