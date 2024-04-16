import pandas as pd
import pytest
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

from survival_data_handler.main import SurvivalEstimation
from survival_data_handler.utils import smooth, process_survival_function, \
    compute_derivative


@pytest.fixture()
def data():
    rossi = load_rossi()
    cph = CoxPHFitter()
    cph.fit(rossi, duration_col='week', event_col='arrest')
    curves = cph.predict_survival_function(rossi).T
    curves.columns = pd.to_timedelta(curves.columns.to_numpy() * 7, unit="D")
    return rossi, curves


def test_utils_smooth(data):
    _, curves = data
    ret = smooth(curves, freq="7D")
    assert all(abs(ret - curves) < 1)


def test_utils_process(data):
    _, curves = data
    ret = process_survival_function(curves)
    assert all(pd.DataFrame(ret == curves))
    curves.iloc[:, 1] = 1
    ret = process_survival_function(curves)
    assert any(pd.DataFrame(ret == curves))


def test_utils_compute_derivative(data):
    _, curves = data
    unit = pd.to_timedelta(7, unit="D").total_seconds()
    ret = compute_derivative(curves, unit)
    assert all(pd.DataFrame(ret < 0))


def test_survival_estimation(data):
    _, curves = data
    se = SurvivalEstimation(
        curves.drop_duplicates(), unit='D',
        n_unit=365.25)
    se.plot_residual_life()


def test_lifespan():
    rossi = load_rossi()
