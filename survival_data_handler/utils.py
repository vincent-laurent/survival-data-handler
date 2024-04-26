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


from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def test_is_survival_curves(data: pd.DataFrame):
    assertion_1 = (data.diff(axis=1) > 0).sum().sum() == 0
    assertion_2 = np.sum((data > 1) | (data < 0)).sum() == 0
    assert assertion_1, "Survival function should be decreasing"
    assert assertion_2, "Survival function should be in [0, 1]"


def smooth(data: pd.DataFrame, freq) -> pd.DataFrame:
    ret = data.T.resample(freq).interpolate(method='linear', order=3)
    ret = pd.DataFrame(
        savgol_filter(ret, window_length=20, polyorder=3, axis=0),
        index=ret.index,
        columns=ret.columns)
    return ret.T


def process_survival_function(data: pd.DataFrame) -> pd.DataFrame:
    data = data.__deepcopy__()
    data[data.columns[0]] = 1
    data[data <= 0] = 0

    # target non decreasing lines
    condition = pd.DataFrame(data.diff(axis=1) > 0)
    crit_lines = condition.sum(axis=1) > 0
    data_correct = data.loc[crit_lines]

    # apply correction
    data_correct = _cut(data_correct.astype("float32").values).astype(
        "float16")
    data.loc[crit_lines] = data_correct
    return data


def compute_derivative(data: pd.DataFrame,
                       unit) -> pd.DataFrame:
    times = data.columns.to_numpy()
    diff = data.T.diff()
    return diff.divide(
        pd.Series(times, index=times).dt.total_seconds().diff() / unit,
        axis=0
    ).T



def _cut(x):
    y = x.copy()
    for i in range(0, y.shape[1] - 1):
        y[:, i] = np.nanmin(y[:, :i + 1], axis=1)
    return y


def residual_life(survival_estimate: pd.DataFrame,
                  precision="float32") -> pd.DataFrame:
    """
       Compute mean residual lifespan according to doi:10.1016/j.jspi.2004.06.012.

       Parameters:
       - survival_estimate (pd.DataFrame): Matrix of the estimate of the survival function.
       - precision (str): Data type precision for calculations (default is "float32").

       Returns:
       pd.DataFrame: Mean residual lifespan computed using the specified formula.

       Notes:
       - The function calculates the mean residual lifespan based on the provided survival estimates.
       - The input DataFrame should represent the estimate of the survival function.
       - The result is a DataFrame containing the mean residual lifespan values.

       Example:
       ```python
       import pandas as pd
       import numpy as np

       # Example usage
       survival_data = pd.DataFrame(...)  # Provide survival function estimates
       result = residual_life(survival_data)
       ```

       References:
       - doi:10.1016/j.jspi.2004.06.012
    """
    deltas = np.diff(survival_estimate.columns.values)

    # days = deltas.astype('timedelta64[D]').astype(int) / 365.25
    dt = 1.*np.ones((len(survival_estimate), 1), dtype=precision) * deltas.reshape(
        1, -1)

    surv_int = survival_estimate.__deepcopy__().astype(float)

    s_left = survival_estimate.iloc[:, 1:].values.astype(float)
    s_right = survival_estimate.iloc[:, :-1].values.astype(float)
    t0 = survival_estimate.columns[0]
    if "time" in str(dt.dtype):
        dt = dt / np.timedelta64(1, 's')
        t0 /= np.timedelta64(1, 's')

    surv_int.iloc[:, :-1] = (s_left + s_right) / 2.
    surv_int.iloc[:, :-1] *= dt
    surv_int = surv_int.drop(surv_int.columns[-1], axis=1)

    surv_int = pd.DataFrame(
        np.cumsum(surv_int[np.sort(surv_int.columns)[::-1]], axis=1),
        index=surv_int.index,
        columns=surv_int.columns)

    ret = (surv_int / survival_estimate).astype(precision)
    ret[survival_estimate == 0] = 0
    return ret - t0


class _PoolShift:
    def __init__(self, dates, starting_dates, data):
        self.dates = dates
        self.starting_dates = starting_dates.values
        self.starting_dates_index = starting_dates.index.to_numpy()
        self.data = data.values
        self.data_index = data.index.to_numpy()

    def __call__(self, index):
        sd = self.starting_dates[index == self.starting_dates_index][0]
        f = self.data[index == self.data_index, 0][0]
        dates = self.dates - sd
        return f(dates.astype(int))


def shift_from_interp(
        data: pd.DataFrame,
        starting_dates: pd.Series,
        period,
        date_final,
        date_initial,
        dtypes="float16"
):
    range_ = np.arange(
        starting_dates.min(),
        date_final + period,
        step=period)

    idx_end, idx_sta = [np.searchsorted(range_, d)
                        for d in (date_final, date_initial)]
    df_data_date = pd.DataFrame(
        index=data.index, columns=range_[idx_sta:idx_end], dtype=dtypes)
    dates = pd.to_datetime(df_data_date.columns.to_numpy())
    try:
        solve1 = _PoolShift(dates, starting_dates, data)
        with Pool(5) as p:
            args = df_data_date.index.to_list()
            results = p.map(solve1, args)
        df_data_date.loc[:, :] = np.array(results)
    except:
        for index in df_data_date.index:
            sd = starting_dates.loc[index]
            f = data.loc[index, "interp"]
            dates = pd.to_datetime(df_data_date.columns.to_numpy()) - sd
            df_data_date.loc[index] = f(dates.astype(int))

    return df_data_date
