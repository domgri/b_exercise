
import bentoml
import numpy as np
import pandas as pd
from bentoml.io import Image, NumpyNdarray, JSON
from sklearn.preprocessing import MinMaxScaler
from pydantic import BaseModel
from pandas import DataFrame
from pandas import concat
from numpy import concatenate

# create a runner for a predictions using just trained model
runner = bentoml.tensorflow.get("tensorflow_lstm:latest").to_runner()

# create a service endpoint
svc = bentoml.Service(
    name="tensorflow_lstm_demo",
    runners=[runner],
)

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  n_vars = 1 if type(data) is list else data.shape[1]
  df = DataFrame(data)
  cols, names = list(), list()
  # input sequence (t-n, ... t-1)
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
  # forecast sequence (t, t+1, ... t+n)
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # put it all together
  agg = concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg

@svc.api(input=JSON(), output=NumpyNdarray(dtype="float32"))
async def predict30(input_data: JSON) -> "np.ndarray":

    # Ingest data
    data = pd.DataFrame(input_data).T
    data = data.drop(['date'], axis=1)

    values = data.values

    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # Reshape
    reframed = series_to_supervised(scaled, 0, 30)

    # drop every column except revenue for prediction
    list_to_drop = list(range(len(reframed.columns) // 2, len(reframed.columns)))
    del list_to_drop [52-1::52]
    reframed.drop(reframed.columns[list_to_drop], axis=1, inplace=True)


    # prepare data for model
    values= reframed.values
    test = values[:len(data), :]
    test_X, test_y = test[:, :-1], test[:, -1]
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # predict
    yhat  = await runner.async_run(test_X)

    # transform and retreive results
    results = []
    for i in range(30):
        test_X[0, 0, -1] = yhat[0, i, 0]
        result_day = scaler.inverse_transform(test_X[:, 0, -52:])
        results.append(result_day[0, -1])

    return np.array(results)

