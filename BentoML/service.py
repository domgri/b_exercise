
import bentoml
import numpy as np
import pandas as pd
from bentoml.io import Image, NumpyNdarray, JSON
from sklearn.preprocessing import MinMaxScaler
from pydantic import BaseModel

runner = bentoml.tensorflow.get("tensorflow_lstm:latest").to_runner()

svc = bentoml.Service(
    name="tensorflow_lstm_demo",
    runners=[runner],
)

@svc.api(input=JSON(), output=NumpyNdarray(dtype="float32"))
async def predict31(input_data: JSON) -> "np.ndarray":

    # Ingest data
    dataset = pd.DataFrame(input_data)
    dataset = dataset.T.values

    print(dataset.shape)

    # Normalize
    sc = MinMaxScaler(feature_range=(0, 1))
    scaled = sc.fit_transform(dataset)

    # Reshape
    test = scaled[:, :dataset.shape[1]-1]
    testX = np.array(test)
    testX = np.reshape(testX, (1, 31, dataset.shape[1]-1))

    testPredict = await runner.async_run(testX)

    testPredict_extended = np.zeros((31, dataset.shape[1]))
    testPredict_extended[:,dataset.shape[1] - 1] = testPredict[:, :, 0]
    testPredict = sc.inverse_transform(testPredict_extended)[:,dataset.shape[1] - 1]

    return testPredict 
