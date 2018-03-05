from numpy import sqrt
from numpy import array
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model
import pickle
import PARAMETERS
from numpy import tile
from pandas import read_csv
from numpy import amax
from numpy import abs
import time
from numpy import vstack
from numpy import std
from pandas import read_csv
import ArtificialSignal
import os
from numpy import concatenate

Fsbias = 8.5
Depthbias = 2.3
TwoOut = True
DROP = True
BIAS = False

#parameters in file name
predic = 100  #predict time(ms) in the file name
# define the delay and predict timestep
predict = predic/5
neural = '500n'
dense = 200
delay = 200
NUM_FEATURE = 2
figureID = 'ValidationForPaper'


TESTDATA_PATH = './data/NNmodelValidationData-0222/*.csv'
#LOAD MODEL
model = load_model(Net_PATH.format(pre = predic, neu = neural, des = dense, dly = delay))

Net_PATH = './TrainedNetwork/TrainedNN_vesselfollow-0222/'
lists = os.listdir(TESTDATA_PATH)
ERROR = array([[0],[0]])
for list in lists:
    dataset = read_csv(TESTDATA_PATH, index_col=None, header=0)
    dataset.dropna(inplace=True)
    values0 = dataset.values
    values0 = array([values0[:,8], values0[:,46]]).T

    # frame as supervised learning
    reframed = PARAMETERS.series_to_supervised(values0, delay, predict)
    # drop columns we don't want to predict
    if DROP == True:
        reframed.drop(reframed.columns[delay*NUM_FEATURE:-NUM_FEATURE], axis=1, inplace=True)

    values = reframed.values
    # split into input and outputs
    groundtruth_X, groundtruth_y = values[:, :delay * NUM_FEATURE], values[:, delay * NUM_FEATURE:]
    # reshape input to be 3D [samples, timesteps, features]
    groundtruth_X = groundtruth_X.reshape((groundtruth_X.shape[0], delay, NUM_FEATURE))
    yhat = model.predict(groundtruth_X)
    # invert scaling for prediction
    #inv_yhat = scaler.inverse_transform(yhat)
    if BIAS == True:
        inv_yhat = array([yhat[:,0]-Fsbias,yhat[:,1]-Depthbias]).T
    else:
        inv_yhat = yhat
    # invert scaling for groundtruth
    #inv_y = scaler.inverse_transform(groundtruth_y)
    inv_y = groundtruth_y
    error = inv_y - inv_yhat
    ERROR = concatenate([ERROR, error], axis=0)
# calculate RMSE
Fs_error = ERROR[0,:]

mae = sum(abs(error))/len(error)
error_std = std(error, axis= 0)
maxerror = amax(abs(error), axis=0)
print('rmse', rmse)
print('mae', mae)
print('error_std', error_std)
print('maxerror', maxerror)