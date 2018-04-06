from matplotlib import pyplot
#from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import PARAMETERS
import pickle
import ArtificialSignal
from numpy import sqrt
from sklearn.metrics import mean_squared_error
from numpy import std
from numpy import amax
from numpy import abs


DROP = True
NUM_FEATURE = 2


BATCH_SIZE = 30
NUM_EPOCH = [300, 400, 500, 600]

NUM_LSTM = [50, 60, 70, 80]
# NUM_DENSE =
# define the delay and predict number ahead
HISTORY = [50, 60, 70, 80]
PRIDICT = 20

HISTORYloaded_PATH = './TrainedNetwork/TrainedNN_vesselfollow-0222/history_2out{pre}ms_noscale_nosmooth{neu}-{des}n-{dly}d-35e-400b.pickle'
Net_PATH = './TrainedNetwork/TrainedNN_vesselfollow-0222/LSTMnet_2out{pre}ms_noscale_nosmooth{neu}-{des}n-{dly}d-35e-400b.h5'

# load dataset
preriodic = ArtificialSignal.PeriodicSingnal(500)
scaled = preriodic
# frame as supervised learning
reframed = PARAMETERS.series_to_supervised(scaled, HISTORY, PRIDICT)
# drop columns we don't want to predict
if DROP == True:
    reframed.drop(reframed.columns[-PRIDICT*NUM_FEATURE:-NUM_FEATURE], axis=1, inplace=True)
    PRIDICT = 1
# split into train and test sets
values = reframed.values
train = values[:int(0.7*values.shape[0]), :]
test = values[int(0.7*values.shape[0]):, :]
# split into input and outputs
train_X0, train_y = train[:, :HISTORY*NUM_FEATURE], train[:, HISTORY*NUM_FEATURE:]
test_X0, test_y = test[:, :HISTORY*NUM_FEATURE], test[:, HISTORY*NUM_FEATURE:]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X0.reshape((train_X0.shape[0], HISTORY, NUM_FEATURE))
test_X = test_X0.reshape((test_X0.shape[0], HISTORY, NUM_FEATURE))
# design network
model = Sequential()
model.add(LSTM(NUM_LSTM, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(PRIDICT*NUM_FEATURE))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=NUM_EPOCH, batch_size=BATCH_SIZE, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
model.save(Net_PATH)
with open(HISTORY_PATH, 'wb') as handle:
    pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
#evaluation
values0 = ArtificialSignal.PeriodicSingnal(10)
reframed = PARAMETERS.series_to_supervised(values0, HISTORY, PRIDICT)
# drop columns we don't want to predict
if DROP == True:
    reframed.drop(reframed.columns[PRIDICT*NUM_FEATURE:-NUM_FEATURE], axis=1, inplace=True)
values = reframed.values
# split into input and outputs
groundtruth_X, groundtruth_y = values[:, :HISTORY * NUM_FEATURE], values[:, HISTORY * NUM_FEATURE:]
# reshape input to be 3D [samples, timesteps, features]
groundtruth_X = groundtruth_X.reshape((groundtruth_X.shape[0], HISTORY, NUM_FEATURE))
yhat = model.predict(groundtruth_X)
# calculate RMSE
rmse = sqrt(mean_squared_error(groundtruth_y, yhat, multioutput='raw_values'))
error = groundtruth_y -yhat
error_std = std(error, axis= 0)
maxerror = amax(abs(error), axis=0)
print('rmse', rmse)
print('error_std', error_std)
print('maxerror', maxerror)
#plot
xx = [x for x in range(len(groundtruth_y[:,0]))]
fig = pyplot.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(xx, groundtruth_y[:, 0], c='r',  label='Fs_groundtruth')
ax1.scatter(xx, yhat[:, 0], s=4, c='m', marker='o',  label='Fs_prediction')
ax1.set_xlabel('Time(s)')
ax1.set_ylabel('Force(mN)')
ax1.legend(loc=2, markerscale=2)
ax2 = ax1.twinx()
ax2.plot(xx, groundtruth_y[:, 1], c='b', label='D_groundtruth')
ax2.scatter(xx, yhat[:, 1], s=4, c='k', marker='o', label='D_prediction')
ax2.set_ylabel('Depth(mm)')
ax2.legend(loc=1, markerscale=2)