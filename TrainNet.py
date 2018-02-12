from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import PARAMETERS
from pandas import read_csv
import pickle

DROP = PARAMETERS.DROP
smootheddata = PARAMETERS.smoothedtrain
Net_PATH = PARAMETERS.Netsaved_PATH
TRAINDATA_PATH = PARAMETERS.TRAINDATA_PATH
HISTORY_PATH = PARAMETERS.HISTORYsaved_PATH
BATCH_SIZE = PARAMETERS.BATCH_SIZE
NUM_EPOCH = PARAMETERS.NUM_EPOCH
NUM_FEATURE = PARAMETERS.NUM_FEATURE
NUM_NEURAL = PARAMETERS.NUM_NEURAL
NUM_DENSE = PARAMETERS.NUM_DENSE
# define the delay and predict number ahead
delay = PARAMETERS.DELAY
predict = PARAMETERS.PREDICT

# load dataset
#values0 = PARAMETERS.loaddata(TRAINDATA_PATH)
values0 = read_csv(smootheddata,index_col=None, header=0) # smoothed data = [Ft, Fs, D]
values0 = values0.values

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values0)
# frame as supervised learning
reframed = PARAMETERS.series_to_supervised(scaled, delay, predict)
# drop columns we don't want to predict
if DROP == True:
    reframed.drop(reframed.columns[-predict*NUM_FEATURE:-NUM_FEATURE], axis=1, inplace=True)
    predict = 1
# split into train and test sets
values = reframed.values
train = values[:int(0.6*values.shape[0]), :]
test = values[int(0.6*values.shape[0]):, :]
# split into input and outputs
train_X0, train_y = train[:, :delay*NUM_FEATURE], train[:, delay*NUM_FEATURE:]
test_X0, test_y = test[:, :delay*NUM_FEATURE], test[:, delay*NUM_FEATURE:]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X0.reshape((train_X0.shape[0], delay, NUM_FEATURE))
test_X = test_X0.reshape((test_X0.shape[0], delay, NUM_FEATURE))
#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(NUM_NEURAL, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(NUM_DENSE))
model.add(Dense(predict*NUM_FEATURE))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=NUM_EPOCH, batch_size=BATCH_SIZE, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
model.save(Net_PATH)
with open(HISTORY_PATH, 'wb') as handle:
    pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
# read the history
#with open(HISTORY_PATH, 'rb') as handle:
#   b = pickle.load(handle)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

