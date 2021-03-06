from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
import PARAMETERS
from pandas import read_csv
import pickle
from kerasify import export_model

exportname = 'test.model'
MULTI_LSTM = PARAMETERS.MULTI_LSTM
singleFs = PARAMETERS.singleFs
DROP = PARAMETERS.DROP
smootheddata = PARAMETERS.smoothedtrain
Net_PATH = PARAMETERS.Netsaved_PATH
TRAINDATA_PATH = PARAMETERS.TRAINDATA_PATH
HISTORY_PATH = PARAMETERS.HISTORYsaved_PATH
BATCH_SIZE = PARAMETERS.BATCH_SIZE
NUM_EPOCH = PARAMETERS.NUM_EPOCH
NUM_FEATURE = PARAMETERS.NUM_FEATURE
NUM_LSTM = PARAMETERS.NUM_LSTM
NUM_LSTM2 = PARAMETERS.NUM_LSTM2
NUM_DENSE = PARAMETERS.NUM_DENSE
# define the delay and predict number ahead
delay = PARAMETERS.DELAY
predict = PARAMETERS.PREDICT

# load dataset
#values0 = PARAMETERS.loaddata(TRAINDATA_PATH)
values0 = read_csv(smootheddata,index_col=None, header=0) # smoothed data = [Ft, Fs, D]
values0 = values0.values
values0 = values0[0:1000,:]
# normalize features
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaled = scaler.fit_transform(values0)
scaled = values0
# frame as supervised learning
reframed = PARAMETERS.series_to_supervised(scaled, delay, predict)
# drop columns we don't want to predict
if DROP == True:
    reframed.drop(reframed.columns[-predict*NUM_FEATURE:-NUM_FEATURE], axis=1, inplace=True)
    predict = 1
if singleFs == True:
#drop the Ft, D in the output
    reframed.drop(reframed.columns[-3], axis=1, inplace=True)
    reframed.drop(reframed.columns[-1], axis=1, inplace=True)
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
#model = Sequential()
#if MULTI_LSTM == False:
#    model.add(LSTM(NUM_LSTM, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
#    model.add(Flatten())
#else:
 #   model.add(LSTM(NUM_LSTM, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
 #   model.add(LSTM(NUM_LSTM2))
#if NUM_DENSE != 0:
 #   model.add(Dense(NUM_DENSE))
#if singleFs == True:
 #   model.add(Dense(1))
#else:
#    model.add(Dense(predict*NUM_FEATURE))

#api model
inputs1 = Input(shape=(train_X.shape[1], train_X.shape[2]))
lstm = LSTM(NUM_LSTM , return_sequences = True)()
model = Model(inputs=inputs1, outputs= lstm)
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=NUM_EPOCH, batch_size=BATCH_SIZE, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
#EXPORT MODEL
export_model(model, exportname)
#SAVE MODEL
#model.save(Net_PATH)
#with open(HISTORY_PATH, 'wb') as handle:
#    pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
# read the history
#with open(HISTORY_PATH, 'rb') as handle:
#   b = pickle.load(handle)
# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()

