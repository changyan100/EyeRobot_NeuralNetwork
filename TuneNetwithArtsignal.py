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
#LR = [0.001, 0.005, 0.01, 0.02, 0.05] #learning rate
NUM_LSTM = [50, 60, 70, 80]
# NUM_DENSE =
# define the delay and predict number ahead
HISTORY = [50, 60, 70, 80]
PRIDICT = 20


Net_PATH = './EyeRobot_NeuralNetwork/TunedNet-mocksignal/Net_Epoch{epoch}-LSTMnurs{neu}-History-{his}.h5'
TrainLossFigname = 'TrainLoss-Epoch{epoch}-LSTMnurs{neu}-History-{his}.svg'
EvaFigname = 'Evaluation-Epoch{epoch}-LSTMnurs{neu}-History-{his}.svg'

# data for train
preriodic = ArtificialSignal.PeriodicSingnal(500)
scaled = preriodic
# data for evaluation
values0 = ArtificialSignal.PeriodicSingnal(10)
# Hyperameters
NUM_EPOCH = [300, 400, 500, 600]
#LR = [0.001, 0.005, 0.01, 0.02, 0.05] #learning rate
NUM_LSTM = [50, 60, 70, 80]
HISTORY = [50, 60, 70, 80]


for e in len(NUM_EPOCH):
    for h in len(HISTORY):
        for n in len(NUM_LSTM):
            # frame as supervised learning
            reframed = PARAMETERS.series_to_supervised(scaled, HISTORY[h], PRIDICT)
            # drop columns we don't want to predict
            if DROP == True:
                reframed.drop(reframed.columns[-PRIDICT * NUM_FEATURE:-NUM_FEATURE], axis=1, inplace=True)
                PRIDICT = 1
            # split into train and test sets
            values = reframed.values
            train = values[:int(0.7 * values.shape[0]), :]
            test = values[int(0.7 * values.shape[0]):, :]
            # split into input and outputs
            train_X0, train_y = train[:, :HISTORY[h] * NUM_FEATURE], train[:, HISTORY[h] * NUM_FEATURE:]
            test_X0, test_y = test[:, :HISTORY[h] * NUM_FEATURE], test[:, HISTORY[h] * NUM_FEATURE:]
            # reshape input to be 3D [samples, timesteps, features]
            train_X = train_X0.reshape((train_X0.shape[0], HISTORY[h], NUM_FEATURE))
            test_X = test_X0.reshape((test_X0.shape[0], HISTORY[h], NUM_FEATURE))
            # design network
            model = Sequential()
            model.add(LSTM(NUM_LSTM[n], input_shape=(train_X.shape[1], train_X.shape[2])))
            model.add(Dense(PRIDICT*NUM_FEATURE))
            model.compile(loss='mae', optimizer='adam')
            # fit network
            history = model.fit(train_X, train_y, epochs=NUM_EPOCH[e], batch_size=BATCH_SIZE, validation_data=(test_X, test_y), verbose=2,
                                shuffle=False)
            # save network
            model.save(Net_PATH.format(epoch=NUM_EPOCH[e], neu=NUM_LSTM[n], his=HISTORY[h]))
            # plot train loss
            fig = pyplot.figure()
            pyplot.plot(history.history['loss'], label='train')
            pyplot.plot(history.history['val_loss'], label='test')
            pyplot.legend()
            figname = TrainLossFigname.format(epoch=NUM_EPOCH[e], neu=NUM_LSTM[n], his=HISTORY[h])
            pyplot.title(figname)
            pyplot.savefig(figname)
            #evaluation
            reframed = PARAMETERS.series_to_supervised(values0, HISTORY[h], PRIDICT)
            # drop columns we don't want to predict
            if DROP == True:
                reframed.drop(reframed.columns[PRIDICT * NUM_FEATURE:-NUM_FEATURE], axis=1, inplace=True)
            values = reframed.values
            # split into input and outputs
            groundtruth_X, groundtruth_y = values[:, :HISTORY[h] * NUM_FEATURE], values[:, HISTORY[h] * NUM_FEATURE:]
            # reshape input to be 3D [samples, timesteps, features]
            groundtruth_X = groundtruth_X.reshape((groundtruth_X.shape[0], HISTORY[h], NUM_FEATURE))
            yhat = model.predict(groundtruth_X)
            # calculate RMSE
            rmse = sqrt(mean_squared_error(groundtruth_y, yhat, multioutput='raw_values'))
            error = groundtruth_y -yhat
            error_std = std(error, axis= 0)
            maxerror = amax(abs(error), axis=0)
            print('epoch',NUM_EPOCH[e])
            print('lstmNural',NUM_LSTM[n])
            print('history',HISTORY[h])
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
            figname = EvaFigname.format(epoch=NUM_EPOCH[e], neu=NUM_LSTM[n], his=HISTORY[h])
            pyplot.title(figname)
            pyplot.savefig(figname)
