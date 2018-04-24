from matplotlib import pyplot
#from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
from keras.callbacks import LearningRateScheduler
from keras.callbacks import Callback
from keras.layers import Dropout
import PARAMETERS
import pickle
import ArtificialSignal
from numpy import sqrt
from sklearn.metrics import mean_squared_error
from numpy import std
from numpy import amax
from numpy import abs
from numpy import array
from keras.utils import plot_model
import math

DROP = True #drop the middle predicted values, and only predict the n time value
NUM_FEATURE = 1

# define the predict number ahead
PRIDICT_initial = 10

suffix = 'LRtimedecay'

Net_PATH = './TuneNet-mocksignal-LRtimedecay/LSTMnurs{neu}-Dense{des}-History{his}-Epoch{epoch}-initialLR{lr}-{suffix}.h5'
ModelFigname = './TuneNet-mocksignal-LRtimedecay/LSTMnurs{neu}-Dense{des}-History{his}-Epoch{epoch}-initialLR{lr}-{suffix}.png'
TrainLossFigname = './TuneNet-mocksignal-LRtimedecay/TrainLoss-LSTMnurs{neu}-Dense{des}-History-{his}-Epoch{epoch}-initialLR{lr}-{suffix}.svg'
EvaFigname = './TuneNet-mocksignal-LRtimedecay/Evaluation-LSTMnurs{neu}-Dense{des}-History-{his}-Epoch{epoch}-initialLR{lr}-{suffix}.svg'
LrdecayFigname = './TuneNet-mocksignal-LRtimedecay/Lrdecay-LSTMnurs{neu}-Dense{des}-History-{his}-Epoch{epoch}-initialLR{lr}-{suffix}.svg'
f= open("./TuneNet-mocksignal-LRtimedecay/ValidationError-lessrange.txt","a+")

# data for train
preriodic = ArtificialSignal.PeriodicSingnal(1000)
scaled = preriodic
# data for evaluation
values0 = ArtificialSignal.PeriodicSingnal(10)
# Hyperameters
Lr_initial = 0.001
BATCH_SIZE = 100
NUM_EPOCH = [10, 20, 40]
NUM_LSTM = [10, 20, 40]
HISTORY = [10, 20, 40]
NUM_Dense = [10, 20, 40]
# NUM_EPOCH = [20, 40, 60, 80, 100, 120, 150, 200]
# NUM_LSTM = [10, 30, 50, 70, 90, 120, 150]
# HISTORY = [20, 40, 60, 80, 100, 150, 200]
# NUM_EPOCH = [60, 80, 100, 120]
# NUM_LSTM = [60, 80]
# HISTORY = [80, 100]
# lr=0.001 (default value of Adam)
# NUM_EPOCH = [20, 40, 60, 80]
# NUM_LSTM = [20, 40, 60, 80]
# HISTORY = [40, 60, 80,100]
# NUM_EPOCH = [10, 15, 20, 25]
# NUM_LSTM = [5, 10, 15, 20]
# HISTORY = [5, 10, 15, 20]


def step_decay(epoch):
   initial_lrate = Lr_initial
   drop = 0.5
   epochs_drop = 5.0  #drop the learning rate by half every 5 epochs
   lrate = initial_lrate * math.pow(drop,
   math.floor((1+epoch)/epochs_drop))
   return lrate


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))

count = 0
f.write("\n*********************************************************************\n")
f.write("Hyperameters:\n")
f.write("\nlstmNural = ")
for k in range(len(NUM_LSTM)):
    f.write("\t%d" % NUM_LSTM[k])
f.write("\nHistory = ")
for k in range(len(HISTORY)):
    f.write("\t%d" % HISTORY[k])
f.write("\nEpoch = ")
for k in range(len(NUM_EPOCH)):
    f.write("\t%d" % NUM_EPOCH[k])
f.write("\nBatch size = %d" % BATCH_SIZE)
f.write("\nInitial learning rate = %f\t" % Lr_initial)
f.write("\nLRprotocal = ", suffix)
f.write("\n****************************************************************************\n")
# f.write("\ncount\t" "lstm\t" "history\t" "epoch\t" "rmseFs\t" "rmseD\t" "std_Fs\t" "std_D\t" "maxe_Fs\t" "maxe_D\t")
f.write("\ncount\t" "lstm\t" "dense\t" "history\t" "epoch\t" "initialLR\t" "rmseFs\t" "std_Fs\t" "maxe_Fs\t")
for n in range(0, len(NUM_LSTM), 1):
    for d in range(0, len(NUM_Dense), 1):
        for h in range(0, len(HISTORY), 1):
            for e in range(0, len(NUM_EPOCH), 1):
                print('LSTM:', NUM_LSTM[n], 'Dense:', NUM_Dense[d], 'History:', HISTORY[h], 'Epoch', NUM_EPOCH[e] )
                count += 1
                PRIDICT = PRIDICT_initial
                # frame as supervised learning
                reframed = PARAMETERS.series_to_supervised(scaled, HISTORY[h], PRIDICT)
                # drop columns we do not want to predict
                if DROP == True:
                    reframed.drop(reframed.columns[-PRIDICT * NUM_FEATURE:-NUM_FEATURE], axis=1, inplace=True)
                    PRIDICT = 1
                # split into train and test sets
                values = reframed.values
                # for k in range(0, 5, 1):
                train = values[:int(0.7 * values.shape[0]), :]
                test = values[int(0.7 * values.shape[0]):, :]
                # train = values
                # test = values
                # split into input and outputs
                train_X0, train_y = train[:, :HISTORY[h] * NUM_FEATURE], train[:, HISTORY[h] * NUM_FEATURE:]
                test_X0, test_y = test[:, :HISTORY[h] * NUM_FEATURE], test[:, HISTORY[h] * NUM_FEATURE:]
                # reshape input to be 3D [samples, timesteps, features]
                train_X = train_X0.reshape((train_X0.shape[0], HISTORY[h], NUM_FEATURE))
                test_X = test_X0.reshape((test_X0.shape[0], HISTORY[h], NUM_FEATURE))
                # design network
                model = Sequential()
                model.add(LSTM(NUM_LSTM[n], input_shape=(train_X.shape[1], train_X.shape[2])))
                model.add(Dense(NUM_Dense[d]))
                model.add(Dense(PRIDICT*NUM_FEATURE))
                # model.compile(loss='mae', optimizer='adam')
                # history = model.fit(train_X, train_y, epochs=NUM_EPOCH[e], batch_size=BATCH_SIZE,
                #                                         validation_data=(test_X, test_y), verbose=2,
                #                                         shuffle=False)
                # time based learning rate decay: lr *= (1. / (1. + self.decay * self.iterations))
                Lr_decay = Lr_initial/NUM_EPOCH[e]
                adam = optimizers.adam(lr=Lr_initial, beta_1=0.9, beta_2=0.999, epsilon=None, decay=Lr_decay, amsgrad=False)
                model.compile(loss='mae', optimizer=adam)
                #fit network
                history = model.fit(train_X, train_y, epochs=NUM_EPOCH[e], batch_size=BATCH_SIZE,
                                    validation_data=(test_X, test_y), verbose=2,
                                    shuffle=False)
                #calculate the learning rate
                lr_inter = Lr_initial
                lr_logger = list()
                for iteration in range(1, NUM_EPOCH[e]+1, 1):
                    lr_inter *= (1. / (1. + Lr_decay*iteration))
                    lr_logger.append(lr_inter)
                # step based learning rate decay: lr = lr0 * drop^floor(epoch / epochs_drop)
                # model.compile(loss='mae', optimizer='adam')
                # loss_history = LossHistory()
                # lrate = LearningRateScheduler(step_decay)
                # callbacks_list = [loss_history, lrate]
                # history = model.fit(train_X, train_y,
                #                     validation_data=(test_X, test_y),
                #                     epochs=NUM_EPOCH[e],
                #                     batch_size=BATCH_SIZE,
                #                     callbacks=callbacks_list,
                #                     verbose=2,
                #                     shuffle=False)
                # save network
                #plot_model(model, to_file=ModelFigname.format(neu=NUM_LSTM[n], des=NUM_Dense[d], his=HISTORY[h], epoch=NUM_EPOCH[e],
                                           # lr=Lr_initial, suffix=suffix))
                model.save(Net_PATH.format(neu=NUM_LSTM[n], des=NUM_Dense[d], his=HISTORY[h], epoch=NUM_EPOCH[e],
                                           lr=Lr_initial, suffix=suffix))
                # plot train loss
                fig = pyplot.figure()
                pyplot.plot(history.history['loss'], label='train')
                pyplot.plot(history.history['val_loss'], label='test')
                pyplot.legend()
                #pyplot.show()
                figname = TrainLossFigname.format(neu=NUM_LSTM[n], des=NUM_Dense[d], his=HISTORY[h], epoch=NUM_EPOCH[e],
                                                  lr=Lr_initial, suffix=suffix)
                pyplot.title(figname)
                pyplot.savefig(figname)
                pyplot.close()
                # plot learning rate decay
                fig = pyplot.figure()
                # pyplot.plot(loss_history.lr, label='lr decay')
                pyplot.plot(lr_logger, label='lr decay')
                pyplot.legend()
                #pyplot.show()
                figname = LrdecayFigname.format(neu=NUM_LSTM[n], des=NUM_Dense[d], his=HISTORY[h], epoch=NUM_EPOCH[e],
                                                lr=Lr_initial, suffix=suffix)
                pyplot.title(figname)
                pyplot.savefig(figname)
                pyplot.close()
                #evaluation
                PRIDICT = PRIDICT_initial
                reframed = PARAMETERS.series_to_supervised(values0, HISTORY[h], PRIDICT)
                # drop columns we don't want to predict
                if DROP == True:
                    reframed.drop(reframed.columns[-PRIDICT * NUM_FEATURE:-NUM_FEATURE], axis=1, inplace=True)
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
                # f.write("*******************************\n")
                # f.write("Count = %d\r\n" % count)
                # f.write("lstmNural = %d\r\n" % NUM_LSTM[n])
                # f.write("history = %d\r\n" % HISTORY[h])
                # f.write("epoch = %d\r\n" % NUM_EPOCH[e])
                # f.write("rmse: Fs = %.2f Depth = %.2f\r\n" % (rmse[0], rmse[1]))
                # f.write("error_std: Fs = %.2f Depth = %.2f\r\n" % (error_std[0],error_std[1]))
                # f.write("maxerror: Fs = %.2f Depth = %.2f\r\n" % (maxerror[0],maxerror[1]))
                # f.write("*******************************\n")
                # f.write("\n%d\t %d\t %d\t %d\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t" % (
                # count, NUM_LSTM[n], HISTORY[h], NUM_EPOCH[e], rmse[0], rmse[1], error_std[0], error_std[1], maxerror[0], maxerror[1]))
                f.write("\n%d\t %d\t %d\t %d\t %d\t %.2f\t %.2f\t %.2f\t " % (
                count, NUM_LSTM[n], NUM_Dense[d], HISTORY[h], NUM_EPOCH[e], rmse, error_std, maxerror))

                #plot
                xx = [x for x in range(len(groundtruth_y[:,0]))]
                fig = pyplot.figure()
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.plot(xx, groundtruth_y[:, 0], c='r',  label='Fs_groundtruth')
                ax1.scatter(xx, yhat[:, 0], s=4, c='m', marker='o',  label='Fs_prediction')
                ax1.set_xlabel('Time(s)')
                ax1.set_ylabel('Force(mN)')
                ax1.legend(loc=2, markerscale=2)
                # ax2 = ax1.twinx()
                # ax2.plot(xx, groundtruth_y[:, 1], c='b', label='D_groundtruth')
                # ax2.scatter(xx, yhat[:, 1], s=4, c='k', marker='o', label='D_prediction')
                # ax2.set_ylabel('Depth(mm)')
                # ax2.legend(loc=1, markerscale=2)
                figname = EvaFigname.format(neu=NUM_LSTM[n], des=NUM_Dense[d], his=HISTORY[h], epoch=NUM_EPOCH[e],
                                                lr=Lr_initial, suffix=suffix)
                pyplot.title(figname)
                pyplot.savefig(figname)
                pyplot.close()
f.close()