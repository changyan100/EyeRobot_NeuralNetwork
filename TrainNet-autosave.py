from matplotlib import pyplot
#from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers
import PARAMETERS
from pandas import read_csv
import pickle
from numpy import sqrt
from sklearn.metrics import mean_squared_error
from numpy import std
from numpy import amax
from numpy import abs

# multi-lstm
MULTI_LSTM = False
# pridict only Fs
singleFs = False
# drop the 1:predict-1 output
DROP = True
# data path
TrainData_Path = './data/prediction data-marina/vesselfollow'
TestData_Path = './data/prediction data-marina/vesselfollow-test'
# load dataset
# data = [RoVel1, RoVel2, RoVel3, RoVel4, RoVel5, depth, Fs1, Fs2]
TrainData = PARAMETERS.loaddata(TrainData_Path)
TestData = PARAMETERS.loaddata(TestData_Path)
inputdatafeature = 8
NUM_FEATURE = 2
# define training process
# Hyperparameters
Lr_initial = 0.001
BATCH_SIZE = 100
NUM_EPOCH = [10, 20, 40]
NUM_LSTM = [50, 60, 80]
NUM_Dense = [50, 60, 80]
NUM_LSTM2 = 0
# Train data parameters
HISTORY = [50, 60, 80]
PRIDICT_initial = 20


suffix = 'VF-Mari'

Net_PATH = './Result-Marinadata/LSTMnurs{neu}-Dense{des}-History{his}-Epoch{epoch}-initialLR{lr}-{suffix}.h5'
ModelFigname = './Result-Marinadata/LSTMnurs{neu}-Dense{des}-History{his}-Epoch{epoch}-initialLR{lr}-{suffix}.png'
TrainLossFigname = './Result-Marinadata/TrainLoss-LSTMnurs{neu}-Dense{des}-History-{his}-Epoch{epoch}-initialLR{lr}-{suffix}.svg'
EvaFigname = './Result-Marinadata/Evaluation-LSTMnurs{neu}-Dense{des}-History-{his}-Epoch{epoch}-initialLR{lr}-{suffix}.svg'
EvaPartFigname = './Result-Marinadata/PartEvaluation-LSTMnurs{neu}-Dense{des}-History-{his}-Epoch{epoch}-initialLR{lr}-{suffix}.svg'
LrdecayFigname = './Result-Marinadata/Lrdecay-LSTMnurs{neu}-Dense{des}-History-{his}-Epoch{epoch}-initialLR{lr}-{suffix}.svg'
f= open("./Result-Marinadata/Result.txt","a+")
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
f.write("\nSuffix = " + suffix)
f.write("\n****************************************************************************\n")
# f.write("\ncount\t" "lstm\t" "history\t" "epoch\t" "rmseFs\t" "rmseD\t" "std_Fs\t" "std_D\t" "maxe_Fs\t" "maxe_D\t")
f.write("\ncount\t" "lstm\t" "dense\t" "history\t" "epoch\t" "initialLR\t" "rmseFs1\t" "maxFs1\t" "rmse_Fs2\t" "maxFs2\t")


for n in range(0, len(NUM_LSTM), 1):
    for d in range(0, len(NUM_Dense), 1):
        for h in range(0, len(HISTORY), 1):
            for e in range(0, len(NUM_EPOCH), 1):
                print('LSTM:', NUM_LSTM[n], 'Dense:', NUM_Dense[d], 'History:', HISTORY[h], 'Epoch', NUM_EPOCH[e] )
                count += 1
                predict = PRIDICT_initial
                # normalize features
                #scaler = MinMaxScaler(feature_range=(0, 1))
                #scaled = scaler.fit_transform(values0)
                # frame as supervised learning
                reframed = PARAMETERS.series_to_supervised(TrainData, HISTORY[h], predict)
                # drop columns we don't want to predict
                if DROP == True:
                    reframed.drop(reframed.columns[-predict*inputdatafeature:-NUM_FEATURE], axis=1, inplace=True)
                    predict = 1
                if singleFs == True:
                #drop the Ft, D in the output
                    reframed.drop(reframed.columns[-3], axis=1, inplace=True)
                    reframed.drop(reframed.columns[-1], axis=1, inplace=True)
                # split into train and test sets
                values = reframed.values
                train = values[:int(0.7*values.shape[0]), :]
                test = values[int(0.7*values.shape[0]):, :]
                # split into input and outputs
                train_X0, train_y = train[:, :HISTORY[h]*inputdatafeature], train[:, HISTORY[h]*inputdatafeature:]
                test_X0, test_y = test[:, :HISTORY[h]*inputdatafeature], test[:, HISTORY[h]*inputdatafeature:]
                # reshape input to be 3D [samples, timesteps, features]
                train_X = train_X0.reshape((train_X0.shape[0], HISTORY[h], inputdatafeature))
                test_X = test_X0.reshape((test_X0.shape[0], HISTORY[h], inputdatafeature))
                #print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

                # design network
                model = Sequential()
                if MULTI_LSTM == False:
                    model.add(LSTM(NUM_LSTM[n], input_shape=(train_X.shape[1], train_X.shape[2])))
                else:
                    model.add(LSTM(NUM_LSTM[n], return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
                    #add dropout
                    model.add(Dropout(0.5))
                    model.add(LSTM(NUM_LSTM2))
                if NUM_Dense[d] != 0:
                    # model.add(Dense(NUM_DENSE,activation='relu'))
                    model.add(Dense(NUM_Dense[d]))
                if singleFs == True:
                    model.add(Dense(1))
                else:
                    model.add(Dense(predict*NUM_FEATURE))
                #model.compile(loss='mae', optimizer='adam')
                # time based learning rate decay: lr *= (1. / (1. + self.decay * self.iterations))
                Lr_decay = Lr_initial / NUM_EPOCH[e]
                adam = optimizers.adam(lr=Lr_initial, beta_1=0.9, beta_2=0.999, epsilon=None, decay=Lr_decay,
                                       amsgrad=False)
                model.compile(loss='mae', optimizer=adam)
                # fit network
                history = model.fit(train_X, train_y, epochs=NUM_EPOCH[e], batch_size=BATCH_SIZE, validation_data=(test_X, test_y), verbose=2,
                                    shuffle=False)
                model.save(Net_PATH.format(neu=NUM_LSTM[n], des=NUM_Dense[d], his=HISTORY[h], epoch=NUM_EPOCH[e],
                                           lr=Lr_initial, suffix=suffix))
                #calculate the learning rate
                lr_inter = Lr_initial
                lr_logger = list()
                for iteration in range(1, NUM_EPOCH[e]+1, 1):
                    lr_inter *= (1. / (1. + Lr_decay*iteration))
                    lr_logger.append(lr_inter)
                # plot train loss
                fig = pyplot.figure()
                pyplot.plot(history.history['loss'], label='train')
                pyplot.plot(history.history['val_loss'], label='test')
                pyplot.legend()
                # pyplot.show()
                figname = TrainLossFigname.format(neu=NUM_LSTM[n], des=NUM_Dense[d], his=HISTORY[h],
                                                  epoch=NUM_EPOCH[e],
                                                  lr=Lr_initial, suffix=suffix)
                pyplot.title(figname)
                pyplot.savefig(figname)
                pyplot.close()
                # plot learning rate decay
                fig = pyplot.figure()
                # pyplot.plot(loss_history.lr, label='lr decay')
                pyplot.plot(lr_logger, label='lr decay')
                pyplot.legend()
                # pyplot.show()
                figname = LrdecayFigname.format(neu=NUM_LSTM[n], des=NUM_Dense[d], his=HISTORY[h],
                                                epoch=NUM_EPOCH[e],
                                                lr=Lr_initial, suffix=suffix)
                pyplot.title(figname)
                pyplot.savefig(figname)
                pyplot.close()

                # evaluation
                PRIDICT = PRIDICT_initial
                reframed = PARAMETERS.series_to_supervised(TestData, HISTORY[h], PRIDICT)
                # drop columns we don't want to predict
                if DROP == True:
                    reframed.drop(reframed.columns[-PRIDICT * inputdatafeature:-NUM_FEATURE], axis=1, inplace=True)
                    PRIDICT = 1
                values = reframed.values
                # split into input and outputs
                groundtruth_X, groundtruth_y = values[:, :HISTORY[h] * inputdatafeature], values[:,
                                                                                     HISTORY[h] * inputdatafeature:]
                # reshape input to be 3D [samples, timesteps, features]
                groundtruth_X = groundtruth_X.reshape((groundtruth_X.shape[0], HISTORY[h], inputdatafeature))
                yhat = model.predict(groundtruth_X)
                # calculate RMSE
                rmseFs = sqrt(mean_squared_error(groundtruth_y, yhat, multioutput='raw_values'))
                error_Fs = groundtruth_y - yhat
                maxFs = amax(abs(error_Fs), axis=0)
                f.write("\n%d\t %d\t %d\t %d\t %d\t %.2f\t %.2f\t %.2f\t %.2f\t" % (
                    count, NUM_LSTM[n], NUM_Dense[d], HISTORY[h], NUM_EPOCH[e], rmseFs[0], maxFs[1], rmseFs[0], maxFs[1]))

                # plot
                xx = [x for x in range(len(groundtruth_y[:, 0]))]
                fig = pyplot.figure()
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.plot(xx, groundtruth_y[:, 0], c='r', label='Fs1_groundtruth')
                ax1.scatter(xx, yhat[:, 0], s=4, c='m', marker='o', label='Fs1_prediction')
                ax1.set_xlabel('Time(s)')
                ax1.set_ylabel('Fs1(mN)')
                ax1.legend(loc=2, markerscale=2)
                ax2 = ax1.twinx()
                ax2.plot(xx, groundtruth_y[:, 1], c='b', label='Fs2_groundtruth')
                ax2.scatter(xx, yhat[:, 1], s=4, c='k', marker='o', label='Fs2_prediction')
                ax2.set_ylabel('Fs2(mN)')
                ax2.legend(loc=1, markerscale=2)
                figname = EvaFigname.format(neu=NUM_LSTM[n], des=NUM_Dense[d], his=HISTORY[h], epoch=NUM_EPOCH[e],
                                            lr=Lr_initial, suffix=suffix)
                pyplot.title(figname)
                pyplot.savefig(figname)
                pyplot.close()

                # plot for the last 50 data
                plotduration = 100
                xx = [x for x in range(plotduration)]
                fig = pyplot.figure()
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.plot(xx, groundtruth_y[-plotduration:, 0], c='r', label='Fs1_groundtruth')
                ax1.scatter(xx, yhat[-plotduration:, 0], s=4, c='m', marker='o', label='Fs1_prediction')
                ax1.set_xlabel('Time(s)')
                ax1.set_ylabel('Fs1(mN)')
                ax1.legend(loc=2, markerscale=2)
                ax2 = ax1.twinx()
                ax2.plot(xx, groundtruth_y[-plotduration:, 1], c='b', label='Fs2_groundtruth')
                ax2.scatter(xx, yhat[-plotduration:, 1], s=4, c='k', marker='o', label='Fs2_groundtruth')
                ax2.set_ylabel('Fs2(mN)')
                ax2.legend(loc=1, markerscale=2)
                figname = EvaPartFigname.format(neu=NUM_LSTM[n], des=NUM_Dense[d], his=HISTORY[h], epoch=NUM_EPOCH[e],
                                                lr=Lr_initial, suffix=suffix)
                pyplot.title(figname)
                pyplot.savefig(figname)
                pyplot.close()
# with open(HISTORY_PATH, 'wb') as handle:
#     pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
# read the history
#with open(HISTORY_PATH, 'rb') as handle:
#   b = pickle.load(handle)
# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()

