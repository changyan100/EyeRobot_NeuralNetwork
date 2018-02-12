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

DROP = True
#parameters in file name
predic = 100
neural = 450
dense = 0
delay = 150


NUM_FEATURE = PARAMETERS.NUM_FEATURE
# define the delay and predict number ahead
predict = PARAMETERS.PREDICT
smoothedtest = './data/smootheddata/smoothed40.csv'

HISTORYloaded_PATH = './TrainedNetwork/history_1out{pre}ms_noscale_smoothed{neu}n-{des}n-{dly}d.pickle'
Net_PATH = './TrainedNetwork/LSTMnet_1out{pre}ms_noscale_smoothed{neu}n-{des}n-{dly}d.h5'

DATASAVE_PATH = './Results/datasaved_LSTMnet_1out{pre}ms_noscale_smoothed{neu}n-{des}n-{dly}d.pickle'
loss_filename = './Results/loss_LSTMnet_1out{pre}ms_noscale_smoothed{neu}n-{des}n-{dly}d.svg'
predict_1out_filename = './Results/prediction_LSTMnet_1out{pre}ms_noscale_smoothed{neu}n-{des}n-{dly}d.svg'
predict_1out_errorname = './Results/prediction_Error_1out{pre}ms_noscale_smoothed{neu}n-{des}n-{dly}d.svg'
predict_allout_filename = './Results/prediction_LSTMnet_smoothed-{neu}n-{dly}d-ahead-{num}.svg'
predicterror_filename = './Results/predictionError_LSTMnet_smoothed-{neu}n-{dly}d.svg'



# read the history
with open(HISTORYloaded_PATH.format(pre = predic, neu = neural, des = dense,  dly = delay), 'rb') as handle:
    history = pickle.load(handle)
# plot history
pyplot.plot(history['loss'], label='train')
pyplot.plot(history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig(loss_filename.format(pre = predic, neu = neural, des = dense, dly = delay))
#pyplot.show()
pyplot.close()

#LOAD MODEL
model = load_model(Net_PATH.format(pre = predic, neu = neural, des = dense, dly = delay))
#load dataset
#values0 = PARAMETERS.loaddata(TESTDATA_PATH)
values0 = read_csv(smoothedtest,index_col=None, header=0) # smoothed data = [Ft, Fs, D]
values0 = values0.values
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
#if DROP == False:
  #  values0 = tile(values0, (1,predict))
  #  scaled = scaler.fit_transform(values0)
   # scaled = scaled[:, :NUM_FEATURE]
#else:
  #  scaled = scaler.fit_transform(values0)
scaled = values0
# frame as supervised learning
reframed = PARAMETERS.series_to_supervised(scaled, delay, predict)
# drop columns we don't want to predict
if DROP == True:
    reframed.drop(reframed.columns[delay*NUM_FEATURE:-NUM_FEATURE], axis=1, inplace=True)
values = reframed.values
# split into input and outputs
groundtruth_X, groundtruth_y = values[:, :delay*NUM_FEATURE], values[:, delay*NUM_FEATURE:]
# reshape input to be 3D [samples, timesteps, features]
groundtruth_X = groundtruth_X.reshape((groundtruth_X.shape[0], delay, NUM_FEATURE))
start_time = time.time()
yhat = model.predict(groundtruth_X)
elapsed_time = time.time() - start_time
# invert scaling for prediction
#inv_yhat = scaler.inverse_transform(yhat)
inv_yhat = yhat
# invert scaling for groundtruth
#inv_y = scaler.inverse_transform(groundtruth_y)
inv_y = groundtruth_y
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat, multioutput='raw_values'))
error = inv_y -inv_yhat
maxerror = amax(abs(error), axis=0)
#save data
pickle.dump([inv_y, inv_yhat, error, maxerror, rmse], open(DATASAVE_PATH.format(pre = predic, neu = neural,  des = dense, dly = delay),'wb'))
xx = [x*0.005 for x in range(len(inv_y[:,0]))]
if DROP == True:
    #plot predicted vs truth
    fig = pyplot.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(xx, inv_y[:, 0], c='g', label='Ft_groundtruth')
    ax1.scatter(xx, inv_yhat[:, 0], s=2, c='c', marker='+',  label='Ft_prediction')
    ax1.plot(xx, inv_y[:, 1], c='r', label='Fs_groundtruth')
    ax1.scatter(xx, inv_yhat[:, 1], s=2, c='m', marker='+',  label='Fs_prediction')
    ax1.set_xlabel('Time(s)')
    ax1.set_ylabel('Force(mN)')
    ax1.legend(loc=2)
    ax2 = ax1.twinx()
    ax2.plot(xx, inv_y[:, 2], c='b', label='D_groundtruth')
    ax2.scatter(xx, inv_yhat[:, 2], s=2, c='k', marker='+', label='D_prediction')
    ax2.set_ylabel('Depth(mm)')
    ax2.legend(loc=1)
    figname = predict_1out_filename.format(pre = predic, neu = neural,  des = dense, dly = delay)
    pyplot.title(figname)
    pyplot.savefig(predict_1out_filename.format(pre = predic, neu = neural,  des = dense, dly = delay))
    pyplot.show()
    pyplot.close()
    # plot predicted error
    fig = pyplot.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(xx, error[:, 0], s=2, c='c', marker='+', label='Ft_Error')
    ax1.scatter(xx, error[:, 1], s=2, c='m', marker='+', label='Fs_Error')
    ax1.set_xlabel('Time(s)')
    ax1.set_ylabel('Force(mN)')
    ax1.legend(loc=2)
    ax2 = ax1.twinx()
    ax2.scatter(xx, error[:, 2], s=2, c='k', marker='+', label='D_Error')
    ax2.set_ylabel('Depth(mm)')
    ax2.legend(loc=1)
    figname = predict_1out_errorname.format(pre=predic, neu=neural, des=dense, dly=delay)
    pyplot.title(figname)
    pyplot.savefig(predict_1out_errorname.format(pre=predic, neu=neural, des=dense, dly=delay))
    pyplot.show()
    pyplot.close()
else:
    # plot the rmse vs time
    Ftrmse, Fsrmse, Drmse = rmse[0::NUM_FEATURE], rmse[1::NUM_FEATURE], rmse[2::NUM_FEATURE]  # [start:end:step]
    fig = pyplot.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(Ftrmse, label='rmse_Ft')
    ax1.plot( Fsrmse, label='rmse_Fs')
    ax1.set_xlabel('Prediction number')
    ax1.set_ylabel('rmse_Force(mN)')
    ax1.legend(loc=2)
    ax2 = ax1.twinx()
    ax2.plot( Drmse, 'y', label='rmse_D')
    ax2.set_ylabel('rmse_Depth(mm)')
    ax2.legend(loc=1)
    pyplot.title('Prediction Error')
    pyplot.savefig(predicterror_filename.format(neu = neural, dly = delay))
    #pyplot.show()
    pyplot.close()
    #plot every prediction value vs. groundtruth value
    for i in range(0,predict*NUM_FEATURE,NUM_FEATURE):
        fig = pyplot.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.plot(inv_y[:,i], label='Ft_groundtruth')
        ax1.plot(inv_yhat[:,i],'.', label='Ft_prediction')
        ax1.plot(inv_y[:,i+1], label='Fs_groundtruth')
        ax1.plot(inv_yhat[:,i+1],'.', label='Fs_prediction')
        ax1.set_xlabel('time(ms)')
        ax1.set_ylabel('Force(mN)')
        ax1.legend(loc=2)
        ax2 = ax1.twinx()
        ax2.plot(inv_y[:,i+2], label='D_groundtruth')
        ax2.plot(inv_yhat[:,i+2],'.', label='D_prediction')
        ax2.set_ylabel('Depth(mm)')
        ax2.legend(loc=1)
        figname = 'prediction_{num}'
        pyplot.title(figname.format(num=i/3+1))
        pyplot.savefig(predict_allout_filename.format(neu = neural, dly = delay, num=i/3+1))
        #pyplot.show()
        pyplot.close()