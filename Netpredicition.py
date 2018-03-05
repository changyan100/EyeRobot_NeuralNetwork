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

Fsbias = 8.5
Depthbias = 2.3
periodictest = True
consttest = False
singleFs = False
TwoOut = True
DROP = True
BIAS = False

#parameters in file name
predict = 20
predic = predict  #predict time(ms) in the file name
# define the delay and predict timestep
neural = '40n'
dense = 0
delay = 50
NUM_FEATURE = 2
figureID = 'PreiodicTest'

if consttest == True:
    figureID = 'consttest-02'
smoothedtest = './data/smootheddata/smoothed40.csv'
TESTDATA_PATH = './data/robot-newdata-trainning-0222/2018-02-21-22-46-48-Predictor-Nofeedback-Newtool-dataalined-changyan-46-RobotLog.csv'
constdata = './data/2017-11-15-03-25-25-Simple Task-Ch-07-RobotLog.csv' #first 1000 samples


if periodictest == True:
    HISTORYloaded_PATH = './TrainedNetwork/PeriodicTrainedNN/history-periodic_1out100_60n-0n-150d-PreriodicTest.pickle'
    Net_PATH = './TrainedNetwork/PeriodicTrainedNN/LSTMnet-periodic_1out100_60n-0n-150d-PreriodicTest.h5'
else:
    HISTORYloaded_PATH = './TrainedNetwork/TrainedNN_vesselfollow-0222/history_2out{pre}ms_noscale_nosmooth{neu}-{des}n-{dly}d-35e-400b.pickle'
    Net_PATH = './TrainedNetwork/TrainedNN_vesselfollow-0222/LSTMnet_2out{pre}ms_noscale_nosmooth{neu}-{des}n-{dly}d-35e-400b.h5'

DATASAVE_PATH = './Result/Result_vesselfollow-0222-correctFs/datasaved_LSTMnet_2out{pre}_noscale_{neu}-{des}n-{dly}d-{index}.pickle'
loss_filename = './Result/Result_vesselfollow-0222-correctFs/loss_LSTMnet_2out{pre}_noscale_{neu}-{des}n-{dly}d.svg'
predict_1out_filename = './Result/Result_vesselfollow-0222-correctFs/prediction_LSTMnet_2out{pre}_noscale_{neu}-{des}n-{dly}d-{index}.svg'
predict_1out_errorname = './Result/Result_vesselfollow-0222-correctFs/prediction_Error_2out{pre}_noscale_{neu}-{des}n-{dly}-{index}.svg'
predict_allout_filename = './Result/Result_vesselfollow/prediction_LSTMnet_smoothed-{neu}-{dly}d-ahead-{index}.svg'
predicterror_filename = './Result/Result_vesselfollow/predictionError_LSTMnet_smoothed-{neu}-{dly}d.svg'



# read the history
# with open(HISTORYloaded_PATH.format(pre = predic, neu = neural, des = dense,  dly = delay), 'rb') as handle:
#     history = pickle.load(handle)
# # plot history
# pyplot.plot(history['loss'], label='train')
# pyplot.plot(history['val_loss'], label='test')
# pyplot.legend()
# pyplot.savefig(loss_filename.format(pre = predic, neu = neural, des = dense, dly = delay))
# #pyplot.show()
# pyplot.close()

#LOAD MODEL
if periodictest == True:
    model = load_model(Net_PATH)
else:
    model = load_model(Net_PATH.format(pre = predic, neu = neural, des = dense, dly = delay))

#load dataset
#values0 = PARAMETERS.loaddata(TESTDATA_PATH)
#values0 = PARAMETERS.loaddata(constdata)
#values0 = values0[0:1000, :]
if periodictest == True:
    values0 = ArtificialSignal.PeriodicSingnal(5)
elif consttest == True:
    const = array([100, 15]*5000)
    values0 = const.reshape(5000,2)
    dataset = read_csv(smoothedtest, index_col=None, header=0)
else:
    dataset = read_csv(TESTDATA_PATH, index_col=None, header=0)
    # drop rows with NaN values
    dataset.dropna(inplace=True)
    values0 = dataset.values
    values0 = array([values0[:,8], values0[:,46]]).T

#values0 = read_csv(smoothedtest,index_col=None, header=0) # smoothed data = [Ft, Fs, D]
#values0 = values0.values
# normalize features
#scaler = MinMaxScaler(feature_range=(0, 1))
#if DROP == False:
  #  values0 = tile(values0, (1,predict))
  #  scaled = scaler.fit_transform(values0)
   # scaled = scaled[:, :NUM_FEATURE]
#else:
  #  scaled = scaler.fit_transform(values0)
#scaled = values0
# frame as supervised learning
reframed = PARAMETERS.series_to_supervised(values0, delay, predict)
# drop columns we don't want to predict
if DROP == True:
    reframed.drop(reframed.columns[delay*NUM_FEATURE:-NUM_FEATURE], axis=1, inplace=True)

if singleFs == True:
    reframed.drop(reframed.columns[-3], axis=1, inplace=True)
    reframed.drop(reframed.columns[-1], axis=1, inplace=True)
values = reframed.values
# split into input and outputs
groundtruth_X, groundtruth_y = values[:, :delay * NUM_FEATURE], values[:, delay * NUM_FEATURE:]
# reshape input to be 3D [samples, timesteps, features]
groundtruth_X = groundtruth_X.reshape((groundtruth_X.shape[0], delay, NUM_FEATURE))
start_time = time.time()
yhat = model.predict(groundtruth_X)
elapsed_time = time.time() - start_time
# invert scaling for prediction
#inv_yhat = scaler.inverse_transform(yhat)
if BIAS == True:
    inv_yhat = array([yhat[:,0]-Fsbias,yhat[:,1]-Depthbias]).T
else:
    inv_yhat = yhat
# invert scaling for groundtruth
#inv_y = scaler.inverse_transform(groundtruth_y)
inv_y = groundtruth_y
# calculate RMSE
if singleFs == True:
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
else:
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat, multioutput='raw_values'))
error = inv_y -inv_yhat
error_std = std(error, axis= 0)
maxerror = amax(abs(error), axis=0)
print('rmse', rmse)
print('error_std', error_std)
print('maxerror', maxerror)
#save data
#pickle.dump([inv_y, inv_yhat, error, maxerror, rmse], open(DATASAVE_PATH.format(pre = predic, neu = neural,  des = dense, dly = delay, index = figureID),'wb'))
xx = [x for x in range(len(inv_y[:,0]))]
if DROP == True:
    #plot predicted vs truth
    fig = pyplot.figure()
    if TwoOut == True:
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(xx, inv_y[:, 0], c='r',  label='Fs_groundtruth')
        ax1.scatter(xx, inv_yhat[:, 0], s=4, c='m', marker='o',  label='Fs_prediction')
        ax1.set_xlabel('Time(s)')
        ax1.set_ylabel('Force(mN)')
        ax1.legend(loc=2, markerscale=2)
        ax2 = ax1.twinx()
        ax2.plot(xx, inv_y[:, 1], c='b', label='D_groundtruth')
        ax2.scatter(xx, inv_yhat[:, 1], s=4, c='k', marker='o', label='D_prediction')
        ax2.set_ylabel('Depth(mm)')
        ax2.legend(loc=1, markerscale=2)
    elif singleFs == True:
        pyplot.plot(xx, inv_y, c='r',marker='o', label='Fs_groundtruth')
        pyplot.scatter(xx, inv_yhat, s=4, c='m', marker='o',  label='Fs_prediction')
        pyplot.xlabel('Time(s)')
        pyplot.ylabel('Force(mN)')
        pyplot.legend(markerscale=2)
    else:
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(xx, inv_y[:, 0], c='g', label='Ft_groundtruth')
        ax1.scatter(xx, inv_yhat[:, 0], s=4, c='c', marker='o',  label='Ft_prediction')
        ax1.plot(xx, inv_y[:, 1], c='r', label='Fs_groundtruth')
        ax1.scatter(xx, inv_yhat[:, 1], s=4, c='m', marker='o',  label='Fs_prediction')
        ax1.set_xlabel('Time(s)')
        ax1.set_ylabel('Force(mN)')
        ax1.legend(loc=2, markerscale=2)
        ax2 = ax1.twinx()
        ax2.plot(xx, inv_y[:, 2], c='b', label='D_groundtruth')
        ax2.scatter(xx, inv_yhat[:, 2], s=4, c='k', marker='o', label='D_prediction')
        ax2.set_ylabel('Depth(mm)')
        ax2.legend(loc=1, markerscale=2)
    figname = predict_1out_filename.format(pre = predic, neu = neural,  des = dense, dly = delay, index = figureID)
    pyplot.title(figname)
    pyplot.savefig(predict_1out_filename.format(pre = predic, neu = neural,  des = dense, dly = delay, index = figureID))
    pyplot.show()
    pyplot.close()
    # plot predicted error
    fig = pyplot.figure()
    if TwoOut == True:
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.scatter(xx, error[:, 0], s=2, c='g', marker='.', label='Fs_Error')
        ax1.set_xlabel('Time(s)')
        ax1.set_ylabel('Force(mN)')
        ax1.legend(loc=2, markerscale=2)
        ax2 = ax1.twinx()
        ax2.scatter(xx, error[:, 1], s=2, c='b', marker='.', label='D_Error')
        ax2.set_ylabel('Depth(mm)')
        ax2.legend(loc=1, markerscale=2)
    elif singleFs == True:
        pyplot.scatter(xx, error, s=2, c='m', marker='.', label='Fs_Error')
        pyplot.xlabel('Time(s)')
        pyplot.ylabel('Force(mN)')
        pyplot.legend(markerscale=2)
    else:
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.scatter(xx, error[:, 0], s=2, c='r', marker='.', label='Ft_Error')
        ax1.scatter(xx, error[:, 1], s=2, c='g', marker='.', label='Fs_Error')
        ax1.set_xlabel('Time(s)')
        ax1.set_ylabel('Force(mN)')
        ax1.legend(loc=2, markerscale=5)
        ax2 = ax1.twinx()
        ax2.scatter(xx, error[:, 2], s=2, c='b', marker='.', label='D_Error')
        ax2.set_ylabel('Depth(mm)')
        ax2.legend(loc=1, markerscale=2)
    figname = predict_1out_errorname.format(pre=predic, neu=neural, des=dense, dly=delay, index = figureID)
    pyplot.title(figname)
    pyplot.savefig(predict_1out_errorname.format(pre=predic, neu=neural, des=dense, dly=delay, index = figureID))
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
