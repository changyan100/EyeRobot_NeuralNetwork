from numpy import array
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import glob
import numpy
from matplotlib import pyplot

# pridict only Fs
singleFs = True
# drop the 1:predict-1 output
DROP = True
# data path
smoothedtrain = './data/smootheddata/smoothed001.csv'
TRAINDATA_PATH = './data/traindata'

# network saved path
Netsaved_PATH = './TrainedNetwork/LSTMnet_smoothed120n-150d.h5'
HISTORYsaved_PATH = './history-120n-150d.pickle'

# define training process
BATCH_SIZE = 500 #3200  #3200*5=17000ms = 17s
NUM_EPOCH = 2
# define network
NUM_FEATURE = 3
NUM_NEURAL = 100
NUM_DENSE = 100
# define the delay and predict number ahead
DELAY = 100
PREDICT = 30

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# load raw data, return smoothed array [Ft, Fs, Depth]
def loaddata(dir):
    filenames = glob.glob(dir + "/*.csv")
    dataset = DataFrame()
    list_ = []
    for file_ in filenames:
        df = read_csv(file_,index_col=None, header=0)
        list_.append(df)
    dataset = concat(list_)
    # drop rows with NaN values
    dataset.dropna(inplace=True)
    rawvalues = dataset.values
    Ft, Fs, Depth = rawvalues[:, 5], rawvalues[:, 6], rawvalues[:, 44]
    #pyplot.plot(Ft)
    #pyplot.plot(Fs)
    #pyplot.plot(Depth)
    #pyplot.show()
    #smooth data
    for i in range(len(Depth)):
        if Depth[i] <= 0:  # or Fs[i] <= 10:
            Depth[i] = 0
    Ft_smooth, Fs_smooth, Depth_smooth = smooth(Ft), smooth(Fs), smooth(Depth)
    #pyplot.plot(Ft_smooth)
    #pyplot.plot(Fs_smooth)
    #pyplot.plot(Depth_smooth)
    #pyplot.show()
    value = array([Ft_smooth, Fs_smooth, Depth_smooth]).T
    # ensure all data is float
    value = value.astype('float32')
    return value

# smooth the raw data
def smooth(x, window_len=10, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.

    http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = numpy.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = numpy.convolve(w / w.sum(), s, mode='valid')
    return y