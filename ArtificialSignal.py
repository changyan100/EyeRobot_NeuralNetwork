import numpy as np
from matplotlib import pyplot

def PeriodicSingnal(num):
    x = np.linspace(-num*np.pi, num*np.pi, 30*num)
    offset = np.linspace(0,1,30*num)
    # Fs = 205*np.sin(0.99*x)+10*np.cos(20*x)
    # Depth = 25*np.sin(1.01*x)+2*np.sin(10*x)
    Fs =20*np.sin(x)+offset*50
    Depth = 5 * np.cos(x) + offset*10
    #Fs = np.absolute(Fs)
    #Depth = np.absolute(Depth)
    fig = pyplot.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(Fs, c='b', label='Fs')
    ax1.set_xlabel('Time(s)')
    ax1.set_ylabel('Force(mN)')
    ax1.legend(loc=2, markerscale=5)
    ax2 = ax1.twinx()
    ax2.plot(Depth, c='r',label='Depth')
    ax2.set_ylabel('Depth(mm)')
    ax2.legend(loc=1, markerscale=5)
    pyplot.show()
    data = np.array([Fs,Depth]).T
    return data