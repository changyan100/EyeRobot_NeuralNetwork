import numpy as np
import matplotlib.pyplot as plt

plt.ion() # interactive mode on
x = np.array([0])
y = np.array([10])
line, = plt.plot(x,y) # plot the data and specify the 2d line
ax = plt.gca() # get most of the figure elements
for tt in range(500):
    new_x = tt
    new_y = tt*5
    if len(x) > 20:
        x = new_x
        y = new_y
    x = np.append(x, new_x)
    y = np.append(y, new_y)
    line.set_xdata(x)
    line.set_ydata(y) # set the curve with new data
    ax.relim() # renew the data limits
    ax.autoscale_view(True, True, True) # rescale plot view
    plt.draw() # plot new figure
    plt.pause(1e-17) # pause to show the figure