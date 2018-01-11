import numpy as np
import matplotlib.pyplot as plt

def show_tensor(t):

    for key in t:
        data = t[key]
        fig, ax = plt.subplots()
        ytick = range(data.shape[0])

        if len(data.shape) < 2:
            data = np.expand_dims(data, axis=1)
            plt.ylabel("dim 0")
            xtick = [' ']
        else:
            xtick = range(data.shape[1])
            plt.ylabel("dim 0")
            plt.xlabel("dim 1")

        cax = ax.pcolor(data, cmap=plt.cm.seismic, edgecolor='black', linewidth=10, vmin = -3.0, vmax = 3.0)
        fig.colorbar(cax)

        fig.canvas.set_window_title(key)
        fig.set_size_inches(data.shape[1] + 2.0, data.shape[0] + 1, forward=True)
        # put the major ticks at the middle of each cell
        ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
        ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)


        # want a more natural, table-like display
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        ax.set_xticklabels(xtick, minor=False)
        ax.set_yticklabels(ytick, minor=False)

    plt.show()