import matplotlib
import matplotlib.pyplot as plt

from collections import OrderedDict

import os

class OrderedDefaultDict(OrderedDict):
    factory = dict

    def __missing__(self, key):
        self[key] = value = self.factory()
        return value


class Plotter():
        _data = OrderedDefaultDict()
        _iter = 0

        def tick(self):
               self._iter += 1

        def plot(self, name, value):
                self._data[name][self._iter] = value

        def save_plots(self, path):
                for name, vals in self._data.items():
                    #prints.append("n{}\t{}".format(name, np.mean(list(vals.values()))))

                    x_vals = list(self._data[name].keys())
                    y_vals = list(self._data[name].values())

                    plt.figure(figsize=[2*6.4, 4.8]) # twice default width
                    ax1 = plt.subplot(1, 2, 1)
                    plt.plot(x_vals, y_vals)
                    plt.xlabel('iteration')
                    plt.ylabel(name)

                    plt.subplot(1, 2, 2)
                    plt.plot(x_vals[len(x_vals)//2:], y_vals[len(y_vals)//2:])
                    plt.xlabel('iteration')
                    ylims = plt.gca().get_ylim()
                    xlims = plt.gca().get_xlim()

                    # rect = matplotlib.patches.Rectangle((xlims[0], ylims[0]),
                    #                              xlims[1]-xlims[0],
                    #                              ylims[1]-ylims[0],
                    #                              linewidth=1,edgecolor='r',facecolor='none')
                    #ax1.add_patch(rect) 
                  
                    plt.savefig(os.path.join(path, name) + '.jpg')
                    plt.close()

