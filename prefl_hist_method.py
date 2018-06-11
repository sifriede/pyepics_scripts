# coding: utf-8
import datetime
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats


def prefl_plot_hist(input, output, amp, power_aver, timestamp, mean, std, no_mean = 10000, delay = 0.2, scale=1e6):
    # Data
    data = input * scale
    mean = mean * scale
    std = std * scale

    # Plot
    fig, ax = plt.subplots(figsize=(12,6), dpi=300)
    if scale == 1e6:
        unit = "\u03bcW"
    else:
        unit = " W/{}".format(scale)
     
    ax.set_xlabel("Reflected laser power in {}".format(unit))
    ax.set_ylabel("Counts".format(unit))
    ax.set_title("Laser amplitude: {} <=> ".format(amp) +
                     "({:3.3g} +/- {:3.3g}) {} ({:.3f}%)".format(mean, std, unit, (std/mean)*100))
    options = { 'bins': None, 
            'hist': True,
            'norm_hist': False,
            'kde' : False, 
            'ax' : ax, 
            'hist_kws' : {"rwidth":0.75,'edgecolor':'black', 'alpha':.75}}
    # #ax1
    plot = sns.distplot(data, label= "Reflected laser power", **options)
    ax.legend(loc='upper right')
    plt.figtext(0.01, 0.01, 
        "Measured: {}, laser amplitude: {}, power meter averaging: {}, ".format(timestamp, amp, power_aver) +
        "number of measurements: {}, delay: {}s".format(no_mean, delay) +
        "\nmean +/- std = ({:3.3g} +/- {:3.3g}) {} ({:.3f}%)".format(mean, std, unit, (std/mean)*100),
        fontsize=6)
    plt.savefig(output, dpi='figure')