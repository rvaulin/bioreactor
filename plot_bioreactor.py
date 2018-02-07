#!/usr/bin/python


import sys
import exceptions
from optparse import *
import glob
import re
import os

import numpy as np
import cPickle

from matplotlib import pyplot as plt



def plot_time_evolution(times, substrate_ts, culture_ts, title="Time evolution", grid=True, savefig=False,
                        output_dir="./Images", **kwargs):
    plt.figure()
    plt.plot(times, substrate_ts, label='substrate', linestyle="dashed", color="black", **kwargs)
    plt.plot(times, culture_ts, label='culture', linestyle="solid", color="black", **kwargs)
    plt.title(title)
    plt.legend(loc=0)
    plt.xlabel("time, hours")
    plt.ylabel("concentration")
    plt.grid(grid)
    if savefig:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        plt.savefig(output_dir + "/time_evolution.png")


def plot_specific_growth_function(specific_growth_func, specific_growth_func_params, substrate_min=0.1, substrate_max=1.0, num_samples=100,
                                  title="Specific growth function", grid=True, savefig=False, output_dir="./Images",
                                  **kwargs):
    plt.figure()
    substrate = np.linspace(substrate_min, substrate_max, num_samples)
    specific_growth = specific_growth_func(substrate, specific_growth_func_params)
    plt.plot(substrate, specific_growth, linestyle="solid", color="black", **kwargs)
    plt.title(title)
    plt.legend(loc=0)
    plt.xlabel("substrate")
    plt.ylabel("specific growth")
    plt.grid(grid)
    if savefig:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        plt.savefig(output_dir + "/specific growth_function.png")




