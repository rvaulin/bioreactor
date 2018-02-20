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



def plot_time_evolution(evolution_dataframe, title="Time evolution", grid=True, savefig=False,
                        output_dir="./Images", **kwargs):
    """
    Generates plot of time evolution for substrate and culture.
    :param times:
    :param substrate_ts:
    :param culture_ts:
    :param title:
    :param grid:
    :param savefig:
    :param output_dir:
    :param kwargs:
    :return:
    """
    plt.figure()
    plt.plot(evolution_dataframe['times'], evolution_dataframe['substrate'], label='substrate', linestyle="dashed", color="black", **kwargs)
    plt.plot(evolution_dataframe['times'], evolution_dataframe['culture'], label='culture', linestyle="solid", color="black", **kwargs)
    plt.title(title)
    plt.legend(loc=0)
    plt.xlabel("time, hours")
    plt.ylabel("concentration")
    plt.grid(grid)
    if savefig:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        plt.savefig(output_dir + "/time_evolution.png")


def plot_family_of_time_evolutions(evolutions, variable_name, markers, predicted_evolutions=None, optimal_evolution=None,
                                   title="Time evolution", grid=True, savefig=False,
                                   output_dir="./Images", **kwargs):
    plt.figure()
    for (variable, evolution_dataframe) in evolutions.items():
        plt.plot(evolution_dataframe['times'], evolution_dataframe['culture'], label=variable_name + " = " + str(variable),
                 marker=markers[variable], linestyle="solid", color="black", **kwargs)
        if predicted_evolutions:
            predicted_evolution_dataframe = predicted_evolutions[variable]
            plt.plot(predicted_evolution_dataframe['times'], predicted_evolution_dataframe['culture'], label=variable_name + " = " + str(variable),
                     marker=markers[variable], linestyle="solid", color="blue", **kwargs)
    if optimal_evolution:
        plt.plot(optimal_evolution[0], optimal_evolution[1],
                 label="optimal",
                 marker="*", linestyle="solid", color="red", **kwargs)
    plt.title(title)
    plt.legend(loc=0)
    plt.xlabel("time, hours")
    plt.ylabel("concentration")
    plt.grid(grid)
    if savefig:
        if predicted_evolutions and optimal_evolution:
            filename = "time_evolution_with_predicion_and_optimal_vs_%s.png" % variable_name
        elif predicted_evolutions:
            filename = "time_evolution_with_predicion_vs_%s.png" % variable_name
        else:
            filename = "time_evolution_vs_%s.png" % variable_name
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        plt.savefig(output_dir + "/" + filename)


def plot_specific_growth_function(specific_growth_func, specific_growth_func_params, substrate_min=0.1, substrate_max=1.0, num_samples=100,
                                  title="Specific growth function", grid=True, savefig=False, output_dir="./Images",
                                  **kwargs):
    """
    Generates plot of specific growth as function of substrate concentration.
    :param specific_growth_func:
    :param specific_growth_func_params:
    :param substrate_min:
    :param substrate_max:
    :param num_samples:
    :param title:
    :param grid:
    :param savefig:
    :param output_dir:
    :param kwargs:
    :return:
    """
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




