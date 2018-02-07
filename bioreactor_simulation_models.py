#!/usr/bin/python


import sys
import exceptions
from optparse import *
import glob
import re

import numpy
import cPickle



def standard_first_order_model(substrate, culture, specific_growth_rate_func, specific_growth_rate_params, model_params):
    """
    Standard first-order model of evolution of the bioreactor process.
    :param substrate:
    :param culture:
    :param specific_growth_rate:
    :param alpha:
    :param decay_rate:
    :return: tuple of first time derivatives of substrate and culture
    """
    alpha = model_params['alpha']
    decay_rate = model_params['decay_rate']

    specific_growth_rate = specific_growth_rate_func(substrate, specific_growth_rate_params)
    substrate_prime = - alpha * specific_growth_rate * culture
    culture_prime = specific_growth_rate * culture - decay_rate * culture
    return (substrate_prime, culture_prime)

def monoid_specific_growth_rate(substrate, params):
    """
    Monoid specific growth rate function.
    :param substrate:
    :return:
    """
    mu_star = params['mu_star']
    K = params['K']
    return mu_star * substrate / (float(K) + substrate)


def run_standard_first_order_model(hamiltionian_func, specific_growth_rate_func, specific_growth_rate_params,
                                   model_params,  initial_substrate, initial_culture, time_step=1.0,
                                   number_of_time_steps=100):
    current_substrate = initial_substrate
    current_culture = initial_culture
    current_time = 0.0
    times = [current_time]
    substrate_ts = [current_substrate]
    culture_ts = [current_culture]

    for step in range(number_of_time_steps):
        (substrate_prime, culture_prime) = hamiltionian_func(current_substrate, current_culture, specific_growth_rate_func, specific_growth_rate_params,
                                   model_params)
        current_time += time_step
        current_substrate += substrate_prime * time_step
        current_culture += culture_prime * time_step
        times.append(current_time)
        substrate_ts.append(current_substrate)
        culture_ts.append(current_culture)
    return (times, substrate_ts, culture_ts)







