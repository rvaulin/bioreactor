#!/usr/bin/python


import sys
import exceptions
from optparse import *
import glob
import re

import numpy as np
import pandas as pd
import copy
import cPickle



def standard_first_order_model(substrate, culture, model_params, noise_covariance=None):
    """
    Standard first-order model of evolution of the bioreactor process.
    :param substrate:
    :param culture:
    :param specific_growth_rate:
    :param alpha:
    :param decay_rate:
    :return: tuple of first time derivatives of substrate and culture
    """
    specific_growth_rate_func = model_params['specific_growth_rate_func']
    specific_growth_rate_params = model_params['specific_growth_rate_params']
    decay_rate_func = model_params['decay_rate_func']
    decay_rate_params = model_params['decay_rate_params']

    alpha = model_params['alpha']
    decay_rate = decay_rate_func(decay_rate_params)

    specific_growth_rate = specific_growth_rate_func(substrate, specific_growth_rate_params)

    substrate_prime = - alpha * specific_growth_rate * culture
    culture_prime = specific_growth_rate * culture - decay_rate * culture
    if noise_covariance is not None:
        noise = np.random.multivariate_normal([0.0, 0.0], noise_covariance, 1)[0]
        substrate_prime_dev, culture_prime_dev = (noise[0], noise[1])
        substrate_prime += substrate_prime_dev
        culture_prime += culture_prime_dev
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

def monoid_quadratic_specific_growth_rate(substrate, params):
    physical_state_vector = params['physical_state_vector']
    rate_inverse_covariance_matrix = params['rate_inverse_covarience_matrix']
    rate_shift_vector = params['shift_vector']
    reference_mu_star = params['reference_mu_star']
    shifted_rate_state_vector = physical_state_vector - rate_shift_vector
    mu_star = reference_mu_star * (1.0 - np.dot(shifted_rate_state_vector, np.dot(rate_inverse_covariance_matrix, shifted_rate_state_vector)))
    k_inverse_covariance_matrix = params['k_inverse_covarience_matrix']
    k_shift_vector = params['k_shift_vector']
    reference_k = params['reference_k']
    shifted_k_state_vector = physical_state_vector - k_shift_vector
    K = reference_k * (1.0 + np.dot(shifted_k_state_vector, np.dot(k_inverse_covariance_matrix, shifted_k_state_vector)))
    return mu_star * substrate / (K + substrate)

def constant_decay_rate(params):
    return params['decay_rate']

def quadratic_decay_rate(params):
    physical_state_vector = params['physical_state_vector']
    decay_inverse_covariance_matrix = params['decay_inverse_covarience_matrix']
    decay_shift_vector = params['shift_vector']
    reference_decay_rate = params['reference_decay_rate']
    shifted_rate_state_vector = physical_state_vector - decay_shift_vector
    decay_rate = reference_decay_rate * (1.0 - np.dot(shifted_rate_state_vector,
                                                      np.dot(decay_inverse_covariance_matrix, shifted_rate_state_vector)))
    #print "PH = ", physical_state_vector[1]
    #print "shifted = ", shifted_rate_state_vector[1]
    #print "decay rate = ", decay_rate
    return decay_rate



def run_standard_first_order_model(hamiltionian_func, model_params,  initial_substrate, initial_culture, time_step=1.0,
                                   number_of_time_steps=100, noise_covariance=None):
    current_substrate = initial_substrate
    current_culture = initial_culture
    current_time = 0.0
    times = [current_time]
    substrate_ts = [current_substrate]
    culture_ts = [current_culture]
    substrate_prime_ts = []
    culture_prime_ts = []

    for step in range(number_of_time_steps):
        (substrate_prime, culture_prime) = hamiltionian_func(current_substrate, current_culture,
                                                             model_params, noise_covariance=noise_covariance)
        substrate_prime_ts.append(substrate_prime)
        culture_prime_ts. append(culture_prime)
        current_time += time_step
        current_substrate += substrate_prime * time_step
        current_culture += culture_prime * time_step
        times.append(current_time)
        substrate_ts.append(current_substrate)
        culture_ts.append(current_culture)
    substrate_prime_ts.append(0.0)
    culture_prime_ts.append(0.0)
    return (times, substrate_ts, culture_ts, substrate_prime_ts, culture_prime_ts)


def get_model_params(temperature, ph):
    rate_inverse_covarience_matrix = np.array([[0.025, 0.0], [0.0, 2.0]])
    rate_shift_vector = np.array([37.0, 7.1])
    k_inverse_covarience_matrix = np.array([[0.025, 0.0], [0.0, 2.0]])
    # k_inverse_covarience_matrix = np.array([[0.0, 0.0], [0.0, 0.0]])
    k_shift_vector = np.array([35.0, 7.1])

    decay_inverse_covarience_matrix = np.array([[0.025, 0.0], [0.0, 3.0]])
    decay_shift_vector = np.array([37.0, 7.1])
    preset_monoid_params = {'rate_inverse_covarience_matrix': rate_inverse_covarience_matrix,
                            'shift_vector': rate_shift_vector,
                            'reference_mu_star': 3.0,
                            'k_inverse_covarience_matrix': k_inverse_covarience_matrix,
                            'k_shift_vector': k_shift_vector,
                            'reference_k': 7.0
                            }
    preset_decay_params = {'decay_inverse_covarience_matrix': decay_inverse_covarience_matrix,
                           'shift_vector': decay_shift_vector,
                           'reference_decay_rate': 0.1
                           }

    noise_covariance = 0.00001 * np.array([[2.0, 0.0], [0.0, 2.0]])
    # noise_covariance = 0.0 * np.array([[2.0, 0.0], [0.0, 2.0]])
    reference_ph = rate_shift_vector[1]
    reference_temperature = rate_shift_vector[0]
    monoid_params = copy.copy(preset_monoid_params)
    state_vector = [temperature, ph]
    monoid_params['physical_state_vector'] = state_vector
    decay_params = copy.copy(preset_decay_params)
    decay_params['physical_state_vector'] = state_vector
    model_params = {'alpha': 0.5,
                    'decay_rate': 0.1,
                    'specific_growth_rate_func': monoid_quadratic_specific_growth_rate,
                    'specific_growth_rate_params': monoid_params,
                    'decay_rate_func': quadratic_decay_rate,
                    'decay_rate_params': decay_params
                    }
    return model_params

def run_family_of_models(initial_substrate, initial_culture,
                         temperatures=[30.0, 33.0, 37.0, 40.0],
                         phs=[6.7, 7.0, 7.3, 7.6],
                         time_step=1.0,
                         number_of_time_steps=100):

    rate_inverse_covarience_matrix = np.array([[0.025, 0.0], [0.0, 2.0]])
    rate_shift_vector = np.array([37.0, 7.1])
    k_inverse_covarience_matrix = np.array([[0.025, 0.0], [0.0, 2.0]])
    #k_inverse_covarience_matrix = np.array([[0.0, 0.0], [0.0, 0.0]])
    k_shift_vector = np.array([35.0, 7.1])

    decay_inverse_covarience_matrix = np.array([[0.025, 0.0], [0.0, 3.0]])
    decay_shift_vector = np.array([37.0, 7.1])
    preset_monoid_params = {'rate_inverse_covarience_matrix': rate_inverse_covarience_matrix,
                     'shift_vector': rate_shift_vector,
                     'reference_mu_star': 3.0,
                     'k_inverse_covarience_matrix': k_inverse_covarience_matrix,
                     'k_shift_vector': k_shift_vector,
                     'reference_k': 7.0
                     }
    preset_decay_params = {'decay_inverse_covarience_matrix': decay_inverse_covarience_matrix,
                           'shift_vector': decay_shift_vector,
                           'reference_decay_rate': 0.1
                           }

    noise_covariance = 0.00001*np.array([[2.0, 0.0], [0.0, 2.0]])
    #noise_covariance = 0.0 * np.array([[2.0, 0.0], [0.0, 2.0]])
    reference_ph = rate_shift_vector[1]
    reference_temperature = rate_shift_vector[0]

    evolutions_vs_temperature = {}
    for temperature in temperatures:
        monoid_params = copy.copy(preset_monoid_params)
        state_vector = [temperature, reference_ph]
        monoid_params['physical_state_vector'] = state_vector
        decay_params = copy.copy(preset_decay_params)
        decay_params['physical_state_vector'] = state_vector
        model_params = {'alpha': 0.5,
                        'decay_rate': 0.1,
                        'specific_growth_rate_func': monoid_quadratic_specific_growth_rate,
                        'specific_growth_rate_params': monoid_params,
                        'decay_rate_func': quadratic_decay_rate,
                        'decay_rate_params': decay_params
                        }


        (times, substrate_ts, culture_ts, substrate_prime_ts, culture_prime_ts) = \
            run_standard_first_order_model(standard_first_order_model,
                                           model_params,
                                           initial_substrate,
                                           initial_culture,
                                           time_step=time_step,
                                           number_of_time_steps=number_of_time_steps,
                                           noise_covariance=noise_covariance)
        evolution_tuple = (times, substrate_ts, culture_ts, substrate_prime_ts, culture_prime_ts)
        evolutions_vs_temperature[temperature] = evolution_to_dataframe(temperature, reference_ph,
                                                                        evolution_tuple)

    evolutions_vs_ph = {}
    for ph in phs:
        monoid_params = copy.copy(preset_monoid_params)
        state_vector = [reference_temperature, ph]
        monoid_params['physical_state_vector'] = state_vector
        decay_params = copy.copy(preset_decay_params)
        decay_params['physical_state_vector'] = state_vector
        model_params = {'alpha': 0.5,
                        'decay_rate': 0.1,
                        'specific_growth_rate_func': monoid_quadratic_specific_growth_rate,
                        'specific_growth_rate_params': monoid_params,
                        'decay_rate_func': quadratic_decay_rate,
                        'decay_rate_params': decay_params
                        }
        (times, substrate_ts, culture_ts, substrate_prime_ts, culture_prime_ts) = \
            run_standard_first_order_model(standard_first_order_model,
                                           model_params,
                                           initial_substrate,
                                           initial_culture,
                                           time_step=time_step,
                                           number_of_time_steps=number_of_time_steps,
                                           noise_covariance=noise_covariance)
        evolution_tuple = (times, substrate_ts, culture_ts, substrate_prime_ts, culture_prime_ts)
        evolutions_vs_ph[ph] = evolution_to_dataframe(reference_temperature, ph,
                                                      evolution_tuple)

    return (evolutions_vs_temperature, evolutions_vs_ph)



def evolution_to_dataframe(temperature, ph, evolution_data):
    data = {'times': np.empty(0),
        'temperature': np.empty(0),
        'ph': np.empty(0),
        'substrate': np.empty(0),
        'culture': np.empty(0),
        'substrate_prime': np.empty(0),
        'culture_prime': np.empty(0)
    }
    (times, substrate, culture, substrate_prime, culture_prime) = evolution_data
    num_samples = len(culture)
    temp_array = temperature * np.ones(num_samples)
    ph_array = ph * np.ones(num_samples)
    data['times'] = np.append(data['times'], times)
    data['temperature'] = np.append(data['temperature'], temp_array)
    data['ph'] = np.append(data['ph'], ph_array)
    data['substrate'] = np.append(data['substrate'], substrate)
    data['culture'] = np.append(data['culture'], culture)
    data['substrate_prime'] = np.append(data['substrate_prime'], substrate_prime)
    data['culture_prime'] = np.append(data['culture_prime'], culture_prime)
    return pd.DataFrame(data)












