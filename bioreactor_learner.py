#!/usr/bin/python


import sys
import exceptions
import glob


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import copy
import cPickle
import bioreactor_simulation_models
from bioreactor_simulation_models import evolution_to_dataframe

def generate_evolution_data(num_evolutions=10):
    initial_substrate = 1.0
    initial_culture = 0.2

    reference_ph = 7.0
    reference_temperature = 37.0
    data = None
    for i in range(num_evolutions):
        (evolutions_vs_temperature, evolutions_vs_ph) = bioreactor_simulation_models.run_family_of_models(initial_substrate,
                                                                                                      initial_culture)
        for temperature in evolutions_vs_temperature.keys():
            if data is not None:
                data = data.append(evolutions_vs_temperature[temperature])
            else:
                data = evolutions_vs_temperature[temperature]


        for ph in evolutions_vs_ph.keys():
            if data is not None:
                data = data.append(evolutions_vs_ph[ph])
            else:
                data = evolutions_vs_ph[ph]
    return data




def split_data_into_features_and_targets(data,
                                         features=['culture', 'ph', 'substrate', 'temperature'],
                                         targets=['culture_prime', 'substrate_prime']):
    return data[features], data[targets]




def train_learner(data,
                  features=['culture', 'ph', 'substrate', 'temperature'],
                  targets=['culture_prime', 'substrate_prime']):
    feature_data, target_data = split_data_into_features_and_targets(data, features=features, targets=targets)
    feature_scaler = StandardScaler()
    feature_scaler.fit(feature_data)
    target_scaler = StandardScaler()
    target_scaler.fit(target_data)
    scaled_feature_data = feature_scaler.transform(feature_data)
    scaled_target_data = target_scaler.transform(target_data)
    regressor = MLPRegressor(activation='tanh')
    regressor.fit(scaled_feature_data, scaled_target_data)
    return (feature_scaler, target_scaler, regressor)


def learner_predict(feature_scaler, target_scaler, learner, data,
                    features=['culture', 'ph', 'substrate', 'temperature'],
                    targets=['culture_prime', 'substrate_prime']):
    feature_data, target_data = split_data_into_features_and_targets(data, features=features, targets=targets)
    scaled_feature_data = feature_scaler.transform(feature_data)
    predicted_data = target_scaler.inverse_transform(learner.predict(scaled_feature_data))
    return pd.DataFrame(predicted_data, columns=targets)


def learner_hamiltonian(substrate, culture, model_params, noise_covariance=None):
    feature_scaler = model_params['feature_scaler']
    target_scaler = model_params['target_scaler']
    learner = model_params['learner']
    temperature = model_params['temperature']
    ph = model_params['ph']
    features = ['culture', 'ph', 'substrate', 'temperature']
    targets = ['culture_prime', 'substrate_prime']
    feature_data = np.array([[culture, ph, substrate, temperature]])
    scaled_feature_data = feature_scaler.transform(feature_data)
    predicted_data = target_scaler.inverse_transform(learner.predict(scaled_feature_data))
    culture_prime = predicted_data[0][0]
    substrate_prime = predicted_data[0][1]
    return (substrate_prime, culture_prime)




def run_family_predicted_models(initial_substrate, initial_culture, feature_scaler, target_scaler, learner,
                         temperatures=[30.0, 33.0, 37.0, 40.0],
                         phs=[6.7, 7.0, 7.3, 7.6],
                         time_step=1.0,
                         number_of_time_steps=100):

    reference_ph = 7.0
    reference_temperature = 37.0

    evolutions_vs_temperature = {}
    for temperature in temperatures:
        model_params = {'feature_scaler': feature_scaler,
                        'target_scaler': target_scaler,
                        'learner': learner,
                        'ph': reference_ph,
                        'temperature': temperature}

        (times, substrate_ts, culture_ts, substrate_prime_ts, culture_prime_ts) = \
            bioreactor_simulation_models.run_standard_first_order_model(learner_hamiltonian,
                                           model_params,
                                           initial_substrate,
                                           initial_culture,
                                           time_step=time_step,
                                           number_of_time_steps=number_of_time_steps)
        evolution_tuple = (times, substrate_ts, culture_ts, substrate_prime_ts, culture_prime_ts)
        evolutions_vs_temperature[temperature] = evolution_to_dataframe(temperature, reference_ph,
                                                                        evolution_tuple)

    evolutions_vs_ph = {}
    for ph in phs:
        model_params = {'feature_scaler': feature_scaler,
                        'target_scaler': target_scaler,
                        'learner': learner,
                        'ph': ph,
                        'temperature': reference_temperature}
        (times, substrate_ts, culture_ts, substrate_prime_ts, culture_prime_ts) = \
            bioreactor_simulation_models.run_standard_first_order_model(learner_hamiltonian,
                                                                        model_params,
                                                                        initial_substrate,
                                                                        initial_culture,
                                                                        time_step=time_step,
                                                                        number_of_time_steps=number_of_time_steps)
        evolution_tuple = (times, substrate_ts, culture_ts, substrate_prime_ts, culture_prime_ts)
        evolutions_vs_ph[ph] = evolution_to_dataframe(reference_temperature, ph,
                                                      evolution_tuple)

    return (evolutions_vs_temperature, evolutions_vs_ph)


def predict_on_grid(substrate, culture, feature_scaler, target_scaler, learner,
                    temperatures=[30.0, 33.0, 37.0, 40.0],
                    phs=[6.7, 7.0, 7.3, 7.6],
                    time_step=1.0):
    predictions = {}
    reference_temperature = 37.0
    reference_ph = 7.0
    for temperature in temperatures:
        model_params = {'feature_scaler': feature_scaler,
                        'target_scaler': target_scaler,
                        'learner': learner,
                        'ph': reference_ph,
                        'temperature': temperature}
        (substrate_prime, culture_prime) = learner_hamiltonian(substrate, culture, model_params)
        next_culture = culture + culture_prime * time_step
        next_substrate = substrate + substrate_prime * time_step
        predictions[(temperature, reference_ph)] = (next_substrate, next_culture)
    for ph in phs:
            model_params = {'feature_scaler': feature_scaler,
                            'target_scaler': target_scaler,
                            'learner': learner,
                            'ph': ph,
                            'temperature': reference_temperature}
            (substrate_prime, culture_prime) = learner_hamiltonian(substrate, culture, model_params)
            next_culture = culture + culture_prime * time_step
            next_substrate = substrate + substrate_prime * time_step
            predictions[(reference_temperature, ph)] = (next_substrate, next_culture)

    return predictions

def optimal_predicted_params(substrate, culture, feature_scaler, target_scaler, learner,
                             temperatures=[30.0, 33.0, 37.0, 40.0],
                             phs=[6.7, 7.0, 7.3, 7.6], time_step=1.0):
    predictions = predict_on_grid(substrate, culture, feature_scaler, target_scaler, learner,
                                  temperatures=temperatures, phs=phs, time_step=time_step)
    params = predictions.keys()
    best_point = params[0]
    best_prediction = predictions[best_point]
    for param in params[1:]:

        if predictions[param][1] > best_prediction[1]:
            best_point = param
            best_prediction = predictions[param]

    return best_point, best_prediction


def generate_optimal_evolution(initial_substrate, initial_culture,
                               feature_scaler, target_scaler, learner,
                               temperatures=[30.0, 33.0, 37.0, 40.0],
                               phs=[6.7, 7.0, 7.3, 7.6],
                               time_step=1.0,
                               number_of_time_steps=100):
    current_substrate = initial_substrate
    current_culture = initial_culture
    current_time = 0.0
    times = [current_time]
    substrate_ts = [current_substrate]
    culture_ts = [current_culture]
    substrate_prime_ts = []
    culture_prime_ts = []
    initial_temperature = 37.0
    initial_ph = 7.0
    temperature_ts = []
    ph_ts = []
    for step in range(number_of_time_steps):
        if step != 0:
            (best_temp, best_ph), best_prediction = optimal_predicted_params(current_substrate, current_culture,
                                                                         feature_scaler, target_scaler, learner,
                                                                         temperatures=[30.0, 33.0, 37.0, 40.0],
                                                                         phs=[6.7, 7.0, 7.3, 7.6], time_step=1.0)
            current_temperature = best_temp
            current_ph = best_ph
        else:
            current_temperature = initial_temperature
            current_ph = initial_ph
        model_params = bioreactor_simulation_models.get_model_params(current_temperature, current_ph)
        (substrate_prime, culture_prime) = bioreactor_simulation_models.standard_first_order_model(current_substrate,
                                                                                                   current_culture,
                                                                                                   model_params)

        substrate_prime_ts.append(substrate_prime)
        culture_prime_ts.append(culture_prime)

        current_time += time_step
        current_substrate += substrate_prime * time_step
        current_culture += culture_prime * time_step

        times.append(current_time)
        substrate_ts.append(current_substrate)
        culture_ts.append(current_culture)

        temperature_ts.append(current_temperature)
        ph_ts.append(current_ph)


    substrate_prime_ts.append(0.0)
    culture_prime_ts.append(0.0)
    temperature_ts.append(current_temperature)
    ph_ts.append(current_ph)

    return (times, substrate_ts, culture_ts, substrate_prime_ts, culture_prime_ts, temperature_ts, ph_ts)