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

def generate_evolution_data(num_evolutions=10):
    initial_substrate = 1.0
    initial_culture = 0.2
    data = {
        'temperature': np.empty(0),
        'ph': np.empty(0),
        'substrate': np.empty(0),
        'culture': np.empty(0),
        'substrate_prime': np.empty(0),
        'culture_prime': np.empty(0)
    }
    reference_ph = 7.0
    reference_temperature = 37.0
    for i in range(num_evolutions):
        (evolutions_vs_temperature, evolutions_vs_ph) = bioreactor_simulation_models.run_family_of_models(initial_substrate,
                                                                                                      initial_culture)
        for temperature in evolutions_vs_temperature.keys():
            (times, substrate, culture, substrate_prime, culture_prime) = evolutions_vs_temperature[temperature]
            num_samples = len(culture)
            temp_array = temperature * np.ones(num_samples)
            ph_array = reference_ph * np.ones(num_samples)
            data['temperature'] = np.append(data['temperature'], temp_array)
            data['ph'] = np.append(data['ph'], ph_array)
            data['substrate'] = np.append(data['substrate'], substrate)
            data['culture'] = np.append(data['culture'], culture)
            data['substrate_prime'] = np.append(data['substrate_prime'], substrate_prime)
            data['culture_prime'] = np.append(data['culture_prime'], culture_prime)

        for ph in evolutions_vs_ph.keys():
            (times, substrate, culture, substrate_prime, culture_prime) = evolutions_vs_ph[ph]
            num_samples = len(culture)
            temp_array = reference_temperature * np.ones(num_samples)
            ph_array = ph * np.ones(num_samples)
            data['temperature'] = np.append(data['temperature'], temp_array)
            data['ph'] = np.append(data['ph'], ph_array)
            data['substrate'] = np.append(data['substrate'], substrate)
            data['culture'] = np.append(data['culture'], culture)
            data['substrate_prime'] = np.append(data['substrate_prime'], substrate_prime)
            data['culture_prime'] = np.append(data['culture_prime'], culture_prime)
    return pd.DataFrame(data)


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
                    targets=['culture_prime', 'substrate_prime'] ):
    feature_data, target_data = split_data_into_features_and_targets(data, features=features, targets=targets)
    scaled_feature_data = feature_scaler.transform(feature_data)
    predicted_data = target_scaler.inverse_transform(learner.predict(scaled_feature_data))
    return pd.DataFrame(predicted_data, columns = targets)


#def evaluate_regression_errors(predicted_data, target_data):




