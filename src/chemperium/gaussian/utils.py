from collections import defaultdict
import numpy as np


def get_dict(values, value_names, min_values: int = 1):
    value_dict = defaultdict(list)
    for name, value in zip(value_names, values):
        value_dict[name].append(value)

    value_dict = dict(value_dict)

    new_value_dict = {}
    for key in value_dict:
        if len(value_dict[key]) > min_values:
            new_value_dict[key] = value_dict[key]

    return new_value_dict


def geometry_type(feature_name):
    type_feature = sum(1 for c in feature_name if c.isupper())
    if type_feature == 4:
        metric = "Dihedral Angle"
        unit = "deg"
    elif type_feature == 3:
        metric = "Angle"
        unit = "deg"
    else:
        metric = "Distance"
        unit = "Å"
    return metric, unit


def gauss(x, x0, sigma):
    """
    This function returns a Gaussian distribution.
    """
    return (1/(sigma*np.sqrt(2 * np.pi))) * np.exp(-(x - x0)**2 / (2 * sigma**2))
