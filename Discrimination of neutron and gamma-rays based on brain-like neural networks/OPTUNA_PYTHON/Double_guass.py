import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def mapminmax(data):
    """ Normalize data to a given range. """
    data_min = np.min(data)
    data_max = np.max(data)
    normalized_data = (data - data_min) / (data_max - data_min)
    return normalized_data

def histogram_fitting_and_compute_fom(pulse_shape_discrimination_factor):
    R = pulse_shape_discrimination_factor

    R = mapminmax(R)
    R = R * 200

    num_components = 2
    gm_model = GaussianMixture(n_components=num_components, max_iter=1000)
    gm_model.fit(R.reshape(-1, 1))

    miu = gm_model.means_.flatten()
    sigma = np.sqrt(gm_model.covariances_).flatten()

    FOM = (miu[1] - miu[0]) / (1.667 * (sigma[1] + sigma[0]))

    return FOM

# Example usage
# pulse_shape_discrimination_factor = np.random.rand(1000)  # Replace with actual data
# The_number_of_bins, miu, sigma, FOM = histogram_fitting_and_compute_fom(pulse_shape_discrimination_factor)