import numpy as np
import pandas as pd
from scipy.stats import expon, gamma, norm, poisson, binom

def generate_exponential_dataset(num_samples, exp_lambda):
    exp_data = expon(scale=1/exp_lambda).rvs(size=num_samples).reshape(-1, 1)
    labels = np.random.randint(2, size=num_samples).reshape(-1, 1)  # Generating random binary labels
    df = pd.DataFrame(np.concatenate([exp_data, labels], axis=1), columns=['var_1', 'result'])
    return df

def generate_gamma_dataset(num_samples, gamma_shape, gamma_scale):
    gamma_data = gamma(a=gamma_shape, scale=gamma_scale).rvs(size=num_samples).reshape(-1, 1)
    labels = np.random.randint(2, size=num_samples).reshape(-1, 1)  # Generating random binary labels
    df = pd.DataFrame(np.concatenate([gamma_data, labels], axis=1), columns=['var_1', 'result'])
    return df

def generate_normal_dataset(num_samples, normal_mean, normal_std):
    normal_data = norm(loc=normal_mean, scale=normal_std).rvs(size=num_samples).reshape(-1, 1)
    labels = np.random.randint(2, size=num_samples).reshape(-1, 1)  # Generating random binary labels
    df = pd.DataFrame(np.concatenate([normal_data, labels], axis=1), columns=['var_1', 'result'])
    return df

def generate_poisson_dataset(num_samples, poisson_lambda):
    poisson_data = poisson(mu=poisson_lambda).rvs(size=num_samples).reshape(-1, 1)
    labels = np.random.randint(2, size=num_samples).reshape(-1, 1)  # Generating random binary labels
    df = pd.DataFrame(np.concatenate([poisson_data, labels], axis=1), columns=['var_1', 'result'])
    return df

def generate_binomial_dataset(num_samples, n_trials, p_success):
    binomial_data = binom(n=n_trials, p=p_success).rvs(size=num_samples).reshape(-1, 1)
    labels = np.random.randint(2, size=num_samples).reshape(-1, 1)  # Generating random binary labels
    df = pd.DataFrame(np.concatenate([binomial_data, labels], axis=1), columns=['var_1', 'result'])
    return df

# Example usage:
num_samples = 20000
exp_lambda = 0.5
gamma_shape = 2
gamma_scale = 2
normal_mean = 0
normal_std = 1
poisson_lambda = 3
n_trials = 10
p_success = 0.5

exp_df = generate_exponential_dataset(num_samples, exp_lambda)
gamma_df = generate_gamma_dataset(num_samples, gamma_shape, gamma_scale)
normal_df = generate_normal_dataset(num_samples, normal_mean, normal_std)
poisson_df = generate_poisson_dataset(num_samples, poisson_lambda)
binomial_df = generate_binomial_dataset(num_samples, n_trials, p_success)
