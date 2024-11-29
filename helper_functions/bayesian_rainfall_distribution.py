import pandas as pd
import numpy as np
import importlib
import os
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as at
import arviz as az
import pickle
import re
import fnmatch
from math import ceil

def plot_posterior(trace,save_fp=None):
    """ plot posterior distribution automatically
    Args:
        trace (pymc's trace)
        save_fp (str): filepath and filename to save the figure
    """
    posterior_vars = [i for i in trace.posterior.variables.keys() if not bool(re.search('chain|draw|dim_',i))]
    print(posterior_vars)
    n_plots = len(trace)
    ncols = 2
    nrows = ceil(n_plots/ncols)
    fig, axes = plt.subplots(nrows,ncols, figsize=(ncols*4, nrows*3))
    for var, ax in zip(posterior_vars, axes.flatten()):
        az.plot_posterior(trace, var_names=[var], kind='hist',bins=25,ax=ax)
    plt.tight_layout()
    if save_fp is not None:
        save_fp = os.path.splitext(save_fp)[0]
        fig.savefig(f'{save_fp}.png', bbox_inches = 'tight')
    plt.show()

def plot_observed_modelled_rainfall(observed_rainfall, modelled_rainfall, 
                                    x_label='Rainfall (mm)',
                                    label_observed = 'observed',
                                    label_modelled_rainfall = 'modelled_averaged',
                                    title='Comparison of distribution between observed and modelled intense rainfall',
                                    percentiles=[50, 75, 90, 95, 99],
                                    save_fp = None):
    """ 
    Args:
        observed_rainfall (np.array): observed historical rainfall from weather stations
        modelled_rainfall (np.array): modelled intense rainfall is derived from bayesian modelling that infers intense rainfall from total daily rainfall
        percentiles (np.array or list): percentile values
    Returns:
        dict: percentiles and associated percentile values of the modelled intense rainfall
    """
    observed_percentiles = np.percentile(observed_rainfall,percentiles)
    modelled_percentiles = np.percentile(modelled_rainfall,percentiles)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.hist(observed_rainfall,
            density=True,histtype='stepfilled',
            label=label_observed)

    ax.hist(modelled_rainfall,
            density=True,histtype='stepfilled',
            label=label_modelled_rainfall,alpha=0.7)
    ax.legend(bbox_to_anchor = (0.9,-0.15), ncol=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel('PDF')
    ax.set_title(title)
    # add text box for the statistics
    stats = []
    for i,perc in enumerate(percentiles):
        s = f'{perc}th percentile: ' '$R_{total} = $' f'{observed_percentiles[i]:.2f}, ' '$R_{intense} = $' f'{modelled_percentiles[i]:.2f}\n'
        # s = f'{perc}th percentile: Obs: {observed_percentiles[i]:.2f}, Pred: {modelled_percentiles[i]:.2f}\n'
        stats.append(s)
    stats = ''.join(stats)
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    ax.text(0.35, 0.65, stats, fontsize=9, bbox=bbox,
            transform=ax.transAxes, horizontalalignment='left')
    plt.tight_layout()
    if save_fp is not None:
        save_fp = os.path.splitext(save_fp)[0]
        fig.savefig(f'{save_fp}.png', bbox_inches = 'tight')
    plt.show()
    return {'percentile':percentiles,'percentile_values':modelled_percentiles}

def bayesian_rainfall_ratio(observed_intense_rainfall,
                            observed_total_daily_rainfall,
                            future_total_daily_rainfall,
                            alpha_mu=1,alpha_tau=0.5,
                            beta_mu=1,beta_tau=0.5,n_sampling=10000):
    """ 
    Bayesian model for rainfall ratios and derived rainfall.
    
    Args:
        observed_intense_rainfall (np.array): Observed intense rainfall e.g. max 30, 60, 120 mins
        observed_total_daily_rainfall (np.array): Observed total daily rainfall.
        future_total_daily_rainfall (np.array): future total daily rainfall
        alpha1, alpha2 (float): Hyperparameters for alpha prior (Gamma distribution).
        beta1, beta2 (float): Hyperparameters for beta prior (Gamma distribution).
        larger alpha and beta values lead to a more peaked distribution
        n_sampling (int): Number of MCMC samples to draw.
    
    Returns:
        trace: PyMC trace object with posterior samples.
    """
    # to ensure global reproducibility
    np.random.seed(42)
    # calculate rainfall
    rainfall_ratio = observed_intense_rainfall/observed_total_daily_rainfall
    # clip values such that ratio is not zero, values out the interval are clipped to the interval edges 
    # since beta distribution is unstable at values 0 and 1
    rainfall_ratio = np.clip(rainfall_ratio, 1e-6, 1 - 1e-6)
    print(observed_total_daily_rainfall.shape)
    with pm.Model() as model:
        # hyperpriors for alpha and beta
        # alpha_prior = pm.Gamma("alpha_prior", alpha=alpha1,beta=alpha2)
        # beta_prior = pm.Gamma("beta_prior", alpha=beta1,beta=beta2)
        beta_prior = pm.Normal("beta_prior", mu=beta_mu, tau=beta_tau, initval=beta_mu)
        alpha_prior = pm.Normal("alpha_prior", mu=alpha_mu, tau=alpha_tau, initval=alpha_mu)
        # beta distribution for ratios (a random variable), connect model with observed data
        # Assume that rainfall ratio follows a beta distribution
        ratio_dist = pm.Beta("modelled_ratio",alpha=alpha_prior,beta=beta_prior)
        # observed ratio, use observed data to update the posterior distribution of alpha and beta distribution
        observed_ratio = pm.Beta("observed_ratio",alpha=alpha_prior, beta=beta_prior,
                                 observed=rainfall_ratio)

        # derived rainfall from ratio and total daily rainfall to obtain intense rainfall distribution
        derived_rainfall = pm.Deterministic("derived_rainfall",ratio_dist*future_total_daily_rainfall)

        # sampling of posterior
        trace = pm.sample(n_sampling)#, step=step, initvals=start) # MCMC sampling 

    return trace

def joint_rainfall_model(observed_intense_rainfall,
                            observed_total_daily_rainfall,
                            future_total_daily_rainfall,
                            alpha_mu=1,alpha_tau=0.5,
                            beta_mu=1,beta_tau=0.5,
                            n_sampling=10000, target_accept = 0.9):
    """ 
    Bayesian model for rainfall ratios and derived rainfall.
    
    Args:
        observed_intense_rainfall (np.array): Observed intense rainfall e.g. max 30, 60, 120 mins
        observed_total_daily_rainfall (np.array): Observed total daily rainfall.
        future_total_daily_rainfall (np.array): future total daily rainfall
        alpha_mu, alpha_tau (float): Hyperparameters for alpha prior (normal distribution).
        beta_mu, beta_tau (float): Hyperparameters for beta prior (normal distribution).
        n_sampling (int): Number of MCMC samples to draw.
        target_accept (float): increasing target sampler make the sampler take smaller steps, reducing the chance of divergences
    
    Returns:
        trace: PyMC trace object with posterior samples.
        predicted_intense_rainfall (np.array): predicted intense rainfall given future total daily rainfall
    """
    # to ensure global reproducibility
    np.random.seed(42)
    
    with pm.Model() as model:
        # Observed daily rainfall as mutable data
        x = pm.MutableData("observed_daily_rainfall", observed_total_daily_rainfall)
        
        # Priors for the parameters of the rainfall ratio (Beta distribution)
        alpha_prior = pm.Normal("alpha", mu=alpha_mu, tau=alpha_tau)
        beta_prior = pm.Normal("beta", mu=beta_mu, tau=beta_tau)
        
        # Define rainfall ratio as a Beta random variable
        rainfall_ratio = pm.Beta("rainfall_ratio", alpha=alpha_prior, beta=beta_prior)
        
        # Define intense rainfall as the product of the ratio and total daily rainfall
        modelled_intense_rainfall = pm.Deterministic(
            "modelled_intense_rainfall", rainfall_ratio * x
        )
        
        # Likelihood: Observed intense rainfall
        pm.Gamma(
            "obs_intense_rainfall",
            mu=modelled_intense_rainfall,
            sigma=0.1 * modelled_intense_rainfall,
            observed=observed_intense_rainfall,
        )
        
        # Sample posterior
        start = pm.find_MAP() # maximum a posterior
        step = pm.Metropolis() #MCMC sampling
        trace = pm.sample(n_sampling, step=step, initvals=start) # MCMC sampling 
        # trace = pm.sample(n_sampling,target_accept=target_accept)

        # Update the total daily rainfall data for future predictions
        pm.set_data({"observed_daily_rainfall": future_total_daily_rainfall})
        
        # Generate posterior predictive samples
        posterior_predictive = pm.sample_posterior_predictive(trace, var_names=["modelled_intense_rainfall"])
        predicted_intense_rainfall = posterior_predictive.posterior_predictive["modelled_intense_rainfall"]
    
    return trace, predicted_intense_rainfall

def zero_inflated_rainfall_distribution(observed_total_daily_rainfall,modelled_intense_rainfall):
    """ Introduce simulated non-rainy data points to the distribution since beta distribution does not accept zero values
    Args:
        observed_total_daily_rainfall (np.array): Observed total daily rainfall.
        modelled_intense_rainfall (np.array): modelled intense rainfall from joint probability distribution
    Returns:
        tuple: (valid total daily rainfall,zero-inflated distribution that accounts proportionally the non-rainy days)
    """
    # remove NA values
    valid_total_daily_rainfall = observed_total_daily_rainfall[~np.isnan(observed_total_daily_rainfall)]
    n_data = valid_total_daily_rainfall.shape[0] # assume to be 100% of the data
    # assume that values below 1e-6 is non-rainfall data
    n_zero_data = np.sum(valid_total_daily_rainfall < 1e-6)
    # proportion_zero = n_zero_data/n_data # proportion of zero data
    proportion_non_zero = (n_data - n_zero_data)/n_data

    n_data_modelled = modelled_intense_rainfall.shape[0]
    n_zeros = np.zeros(int(1/proportion_non_zero*n_data_modelled))
    # returns distribution excluding NA data, zero-inflated modelled distribution
    return valid_total_daily_rainfall,np.concatenate((n_zeros,modelled_intense_rainfall))

