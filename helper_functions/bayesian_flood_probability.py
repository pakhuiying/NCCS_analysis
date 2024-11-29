import pandas as pd
import numpy as np
import importlib
import os
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as at
import arviz as az
from scipy.stats.mstats import mquantiles
import pickle

def get_non_flood_dates(weather_dates, flood_dates):
    """ returns a list of non_flood dates
    Args:
        weather_dates (list of str): list of all weather dates
        flood_dates (list of str): list of all flood dates
    Returns:
        list: a list of flood dates (str)
    """
    unique_weather_dates = set(weather_dates)
    unique_flood_dates = set(flood_dates)
    return sorted(unique_weather_dates^unique_flood_dates)

def subset_nonflood_weather(weather_df,non_flood_dates):
    """ returns a pd.DataFrame which shows non flood weather
    Args:
        weather_df (pd.DataFrame): weather df with the column Date
        non_flood_dates (list): list of str of non_flood_dates
    Returns:
        pd.DataFrame: non-flood weather, averaged across weather stations
    """
    return weather_df[weather_df['Date'].isin(non_flood_dates)].groupby('Date').mean(numeric_only=True).reset_index()

def get_binary_flood_occurrence(flood_data, non_flood_data):
    """ returns paired data (tuple) with NAs removed. (binary flood occurrence, corresponding rainfall)
    Args:
        flood_data (dict): where keys = name of drainage catchment, values = df (nx4) of rainfall corresponding to flood events
        non_flood_data (dict): where keys = name of drainage catchment, values = df (nx4) of rainfall corresponding to non flood events
    Returns:
        dict: keys = name of drainage catchment, 2nd-level keys = rainfall type, values = tuple of (binary flood occurrence, corresponding rainfall)
    """
    assert len(list(flood_data)) == len(list(non_flood_data)), f"Number of drainage catchment names do not match: {list(flood_data)}, {list(non_flood_data)}"

    flood_occurrences_dict = {drainage_name: {rf: None for rf in df.columns.to_list()} for drainage_name,df in flood_data.items()}
    for drainage_name, rf_dict in flood_occurrences_dict.items():
        for rainfall_type in list(rf_dict):
            # prepare flood data
            flood_ppt = flood_data[drainage_name][rainfall_type].values
            flood_ppt = flood_ppt[~np.isnan(flood_ppt)] # remove NAs
            flood_binary = np.ones(flood_ppt.shape,dtype=int) # where 1 = flood occurred
            # prepare non-flood data
            non_flood_ppt = non_flood_data[drainage_name][rainfall_type].values
            non_flood_ppt = non_flood_ppt[~np.isnan(non_flood_ppt)] # remove NAs
            non_flood_binary = np.zeros(non_flood_ppt.shape,dtype=int) # where 0 = flood did not occur
            # concat non-flood and flood data
            flood_occurrences_dict[drainage_name][rainfall_type] = (np.hstack([flood_binary,non_flood_binary]),np.hstack([flood_ppt, non_flood_ppt]))

    return flood_occurrences_dict

def bayesian_flood_model(x,y,
                         beta_mu,beta_tau,beta_initval,
                         alpha_mu,alpha_tau,alpha_initval,
                         n_sampling=10000):
    """ returns bayesian model after initialising the priors and fitting a logistic curve
    Args:
        x (np.array): x-variable e.g. precipitation
        y (np.array): y-variable e.g. observations of flood in binary, where 1=flood, 0=non-flood
        beta_mu (float): mean value for prior that determines the steepness of the gradient
        beta_tau (float): precision for prior, the higher the precision, the smaller the standard dev
        beta_initval (float): initial value
        alpha_mu (float): mean value for prior that determines the x-shift of the logistic curve
        alpha_tau (float): precision for prior, the higher the precision, the smaller the standard dev
        alpha_initval (float): initial value
    Returns:
        trace: posterior distribution
    """
    # to ensure global reproducibility
    np.random.seed(42)
    with pm.Model() as model:
        beta = pm.Normal("beta", mu=beta_mu, tau=beta_tau, initval=beta_initval)
        alpha = pm.Normal("alpha", mu=alpha_mu, tau=alpha_tau, initval=alpha_initval)
        # fits a logistic curve where p(x) is the probability of flood given rainfall
        p = pm.Deterministic("p", 1.0/(1. + at.exp(beta*x + alpha)))

    with model:
        # fit a bernoulli distribution where Ber(p) is a random variable that takes 1 with a probability and 0 woth another probability 
        observed = pm.Bernoulli("bernoulli_obs", p, observed=y)
        start = pm.find_MAP() # maximum a posterior
        step = pm.Metropolis() #MCMC sampling
        trace = pm.sample(n_sampling, step=step, initvals=start) # MCMC sampling 
    return trace

def plot_posterior_arviz(trace,title,save_fp=None):
    """ plot distribution of parameters using trace
    Args:
        trace: posterior distribution
        title (str): title of figure
        save_fp (str): filepath of where to save the file
    """
    #Here is the ArviZ version 
    fig,ax = plt.subplots(2,1,figsize=(8,6))
    az.plot_posterior(trace, var_names=['beta'], kind='hist',bins=25,color="#7A68A6",ax=ax[0])
    az.plot_posterior(trace, var_names=['alpha'], kind='hist',bins=25,color="#A60628",ax=ax[1])
    plt.suptitle(f'{title}\n' + r"Posterior distributions of the variables $\alpha, \beta$",
                 fontsize=20)
    ax[0].set_title(r"posterior of $\beta$")
    ax[1].set_title(r"posterior of $\alpha$")
    plt.plot()
    plt.tight_layout()
    if save_fp is not None:
        plt.savefig(f'{save_fp}.png', bbox_inches = 'tight')
    plt.show()
    return

def plot_posterior(trace,title,save_fp=None):
    """ plot distribution of parameters using trace
    Args:
        trace: posterior distribution
        title (str): title of figure
        ax (Axes): if ax is None, plot on a new figure, else, draw on supplied ax
        save_fp (str): filepath of where to save the file
    """
    alpha_samples = np.concatenate(trace.posterior.alpha.data[:,1000::2])[:, None]  # best to make them 1d
    beta_samples = np.concatenate(trace.posterior.beta.data[:,1000::2])[:, None]

    #histogram of the samples:
    fig, axes = plt.subplots(2,1,figsize=(8,6))
    
    fig.suptitle(f'{title}\n' + r"Posterior distributions of the variables $\alpha, \beta$")
    axes[0].hist(beta_samples, histtype='stepfilled', bins=35, alpha=0.85,
            label=r"posterior of $\beta$", color="#7A68A6", density=True)
    axes[1].hist(alpha_samples, histtype='stepfilled', bins=35, alpha=0.85,
            label=r"posterior of $\alpha$", color="#A60628", density=True)
    plt.tight_layout()
    for ax in axes.flatten():
        ax.legend(loc='upper right')
    if save_fp is not None:
        plt.savefig(f'{save_fp}.png', bbox_inches = 'tight')
    plt.show()
    return

def sample_posterior(x,trace):
    """ 
    Args:
        x (np.array): x-variable e.g. precipitation
        trace: posterior distribution from MCMC sampling
    Returns:
        np.arrays: posterior distributions of alpha, beta, and expected probabilities
    """
    # only take samples from the trace from 1000 onwards and sample every other
    alpha_samples = np.concatenate(trace.posterior.alpha.data[:,1000::2])[:, None]  # best to make them 1d
    beta_samples = np.concatenate(trace.posterior.beta.data[:,1000::2])[:, None]
    t = np.linspace(x.min(), x.max(), 50)[:, None]
    p_t = utils.logistic(t.T, beta_samples, alpha_samples)
    return alpha_samples, beta_samples,p_t

def plot_fitted_logistic(x,y,sampled_posterior,title,ax=None,**kwargs):
    """ plot fitted logistic
    Args:
        x (np.array): x-variable e.g. precipitation
        y (np.array): y-variable e.g. observations of flood in binary, where 1=flood, 0=non-flood
        sampled_posterior: posterior distribution sampled from the trace from 1000 onwards and sample every other
        title (str): title of figure
        ax (Axes): if ax is None, plot on a new figure, else, draw on supplied ax
    Returns:
        np.array: returns expected probability
    """
    
    t = np.linspace(x.min(), x.max(), 50)[:, None]
    p_t = sampled_posterior
    # p_t = logistic(t.T, beta_samples, alpha_samples)
    mean_prob_t = p_t.mean(axis=0) # expected probability
    # vectorized bottom and top 2.5% quantiles for "confidence interval"
    qs = mquantiles(p_t, [0.025, 0.975], axis=0)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,6))
    # plot expected probability
    ax.plot(t, mean_prob_t, lw=1, ls="--", color="k",
         label="average posterior probability of flood")
    # plot uncertainty range
    ax.fill_between(t[:, 0], *qs, alpha=0.7,**kwargs)
    ax.plot(t[:, 0], qs[0], label="95% CI",alpha=0.7,**kwargs)
    # plot observations
    ax.scatter(x, y, color="k", s=50, alpha=0.5)
    ax.set_ylabel("probability estimate")
    ax.set_xlabel("Precipitation (mm)")
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    if ax is None:
        plt.legend(loc="lower left")
        plt.show()
    return p_t