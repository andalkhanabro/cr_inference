import jax
import matplotlib.pyplot as plt
import nifty8.re as jft
from jax import numpy as jnp
from jax import random
import os
import subprocess
from nifty8.re.prior import *
from truncated_normal_prior import TruncatedNormalPrior
import numpy as np
jax.config.update("jax_enable_x64", True)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm" 
from signal_model_modified import SignalModel
from inference_utils import *

# priors

xmax1_prior = TruncatedNormalPrior(mean= 380, std= 120, a_min= 200, a_max= 550, name = "xmax1", shape=(1,)) 
delta_xmax_prior = TruncatedNormalPrior(mean = 450, std = 150, a_min = 200, a_max = 700, name = "delta_xmax", shape=(1,))
# nmax1_prior = LogNormalPrior(mean = np.log(2e5), std = 4, name = "nmax1", shape=(1,))
nmax1_prior = UniformPrior(a_min = 1e5, a_max = 4e5)
n_fac_prior = TruncatedNormalPrior(mean = 0.8, std = 0.4, a_min = 0.35, a_max = 0.9, name = "n_fac", shape=(1,))

# signal model for mock data 

signal_model = SignalModel(xmax1_prior, nmax1_prior, delta_xmax_prior, n_fac_prior)   

# generating a seed for inference 

seed = 50
key = random.PRNGKey(seed)
key, subkey = jax.random.split(key)
xi = jft.random_like(subkey, signal_model.domain)

# parameters for a single draw of the mock data 

parameters = {
    "xmax1": 600,
    "delta_xmax": 50,
    "nmax1": 1e5,
    "nmax_fac": 0.85
}

mock_data, db_info = signal_model.call_with_parameters(parameters) # these are just efield traces 
(long_profile_1, long_profile_2, grammages, template_time_traces, e_field_traces_1, e_field_traces_2, template) = db_info
tag = "mock_data_2"
title = r"Mock Data 2; $X_{max_1} = 600$, $\Delta X_{max} = 50$, $N_{max_1} = 1e^5$, $N_{fac} = 0.85$"

plot_mock_data = False

if plot_mock_data:

    plot_mock_data(long_profile_1, 
    long_profile_2, 
    grammages, 
    template_time_traces, 
    e_field_traces_1, 
    e_field_traces_2, 
    template, 
    tag, 
    title)
    

# likelihood

noise_cov = lambda x: 0.1**2 * x
noise_cov_inv = lambda x: 0.1**-2 * x

# create synthetic data

key, subkey = random.split(key)
noise_truth = (
    (noise_cov(jft.ones_like(signal_model.target))) ** 0.5
) * jft.random_like(key, signal_model.target)

data = mock_data + noise_truth
lh_model = jft.Gaussian(data, noise_cov_inv).amend(signal_model)

# inference 

n_vi_iterations = 6                     
delta = 1e-4
n_samples = 0                  
key, k_i, k_o = random.split(key, 3)

# NOTE, changing the number of samples always triggers a resampling even if
# `resamples=False`, as more samples have to be drawn that did not exist before.
samples, state = jft.optimize_kl(
    lh_model,
    jft.Vector(lh_model.init(k_i)),  # Initial point for the optimization
    n_total_iterations=n_vi_iterations,
    n_samples=lambda i: n_samples // 2 if i < 2 else n_samples,
    # Source for the stochasticity for sampling
    key=k_o,
    # Names of parameters that should not be sampled but still optimized
    # can be specified as point_estimates (effectively we are doing MAP for
    # these degrees of freedom).
    # point_estimates=("cfax1flexibility", "cfax1asperity"),
    # Arguments for the conjugate gradient method used to drawing samples
    draw_linear_kwargs=dict(
        cg_name="SL",
        cg_kwargs=dict(absdelta=delta * jft.size(lh_model.domain) / 10.0, maxiter=100),
    ),
    # Arguments for the minimizer of the KL-divergence cost potential
    kl_kwargs=dict(
        minimize_kwargs=dict(
            name="M", xtol=delta, cg_kwargs=dict(name=None), maxiter=35
        )
    ),
    sample_mode="linear_resample",
    odir="results_intro",
    resume=False,
)












