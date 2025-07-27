import jax
import matplotlib.pyplot as plt
import nifty.re as jft
from jax import numpy as jnp
from jax import random
import os
import subprocess
from nifty.re.prior import *
from truncated_normal_prior import TruncatedNormalPrior
jax.config.update("jax_enable_x64", True)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm" 
make_priors = False

# priors as instances of jft.Models

trunc = TruncatedNormalPrior(mean= 3.5, std= 0.2, a_min= 1, a_max= 5) 
uniform = UniformPrior(a_min = 2e5, a_max = 7e5)
log_norm = LogNormalPrior(mean = 2, std = 3)
lap_prior = LaplacePrior(alpha = 3)

seed = 40
key = random.PRNGKey(seed)

key, subkey = jax.random.split(key)

samples=[]
uniforms=[]
log_norms=[]
laps=[]

if make_priors:

    for _ in range(10000):

        key, subkey = random.split(key)
        xi = jft.random_like(subkey, trunc.domain)
        realisation_trunc = trunc(xi)
        realisation_uniform = uniform(xi)
        realisation_ln = log_norm(xi)
        realisation_lap = lap_prior(xi)
        samples.append(realisation_trunc)
        uniforms.append(realisation_uniform)
        log_norms.append(realisation_ln)
        laps.append(realisation_lap)

    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (10,6))
    print(ax.shape)
    ax1, ax2, ax3, ax4 = ax.flatten()

    ax1.hist(samples, color='skyblue', edgecolor='black', bins = 30)
    ax1.set_xlabel("Value Of TNP")
    ax1.set_ylabel("Frequency")

    ax2.hist(uniforms, color='orange', edgecolor='black', bins = 30)
    ax2.set_xlabel("Value Of UP")
    ax2.set_ylabel("Frequency")

    ax3.hist(log_norms, color='lightgreen', edgecolor='black', bins = 30)
    ax3.set_xlabel("Value Of LNP")
    ax3.set_ylabel("Frequency")

    ax4.hist(laps, color='turquoise', edgecolor='black', bins = 30)
    ax4.set_xlabel("Value Of LP")
    ax4.set_ylabel("Frequency")
    plt.tight_layout()

    output_dir = '/cr/users/abro/cr_inference/cr_inference/plots'
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, f"trunc_prior_draws_.png")
    plt.savefig(full_path, dpi=300, bbox_inches='tight', transparent=False)

else:
        key, subkey = random.split(key)
        xi = jft.random_like(subkey, trunc.domain)

        TODO: # sample from the prior and pass to the plotting script tmrw 



