import jax
import matplotlib.pyplot as plt
import nifty.re as jft
from jax import numpy as jnp
from jax import random
import os
import subprocess
from nifty.re.prior import *
from truncated_normal_prior import TruncatedNormalPrior
import numpy as np
jax.config.update("jax_enable_x64", True)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm" 
make_priors = True

# priors as instances of jft.Models

xmax1 = TruncatedNormalPrior(mean= 380, std= 120, a_min= 200, a_max= 550) 
delta_xmax = TruncatedNormalPrior(mean = 450, std = 150, a_min = 200, a_max = 700)
nmax1 = LogNormalPrior(mean = np.log(2e5), std = 4)
n_fac = TruncatedNormalPrior(mean = 0.8, std = 0.4, a_min = 0.35, a_max = 0.9)

seed = 40
key = random.PRNGKey(seed)

samples=[]
uniforms=[]
log_norms=[]
laps=[]

if make_priors:

    for _ in range(10000):

        key, subkey = random.split(key)
        xi = jft.random_like(subkey, xmax1.domain)
        realisation_trunc = xmax1(xi)
        realisation_uniform = delta_xmax(xi)
        realisation_ln = nmax1(xi)
        realisation_lap = n_fac(xi)
        samples.append(realisation_trunc)
        uniforms.append(realisation_uniform)
        log_norms.append(realisation_ln)
        laps.append(realisation_lap)

    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (10,6))
    ax1, ax2, ax3, ax4 = ax.flatten()

    ax1.hist(samples, color='skyblue', edgecolor='black', bins = 30)
    ax1.set_xlabel(r"$X_{max_1}$")
    ax1.set_ylabel("Frequency")

    ax2.hist(uniforms, color='orange', edgecolor='black', bins = 30)
    ax2.set_xlabel(r"$\Delta X_{max}$")
    ax2.set_ylabel("Frequency")

    ax3.hist(log_norms, color='lightgreen', edgecolor='black', bins = 30)
    ax3.set_xlabel(r"$ln(N_{max_1})$")
    ax3.set_ylabel("Frequency")

    ax4.hist(laps, color='turquoise', edgecolor='black', bins = 30)
    ax4.set_xlabel(r"$N_{fac}$")
    ax4.set_ylabel("Frequency")
    plt.tight_layout()

    output_dir = '/cr/users/abro/cr_inference/cr_inference/plots/prior_samples'
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, f"parameter_distributions_.png")
    plt.savefig(full_path, dpi=300, bbox_inches='tight', transparent=False)

else:
    pass
        # key, subkey = random.split(key)
        # xi = jft.random_like(subkey, trunc.domain)

        # xmax1 = TruncatedNormalPrior(mean= 3.5, std= 0.2, a_min= 1, a_max= 5) 
        # delta_xmax = UniformPrior(a_min = 2e5, a_max = 7e5)
        # nmax1 = TruncatedNormalPrior(mean = 3e5, a_min = 1e5, a_max = 5e5, std = 2)
        # n_fac = UniformPrior(a=0.1, b = 0.9)


       



