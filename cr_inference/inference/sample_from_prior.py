import jax
import matplotlib.pyplot as plt
import nifty.re as jft
from jax import numpy as jnp
from jax import random
import os
import numpy as np
import subprocess
from nifty.re.prior import *
import time
from truncated_normal_prior import TruncatedNormalPrior
jax.config.update("jax_enable_x64", True)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm" 

# parameters to parameterise for user later 

verbose = True

for iteration in range(1, 20):
    tag = f"prior_sample_{iteration}"
    title = f"Prior Sample {iteration}"
    seed = int(time.time())

    xmax1 = TruncatedNormalPrior(mean= 380, std= 120, a_min= 200, a_max= 550) 
    delta_xmax = TruncatedNormalPrior(mean = 450, std = 150, a_min = 200, a_max = 700)
    nmax1 = LogNormalPrior(mean = np.log(2e5), std = 4)
    n_fac = TruncatedNormalPrior(mean = 0.8, std = 0.4, a_min = 0.35, a_max = 0.9)

    key = random.PRNGKey(seed)

    key, subkey = random.split(key)
    xi = jft.random_like(subkey, xmax1.domain)
    xmax1_sample = xmax1(xi)
    delta_sample = delta_xmax(xi)
    nmax1_sample = nmax1(xi)
    n_fac_sample = n_fac(xi)

    if verbose:
        print(f"\n X_max_1: {xmax1_sample}\n Delta_xmax: {delta_sample}\n N_max_1: {jnp.exp(nmax1_sample):2e}\n N_fac: {n_fac_sample}\n")

    xmax1 = xmax1_sample
    nmax1 = jnp.exp(nmax1_sample)
    delta_xmax = delta_sample
    n_fac = n_fac_sample

    xmax2 = xmax1 + delta_xmax
    nmax2 = n_fac * nmax1
    L = 200
    R = 0.25                                                                                                            # FIXME: constant for now, make distributions later 

    antenna_pos = 14 
    f_min = 200
    f_max = 350 

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    dir = "/cr/users/abro/cr_inference/cr_inference/plots/prior_samples"


    cmd = [
        "python", "-m", "cr_inference.executables.forward_model_t_jax",
        "--xmax1", str(xmax1),
        "--xmax2", str(xmax2),
        "--nmax1", str(nmax1),
        "--nmax2", str(nmax2),
        "--antenna_pos", str(antenna_pos),
        "--f_min", str(f_min),
        "--f_max", str(f_max),
        "--tag", tag,
        "--title", title,
        "--dir", dir
    ]

    subprocess.run(cmd, cwd = project_root)
