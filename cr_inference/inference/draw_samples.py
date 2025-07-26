import jax
import matplotlib.pyplot as plt
import nifty.re as jft
from jax import numpy as jnp
from jax import random
import os

from truncated_normal_prior import TruncatedNormalPrior

jax.config.update("jax_enable_x64", True)

trunc = TruncatedNormalPrior(mean= 3.5, std= 2, a_min= 1, a_max= 5)

seed = 40
key = random.PRNGKey(seed)

key, subkey = jax.random.split(key)

samples=[]

for _ in range(100000):

    key, subkey = random.split(key)
    xi = jft.random_like(subkey, trunc.domain)
    realisation = trunc(xi)
    samples.append(realisation)


plt.figure(figsize=(6,6))
plt.hist(samples, color='skyblue', edgecolor='black')
plt.xlabel("Value")
plt.ylabel("Frequency")

output_dir = '/cr/users/abro/cr_inference/cr_inference/plots'
os.makedirs(output_dir, exist_ok=True)

full_path = os.path.join(output_dir, f"trunc_prior_draws_.png")
plt.savefig(full_path, dpi=300, bbox_inches='tight', transparent=False)