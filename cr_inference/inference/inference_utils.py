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
from signal_model_modified import SignalModel





### TODO: add signature and generalise this 
### TODO: add fluence plots, and circles for the antenna positions chosen... 

def plot_mock_data(long_profile_1, 
    long_profile_2, 
    grammages, 
    template_time_traces, 
    e_field_traces_1, 
    e_field_traces_2, 
    template, 
    tag, 
    title):

    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize=(8,6))
    ax1, ax2, ax3, ax4 = ax.flatten()

    ax1.plot(grammages, long_profile_1, color = "darkblue", label = "S1", alpha=0.4)
    ax1.plot(grammages, long_profile_2, color = "red", label = "S2", alpha = 0.4)
    ax1.plot(grammages, long_profile_2 + long_profile_1, color = "green", label = "DB", lw = 1.8)
    ax1.legend()
    ax1.set_xlabel(r"$X \ [g/cm^2]$")
    ax1.set_ylabel(r"$N(x)$")

    GEOMAGNETIC = 0
    antenna_position = 14
    geomagnetic_antenna_ef_1 = e_field_traces_1[GEOMAGNETIC, antenna_position, :]     # the geomagnetic e-field traces for THIS antenna 
    geomagnetic_antenna_ef_2 = e_field_traces_2[GEOMAGNETIC, antenna_position, :] 
    time_traces_per_antenna = template_time_traces[antenna_position, :]     

    ax2.plot(time_traces_per_antenna, geomagnetic_antenna_ef_1, color = "darkblue", label = f"S1", alpha = 0.5)
    ax2.plot(time_traces_per_antenna, geomagnetic_antenna_ef_2, color = "red", label = f"S2", alpha = 0.5)
    ax2.plot(time_traces_per_antenna, geomagnetic_antenna_ef_2 + geomagnetic_antenna_ef_1, color = "green", label = "DB", lw = 1.8)
    ax2.set_xlim(-180, -140)
    ax2.set_xlabel("t")
    ax2.set_ylabel(r"$E_{geo}$")
    ax2.set_title(f"antenna position: {antenna_position}")
    ax2.legend()

    antenna_position_2 = 35
    geomagnetic_antenna_ef_1 = e_field_traces_1[GEOMAGNETIC, antenna_position_2, :]     # the geomagnetic e-field traces for THIS antenna 
    geomagnetic_antenna_ef_2 = e_field_traces_2[GEOMAGNETIC, antenna_position_2, :] 
    time_traces_per_antenna = template_time_traces[antenna_position_2, :]     

    ax3.plot(time_traces_per_antenna, geomagnetic_antenna_ef_1, color = "darkblue", label = f"S1", alpha = 0.5)
    ax3.plot(time_traces_per_antenna, geomagnetic_antenna_ef_2, color = "red", label = f"S2", alpha = 0.5)
    ax3.plot(time_traces_per_antenna, geomagnetic_antenna_ef_2 + geomagnetic_antenna_ef_1, color = "green", label = "DB", lw = 1.8)
    ax3.set_xlim(133, 155)
    ax3.set_xlabel("t")
    ax3.set_ylabel(r"$E_{geo}$")
    ax3.set_title(f"antenna position: {antenna_position_2}")
    ax3.legend()

    fig.suptitle(title, fontsize=12)

    output_dir = '/cr/users/abro/cr_inference/cr_inference/plots'
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, f'62804598_30_500_antenna_{antenna_position}_{tag}.png')
    plt.tight_layout()
    print("plot saved.")
    plt.savefig(full_path, dpi=300, bbox_inches='tight', transparent=False)