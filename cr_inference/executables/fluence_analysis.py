#### Fluence comparison bw DB, S1 and S2 (JAX)

from smiet.jax.synthesis import TemplateSynthesis
from smiet.jax.io import BaseShower
from smiet.numpy import geo_ce_to_e
from smiet import units
from jax_radio_tools.shower_utils import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from argparse import ArgumentParser
from cr_pulse_interpolator.interpolation_fourier import interp2d_fourier 
import numpy as np
from smiet.numpy.utilities import bandpass_filter_trace
from cr_inference.utils import *

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm" 

parser = ArgumentParser()

parser.add_argument("--xmax1", type=float)
parser.add_argument("--xmax2", type=float)
parser.add_argument("--nmax1", type=float)
parser.add_argument("--nmax2", type=float)
parser.add_argument("--f_min", type=int)
parser.add_argument("--f_max", type=int)
parser.add_argument("--antenna_pos", type=int)
parser.add_argument("--tag", type=str)
parser.add_argument("--title", type=str)
parser.add_argument("--dir", type=str)

args = parser.parse_args()

# MACROS 

GEOMAGNETIC = 0 
CHARGE_EXCESS = 1

# / MACROS 

template_path = "/cr/users/abro/cr_inference/cr_inference/data/templates"                            
template_name = "template_62804598_proton_30_500_100_dt4.h5"                                                           

template = TemplateSynthesis(
    freq_ar = [30 * units.MHz, 500 * units.MHz, 100 * units.MHz]                                                         
)

loaded_template = template.load_template(
    template_file=template_name,
    save_dir=template_path
)

shower_1 = BaseShower()
shower_2 = BaseShower()
origin_information = template.template_information

parameters_1 = {
    "xmax": args.xmax1,  
    "nmax": args.nmax1,
    "zenith": origin_information["zenith"],
    "azimuth": origin_information["azimuth"],
    "magnetic_field_vector": origin_information["magnetic_field_vector"],
    "core": origin_information["core"]
}

parameters_2 = {
    "xmax": args.xmax2,  
    "nmax": args.nmax2,
    "zenith": origin_information["zenith"],
    "azimuth": origin_information["azimuth"],
    "magnetic_field_vector": origin_information["magnetic_field_vector"],
    "core": origin_information["core"]
}

grammages = template.grammages

shower_1.set_parameters(grammages, parameters_1)
shower_2.set_parameters(grammages, parameters_2)

long_profile_1 = gaisser_hillas_function_LR(
    x=template.grammages,
    nmax=parameters_1["nmax"],
    xmax=parameters_1["xmax"],
    L = 200,
    R = 0.25 
)

long_profile_2 = gaisser_hillas_function_LR(
    x=template.grammages,
    nmax=parameters_2["nmax"],
    xmax=parameters_2["xmax"],
    L = 200,
    R = 0.25 
)

origin_xmax = template.template_information["xmax"] 

if np.abs(origin_xmax - parameters_1["xmax"]) > 200:
    print("\nShower 1's xmax is violating accuracy assumptions.")

if np.abs(origin_xmax - parameters_2["xmax"]) > 200:
    print("\nShower 2's xmax is violating accuracy assumptions.")


fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize=(8,6))

ax1, ax2, ax3, ax4 = ax.flatten()

shower_1.set_longitudinal_profile(long_profile_1)
shower_2.set_longitudinal_profile(long_profile_2)

e_field_traces_1 = template.map_template(shower_1)
e_field_traces_2 = template.map_template(shower_2)
template_time_traces = template.get_time_axis()

print("INTERFACE VERSION: JAX")

antenna_position = args.antenna_pos
geomagnetic_antenna_ef_1 = e_field_traces_1[GEOMAGNETIC, antenna_position, :]     # the geomagnetic e-field traces for THIS antenna 
geomagnetic_antenna_ef_2 = e_field_traces_2[GEOMAGNETIC, antenna_position, :] 
time_traces_per_antenna = template_time_traces[antenna_position, :]             # all the t data points for the given antenna


ax1.plot(grammages, long_profile_1, color = "darkblue", label = "S1", alpha=0.4)
ax1.plot(grammages, long_profile_2, color = "red", label = "S2", alpha = 0.4)
ax1.plot(grammages, long_profile_2 + long_profile_1, color = "green", label = "DB", lw = 1.8)
ax1.legend()
ax1.set_xlabel(r"$X \ [g/cm^2]$")
ax1.set_ylabel(r"$N(x)$")

def fourier_on_ef(e_field_magnitude, time_traces):
    
    n_samples = len(e_field_magnitude)
    dt = (time_traces[1] - time_traces[0]) * 1e-9
    n_freq = 1/(2 * dt) * 1e-6
    frequencies = np.fft.rfftfreq(n_samples, d=dt)  
    spectrum = np.fft.rfft(e_field_magnitude)  

    return frequencies, np.abs(spectrum), n_freq


### fluence calculation

dt = (time_traces_per_antenna[1] - time_traces_per_antenna[0])
db_traces = e_field_traces_1 + e_field_traces_2



f_min = args.f_min * units.MHz
f_max = args.f_max * units.MHz
sp_positions = template.ant_positions_vvB
ant_x = sp_positions[:, 0]
ant_y = sp_positions[:, 1]




### S1 fluence 


fluence_in_band_1 = get_fluence_in_band(e_field_traces_1[0, :, :], 
                  e_field_traces_1[1, :, :], 
                  ant_x, ant_y)

XI, YI, ZI = interpolate_fluence(ant_x, ant_y, fluence_in_band_1)

fluence_cmap = plt.get_cmap('jet')
                                                                                 
fluence_norm = mcolors.Normalize(vmin=min(fluence_in_band_1), vmax=max(fluence_in_band_1))

mesh = ax2.pcolormesh(
    XI, YI, ZI,
    cmap=fluence_cmap,
    norm=fluence_norm,
    shading='auto',
    zorder=0,
    alpha=0.9
)

#ax4.set_aspect('equal', adjustable='datalim')  # now adjusts the data limits to keep 1:1 aspect
ax2.set_xlim(-300, 300)
ax2.set_ylim(-300, 300)

ax2.set_title(f"S1 ({int(f_min * 1000)} - {int(f_max * 1000)} MHz)")
ax2.set_xlabel("(v x B) [m]")
ax2.set_ylabel("(v x v x B) [m]")
cbar = fig.colorbar(mesh, ax=ax2)
cbar.set_label(r'Fluence (eV / $m^2$)')


### S2 fluence 

fluence_in_band_2 = get_fluence_in_band(e_field_traces_2[0, :, :], 
                  e_field_traces_2[1, :, :], 
                  ant_x, ant_y)

XI, YI, ZI = interpolate_fluence(ant_x, ant_y, fluence_in_band_2)

fluence_cmap = plt.get_cmap('jet')
                                                                                 
fluence_norm = mcolors.Normalize(vmin=min(fluence_in_band_2), vmax=max(fluence_in_band_2))

mesh = ax3.pcolormesh(
    XI, YI, ZI,
    cmap=fluence_cmap,
    norm=fluence_norm,
    shading='auto',
    zorder=0,
    alpha=0.9
)

#ax4.set_aspect('equal', adjustable='datalim')  # now adjusts the data limits to keep 1:1 aspect
ax3.set_xlim(-300, 300)
ax3.set_ylim(-300, 300)

ax3.set_title(f"S2 ({int(f_min * 1000)} - {int(f_max * 1000)} MHz)")
ax3.set_xlabel("(v x B) [m]")
ax3.set_ylabel("(v x v x B) [m]")
cbar = fig.colorbar(mesh, ax=ax3)
cbar.set_label(r'Fluence (eV / $m^2$)')


#### DB fluence 


fluence_in_band_db = get_fluence_in_band(db_traces[0, :, :], 
                  db_traces[1, :, :], 
                  ant_x, ant_y)

XI, YI, ZI = interpolate_fluence(ant_x, ant_y, fluence_in_band_db)

fluence_cmap = plt.get_cmap('jet')
                                                                                 
fluence_norm = mcolors.Normalize(vmin=min(fluence_in_band_db), vmax=max(fluence_in_band_db))

mesh = ax4.pcolormesh(
    XI, YI, ZI,
    cmap=fluence_cmap,
    norm=fluence_norm,
    shading='auto',
    zorder=0,
    alpha=0.9
)

#ax4.set_aspect('equal', adjustable='datalim')  # now adjusts the data limits to keep 1:1 aspect
ax4.set_xlim(-300, 300)
ax4.set_ylim(-300, 300)

ax4.set_title(f"DB ({int(f_min * 1000)} - {int(f_max * 1000)} MHz)")
ax4.set_xlabel("(v x B) [m]")
ax4.set_ylabel("(v x v x B) [m]")

cbar = fig.colorbar(mesh, ax=ax4)
cbar.set_label(r'Fluence (eV / $m^2$)')

if args.title:
    fig.suptitle(args.title, fontsize=12)

### / fluence calc ends here ####

plt.tight_layout()

if args.dir is not None:
    output_dir = args.dir
else:
    output_dir = '/cr/users/abro/cr_inference/cr_inference/plots'

print(output_dir)
os.makedirs(output_dir, exist_ok=True)
full_path = os.path.join(output_dir, f'62804598_30_500_antenna_{antenna_position}_{args.tag}.png')
plt.savefig(full_path, dpi=300, bbox_inches='tight', transparent=False)
