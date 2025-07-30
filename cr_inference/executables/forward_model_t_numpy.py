

from smiet.numpy.synthesis import TemplateSynthesis, SlicedShower
from smiet.numpy.io import Shower
from smiet.numpy import geo_ce_to_e
from smiet import units
import numpy as np
from jax_radio_tools.shower_utils import *
import matplotlib.pyplot as plt
from smiet.numpy.utilities import bandpass_filter_trace
from cr_pulse_interpolator.interpolation_fourier import interp2d_fourier            # look at these files? 
import matplotlib.colors as mcolors
from argparse import ArgumentParser
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm" 


gdas_filepath="/cr/users/abro/cr_inference/cr_inference/data/atmosphere_models/atmosphere_files/ATMOSPHERE_62804598.DAT" 
template_path = "/cr/users/abro/cr_inference/cr_inference/data/templates/SIM000000.npz"
origin_shower_path = "/cr/tempdata01/kwatanabe/origin_showers/62804598/proton/SIM000000.hdf5"

sliced_origin_shower = SlicedShower(origin_shower_path, gdas_file=gdas_filepath)

template = TemplateSynthesis(freq_ar=[30,500,100])
template.load_template(file_path = template_path, gdas_file = gdas_filepath)

grammages = sliced_origin_shower.long[:, 0]                 # grammages from origin shower used to create the template 

xmax_1 = 855
nmax_1 = 2e5
xmax_2 = 376.41
nmax_2 = 2e5

first_bump = gaisser_hillas_function_LR(grammages, nmax_1, xmax_1, 200, 0.25)
shower_1 = Shower()
shower_1.copy_settings(sliced_origin_shower)                # TODO: what is this doing? 
shower_1.long = np.stack((grammages, first_bump), axis=1)   # TODO: what does .long expect? and what does np.stack do?

second_bump = gaisser_hillas_function_LR(grammages, nmax_2, xmax_2, 200, 0.25)
shower_2 = Shower()
shower_2.copy_settings(sliced_origin_shower)                # TODO: what is this doing? 
shower_2.long = np.stack((grammages, second_bump), axis=1)   # TODO: what does .long expect? and what does np.stack do?

time_traces = template.get_time_axis()

antenna_position = 0

geo_first, ce_first = template.map_template(shower_1) 
synthesized_geo_ef_1 = geo_first[antenna_position, :]

geo_second, ce_second = template.map_template(shower_2) 
synthesized_geo_ef_2 = geo_second[antenna_position, :]

time_traces_per_antenna = time_traces[antenna_position, :]

fig, ax = plt.subplots(nrows=2, ncols = 2, figsize = (8, 6))

ax1, ax2, ax3, ax4 = ax.flatten()

### move this function somewhere! 

def fourier_on_ef(e_field_magnitude, time_traces):
    
    n_samples = len(e_field_magnitude)
    dt = (time_traces[1] - time_traces[0]) * 1e-9
    n_freq = 1/(2 * dt) * 1e-6
    frequencies = np.fft.rfftfreq(n_samples, d=dt)  
    spectrum = np.fft.rfft(e_field_magnitude)  

    return frequencies, np.abs(spectrum), n_freq


# shower profiles 

ax1.plot(grammages, first_bump, color = "darkblue", label = "S1")
ax1.plot(grammages, second_bump, color = "red", label = "S2")
ax1.plot(grammages, second_bump + first_bump, color = "green", label = "DB")
ax1.axvline(x = xmax_1, color = "black", lw = 1.0, linestyle = "--")
ax1.axvline(x = xmax_2, color = "black", lw = 1.0, linestyle = "--")

frequencies_geo_1, amp_spectrum_geo_1, _ = fourier_on_ef(synthesized_geo_ef_1, time_traces_per_antenna)
frequencies_geo_2, amp_spectrum_geo_2, _ = fourier_on_ef(synthesized_geo_ef_2, time_traces_per_antenna)
frequencies_geo_db, amp_spectrum_geo_db, _ = fourier_on_ef(synthesized_geo_ef_2 + synthesized_geo_ef_1, time_traces_per_antenna)
ax2.plot(frequencies_geo_1 / 1e6, amp_spectrum_geo_1, color = "darkblue", label = "S1")
ax2.plot(frequencies_geo_2 / 1e6, amp_spectrum_geo_2, color = "red", label = "S2")
ax2.plot(frequencies_geo_db / 1e6, amp_spectrum_geo_db, color = "green", label = "DB")
ax2.set_xlim(20, 550)
ax2.set_xlabel("frequencies")
ax2.set_ylabel("spectrum")
ax2.set_title(f"antenna position: {antenna_position}, version: NUMPY")

ax3.plot(time_traces_per_antenna, synthesized_geo_ef_1, color = "darkblue", label = "S1")
ax3.plot(time_traces_per_antenna, synthesized_geo_ef_2, color = "red", label = "S2")
ax3.plot(time_traces_per_antenna, synthesized_geo_ef_2 + synthesized_geo_ef_1, color = "green", label = "DB")
ax3.set_xlabel("t")
ax3.set_ylabel("E-field")
ax3.set_xlim(-80, 0)


### fluence calculation, numpy version 

def get_fluence_in_band(
    geo_signal: np.ndarray, ce_signal: np.ndarray, ant_x, ant_y, f_min=200 * units.MHz, f_max=250 * units.MHz
):
    signal = geo_ce_to_e(
        geo_signal, ce_signal, ant_x, ant_y
    )  # shape = (ANT, SAMPLES, 3)

    filtered_signal = bandpass_filter_trace(
        signal, 0.2 * units.ns, f_min, f_max, sample_axis=1
    )

    fluence_in_band = np.sum(filtered_signal**2, axis=(1, 2))

    return fluence_in_band

ant_x, ant_y = template.antenna_information["position_showerplane"].T

f_min = 200 * units.MHz
f_max = 350 * units.MHz

fluence_in_band = get_fluence_in_band(geo_first + geo_second, ce_first + ce_second, ant_x, ant_y, f_min, f_max)

fourier_interpolator = interp2d_fourier(ant_x, ant_y, fluence_in_band, fill_value="extrapolate")

dist_scale = 600.0
ti = np.linspace(-dist_scale, dist_scale, 2000)
XI, YI = np.meshgrid(ti, ti)
ZI = fourier_interpolator(XI, YI)

fluence_cmap = plt.get_cmap('jet')
# vmin = 1e-12
# vmax_cap = 1e-9                                                                                       # FIXME: some antennas are blowing up, inspect in origin shower. OMITTING FOR NOW in fluence print
fluence_norm = mcolors.Normalize(vmin=min(fluence_in_band), vmax=max(fluence_in_band))

mesh = ax4.pcolormesh(
    XI, YI, ZI,
    cmap=fluence_cmap,
    norm=fluence_norm,
    shading='auto',
    zorder=0,
    alpha=0.9
)

ax4.set_aspect('equal', adjustable='datalim')  # now adjusts the data limits to keep 1:1 aspect
ax4.set_xlim(-300, 300)
ax4.set_ylim(-300, 300)

ax4.set_title(f"Footprint[{f_min * 1000} - {f_max * 1000} MHz]")
ax4.set_xlabel("(v x B) [m]")
ax4.set_ylabel("(v x v x B) [m]")

cbar = fig.colorbar(mesh, ax=ax4)
cbar.set_label(r'Fluence (eV / $m^2$)')

plt.tight_layout()

output_dir = '/cr/users/abro/cr_inference/cr_inference/plots'
os.makedirs(output_dir, exist_ok=True)
full_path = os.path.join(output_dir, f'62804598_30_500_antenna_{antenna_position}_NUMPY.png')
plt.savefig(full_path, dpi=300, bbox_inches='tight', transparent=False)


