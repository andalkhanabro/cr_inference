'''Utility functions to abstract out commonly used functionalities like fluence calculation and shower parameterisations.'''

from cr_pulse_interpolator.interpolation_fourier import interp2d_fourier 
from smiet.numpy import geo_ce_to_e
from smiet import units
from smiet.numpy.utilities import bandpass_filter_trace
import numpy as np


# fourier utility functions 

"""
fourier_on_ef()

A function to determine fourier spectra for electric fields per antenna. 

- e_fields_per_antenna: an array of an e-field (geomagnetic/ce) for a given antenna i               
- time_traces : time traces sampled for the e-field (set from the template)

Returns:

- frequencies: frequencies for the spectra 
- spectrum: the amplitude spectrum for the transform 
- n_freq: the Nynist frequency calculated for the transform (based on the sampling rate)

"""

def fourier_on_ef(e_fields_per_antenna, time_traces):
    
    n_samples = len(e_fields_per_antenna)
    dt = (time_traces[1] - time_traces[0]) * 1e-9
    n_freq = 1/(2 * dt) * 1e-6
    frequencies = np.fft.rfftfreq(n_samples, d=dt)  
    spectrum = np.fft.rfft(e_fields_per_antenna)  

    return frequencies, np.abs(spectrum), n_freq


# fluence utility functions 

"""
get_fluence_in_band()

A function to determine fluence in a given radio band. Defaults to 200-250 MHz, and assumes bands are given without dimensions. 

- geo_signal: an array of geomagnetic e-fields for all antennas                 
- ce_signal: an array of charge-excess e-fields for all antennas 
- ant_x: an array of antennas in the shower plane, along the x-axis
- ant_y: an array of antennas in the shower plane, along the y-axis

Returns a fluence array for all antennas. 

"""

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


"""
interpolate_fluence()

A function to intepolate fluence footprints in a given radio band in a radially symmetric manner on finer grids. 

- ant_x: an array of antennas in the shower plane, along the x-axis
- ant_y: an array of antennas in the shower plane, along the y-axis
- fluence_in_band: fluence determined for all antennas in get_fluence_in_band
- dist_scale: radius in meters to be used for interpolation

Returns the interpolated values (ZI) as a function of the coordinate grid (XI, YI) in the shower plane. 

"""


def interpolate_fluence(ant_x, ant_y, fluence_in_band, dist_scale=600.0):

    fourier_interpolator = interp2d_fourier(ant_x, ant_y, fluence_in_band, fill_value="extrapolate")
    ti = np.linspace(-dist_scale, dist_scale, 2000)
    XI, YI = np.meshgrid(ti, ti)
    ZI = fourier_interpolator(XI, YI)

    return XI, YI, ZI











