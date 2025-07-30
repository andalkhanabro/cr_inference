from smiet.numpy.synthesis import TemplateSynthesis, SlicedShower
from smiet.numpy.io import Shower
from smiet.numpy import geo_ce_to_e
from smiet import units
import numpy as np
from jax_radio_tools.shower_utils import *
import matplotlib.pyplot as plt
from smiet.numpy.utilities import bandpass_filter_trace
import matplotlib.colors as mcolors
from argparse import ArgumentParser

gdas_filepath="/cr/users/abro/cr_inference/cr_inference/data/atmosphere_models/atmosphere_files/ATMOSPHERE_62804598.DAT"        
save_dir = "/cr/users/abro/cr_inference/cr_inference/data/templates"
origin_shower = "/cr/tempdata01/kwatanabe/origin_showers/62804598/proton/SIM000000.hdf5"      

sliced_origin_shower = SlicedShower(origin_shower, gdas_file=gdas_filepath)
template = TemplateSynthesis(freq_ar=[30,500,100])
template.make_template(sliced_origin_shower)

template.save_template(save_dir = save_dir)
print(f"Template has been saved to {save_dir}")