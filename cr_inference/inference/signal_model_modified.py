import logging
import operator
import os
from typing import TypeVar
import jax
jax.config.update("jax_enable_x64", True)
from jax_radio_tools.shower_utils import *
import jax.numpy as jnp
import numpy as np
import nifty8.re as jft
from smiet.jax import Shower, TemplateSynthesis
from jax_radio_tools import units, cstrafo, geo_ce_to_vB_vvB_v
from nifty8.re.model import Initializer

# modify your path to the templates and atmospheric model path

path_to_templates = "/cr/users/abro/cr_inference/cr_inference/data/templates/template_62804598_proton_30_500_100_dt4.h5"  # use the single template 

class SignalModel(jft.Model):

    """Forward model that characterises the radio emission from a longitudinal profile."""

# pass in all priors for shower parameters as jft.Model objects

    def __init__(
        self,
        xmax_prior,                                               
        nmax_prior,
        delta_xmax_prior,
        Nmax_fac_prior,
        origin_num : int = 62804598,                                                                      # DETAIL: im using 62804598 here because of the template we are using                
        primary_ptype : str = "proton",
        freq_range: list = [50 * units.MHz, 200 * units.MHz, 50 * units.MHz],                            # KEITO: changed it to this because of the template we have
                                                                                                          # KEITO: given origin shower has 42.8 degrees? 
                                                                                                         
        delta_t: float = 2 * units.ns,                                                                                                   

    ) -> None:

        """
        Signal model class to generate electric field traces from shower observable parameters.

        Parameter:
        ----------
        origin_num : int
            the event number of the origin shower
        primary_ptype : str
            the primary composition
        freq_range : list
            the frequency range in which the signal is generated
            in units of MHz
        delta_t : float
            the timing resolution in which the signal is generated
            in units of ns
        """

        self.freq_range = freq_range      # frequency bandwidth in MHz
        self.delta_t = delta_t            # timing resolution in ns
        self.f_smiet_sys = None           # systematic uncertainty for smiet


        self.xmax_prior = xmax_prior
        self.nmax_prior = nmax_prior
        self.delta_xmax_prior = delta_xmax_prior
        self.Nmax_fac_prior = Nmax_fac_prior

        self.shower_params = {
            "long": None,
            "zenith": None,  
            "azimuth": None,  
            "magnetic_field_vector": jnp.array([0.0, 27.6, 48.27])                  # DETAIL: these specific values? (for LOFAR)
            * units.gauss,  
            "core": jnp.zeros(3),
        }  
        
        # shower parameters that are fixed, maybe vary in the future?

        # create template synthesis object & load the template with that frequency band
        print(f"Initialising model for {primary_ptype} event {origin_num} with timing resolution of {delta_t:.1f} ns")
        print(f"within frequency range of [{freq_range[0] / units.MHz:.0f}, {(freq_range[1] / units.MHz):.0f}] MHz with central frequency of {(freq_range[2] / units.MHz):.0f} MHz.")

        self.synthesis = self.__load_template(origin_num, primary_ptype)

        # also set the geometrical parameters from template synthesis (origin shower)
        # in principle we can set all of these free in the future

        print(f"This origin shower has a zenith angle of {np.rad2deg(self.synthesis.template_information['zenith']):.1f} degrees")
        print(f"and an azimuth angle of {np.rad2deg(self.synthesis.template_information['azimuth']):.1f} degrees in NRR coordinates (East-North-Vertical)")
        print(f"the Xmax and Nmax of the event are {self.synthesis.template_information['xmax']:.1f} g/cm^2 and {self.synthesis.template_information['nmax'] / 1e8:.1f} 10^8 particles, respectively.")
        self.shower_params["zenith"] = self.synthesis.template_information["zenith"]
        self.shower_params["azimuth"] = self.synthesis.template_information["azimuth"]
        self.shower_params["magnetic_field_vector"] = (
            self.synthesis.template_information["magnetic_field_vector"]
        )
        self.shower_params["core"] = self.synthesis.template_information["core"]

        # defining transformer to map from shower <-> ground here
        self.transformer = cstrafo(
            self.shower_params["zenith"] / units.rad,
            self.shower_params["azimuth"] / units.rad,
            magnetic_field_vector=self.shower_params["magnetic_field_vector"],
        )
        self.grammages = self.synthesis.grammages


        combined_struct = {                                                 # DETAIL: non-trivial fix, sus area for future bugs! 
            **self.xmax_prior.init._call_or_struct,
            **self.nmax_prior.init._call_or_struct,
            **self.delta_xmax_prior.init._call_or_struct,
            **self.Nmax_fac_prior.init._call_or_struct,
        }

        combined_init = Initializer(combined_struct)

        super().__init__(init=combined_init)

	# # !! modify this line to incorporate each shower prior using prior1.init | prior2.init
    #     super().__init__(init= (self.xmax_prior.init | self.nmax_prior.init | self.delta_xmax_prior.init | self.Nmax_fac_prior.init))                  # KEITO: ask Keito about this line. How may I combine the priors? 

 
    def __load_template(self, origin_num: int, primary_ptype: str, starting_grammage : float = 200) -> None:
        """
        Load the templates for the given origin number and primary particle type.

        Parameter:
        -----------
        origin_num : int
            the event number of the origin shower
        primary_ptype : str
            the primary composition
        
        Return:
        -------
        synthesis : TemplateSynthesis
            the synthesis object that consists of the loaded template
        """

        synthesis = TemplateSynthesis()

        # "template_62804598_proton_30_500_100_dt4.h5" 

        template_name = f"template_{origin_num}_{primary_ptype}_{(self.freq_range[0] / units.MHz):.0f}_{(self.freq_range[1] / units.MHz):.0f}_{(self.freq_range[2] / units.MHz):.0f}_dt{self.delta_t * 10:.0f}.h5"
       # synthesis.load_template(template_name, os.path.join(path_to_templates, "templates"))  

        template_path = "/cr/users/abro/cr_inference/cr_inference/data/templates"                            
        template_name = "template_62804598_proton_30_500_100_dt4.h5"
        template_name = "template_62804598_proton_50_200_50_dt20.h5"                    # FIXME: make this dynamic after discussing with Keito 

        synthesis.load_template(template_file=template_name,
                                save_dir=template_path) 

        synthesis.truncate_atmosphere(starting_grammage=starting_grammage)   

        return synthesis                            


    def __call__(self, xi: jax.typing.ArrayLike) -> jax.Array:
        """
        Return the traces evaluated from each sample.
        
        Parameters:
        -----------
        xi : jax.typing.ArrayLike
            the hyperpriors, sampled via a normal distribution
        
        Returns:
        --------
        final_efield : jax.typing.ArrayLike
            the electric field traces in the shower plane.
        """
        
        # evaluate the parameters here using your priors
        xmax1 = self.xmax_prior(xi)
        nmax1 = jnp.exp(self.nmax_prior(xi))
        delta_xmax = self.delta_xmax_prior(xi)
        nmax_fac = self.Nmax_fac_prior(xi)

	    # calculate xmax2, nmax2

        xmax2 = xmax1 + delta_xmax                          
        nmax2 = nmax1 * nmax_fac
        L = 200                                                                     #FIXME: dont hardcode for later
        R = 0.25                                                                    #FIXME: dont hardcode for later 

        # print(f"\n X_max_1: {xmax1}\n Delta_xmax: {delta_xmax}\n N_max_1: {nmax1:2e}\n N_fac: {nmax_fac}\n")        
	
	    # parameterise the two longitudinal profiles (DB)

        shower1 = Shower()
        shower2 = Shower()

        origin_information = self.synthesis.template_information    # self.synthesis is the template 

        parameters_1 = {
            "xmax": xmax1,  
            "nmax": nmax1,
            "zenith": origin_information["zenith"],
            "azimuth": origin_information["azimuth"],
            "magnetic_field_vector": origin_information["magnetic_field_vector"],
            "core": origin_information["core"]
        }

        parameters_2 = {
            "xmax": xmax2,  
            "nmax": nmax2,
            "zenith": origin_information["zenith"],
            "azimuth": origin_information["azimuth"],
            "magnetic_field_vector": origin_information["magnetic_field_vector"],
            "core": origin_information["core"]
        }

        grammages = self.synthesis.grammages

        shower1.set_parameters(grammages, parameters_1)
        shower2.set_parameters(grammages, parameters_2)

        long_profile_1 = gaisser_hillas_function_LR(
            x=self.synthesis.grammages,
            nmax=parameters_1["nmax"],
            xmax=parameters_1["xmax"],
            L = 200,
            R = 0.25 
        )

        long_profile_2 = gaisser_hillas_function_LR(
            x=self.synthesis.grammages,
            nmax=parameters_2["nmax"],
            xmax=parameters_2["xmax"],
            L = 200,
            R = 0.25 
        )

        origin_xmax = self.synthesis.template_information["xmax"] 

        shower1.set_longitudinal_profile(long_profile_1)
        shower2.set_longitudinal_profile(long_profile_2)

        e_field_traces_1 = self.synthesis.map_template(shower1)
        e_field_traces_2 = self.synthesis.map_template(shower2)
        template_time_traces = self.synthesis.get_time_axis()

        db_traces = e_field_traces_1 + e_field_traces_2

        return db_traces        # final e field 

    def call_with_parameters(self, parameters) -> jax.Array:
        """
        Return the traces evaluated from each sample.
        
        Parameters:
        -----------
        xi : jax.typing.ArrayLike
            the hyperpriors, sampled via a normal distribution
        
        Returns:
        --------
        final_efield : jax.typing.ArrayLike
            the electric field traces in the shower plane.
        """
        
        # evaluate the parameters here using your priors
        xmax1 = parameters["xmax1"]
        nmax1 = parameters["nmax1"]
        delta_xmax = parameters["delta_xmax"]
        nmax_fac = parameters["nmax_fac"]

	    # calculate xmax2, nmax2

        xmax2 = xmax1 + delta_xmax                          
        nmax2 = nmax1 * nmax_fac
        L = 200                                                                     #FIXME: dont hardcode for later
        R = 0.25                                                                    #FIXME: dont hardcode for later 

        # print(f"\n X_max_1: {xmax1}\n Delta_xmax: {delta_xmax}\n N_max_1: {nmax1:2e}\n N_fac: {nmax_fac}\n")        
	
	    # parameterise the two longitudinal profiles (DB)

        shower1 = Shower()
        shower2 = Shower()

        origin_information = self.synthesis.template_information    # self.synthesis is the template 

        parameters_1 = {
            "xmax": xmax1,  
            "nmax": nmax1,
            "zenith": origin_information["zenith"],
            "azimuth": origin_information["azimuth"],
            "magnetic_field_vector": origin_information["magnetic_field_vector"],
            "core": origin_information["core"]
        }

        parameters_2 = {
            "xmax": xmax2,  
            "nmax": nmax2,
            "zenith": origin_information["zenith"],
            "azimuth": origin_information["azimuth"],
            "magnetic_field_vector": origin_information["magnetic_field_vector"],
            "core": origin_information["core"]
        }

        grammages = self.synthesis.grammages

        shower1.set_parameters(grammages, parameters_1)
        shower2.set_parameters(grammages, parameters_2)

        long_profile_1 = gaisser_hillas_function_LR(
            x=self.synthesis.grammages,
            nmax=parameters_1["nmax"],
            xmax=parameters_1["xmax"],
            L = 200,
            R = 0.25 
        )

        long_profile_2 = gaisser_hillas_function_LR(
            x=self.synthesis.grammages,
            nmax=parameters_2["nmax"],
            xmax=parameters_2["xmax"],
            L = 200,
            R = 0.25 
        )

        origin_xmax = self.synthesis.template_information["xmax"] 

        shower1.set_longitudinal_profile(long_profile_1)
        shower2.set_longitudinal_profile(long_profile_2)

        e_field_traces_1 = self.synthesis.map_template(shower1)
        e_field_traces_2 = self.synthesis.map_template(shower2)
        template_time_traces = self.synthesis.get_time_axis()

        db_traces = e_field_traces_1 + e_field_traces_2

        return db_traces, (long_profile_1, long_profile_2, grammages, template_time_traces, e_field_traces_1, e_field_traces_2, self.synthesis)       # final e field 

