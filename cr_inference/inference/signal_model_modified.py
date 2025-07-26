import logging
import operator
import os
# from typing import Self, Union, Callable
from typing import TypeVar
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import nifty8.re as jft
from smiet.jax import Shower, TemplateSynthesis
from jax_radio_tools import units, cstrafo, geo_ce_to_vB_vvB_v


# modify your path to the templates and atmospheric model path

path_to_templates = "/cr/users/abro/cr_inference/cr_inference/data/templates/template_62804598_proton_30_500_100_dt4.h5"  # use the single template 


class SignalModel(jft.Model):

    """Forward model that characterises the radio emission from a longitudinal profile."""

# pass in all priors for shower parameters as jft.Model objects

    def __init__(
        self,
        xmax_prior,                                               # data types? are these of type jft.Model instances? (or classes that extend jft.Model?)
        nmax_prior,
        delta_xmax_prior,
        Nmax_fac_prior,
        origin_num : int = 000000,                                                                      # DETAIL: im using SIM000000 here                    
        primary_ptype : str = "proton",
        freq_range: list = [30 * units.MHz, 80 * units.MHz, 50 * units.MHz],                            # DETAIL: can this be used with templates which have different frequencies?
                                                                                                        # TODO: change to 50-200, central = 50
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
	
	# !! modify this line to incorporate eahc shower prior using prior1.init | prior2.init
        super().__init__(init=self.xmax_prior.init | self.delta_xmax_prior.init |  ) # TODO: add other priors here 
 
    def __load_template(self : Self, origin_num: int, primary_ptype: str, starting_grammage : float = 200) -> None:
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
        template_name = f"template_{origin_num}_{primary_ptype}_{(self.freq_range[0] / units.MHz):.0f}_{(self.freq_range[1] / units.MHz):.0f}_{(self.freq_range[2] / units.MHz):.0f}_dt{self.delta_t * 10:.0f}.h5"
        synthesis.load_template(template_name, os.path.join(path_to_templates, "templates"))
        synthesis.truncate_atmosphere(starting_grammage=starting_grammage)          #TODO: for both showers  
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
        final_eifld : jax.typing.ArrayLike
            the electric field traces in the shower plane.
        """
        
        # evaluate the parameters here using your prirors
        xmax1 = self.xmax_prior(xi)
        nmax1 = self.nmax_prior(xi)
        delta_xmax = self.delta_xmax_prior(xi)
        nmax_fac = self.Nmax_fac_prior.(xi)

	# calculate xmax2, nmax2

        xmax2 = xmax1 + delta_xmax                           # delta xmax should be bw -300 and 300 then (uniform, transformed in xi)
        nmax2 = nmax1 * nmax_fac
	
	# calculate the two longitudinal profiles

        long_prof1 = gaisser_hillas(xmax1, nmax1, L=200 R=0.25)  # for now          #TODO: just follow forward_jax_t.py as earlier for both 
        long_prof2 = gaisser_hillas(xmax2, nmax2, L=200, R=0.25) # for now

        # TODO: need some abstraction/interface here to visualise samples from the prior 
	
	# perform TS for each long profile 

        final_efield_geoce = jnp.array([])

        for long_prof in [long_prof1, long_prof2]:

            synthesis_shower = Shower()
            synthesis_shower.set_parameters(self.grammages, self.shower_params)

            synthesis_shower.long_profile = long_prof

            self.shower_params["long"] = synthesis_shower.long_profile           # match other things of shower with origin shower 

            final_efield_geoce += self.synthesis.map_template(synthesis_shower)
	
	# add the efields together

        return final_efield_geoce
