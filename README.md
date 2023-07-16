# Ellipsometry-Simulation

Ellipsometry-Simulation is tool that offers simulation of transparent coatings with variable refractive index profiles.

It returns three types of simulations:
* **Abeles Matrix** exact numerical simulation
* **Thin Film Approximation** to 2nd order in thickness
* **Weak Contrast Limit Approximation** where the refractive index is very near the refractive index of the transmitted medium (the substrate)

Two types of tools are offered: A class called **Simulation** and a function called **plot**, which can be used to plot multiple simulations. 

To run the program:

`import rho_simulation_class as Elli`

You will have to install the appropriate packages as defined in "requirements.txt" 


## The **Simulation** Class

*class* **Simulation**(*ri_model*,*n_i*,*substrate*,_**params_)

**Parameters**
* **ri_model**: (*str*) The name of refractive index profile as defined in "ri_models.py"
* **n_i**: The refractive index of the incident medium
* **substrate**: (*str*) The name of the substrate as defined in "substrate_list.json" 
* __**params__: Additional keyword arguments for the parameters used by the model defining the refractive index profile  

When invoked, the class will return 3 plots: The refractive index profile, $\Psi$ and $\Delta$ plotted between an angle of incidence of 0 to 90 degrees, and both the imaginary and real parts of $\rho-\rho_o$ as based on the Abeles Matrix numerical method

### Example ###

`output = Elli.Simulation("tanh_profile",1.0,"silicon",n_avg = 1.5,width = 40, dn = 0.02, sigma = 5)`

## **Simulation** class Methods ##



Simulation.**get_info**()

Simulation.**show_ri_profile**()

Simulation.**show_psi_delta**()

Simulation.**show_rho**(name_of_comparison="none", **kwargs)

Simulation.**rho**(self,type = "coating",**kwargs)

Simulation.**psi**(self, with_coating = "yes",**kwargs)

Simulation.**delta**(self, with_coating = "yes",**kwargs )

Simulation.**uncertainity**(self,type = "coating",**kwargs)

Simulation.**ri_profile**(self)

## **Simulation** class Variables ##

* **ri_model**
  * String containing the name of the refractive index model found in "ri_models.py"
* **n_i**
  * Refractive index of the incident medium
* **substrate**
  * String containing the name of the substrate as found in  "substrate_list.json"
* **params**
  * Dictionary of parameter names and their values used in the ri_model
* **angles**
  * Numpy array of angles of incidence to be used in the simulation
  * Default is an array from 1 to 90 with a step-size of 1 degree. Note that the final value is 89.999 (not 90) to avoid division errors in the simulation
  * Can be set to a single angle. To set to a single angle of 75.55, for instance, use `Simulaton.angles=[75.55]`
* **wavelength**
  * Wavelength of incident radiation
  * Default is 632.8 nm