# Ellipsometry-Simulation

Ellipsometry-Simulation is tool that offers simulation of transparent coatings with variable refractive index profiles.

Two types of tools are offered: A class called **Simulation** and a function called **plot**, which can be used to plot multiple simulations. 

To run the program:

`import rho_simulation_class as Elli`

You will have to install the appropriate packages as defined in "requirements.txt" 


## The **Simulation** Class

*class* **Simulation**(*ri_model*,*n_i*,*substrate*,_**params_)

**Parameters**
* **ri_model**: (*str*) The name of refractive index profile as defined in "ri_models.py"
* **n_i**: The refractive index of the incident medium
* **substrate**: (*str) The name of the substrate as defined in "substrate_list.json" 
* __**params__: Additional keyword arguments for the parameters used by the model defining the refractive index profile  

When invoked, the class will return 3 plots: The refractive index profile, $\Psi$ and $\Delta$ plotted between an angle of incidence of 0 to 90 degrees, and both the imaginary and real parts of $\rho-\rho_o$

