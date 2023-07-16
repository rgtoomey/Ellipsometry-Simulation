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

Note: For methods that take **kwargs as a parameter, the following are used. All others will be ignored.
* **ri_over_ride**: Overrides the refractive index of the substrate as given in "substrate_list.json." To use `ri_over_ride = [value_real,value_imag]`
* **d_psi**: The default uncertainity or precision in the measurement of psi is 0.01 degrees. To change this default use `d_psi = value`
* **d_delta**: The uncertainity or precision in the measurement of delta is 0.01 degrees. To change this default use `d_delta = value`

___

Simulation.**get_info**()
* Returns a printout of the ri_model used, wavelength, substrate, and a dictionary of the parameters used for the ri_model

___

Simulation.**show_ri_profile**()
* Returns a plot of the refractive index profile

___

Simulation.**show_psi_delta**()
* Returns a plot of Psi and Delta for the Abeles Matrix (exact) numerical simulation

___

Simulation.**show_rho**(_name_of_comparison="none", **kwargs_)

**Returns**
* A plot based on the imaginary and real parts of $\rho-\rho_o$ for the Abeles Matrix numerical method (solid lines), confidence bars, and the approximation model (dashed lines) if chosen._

**Parameters**
* **name_of_comparison** Can select "tf" for the thin film approximation or "weak_t" for the weak contrast approximation
* **kwargs**
  * If a d_psi and a d_delta are provided, the confidence limits will be changed from the default to the given values

___

Simulation.**ri_profile**()

**Returns**
* Two numpy arrays: position(in nm) and refractive index corresponding the the ri_model

___

Simulation.**rho**(_type = "coating",**kwargs_)

**Returns**
* A complex numpy array of rho based on the Abeles Matrix numerical method.

**Paramaters**
* **type**: Default is "coating" which returns $\rho$. 
  * If type="no_coating", the method returns $\rho_o$,
  * If type="difference", the method returns $\rho-\rho_o$

___

Simulation.**psi**(_with_coating = "yes",**kwargs_)

**Returns**
* A numpy array of psi based on the Abeles Matrix numerical method.

**Parameters**
* **with_coating**: Default is "yes"  
  * If with_coating="no", the method returns psi just for the interface between the incident medium and the substrate

___

Simulation.**delta**(_with_coating = "yes",**kwargs_)

**Returns**
* A numpy array of delta based on the Abeles Matrix numerical method.

**Paramaters**
* **with_coating**: Default is "yes"  
  * If with_coating="no", the method returns delta just for the interface between the incident medium and the substrate

___

Simulation.**uncertainity**(self,type = "coating",**kwargs)

**Returns**




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