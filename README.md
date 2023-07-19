# Ellipsometry-Simulation

Ellipsometry-Simulation is tool that offers simulation of transparent coatings with variable refractive index profiles.

It returns three types of simulations:
* **Abeles Matrix** exact numerical simulation
* **Thin Film Approximation** to 2nd order in thickness
* **Weak Contrast Limit Approximation** where the refractive index is very near the refractive index of the transmitted medium (the substrate)

Two types of tools are offered: A class called **Simulation** and a function called **plot**, which can be used to plot multiple simulations. 

To run the program:

  `import rho_simulation_class as Elli`

You will have to install the appropriate packages as defined in "requirements.txt". 
The version of Python is 3.9.6


## The **Simulation** Class

*class* **Simulation**(*ri_model*,*n_i*,*substrate*)

**Parameters**
* **ri_model**: (*str*) The name of refractive index profile as defined in "ri_models.py"
* **n_i**: The refractive index of the incident medium
* **substrate**: (*str*) The name of the substrate as defined in "substrate_list.json" 

When invoked, the class will ask for the inputs for the ri_model and return 3 plots: The refractive index profile, $\Psi$ and $\Delta$ plotted between an angle of incidence of 0 to 90 degrees, and both the imaginary and real parts of $\rho-\rho_o$ as based on the Abeles Matrix numerical method. The default wavelength is 632.8 nm.

### Example ###

`elli_simulation = Elli.Simulation("tanh_profile",1.0,"silicon")`

You will be asked through a dialogue prompt to enter the parameters for the ri_model chosen

Three plots will be displayed that correspond to the ri_model. These represent the Abeles Matrix Numerical Simulation of the ri_model. 

1. <img alt="ri_profile" height="150" src="/Examples/Substrate_silicon.png" width="150"/>
2. <img alt="psi_delta" height="150" src="/Examples/psi_delta.png" width="150"/>
3. <img alt="rho2" height="150" src="/Examples/rho2.png" width="150"/>

___

## **Simulation** class Methods ##

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
* A plot based on the imaginary and real parts of $\rho-\rho_o$ for the Abeles Matrix numerical method (solid lines), confidence bars, and the approximation model (dashed lines) if chosen. If an approximation model is chosen, the term "analytic" is used in the legend.

**Parameters**
* **name_of_comparison** Can select "tf" for the thin film approximation or "weak_t" for the weak contrast approximation
* **kwargs**
  * If a d_psi and a d_delta are provided, the confidence limits will be changed from the default (0.01) to the given values
  * To select the number of terms (or order) of the weak contrast approximation, use 'orders' = *num*

**Example**

`elli_simulation.show_rho(name_of_comparison = 'tf')`

<img alt="tf_simulation" height="150" src="/Examples/tf_simulation.png" width="150"/>

___

Simulation.**ri_profile**()

**Returns**
* Two numpy arrays: position(in nm) and refractive index corresponding to the ri_model

___

Simulation.**rho**(_type = "coating",**kwargs_)

**Returns**
* A complex numpy array of rho based on the Abeles Matrix numerical method.

**Parameters**
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

Simulation.**delta**(_with_coating = "yes", **kwargs_)

**Returns**
* A numpy array of delta based on the Abeles Matrix numerical method.

**Parameters**
* **with_coating**: Default is "yes"  
  * If with_coating="no", the method returns delta just for the interface between the incident medium and the substrate

___

Simulation.**d_rho_tf**(_**kwargs_)

**Returns**
* A complex numpy array of rho based on the thin film approximation

___

Simulation.**d_rho_weak_t**(_**kwargs_)

**Returns**
* A complex numpy array of rho based on the weak contrast approximation

**Parameters**
* To select the number of terms (or order) of the weak contrast approximation, use 'orders' = *num*

___

Simulation.**vary**(_param_name,value_low,value_high,num_points,angle,name_of_comparison = 'none',**kwargs_)

Method that varies a parameter and returns the appropriate plot. 

**Returns**
* A plot based on the imaginary and real parts of $\rho-\rho_o$ for the Abeles Matrix numerical method (solid lines), confidence bars, and the approximation model (dashed lines) if chosen. If an approximation model is chosen, the term "analytic" is used in the legend.

**Parameters**
* **param_name**: Name of parameter to vary in the ri_model. Note that the refractive index of the substrate can also be varied. Use param_name = 'n_t'.
* **value_low**: The lower value (_float_) of the parameter being varied. 
* **value_high**: The upper value (_float_) of the parameter being varied.
* **num_points**: The number of points (_int_) to include in the variation.
* **angle**: The angle of incidence to use
* **name_of_comparison** Select "tf" for the thin film approximation or "weak_t" for the weak contrast approximation
* **kwargs**: To select the number of terms (or order) of the weak contrast approximation, use 'orders' = *num*

**Example**

`elli_simulation.vary('n_avg',1.4,1.5,20,70,name_of_comparison = 'tf')`

<img alt="variation" height="150" src="/Examples/variation.png" width="150"/>


___

## **Simulation** class Variables ##

* **ri_model**
  * String containing the name of the refractive index model found in "ri_models.py"
* **n_i**
  * Refractive index of the incident medium
* **substrate**
  * String containing the name of the substrate as found in  "substrate_list.json"
* **params**
  * Dictionary of parameter names and their values used in the ri_model.
  * To change a parameter value use: `Simulation.params.update({param_name : value})`
* **angles**
  * Numpy array of angles of incidence to be used in the simulation
  * Default is an array from 1 to 90 with a step-size of 1 degree. Note that the final value is 89.999 (not 90) to avoid division errors in the simulation
  * Can be set to a single angle. To set to a single angle of 75.55, for instance, use `Simulaton.angles=[75.55]`
* **wavelength**
  * Wavelength of incident radiation
  * Default is 632.8 nm