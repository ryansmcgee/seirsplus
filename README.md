# SEIRS+ Model

This package implements generalized SEIRS infectious disease dynamics models with extensions that model the effect of factors including population structure, social distancing, testing, contact tracing, and quarantining detected cases. Notably, this package includes both standard deterministic as well as stochastic network implementations of these models.

**README Contents:**
* [ Model Description ](#model)
    * [ Deterministic Model ](#model-standard)
    * [ Network Model ](#model-network)
* [ Code Usage ](#usage)

<a name="model"></a>
## Model Description

### SEIRS Dynamics

The foundation of the models in this package is the classic SEIR model of infectious disease. The SEIR model is a standard compartmental model in which the population is divided into **susceptible (S)**, **exposed (E)**, **infectious (I)**, and **recovered (R)** individuals. A susceptible member of the population becomes exposed (latent infection) when coming into contact with an infectious individual, and progresses to the infectious and then recovered states. In the SEIRS model, recovered individuals may become resusceptible some time after recovering (although re-susceptibility can be excluded if not applicable or desired). 
<p align="center">
  <img src="https://github.com/ryansmcgee/seirsplus/blob/master/images/SEIRS_diagram.png" width="400"></div>
</p>

The rates of transition between the states are given by the parameters:
* β: rate of transmission (transmissions per S-I contact per time)
* σ: rate of progression (inverse of incubation period)
* γ: rate of recovery (inverse of infectious period)
* ξ: rate of re-susceptibility (inverse of temporary immunity period; 0 if permanent immunity)
* μ<sub>I</sub>: rate of mortality from the disease (deaths per infectious individual per time)

#### SEIRS Dyanmics with Testing

The effect of testing for infection on the dynamics can be modeled by introducing states corresponding to **detected exposed (D<sub>E</sub>)** and **detected infectious (D<sub>I</sub>)**. Exposed and infectious individuals are tested at rates θ<sub>E</sub> and θ<sub>I</sub>, respectively, and test positively for infection with rates ψ<sub>E</sub> and ψ<sub>I</sub>, respectively. Testing positive moves an individual into the appropriate detected case state, where rates of transmission, progression, recovery, and/or mortality (as well as network connectivity in the network model) may be different than those of undetected cases.

<p align="center">
  <img src="https://github.com/ryansmcgee/seirsplus/blob/master/images/SEIRStesting_diagram.png" width="400"></div>
</p>

The rates of transition between the states are given by the parameters:
* β: rate of transmission (transmissions per S-I contact per time)
* σ: rate of progression (inverse of incubation period)
* γ: rate of recovery (inverse of infectious period)
* μ<sub>I</sub>: rate of mortality from the disease (deaths per infectious individual per time)
* ξ: rate of re-susceptibility (inverse of temporary immunity period; 0 if permanent immunity)
* θ<sub>E</sub>: rate of testing for exposed individuals 
* θ<sub>I</sub>: rate of testing for infectious individuals 
* ψ<sub>E</sub>: rate of positive test results for exposed individuals 
* ψ<sub>I</sub>: rate of positive test results for infectious individuals 
* β<sub>D</sub>: rate of transmission for detected cases (transmissions per S-D<sub>I</sub> contact per time)
* σ<sub>D</sub>: rate of progression for detected cases (inverse of incubation period)
* γ<sub>D</sub>: rate of recovery for detected cases (inverse of infectious period)
* μ<sub>D</sub>: rate of mortality from the disease for detected cases (deaths per infectious individual per time)

*Vital dynamics are also supported in these models (optional, off by default), but aren't discussed in the README.* 

*See [model equations documentation](https://github.com/ryansmcgee/seirsplus/blob/master/docs/SEIRSplus_Model.pdf) for more information about the  model equations.*


<a name="model-standard"></a>
### Deterministic Model

The evolution of the SEIRS dynamics described above can be described by the following systems of differential equations. Importantly, this version of the model is deterministic and assumes a uniformly-mixed population. 

### SEIRS Dynamics

<p align="center">
  <img src="https://github.com/ryansmcgee/seirsplus/blob/master/images/SEIRStesting_deterministic_equations.png" width="400"></div>
</p>

### SEIRS Dynamics with Testing

<p align="center">
  <img src="https://github.com/ryansmcgee/seirsplus/blob/master/images/SEIRStesting_deterministic_equations.png" width="400"></div>
</p>

<a name="model-network"></a>
### Network Model

sometext

<a name="usage"></a>
## Code Usage

sometext
