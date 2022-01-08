# SEIRS+ Model Framework

This package implements models of generalized SEIRS infectious disease dynamics with extensions that allow us to study the effect of social contact network structures, heterogeneities, stochasticity, and interventions, such as social distancing, testing, contact tracing, and isolation. 

#### *Latest Major Release: 1.0 (9 Aug 2020)*

Version 1.0 of the seirsplus package as well as the code and documentation presented in this repository represent COVID modeling frameworks developed in 2020 and early 2021. 

#### 🔶 *Upcoming Major Release: 2.0 (Feb 2022)*

Ongoing modeling and code development with collaborators have advanced the SEIRS+ framework we use internally considerably since the last major package release. 

A major Version 2.0 public release of the SEIRS+ framework package and documentation is planned for early February — watch this space! If you would like access to the Version 2.0 code or further information before this public release, please do not hesitate to reach out.

*New Features in Version 2.0 include:*
* Top-to-bottom redesign of package to make SEIRS+ a versatile framework not only for accessible rapid COVID modeling but also for network epidemiology research more broadly.
* Fully customizable compartment models for flexible implementation of any disease state model, and pre-configured compartment models for COVID-19 contexts and other common diseases.
* Features for tracking transmission chain lineages and infector/infectee characteristics.
* Expanded support for parameterizable implementations of testing, tracing, isolation, deisolation, and social distancing non-pharmaceutical intervention scenarios.
* Support for vaccination, including multiple vaccine types and multi-dose vaccine courses.
* Support for scenarios with emerging disease variants
* Run-time improvements (i.e., Gillespie tau leaping supported)
* Many other features
* Comprehensive testing suite for developers and contributors
* As always, emphasis on ease of use.


## Overview

#### Full documentation of this package's models, code, use cases, examples, and more can be found on [the wiki](https://github.com/ryansmcgee/seirsplus/wiki/)

[**Basic SEIRS Model**](https://github.com/ryansmcgee/seirsplus/wiki/SEIRS-Model-Description) | [**Extended SEIRS Model**](https://github.com/ryansmcgee/seirsplus/wiki/Extended-SEIRS-Model-Description)
:-----:|:-----:
<img src="https://github.com/ryansmcgee/seirsplus/blob/master/images/BasicSEIRS_compartments_padded.png" width="400"> | <img src="https://github.com/ryansmcgee/seirsplus/blob/master/images/ExtSEIRS_compartments.png" width="400">
[Model Description](https://github.com/ryansmcgee/seirsplus/wiki/SEIRS-Model-Description) | [Model Description](https://github.com/ryansmcgee/seirsplus/wiki/Extended-SEIRS-Model-Description)
[`SEIRSNetworkModel` docs](https://github.com/ryansmcgee/seirsplus/wiki/SEIRSModel-class)<br>[`SEIRSModel` docs](https://github.com/ryansmcgee/seirsplus/wiki/SEIRSNetworkModel-class) | [`ExtSEIRSNetworkModel` docs](https://github.com/ryansmcgee/seirsplus/wiki/ExtSEIRSNetworkModel-class)
[Basic SEIRS Mean-field Model Demo](https://github.com/ryansmcgee/seirsplus/blob/master/examples/Basic_SEIRS_Meanfield_Model_Demo.ipynb)<br>[Basic SEIRS Network Model Demo](https://github.com/ryansmcgee/seirsplus/blob/master/examples/Basic_SEIRS_Network_Model_Demo.ipynb) | [Extended SEIRS Community TTI Demo](https://github.com/ryansmcgee/seirsplus/blob/master/examples/Extended_SEIRS_Community_TTI_Demo.ipynb)<br>[Extended SEIRS Workplace TTI Demo](https://github.com/ryansmcgee/seirsplus/blob/master/examples/Extended_SEIRS_Workplace_TTI_Demo.ipynb)

#

### SEIRS Dynamics

The foundation of the models in this package is the classic SEIR model of infectious disease. The SEIR model is a standard compartmental model in which the population is divided into **susceptible (S)**, **exposed (E)**, **infectious (I)**, and **recovered (R)** individuals. A susceptible member of the population becomes infected (exposed) when making a transmissive contact with an infectious individual and then progresses to the infectious and finally recovered states. In the SEIRS model, recovered individuals may become resusceptible some time after recovering (although re-susceptibility can be excluded if not applicable or desired). 
<p align="center">
  <img src="https://github.com/ryansmcgee/seirsplus/blob/master/images/BasicSEIRS_compartments_resus.png" width="400"></div>
</p>

### Extended SEIRS Model

This model extends the classic SEIRS model of infectious disease to represent pre-symptomatic, asymptomatic, and severely symptomatic disease states, which are of particular **relevance to the SARS-CoV-2 pandemic**. In this extended model, the infectious subpopulation is subdivided into **pre-symptomatic (*I<sub>pre</sub>*)**, **asymptomatic (*I<sub>asym</sub>*)**, **symptomatic (*I<sub>sym</sub>*)**, and **hospitalized (severely symptomatic, *I<sub>H</sub>*)**. All of these *I* compartments represent contagious individuals, but transmissibility, rates of recovery, and other parameters may vary between these disease states.

<p align="center">
  <img src="https://github.com/ryansmcgee/seirsplus/blob/master/images/ExtSEIRS_compartments.png" width="600"></div>
</p>


### Testing, Tracing, & Isolation

The effect of isolation-based interventions (e.g., isolating individuals in response to testing or contact tracing) are modeled by introducing compartments representing quarantined individuals. An individual may be quarantined in any disease state, and every disease state has a corresponding quarantine compartment (with the exception of the hospitalized state, which is considered a quarantine state for transmission and other purposes). Quarantined individuals follow the same progression through the disease states, but the rates of transition or other parameters may be different. There are multiple methods by which individuals can be moved into or out of a quarantine state in this framework.

<p align="center">
  <img src="https://github.com/ryansmcgee/seirsplus/blob/master/images/BothSEIRS_compartments_quarantine.png" width="800"></div>
</p>

<a name="model-network"></a>
### Network Models

<img align="right" src="https://github.com/ryansmcgee/seirsplus/blob/master/images/network_p.png" height="220">

Standard compartment models capture important features of infectious disease dynamics, but they are deterministic mean-field models that assume uniform mixing of the population (i.e., every individual in the population is equally likely to interact with every other individual). However, it is often important to consider stochasticity, heterogeneity, and the structure of contact networks when studying disease transmission, and many strategies for mitigating spread can be thought of as perturbing the contact network (e.g., social distancing) or making use of it (e.g., contact tracing).

This package includes implementation of SEIRS models on stochastic dynamical networks. Individuals are represented as nodes in a network, and parameters, contacts, and interventions can be specified on a targeted individual basis. The network model enables rigorous analysis of transmission patterns and network-based interventions with respect to the properties of realistic contact networks. These SEIRS models can be simulated on any network. Network generation is largely left to the user, but some tools for [Network Generation](https://github.com/ryansmcgee/seirsplus/wiki/network-generation) are included in this package. 

Unlike mean-field compartment models, which do not model individual members of the population, the network model explicitly represents individuals as discrete nodes. All model parameters can be assigned to each node on an individual basis. Therefore, the network models support arbitrary parameter heterogeneity at the user's discretion. In addition, the specification of the contact network allows for heterogeneity in interaction patterns to be explicitly modeled as well. 


<a name="usage"></a>
## Code Usage

This package was designed with broad usability in mind. Complex scenarios can be simulated with very few lines of code or, in many cases, no new coding or knowledge of Python by simply modifying the parameter values in the [example notebooks](https://github.com/ryansmcgee/seirsplus/tree/master/examples) provided. See the Quick Start section and the rest of the wiki documentation for more details.

Don't be intimidated by the length of the wiki pages, running these models is quick and easy. The package does all the hard work for you. As an example, here's a complete script that simulates the SEIRS dynamics on a network with forms of social distancing, testing, contact tracing, and quarantining in only 10 lines of code.:
```python
from seirsplus.models import *
import networkx

numNodes = 10000
baseGraph    = networkx.barabasi_albert_graph(n=numNodes, m=9)
G_normal     = custom_exponential_graph(baseGraph, scale=100)
# Social distancing interactions:
G_distancing = custom_exponential_graph(baseGraph, scale=10)
# Quarantine interactions:
G_quarantine = custom_exponential_graph(baseGraph, scale=5)

model = SEIRSNetworkModel(G=G_normal, beta=0.155, sigma=1/5.2, gamma=1/12.39, mu_I=0.0004, p=0.5,
                          Q=G_quarantine, beta_D=0.155, sigma_D=1/5.2, gamma_D=1/12.39, mu_D=0.0004,
                          theta_E=0.02, theta_I=0.02, phi_E=0.2, phi_I=0.2, psi_E=1.0, psi_I=1.0, q=0.5,
                          initI=10)

checkpoints = {'t': [20, 100], 'G': [G_distancing, G_normal], 'p': [0.1, 0.5], 'theta_E': [0.02, 0.02], 'theta_I': [0.02, 0.02], 'phi_E':   [0.2, 0.2], 'phi_I':   [0.2, 0.2]}

model.run(T=300, checkpoints=checkpoints)

model.figure_infections()
```



### Quick Start

Perhaps the best way to get started with these models is to dive into the [examples](https://github.com/ryansmcgee/seirsplus/tree/master/examples). These example notebooks walk through full simulations using each of the models  included in this package with description of the steps involved. **These notebooks can also serve as ready-to-run sandboxes for trying your own simulation scenarios simply by changing the parameter values in the notebook.**

<a name="usage-install"></a>
### Installing and Importing the Package

All of the code needed to run the models is imported from the ```models``` module of this package. Other features that may be of interest are implemented in the `networks`, `sim_loops`, and `utilities` modules.

#### Install the package using ```pip```
The package can be installed on your machine by entering this in the command line:

```> pip install seirsplus```

Then, the ```models``` module (and other modules) can be imported into your scripts as shown here:

```python
from seirsplus.models import *
from seirsplus.networks import *
from seirsplus.sim_loops import *
from seirsplus.utilities import *
import networkx
```

#### *Alternatively, manually copy the code to your machine*

*You can use the model code without installing a package by copying the ```models.py``` module file to a directory on your machine. For some of the features included in this package you will also need the `networks`, `sim_loops`, and `utilities` modules. In this case, the easiest way to use the modules is to place your scripts in the same directory as the modules, and import the modules as shown here:*
```python
from models import *
from networks import *
from sim_loops import *
from utilities import *
```

### Full Code Documentation

Complete documentation for all package classes and functions can be found throughout this wiki, including in-depth descriptions of concept, parameters, and how to initialize, run, and interface with the models. Some pages of note:

 * [`SEIRSModel`](https://github.com/ryansmcgee/seirsplus/wiki/SEIRSModel-class)
 * [`SEIRSNetworkModel`](https://github.com/ryansmcgee/seirsplus/wiki/SEIRSNetworkModel-class)
 * [`ExtSEIRSNetworkModel`](https://github.com/ryansmcgee/seirsplus/wiki/ExtSEIRSNetworkModel-class)
 * [Network Generation](https://github.com/ryansmcgee/seirsplus/wiki/Network-Generation)
 * [TTI Simulation Loop](https://github.com/ryansmcgee/seirsplus/wiki/TTI-Simulation-Loop)

