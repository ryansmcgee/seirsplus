# SEIRS+ Model Framework

<span style="background-color: #FFFF00">**_UPDATE 7 Aug 2020: A new version of the `seirsplus` package featuring extended models, simulations, and other new features is being prepared for release in the coming days. In conjunction with this release, this readme is being transitioned to a more thorough wiki. Please bear with us during these updates, and check back shortly for more information._**</span>

This package implements generalized SEIRS infectious disease dynamics models with extensions that model the effect of factors including population structure, social distancing, testing, contact tracing, and quarantining detected cases. 

Notably, this package includes stochastic implementations of these models on dynamic networks.

**README Contents:**
* [ Model Description ](#model)
   * [ SEIRS Dynamics ](#model-seirs)
   * [ Extended SEIRS Model ](#model-extseirs)
      * [ Testing, Tracing, & Isolation ](#model-tti)
   * [ Network Model ](#model-network)
* [ Code Usage ](#usage)
   * [ Quick Start ](#usage-start)
   * [ Installing and Importing the Package ](#usage-install)
   * [ Initializing the Model ](#usage-init)
      * [ Deterministic Model ](#usage-init-determ)
      * [ Network Model ](#usage-init-network)
   * [ Running the Model ](#usage-run)
   * [ Accessing Simulation Data ](#usage-data)
   * [ Changing parameters during a simulation ](#usage-checkpoints)
   * [ Specifying Interaction Networks ](#usage-networks)
   * [ Vizualization ](#usage-viz)
  
<a name="model"></a>
## Model Description

<a name="model-seirs"></a>
### SEIRS Dynamics

The foundation of the models in this package is the classic SEIR model of infectious disease. The SEIR model is a standard compartmental model in which the population is divided into **susceptible (S)**, **exposed (E)**, **infectious (I)**, and **recovered (R)** individuals. A susceptible member of the population becomes exposed (latent infection) when coming into contact with an infectious individual, and progresses to the infectious and then recovered states. In the SEIRS model, recovered individuals may become resusceptible some time after recovering (although re-susceptibility can be excluded if not applicable or desired). 
<p align="center">
  <img src="https://github.com/ryansmcgee/seirsplus/blob/master/images/SEIRS_diagram.png" width="400"></div>
</p>

The rates of transition between the states are given by the parameters:
* σ: rate of progression (inverse of incubation period)
* γ: rate of recovery (inverse of infectious period)
* ξ: rate of re-susceptibility (inverse of temporary immunity period; 0 if permanent immunity)
* μ<sub>I</sub>: rate of mortality from the disease (deaths per infectious individual per time)

*Vital dynamics are also supported in these models (optional, off by default), but aren't discussed in the README.* 

<a name="model-extseirs"></a>
### Extended SEIRS Model

This model extends the classic SEIRS model of infectious disease to represent pre-symptomatic, asymptomatic, and severely symptomatic disease states, which are of particular relevance to the SARS-CoV-2 pandemic. The standard SEIR model divides the population into **susceptible (*S*)**, **exposed (*E*)**, **infectious (*I*)**, and **recovered (*R*)** individuals. In this extended model, the infectious subpopulation is further subdivided into **pre-symptomatic (*I<sub>pre</sub>*)**, **asymptomatic (*I<sub>asym</sub>*)**, **symptomatic (*I<sub>sym</sub>*)**, and **hospitalized (severely symptomatic, *I<sub>H</sub>*)**. All of these *I* compartments represent contagious individuals, but transmissibility, rates of recovery, and other parameters may vary between these disease states.

A susceptible (*S*) member of the population becomes infected (exposed) when making a transmissive contact with an infectious individual. Newly exposed (*E*) individuals first experience a latent period during which time they are not contagious (e.g., while a virus is replicating, but before shedding). Infected individuals then progress to a pre-symptomatic infectious state (*I<sub>pre</sub>*), in which they are contagious but not yet presenting symptoms. Some infectious individuals go on to develop symptoms (*I<sub>sym</sub>*), while a portion of the population never develops symptoms despite being contagious (*I<sub>asym</sub>*). A subset of symptomatic individuals progress to a severe clinical state and must be hospitalized (*I<sub>H</sub>*), and some fraction severe cases are fatal (*F*). At the conclusion of the infectious period, infected individuals enter the recovered state (*R*) and are no longer contagious or susceptible to infection. As in a SEIR*S* model, recovered individuals may become resusceptible some time after recovering (although re-susceptibility can be excluded if not applicable or desired).

<p align="center">
  <img src="https://github.com/ryansmcgee/seirsplus/blob/master/images/ExtSEIRS_compartments.png" width="700"></div>
</p>

The rates of transition between the states are governed by the parameters:
* σ: rate of progression to infectiousness (inverse of latent period)
* λ: rate of progression to (a)symptomatic state (inverse of pre-symptomatic period)
* a: probability of an infected individual remaining asymptomatic
* h: probability of a symptomatic individual being hospitalized
* η: rate of progression to hospitalized state (inverse of onset-to-admission period)
* γ: rate of recovery for non-hospitalized symptomatic individuals (inverse of symptomatic infectious period)
* γ<sub>A</sub>: rate of recovery for asymptomatic individuals (inverse of asymptomatic infectious period)
* γ<sub>H</sub>: rate of recovery hospitalized symptomatic individuals (inverse of hospitalized infectious period)
* f: probability of death for hospitalized individuals (case fatality rate)
* μ<sub>H</sub>: rate of death for hospitalized individuals (inverse of admission-to-death period)
* ξ: rate of re-susceptibility (inverse of temporary immunity period; 0 if permanent immunity)

Note that the extended model reduces to the basic SEIRS model for *a = 0*, *h = 0*, and *σ → 0*.

<a name="model-tti"></a>
#### Testing, Tracing, & Isolation

The effect of isolation-based interventions (e.g., isolating individuals in response to testing or contact tracing) are modeled by introducing compartments representing quarantined individuals. An individual may be quarantined in any disease state, and every disease state has a corresponding quarantine compartment (with the exception of the hospitalized state, which is considered a quarantine state for transmission and other purposes). Quarantined individuals follow the same progression through the disease states, but the rates of transition or other parameters may be different. There are multiple methods by which individuals can be moved into or out of a quarantine state in this framework.

<p align="center">
  <img src="https://github.com/ryansmcgee/seirsplus/blob/master/images/ExtSEIRS_compartments_quarantine.png" width="700"></div>
</p>

In addition to the parameters given above, transitions between quarantine states are governed by the parameters:
* σ<sub>Q</sub>: rate of progression to infectiousness for quarantined individuals (inverse of latent period)
* λ<sub>Q</sub>: rate of progression to (a)symptomatic state for quarantined individuals (inverse of pre-symptomatic period)
* γ<sub>Q<sub>S</sub></sub>: rate of recovery for quarantined non-hospitalized symptomatic individuals (inverse of symptomatic infectious period)
* γ<sub>Q<sub>A</sub></sub>: rate of recovery for non-hospitalized asymptomatic individuals (inverse of asymptomatic infectious period)







<a name="model-network"></a>
### Network Model

The standard SEIRS model captures important features of infectious disease dynamics, but it is deterministic and assumes uniform mixing of the population (every individual in the population is equally likely to interact with every other individual). However, it is often important to consider stochastic effects and the structure of contact networks when studying disease transmission and the effect of interventions such as social distancing and contact tracing.

This package includes implementation of the SEIRS dynamics on stochastic dynamical networks. This avails analysis of the realtionship between network structure and effective transmission rates, including the effect of network-based interventions such as social distancing, quarantining, and contact tracing.

<img align="right" src="https://github.com/ryansmcgee/seirsplus/blob/master/images/network_p.png" height="220">

Consider a graph **_G_** representing individuals (nodes) and their interactions (edges). Each individual (node) has a state (*S, E, I<sub>pre</sub>, I<sub>sym</sub>, I<sub>asym</sub>, H, R, F*, etc). The set of nodes adjacent (connected by an edge) to an individual defines their set of "close contacts" (highlighted in black).  At a given time, each individual makes contact with a random individual from their set of close contacts with probability *(1-p)* or with a random individual from anywhere in the network (highlighted in teal) with probability *p*. The latter global contacts represent individuals interacting with the population at large (i.e., individuals outside of ones' social circle, such as on public transit, at an event, etc.) with some probability. The parameter *p* defines the locality of the network: for *p=0* an individual only interacts with their close contacts, while *p=1* represents a uniformly mixed population. Social distancing interventions may increase the locality of the network (i.e., decrease *p*) and/or decrease local connectivity of the network (i.e., decrease the degree of individuals).

When a susceptible individual interacts with an infectious individual they may become exposed. The probability of transmission from an infectious individual *i* to a susceptible individual *j* depends in general on the transmissibility of the infectious individual and the susceptibility of the susceptible individual (the product of these parameters weight the interaction edges).


#### Quarantine Contacts

<img align="right" src="https://github.com/ryansmcgee/seirsplus/blob/master/images/network_qp.png" height="220">

Now we also consider another graph **_G<sub>Q</sub>_** which represents the interactions that each individual has while in a quarantine state. The quarantine has the effect of dropping some fraction of the edges connecting the quarantined individual to others (according to a rule of the user's choice when generating the graph *G<sub>Q</sub>*). The edges of *G<sub>Q</sub>* (highlighted in purple) for each individual are then a subset of the normal edges of *G* for that individual. The set of nodes that are adjacent to a quarantined individual define their set of "quarantine contacts" (highlighted in purple). At a given time, a quarantined individual comes into contact with an individual in their set of quarantine contacts with probability *(1-p)* or comes into contact with a random individual from anywhere in the network with probability *p*. The parameter *q* (down)weights the intensity of interactions with the population at large while in quarantine relative to baseline. The transmissibility, susceptibility, and other parameters may be different for individuals in quarantine. 



<a name="usage"></a>
## Code Usage

This package was designed with broad usability in mind. Complex scenarios can be simulated with very few lines of code or, in many cases, no new coding or knowledge of Python by simply modifying the parameter values in the example notebooks provided. See the Quick Start section and the rest of this documentation for more details.

Don't be fooled by the length of the README, running these models is quick and easy. The package does all the hard work for you. As an example, here's a complete script that simulates the SEIRS dyanmics on a network with social distancing, testing, contact tracing, and quarantining in only 10 lines of code (see the [example notebooks](https://github.com/ryansmcgee/seirsplus/tree/master/examples) for more explanation of this example):
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

<a name="usage-start"></a>
### Quick Start

The [```examples```](https://github.com/ryansmcgee/seirsplus/tree/master/examples) directory contains two Jupyter notebooks: one for the [deterministic model](https://github.com/ryansmcgee/seirsplus/blob/master/examples/deterministic_model_demo.ipynb) and one for the [network model](https://github.com/ryansmcgee/seirsplus/blob/master/examples/network_model_demo.ipynb). These notebooks walk through full simulations using each of these models with description of the steps involved.

**These notebooks can also serve as ready-to-run sandboxes for trying your own simulation scenarios by simply changing the parameter values in the notebook.**

<a name="usage-install"></a>
### Installing and Importing the Package

All of the code needed to run the model is imported from the ```models``` module of this package.

#### Install the package using ```pip```
The package can be installed on your machine by entering this in the command line:

```> sudo pip install seirsplus```

Then, the ```models``` module can be imported into your scripts as shown here:

```python
from seirsplus.models import *
import networkx
```

#### *Alternatively, manually copy the code to your machine*

*You can use the model code without installing a package by copying the ```models.py``` module file to a directory on your machine. In this case, the easiest way to use the module is to place your scripts in the same directory as the module, and import the module as shown here:*

```python
from models import *
```
<a name="usage-init"></a>
### Initializing the Model

<a name="usage-init-determ"></a>
#### Deterministic Model

All model parameter values, including the normal and (optional) quarantine interaction networks, are set in the call to the ```SEIRSModel``` constructor. The basic SEIR parameters ```beta```, ```sigma```, ```gamma```, and ```initN``` are the only required arguments. All other arguments represent parameters for optional extended model dynamics; these optional parameters take default values that turn off their corresponding dynamics when not provided in the constructor. 

Constructor Argument | Parameter Description | Data Type | Default Value
-----|-----|-----|-----
```beta   ``` | rate of transmission | float | REQUIRED
```sigma  ``` | rate of progression | float | REQUIRED
```gamma  ``` | rate of recovery | float | REQUIRED
```xi     ``` | rate of re-susceptibility | float | 0
```mu_I   ``` | rate of infection-related mortality | float | 0
```mu_0   ``` | rate of baseline mortality | float | 0 
```nu     ``` | rate of baseline birth | float | 0 
```beta_D ``` | rate of transmission for detected cases | float | None (set equal to ```beta```) 
```sigma_D``` | rate of progression for detected cases | float | None (set equal to ```sigma```)  
```gamma_D``` | rate of recovery for detected cases | float | None (set equal to ```gamma```)  
```mu_D   ``` | rate of infection-related mortality for detected cases | float | None (set equal to ```mu_I```) 
```theta_E``` | rate of testing for exposed individuals | float | 0 
```theta_I``` | rate of testing for infectious individuals | float | 0 
```psi_E  ``` | probability of positive tests for exposed individuals | float | 0 
```psi_I  ``` | probability of positive tests for infectious individuals | float | 0
```initN  ``` | initial total number of individuals | int | 10
```initI  ``` | initial number of infectious individuals | int | 10
```initE  ``` | initial number of exposed individuals | int | 0 
```initD_E``` | initial number of detected exposed individuals | int | 0 
```initD_I``` | initial number of detected infectious individuals | int | 0 
```initR  ``` | initial number of recovered individuals | int | 0
```initF  ``` | initial number of deceased individuals | int | 0

##### Basic SEIR

```python
model = SEIRSModel(beta=0.155, sigma=1/5.2, gamma=1/12.39, initN=100000, initI=100)
```


##### Basic SEIRS

```python
model = SEIRSModel(beta=0.155, sigma=1/5.2, gamma=1/12.39, xi=0.001, initN=100000, initI=100)
```

##### SEIR with testing and different progression rates for detected cases (```theta``` and ```psi``` testing params > 0, rate parameters provided for detected states)

```python
model = SEIRSModel(beta=0.155, sigma=1/5.2, gamma=1/12.39, initN=100000, initI=100,
                   beta_D=0.100, sigma_D=1/4.0, gamma_D=1/9.0, theta_E=0.02, theta_I=0.02, psi_E=1.0, psi_I=1.0)
```


<a name="usage-init-network"></a>
#### Network Model

All model parameter values, including the interaction network and (optional) quarantine network, are set in the call to the ```SEIRSNetworkModel``` constructor. The interaction network ```G``` and the basic SEIR parameters ```beta```, ```sigma```, and ```gamma``` are the only required arguments. All other arguments represent parameters for optional extended model dynamics; these optional parameters take default values that turn off their corresponding dynamics when not provided in the constructor. 

**_Heterogeneous populations:_** Nodes can be assigned different values for a given parameter by passing a list of values (with length = number of nodes) for that parameter in the constructor.

All constructor parameters are listed and described below, followed by examples of use cases for various elaborations of the model are shown below (non-exhaustive list of use cases).

Constructor Argument | Parameter Description | Data Type | Default Value
-----|-----|-----|-----
```G      ``` | graph specifying the interaction network | ```networkx Graph``` or ```numpy 2d array```  | REQUIRED 
```beta   ``` | rate of transmission | float | REQUIRED
```sigma  ``` | rate of progression | float | REQUIRED
```gamma  ``` | rate of recovery | float | REQUIRED
```xi     ``` | rate of re-susceptibility | float | 0
```mu_I   ``` | rate of infection-related mortality | float | 0
```mu_0   ``` | rate of baseline mortality | float | 0 
```nu     ``` | rate of baseline birth | float | 0 
```p      ``` | probability of global interactions (network locality) | float | 0
```Q      ``` | graph specifying the quarantine interaction network | ```networkx Graph``` or ```numpy 2d array``` | None 
```beta_D ``` | rate of transmission for detected cases | float | None (set equal to ```beta```) 
```sigma_D``` | rate of progression for detected cases | float | None (set equal to ```sigma```)  
```gamma_D``` | rate of recovery for detected cases | float | None (set equal to ```gamma```)  
```mu_D   ``` | rate of infection-related mortality for detected cases | float | None (set equal to ```mu_I```) 
```theta_E``` | rate of testing for exposed individuals | float | 0 
```theta_I``` | rate of testing for infectious individuals | float | 0 
```phi_E  ``` | rate of contact tracing testing for exposed individuals | float | 0 
```phi_I  ``` | rate of contact tracing testing for infectious individuals | float | 0 
```psi_E  ``` | probability of positive tests for exposed individuals | float | 0 
```psi_I  ``` | probability of positive tests for infectious individuals | float | 0
```q      ``` | probability of global interactions for quarantined individuals | float | 0
```initI  ``` | initial number of infectious individuals | int | 10
```initE  ``` | initial number of exposed individuals | int | 0 
```initD_E``` | initial number of detected exposed individuals | int | 0 
```initD_I``` | initial number of detected infectious individuals | int | 0 
```initR  ``` | initial number of recovered individuals | int | 0
```initF  ``` | initial number of deceased individuals | int | 0

##### Basic SEIR on a network

```python
model = SEIRSNetworkModel(G=myGraph, beta=0.155, sigma=1/5.2, gamma=1/12.39, initI=100)
```


##### Basic SEIRS on a network

```python
model = SEIRSNetworkModel(G=myGraph, beta=0.155, sigma=1/5.2, gamma=1/12.39, xi=0.001, initI=100)
```


##### SEIR on a network with global interactions (p>0)

```python
model = SEIRSNetworkModel(G=myGraph, beta=0.155, sigma=1/5.2, gamma=1/12.39, p=0.5, initI=100)
```

##### SEIR on a network with testing and quarantining (```theta``` and ```psi``` testing params > 0, quarantine network ```Q``` provided)

```python
model = SEIRSNetworkModel(G=myNetwork, beta=0.155, sigma=1/5.2, gamma=1/12.39, p=0.5,
                          Q=quarantineNetwork, q=0.5,
                          theta_E=0.02, theta_I=0.02, psi_E=1.0, psi_I=1.0, 
                          initI=100)
```

##### SEIR on a network with testing, quarantining, and contact tracing (```theta``` and ```psi``` testing params > 0, quarantine network ```Q``` provided, ```phi``` contact tracing params > 0)

```python
model = SEIRSNetworkModel(G=myNetwork, beta=0.155, sigma=1/5.2, gamma=1/12.39, p=0.5,
                          Q=quarantineNetwork, q=0.5,
                          theta_E=0.02, theta_I=0.02, phi_E=0.2, phi_I=0.2, psi_E=1.0, psi_I=1.0,  
                          initI=100)
```

<a name="usage-run"></a>
### Running the Model

Stochastic network SEIRS dynamics are simulated using the Gillespie algorithm.

Once a model is initialized, a simulation can be run with a call to the following function:

```python
model.run(T=300)
```

The ```run()``` function has the following arguments

Argument | Description | Data Type | Default Value
-----|-----|-----|-----
```T``` | runtime of simulation | numeric | REQUIRED
```checkpoints``` | dictionary of checkpoint lists (see section below) | dictionary | ```None```
```print_interval``` | (network model only) time interval to print sim status to console | numeric | 10
```verbose``` | if ```True```, print count in each state at print intervals, else just the time | bool | ```False```

<a name="usage-data"></a>
### Accessing Simulation Data

Model parameter values and the variable time series generated by the simulation are stored in the attributes of the ```SEIRSModel``` or ```SEIRSNetworkModel``` being used and can be accessed directly as follows:

```python
S = model.numS      # time series of S counts
E = model.numE      # time series of E counts
I = model.numI      # time series of I counts
D_E = model.numD_E    # time series of D_E counts
D_I = model.numD_I    # time series of D_I counts
R = model.numR      # time series of R counts
F = model.numF      # time series of F counts

t = model.tseries   # time values corresponding to the above time series

G_normal     = model.G    # interaction network graph
G_quarantine = model.Q    # quarantine interaction network graph

beta = model.beta   # value of beta parameter (or list of beta values for each node if using network model)
# similar for other parameters
```
*Note: convenience methods for plotting these time series are included in the package. See below.*

<a name="usage-networks"></a>
### Specifying Interaction Networks

This model includes a model of SEIRS dynamics for populations with a structured interaction network (as opposed to standard deterministic SIR/SEIR/SEIRS models, which assume uniform mixing of the population). When using the network model, a graph specifying the interaction network for the population must be specified, where each node represents an individual in the population and edges connect individuals who have regular interactions.

The interaction network can be specified by a **```networkx``` ```Graph```** object or a **```numpy``` 2d array** representing the adjacency matrix, either of which can be defined and generated by any method.

This SEIRS+ model also implements dynamics corresponding to testing individuals for the disease and moving individuals with detected infections into a state where their rate of recovery, mortality, etc may be different. In addition, given that this model considers individuals in an interaction network, a separate graph defining the interactions for individuals with detected cases can be specified (i.e., the "quarantine interaction" network).

Epidemic scenarios of interest often involve interaction networks that change in time. Multiple interaction networks can be defined and used at different times in the model simulation using the checkpoints feature (described in the section below).

**_Note:_** *Simulation time increases with network size. Small networks simulate quickly, but have more stochastic volatility. Networks with ~10,000 are large enough to produce per-capita population dynamics that are generally consistent with those of larger networks, but small enough to simulate quickly. We recommend using networks with ~10,000 nodes for prototyping parameters and scenarios, which can then be run on larger networks if more precision is required.*


<a name="usage-checkpoints"></a>
### Changing parameters during a simulation

Model parameters can be easily changed during a simulation run using checkpoints. A dictionary holds a list of checkpoint times (```checkpoints['t']```) and lists of new values to assign to various model parameters at each checkpoint time. 

Example of running a simulation with ```checkpoints```:
```python
checkpoints = {'t':       [20, 100], 
               'G':       [G_distancing, G_normal], 
               'p':       [0.1, 0.5], 
               'theta_E': [0.02, 0.02], 
               'theta_I': [0.02, 0.02], 
               'phi_E':   [0.2, 0.2], 
               'phi_I':   [0.2, 0.2]}
```

*The checkpoints shown here correspond to starting social distancing and testing at time ```t=20``` (the graph ```G``` is updated to ```G_distancing``` and locality parameter ```p``` is decreased to ```0.1```; testing params ```theta_E```, ```theta_I```, ```phi```, and ```phi_I``` are set to non-zero values) and then stopping social distancing at time ```t=100``` (```G``` and ```p``` changed back to their "normal" values; testing params remain non-zero).*

Any model parameter listed in the model constructor can be updated in this way. Only model parameters that are included in the checkpoints dictionary have their values updated at the checkpoint times, all other parameters keep their pre-existing values.

Use cases of this feature include: 

* Changing parameters during a simulation, such as changing transition rates or testing parameters every day, week, on a specific sequence of dates, etc.
* Starting and stopping interventions, such as social distancing (changing interaction network), testing and contact tracing (setting relevant parameters to non-zero or zero values), etc.

**_Consecutive runs_**: *You can also run the same model object multiple times. Each time the ```run()``` function of a given model object is called, it starts a simulation from the state it left off in any previous simulations. 
For example:*
```python
model.run(T=100)    # simulate the model for 100 time units
# ... 
# do other things, such as processing simulation data or changing parameters 
# ...
model.run(T=200)    # simulate the model for an additional 200 time units, picking up where the first sim left off
```

<a name="usage-viz"></a>
## Visualization

### Visualizing the results
The ```SEIRSModel``` and ```SEIRSNetworkModel``` classes have a ```plot()``` convenience function for plotting simulation results on a matplotlib axis. This function generates a line plot of the frequency of each model state in the population by default, but there are many optional arguments that can be used to customize the plot.

These classes also have convenience functions for generating a full figure out of model simulation results (optionally, arguments can be provided to customize the plots generated by these functions, see below). 

- ```figure_basic()``` calls the ```plot()``` function with default parameters to generate a line plot of the frequency of each state in the population.
- ```figure_infections()``` calls the ```plot()``` function with default parameters to generate a stacked area plot of the frequency of only the infection states (*E*, *I*, *D<sub>E</sub>*, *D<sub>I</sub>*) in the population.

Parameters that can be passed to any of the above functions include:
Argument | Description 
-----|-----
```plot_S``` | ```'line'```, ```shaded```, ```'stacked'```, or ```False``` 
```plot_E``` | ```'line'```, ```shaded```, ```'stacked'```, or ```False``` 
```plot_I``` | ```'line'```, ```shaded```, ```'stacked'```, or ```False``` 
```plot_R``` | ```'line'```, ```shaded```, ```'stacked'```, or ```False``` 
```plot_F``` | ```'line'```, ```shaded```, ```'stacked'```, or ```False``` 
```plot_D_E``` | ```'line'```, ```shaded```, ```'stacked'```, or ```False``` 
```plot_D_I``` | ```'line'```, ```shaded```, ```'stacked'```, or ```False``` 
```combine_D``` | ```True``` or ```False```
```color_S``` | matplotlib color of line or stacked area
```color_E``` | matplotlib color of line or stacked area
```color_I``` | matplotlib color of line or stacked area
```color_R``` | matplotlib color of line or stacked area
```color_F``` | matplotlib color of line or stacked area
```color_D_E``` | matplotlib color of line or stacked area
```color_D_I``` | matplotlib color of line or stacked area
```color_reference``` | matplotlib color of line or stacked area
```dashed_reference_results``` | ```seirsplus``` model object containing results to be plotted as a dashed-line reference curve
```dashed_reference_label``` | ```string``` for labeling the reference curve in the legend
```shaded_reference_results``` | ```seirsplus``` model object containing results to be plotted as a dashed-line reference curve
```shaded_reference_label``` | ```string``` for labeling the reference curve in the legend
```vlines``` | ```list``` of x positions at which to plot vertical lines
```vline_colors``` | ```list``` of ```matplotlib``` colors corresponding to the vertical lines
```vline_styles``` | ```list``` of ```matplotlib``` ```linestyle``` ```string```s corresponding to the vertical lines
```vline_labels``` | ```list``` of ```string``` labels corresponding to the vertical lines
```ylim``` | max y-axis value 
```xlim``` | max x-axis value
```legend``` | display legend, ```True``` or ```False```
```title``` | ```string``` plot title
```side_title``` | ```string``` plot title along y-axis
```plot_percentages``` | if ```True``` plot percentage of population in each state, else plot absolute counts
```figsize``` | ```tuple``` specifying figure x and y dimensions
```use_seaborn``` | if ```True``` import ```seaborn``` and use ```seaborn``` styles
