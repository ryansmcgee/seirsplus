# SEIRS+ Model

This package implements generalized SEIRS infectious disease dynamics models with extensions that model the effect of factors including population structure, social distancing, testing, contact tracing, and quarantining detected cases. 

Notably, this package includes stochastic implementations of these models on dynamic networks.

**README Contents:**
* [ Model Description ](#model)
   * [ SEIRS Dynamics ](#model-seirs)
   * [ SEIRS Dynamics with Testing ](#model-seirstesting)
   * [ Deterministic Model ](#model-determ)
   * [ Network Model ](#model-network)
      * [ Network Model with Testing, Contact Tracing, and Quarantining ](#model-network-ttq)
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
* β: rate of transmission (transmissions per S-I contact per time)
* σ: rate of progression (inverse of incubation period)
* γ: rate of recovery (inverse of infectious period)
* ξ: rate of re-susceptibility (inverse of temporary immunity period; 0 if permanent immunity)
* μ<sub>I</sub>: rate of mortality from the disease (deaths per infectious individual per time)

<a name="model-seirstesting"></a>
### SEIRS Dynamics with Testing

The effect of testing for infection on the dynamics can be modeled by introducing states corresponding to **detected exposed (*D<sub>E</sub>*)** and **detected infectious (*D<sub>I</sub>*)**. Exposed and infectious individuals are tested at rates *θ<sub>E</sub>* and *θ<sub>I</sub>*, respectively, and test positively for infection with rates *ψ<sub>E</sub>* and *ψ<sub>I</sub>*, respectively  (the false positive rate is assumed to be zero, so susceptible individuals never test positive). Testing positive moves an individual into the appropriate detected case state, where rates of transmission, progression, recovery, and/or mortality (as well as network connectivity in the network model) may be different than those of undetected cases.

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


<a name="model-determ"></a>
### Deterministic Model

The evolution of the SEIRS dynamics described above can be described by the following systems of differential equations. Importantly, this version of the model is deterministic and assumes a uniformly-mixed population. 

#### SEIRS Dynamics

<p align="center">
  <img src="https://github.com/ryansmcgee/seirsplus/blob/master/images/SEIRS_deterministic_equations.png" width="210"></div>
</p>

where *S*, *E*, *I*, *R*, and *F* are the numbers of susceptible, exposed, infectious, recovered, and deceased individuals, respectively, and *N* is the total number of individuals in the population (parameters are described above).

#### SEIRS Dynamics with Testing

<p align="center">
  <img src="https://github.com/ryansmcgee/seirsplus/blob/master/images/SEIRStesting_deterministic_equations.png" width="400"></div>
</p>

where *S*, *E*, *I*, *D<sub>E</sub>*, *D<sub>I</sub>*, *R*, and *F* are the numbers of susceptible, exposed, infectious, detected exposed, detected infectious, recovered, and deceased individuals, respectively, and *N* is the total number of individuals in the population (parameters are described above).

<a name="model-network"></a>
### Network Model

The standard SEIRS model captures important features of infectious disease dynamics, but it is deterministic and assumes uniform mixing of the population (every individual in the population is equally likely to interact with every other individual). However, it is often important to consider stochastic effects and the structure of contact networks when studying disease transmission and the effect of interventions such as social distancing and contact tracing.

This package includes implementation of the SEIRS dynamics on stochastic dynamical networks. This avails analysis of the realtionship between network structure and effective transmission rates, including the effect of network-based interventions such as social distancing, quarantining, and contact tracing.

<img align="right" src="https://github.com/ryansmcgee/seirsplus/blob/master/images/network_contacts.png" height="250">

Consider a graph **_G_** representing individuals (nodes) and their interactions (edges). Each individual (node) has a state (*S, E, I, D<sub>E</sub>, D<sub>I</sub>, R, or F*). The set of nodes adjacent (connected by an edge) to an individual defines their set of "close contacts" (highlighted in black).  At a given time, each individual makes contact with a random individual from their set of close contacts with probability *(1-p)β* or with a random individual from anywhere in the network (highlighted in blue) with probability *pβ*. The latter global contacts represent individuals interacting with the population at large (i.e., individuals outside of ones social circle, such as on public transit, at an event, etc.) with some probability. When a susceptible individual interacts with an infectious individual they become exposed. The parameter *p* defines the locality of the network: for *p=0* an individual only interacts with their close contacts, while *p=1* represents a uniformly mixed population. Social distancing interventions may increase the locality of the network (i.e., decrease *p*) and/or decrease local connectivity of the network (i.e., decrease the degree of individuals).

Each node *i* has a state *X<sub>i</sub>* that updates according to the following probability transition rates: 

<p align="center">
  <img src="https://github.com/ryansmcgee/seirsplus/blob/master/images/SEIRSnetwork_transitions.png" width="500"></div>
</p>

where *δ<sub>Xi=A</sub> = 1* if the state of *X<sub>i</sub>* is *A*, or *0* if not, and where *C<sub>G</sub>(i)* denotes the set of close contacts of node *i*. For large populations and *p=1*, this stochastic model approaches the same dynamics as the deterministic SEIRS model.

This implementation builds on the work of Dottori et al. (2015).
* Dottori, M. and Fabricius, G., 2015. SIR model on a dynamical network and the endemic state of an infectious disease. Physica A: Statistical Mechanics and its Applications, 434, pp.25-35.

<a name="model-network-ttq"></a>
#### Network Model with Testing, Contact Tracing, and Quarantining

##### Testing & Contact Tracing

As with the deterministic model, exposed and infectious individuals are tested at rates *θ<sub>E</sub>* and *θ<sub>I</sub>*, respectively, and test positively for infection with rates *ψ<sub>E</sub>* and *ψ<sub>I</sub>*, respectively (the false positive rate is assumed to be zero, so susceptible individuals never test positive). Testing positive moves an individual into the appropriate detected case state (*D<sub>E</sub>* or *D<sub>I</sub>*), where rates of transmission, progression, recovery, and/or mortality (as well as network connectivity in the network model) may be different than those of undetected cases.

Consideration of interaction networks allows us to model contact tracing, where the close contacts of an positively-tested individual are more likely to be tested in response. In this model, an individual is tested due to contact tracing at a rate equal to *φ* times the number of its close contacts who have tested positively.

##### Quarantining

<img align="right" src="https://github.com/ryansmcgee/seirsplus/blob/master/images/network_contacts_quarantine.png" height="250">

Now we also consider another graph **_Q_** which represents the interactions that each individual has if they test positively for the disease (i.e., individuals in the *D<sub>E</sub>* or *D<sub>I</sub>* states) and enter into a form of quarantine. The quarantine has the effect of dropping some fraction of the edges connecting the quarantined individual to others (according to a rule of the user's choice when generating the graph *Q*). The edges of *Q* (highlighted in purple) for each individual are then a subset of the normal edges of *G* for that individual. The set of nodes that are adjacent to a quarantined individual define their set of "quarantine contacts" (highlighted in purple). At a given time, a quarantined individual may come into contact with another individual in this quarantine contact set with probability *(1-p)β<sub>D</sub>*. A quarantined individual may also be come in contact with a random individual from anywhere in the network with rate *qpβ<sub>D</sub>*.

Each node *i* has a state *X<sub>i</sub>* that updates according to the following probability transition rates: 
<p align="center">
  <img src="https://github.com/ryansmcgee/seirsplus/blob/master/images/SEIRSnetworktesting_transitions.png" width="800"></div>
</p>

where *δ<sub>Xi=A</sub>=1* if the state of *X<sub>i</sub>* is *A*, or *0* if not, and where *C<sub>G</sub>(i)* and *C<sub>Q</sub>(i)* denotes the set of close contacts and quarantine contacts of node *i*, respectively. For large populations and *p=1*, this stochastic model approaches the same dynamics as the deterministic SEIRS model (sans contact tracing, which is not included in the uniformly-mixed model).

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

<a name="usage-run"></a>
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

#### Custom Exponential Graph

Human interaction networks often resemble scale-free power law networks with exponential degree distributions.
This package includes a ```custom_exponential_graph()``` convenience funciton that generates power-law-like graphs that have degree distributions with two exponential tails. The method of generating these graphs also makes it easy to remove edges from a reference graph and decrease the degree of the network, which is useful for generating networks representing social distancing and quarantine conditions.

Common algorithms for generating power-law graphs, such as the Barabasi-Albert preferential attachment algorithm, produce graphs that have a minimum degree; that is, no node has fewer than *m* edges for some value of *m*, which is unrealistic for interaction networks. This ```custom_exponential_graph()``` function simply produces graphs with degree distributions that have a peak near their mean and exponential tails in the direction of both high and low degrees. (No claims about the realism or rigor of these graphs are made.)

<img align="right" src="https://github.com/ryansmcgee/seirsplus/blob/master/images/degreeDistn_compareToBAGraph1.png" height="250">

This function generates graphs using the following algorithm:
* Start with a Barabasi-Albert preferential attachment power-law graph (or any graph that is optionally provided by the user).
* For each node:
    * Count the number of neighbors *n* of the node 
    * Draw a random number *r* from an exponential distribution with some mean=```scale```. If *r>n*, set *r=n*. 
    * Randomly select *r* of this node’s neighbors to keep, delete the edges to all other neighbors. 
When starting from a Barabasi-Albert (BA) graph, this generates a new graphs that have a peak at their mean and approximately exponential tails in both directions, as shown to the right.

<img align="right" src="https://github.com/ryansmcgee/seirsplus/blob/master/images/degreeDistn_compareToBAGraph4.png" height="250">

Since this algorithm starts with a graph with defined connections and makes a new graph by breaking some number of connections, it also makes it easy to take an existing graph and make a subgraph of it that also has exponential-ish tails and a left-shifted mean. This can be used for generating social distancing and quarantine subgraphs. The amount of edge breaking/degree reduction is modulated by the ```scale``` parameter. To the right are some examples of graphs with progressively lower mean degree generated using the same reference Barabasi-Albert graph, which therefore are all subsets of a common reference set of edges.

The ```custom_exponential_graph()``` function has the following arguments

base_graph=None, scale=100, min_num_edges=0, m=9, n=None
Argument | Description | Data Type | Default Value
-----|-----|-----|-----
```base_graph``` | Graph to use as the starting point for the algorithm. If ```None```, generate a Barabasi-Albert graph to use as the starting point using arguments ```n``` and ```m``` as parameters | ```networkx``` ```Graph``` object | ```None```
```scale``` | Mean of the exponential distribution used to draw ```base_graph``` to keep. Large values result in graphs that approximate the original ```base_graph```, small values result in sparser subgraphs of ```base_graph```  | numeric | 100
```min_num_edges``` | Minimum number of edges that all nodes must have in the generated graph | int | 0
```n``` | *n* parameter for teh Barabasi-Albert algorithm (number of nodes to add) | int | ```None``` (value required when no ```base_graph``` is given)
```m``` | *m* parameter for the Barabasi-Albert algorithm (number of edges added with each added node) | int | 9

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
