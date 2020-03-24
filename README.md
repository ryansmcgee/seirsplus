# SEIRS+ Model

This package implements generalized SEIRS infectious disease dynamics models with extensions that model the effect of factors including population structure, social distancing, testing, contact tracing, and quarantining detected cases. 

Notably, this package includes stochastic implementations of these models on dynamic networks.

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


<a name="model-standard"></a>
### Deterministic Model

The evolution of the SEIRS dynamics described above can be described by the following systems of differential equations. Importantly, this version of the model is deterministic and assumes a uniformly-mixed population. 

### SEIRS Dynamics

<p align="center">
  <img src="https://github.com/ryansmcgee/seirsplus/blob/master/images/SEIRS_deterministic_equations.png" width="210"></div>
</p>

where *S*, *E*, *I*, *R*, and *F* are the numbers of susceptible, exposed, infectious, receovered, and deceased individuals, respectively, and *N* is the total number of individuals in the population (parameters are described above).

### SEIRS Dynamics with Testing

<p align="center">
  <img src="https://github.com/ryansmcgee/seirsplus/blob/master/images/SEIRStesting_deterministic_equations.png" width="400"></div>
</p>

where *S*, *E*, *I*, *D<sub>E</sub>*, *D<sub>I</sub>*, *R*, and *F* are the numbers of susceptible, exposed, infectious, detected exposed, detected infectious, receovered, and deceased individuals, respectively, and *N* is the total number of individuals in the population (parameters are described above).

<a name="model-network"></a>
### Network Model

The standard SEIRS model captures important features of infectious disease dynamics, but it is deterministic and assumes uniform mixing of the population (every individual in the population is equally likely to interact with every other individual). However, it is often important to consider stochastic effects and the structure of contact networks when studying disease transmission and the effect of interventions such as social distancing and contact tracing.

This package includes implementation of the SEIRS dynamics on stochastic dynamical networks. This avails analysis of the realtionship between network structure and effective transmission rates, including the effect of network-based interventions such as social distancing, quarantining, and contact tracing.

<img align="right" src="https://github.com/ryansmcgee/seirsplus/blob/master/images/network_contacts.png" height="250">

Consider a graph **_G_** representing individuals (nodes) and their interactions (edges). 
Each individual (node) has a state (*S, E, I, D<sub>E</sub>, D<sub>I</sub>, R, or F*).
The set of nodes adjacent (connected by an edge) to an individual defines their set of "close contacts" (highlighted in black).  At a given time, each individual makes contact with a random individual from their set of close contacts with probability *(1-p)β* or with a random individual from anywhere in the network (highlighted in blue) with probability *pβ*. 
The latter global contacts represent individuals interacting with the population at large (i.e., individuals outside of ones social circle, such as on public transit, at an event, etc.) with some probability.
When a susceptible individual interacts with an infectious individual they become exposed. 
The parameter *p* defines the locality of the network: for *p=0* an individual only interacts with their close contacts, while *p=1* represents a uniformly mixed population. Social distancing interventions may increase the locality of the network (i.e., decrease *p*) and/or decrease local connectivity of the network (i.e., decrease the degree of individuals).

Each node *i* has a state *X<sub>i</sub>* that updates according to the following probability transition rates: 

<p align="center">
  <img src="https://github.com/ryansmcgee/seirsplus/blob/master/images/SEIRSnetwork_transitions.png" width="500"></div>
</p>

where *δ<sub>Xi=A</sub> = 1* if the state of *X_i* is *A*, or *0* if not, and where *C<sub>G</sub>(i)* denotes the set of close contacts of node *i*. For large populations and *p=1*, this stochastic model approaches the same dynamics as the deterministic SEIRS model.

This implementation is based on the work of Dottori et al. (2018)
* Dottori, M. and Fabricius, G., 2015. SIR model on a dynamical network and the endemic state of an infectious disease. Physica A: Statistical Mechanics and its Applications, 434, pp.25-35.

#### Network Model with Testing, Contact Tracing, and Quarantining

<img align="right" src="https://github.com/ryansmcgee/seirsplus/blob/master/images/network_contacts_quarantine.png" height="250">

##### Testing & Contact Tracing

As with the deterministic model, exposed and infectious individuals are tested at rates *θ<sub>E</sub>* and *θ<sub>I</sub>*, respectively, and test positively for infection with rates *ψ<sub>E</sub>* and *ψ<sub>I</sub>*, respectively (the false positive rate is assumed to be zero, so susceptible individuals never test positive). Testing positive moves an individual into the appropriate detected case state (*D<sub>E</sub>* or *D<sub>I</sub>*), where rates of transmission, progression, recovery, and/or mortality (as well as network connectivity in the network model) may be different than those of undetected cases.

Consideration of interaction networks allows us to model contact tracing, where the close contacts of an positively-tested individual are more likely to be tested in response. In this model, an individual is tested due to contact tracing at a rate equal to *φ* times the number of its close contacts who have tested positively.

##### Quarantining

Now we also consider another graph **_Q_** which represents the interactions that each individual has if they test positively for the disease (i.e., individuals in the *D_E* or *D_I* states) and enter into a form of quarantine.  
The quarantine has the effect of dropping some fraction of the edges connecting the quarantined individual to others (according to some rule to be specified elsewhere). 
The edges of *Q* for each individual are then a subset of the edges of *G* for that individual.
The set of nodes that are adjacent to a quarantined individual define their set of "quarantine contacts" (highlighted in purple). 
At a given time, a quarantined individual may come into contact with another individual in this quarantine contact set with probability *(1-p)β<sub>D</sub>*. 
A quarantined individual may also be come in contact with a random individual from anywhere in the network with rate *qpβ<sub>D</sub>*..

Each node *i* has a state *X<sub>i</sub>* that updates according to the following probability transition rates: 
<p align="center">
  <img src="https://github.com/ryansmcgee/seirsplus/blob/master/images/SEIRSnetworktesting_transitions.png" width="800"></div>
</p>

where *δ<sub>Xi=A</sub>=1* if the state of *X<sub>i</sub>* is *A*, or *0* if not, and where *C<sub>G</sub>(i)* and *C<sub>Q</sub>(i)* denotes the set of close contacts and quarantine contacts of node *i*, respectively. For large populations and *p=1*, this stochastic model approaches the same dynamics as the deterministic SEIRS model (sans contact tracing, which is not included in the uniformly-mixed model).

<a name="usage"></a>
## Code Usage

sometext
