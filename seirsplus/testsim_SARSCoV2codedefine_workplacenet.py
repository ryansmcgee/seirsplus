from models import *
from networks import *
import matplotlib.pyplot

#------------------------

# Instantiate a FARZ network
N = 200
MEAN_DEGREE                = 10
MEAN_CLUSTER_SIZE          = 10
CLUSTER_INTERCONNECTEDNESS = 0.25
network, network_info  = generate_workplace_contact_network(num_cohorts=1, num_nodes_per_cohort=N, 
                                                            num_teams_per_cohort=int(N/MEAN_CLUSTER_SIZE),
                                                            mean_intracohort_degree=MEAN_DEGREE, 
                                                            farz_params={'beta':(1-CLUSTER_INTERCONNECTEDNESS), 'alpha':5.0, 'gamma':5.0, 'r':1, 'q':0.0, 'phi':50, 
                                                                         'b':0, 'epsilon':1e-6, 'directed': False, 'weighted': False})
networks = {"network": network}


# Instantiate the model:
#  - The CompartmentModelBuilder class gives helper functions for defining a new 
#    compartment model from scratch from within a Python script, as demonstrated here.
compartmentBuilder = CompartmentModelBuilider()

compartmentBuilder.add_compartments(['S', 'E', 'P', 'I', 'A', 'R'])

latent_period         = gamma_dist(mean=3.0, coeffvar=0.6, N=N) 
presymptomatic_period = gamma_dist(mean=2.2, coeffvar=0.5, N=N) 
symptomatic_period    = gamma_dist(mean=4.0, coeffvar=0.4, N=N) 
infectious_period     = presymptomatic_period + symptomatic_period
pct_asymptomatic      = 0.3

R0_mean           = 3.0
R0_cv             = 2.0
R0                = gamma_dist(R0_mean, R0_cv, N)
transmissibility  = 1/infectious_period * R0

compartmentBuilder.set_susceptibility('S', to=['P', 'I', 'A'], susceptibility=1.0)

compartmentBuilder.set_transmissibility(['P', 'I', 'A'], 'network', transmissibility) 

compartmentBuilder.add_transition('S', to='E', upon_exposure_to=['P', 'I', 'A'], prob=1.0)
compartmentBuilder.add_transition('E', to='P', time=latent_period, prob=1.0)
compartmentBuilder.add_transition('P', to='I', time=presymptomatic_period, prob=1-pct_asymptomatic)
compartmentBuilder.add_transition('P', to='A', time=presymptomatic_period, prob=pct_asymptomatic)
compartmentBuilder.add_transition('I', to='R', time=symptomatic_period, prob=1.0)
compartmentBuilder.add_transition('A', to='R', time=symptomatic_period, prob=1.0)

model = CompartmentNetworkModel(compartmentBuilder, networks)

print(model.compartments)
# exit()


# Set up the initial state:
model.set_initial_prevalence('E', 0.01)

# Run the model
model.run(T=100)

# Plot results
fig, ax = pyplot.subplots()
ax.fill_between(model.tseries, model.counts['E']+model.counts['P']+model.counts['I']+model.counts['A'], np.zeros_like(model.tseries), label='A', color='pink')
ax.fill_between(model.tseries, model.counts['E']+model.counts['P']+model.counts['I'], np.zeros_like(model.tseries), label='I', color='crimson')
ax.fill_between(model.tseries, model.counts['E']+model.counts['P'], np.zeros_like(model.tseries), label='P', color='orange')
ax.fill_between(model.tseries, model.counts['E'], np.zeros_like(model.tseries), label='E', color='gold')
ax.legend()
pyplot.show()





