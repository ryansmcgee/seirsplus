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
#  - The following class defines a pre-configured SARSCoV2 model with heterogeneous 
#    and realistic (at least for pre-delta) parameter distributions
#     - Transmissibility is overdispersed in this configuration, so most individuals 
#       are not expected to generate secondary cases, and many runs with low initial 
#       prevalence never take off into epidemics.
model = SARSCoV2NetworkModel(networks=networks)

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





