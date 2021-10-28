from models import *
from networks import *
import matplotlib.pyplot

#------------------------

# Instantiate a FARZ network
N = 100
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
model = CompartmentNetworkModel(compartments='compartments_SIR_workplacenet.json', networks=networks, 
								transition_mode='time_in_state')

# Set up the initial state:
model.set_initial_prevalence('I', 0.05)

# Run the model
model.run(T=50)

# Plot results
fig, ax = pyplot.subplots()
for compartment in list(model.counts.keys())[::-1]:
    ax.plot(model.tseries, model.counts[compartment], label=compartment)
ax.legend()
pyplot.show()





