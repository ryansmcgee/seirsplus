from models import *
from networks import *
import matplotlib.pyplot

#------------------------

# Define a simple network with three layers:
adj_toywork   = numpy.array([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							 [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							 [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
							 [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
							 [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
							 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

adj_toyschool = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							 [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
							 [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
							 [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
							 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
							 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
							 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]])

adj_toyhome   = numpy.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
							 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
							 [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
							 [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
							 [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
							 [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
							 [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
							 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							 [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
							 [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
							 [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]])

networks = {'work': adj_toywork, 'school': adj_toyschool, 'home': adj_toyhome}

# Instantiate the model:
model = CompartmentNetworkModel(compartments='compartments_SIR_toynet.json', networks=networks, 
								transition_mode='exponential_rates', # or 'time_in_state'
								isolation_period=10)

# Set up a particular initial state:
model.set_state(node=[1, 5, 8, 10], state='I')
model.set_isolation(node=[1, 5, 8, 11], isolation=True)
model.set_network_activity(network=['work', 'school'], active=True, active_isolation=False)
model.set_network_activity(network='home', active=True, active_isolation=True)

# Run the model
model.run(T=50)

# Plot results
fig, ax = pyplot.subplots()
for compartment in list(model.counts.keys())[::-1]:
    ax.plot(model.tseries, model.counts[compartment], label=compartment)
ax.legend()
pyplot.show()





