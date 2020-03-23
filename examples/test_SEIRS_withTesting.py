from models import SEIRSGraphModel # Import the SEIRS graph model.
from models import custom_exponential_graph
import numpy
import networkx
import matplotlib.pyplot as pyplot
import seaborn


# Generate the interaction graphs to be used:
print "Generating graphs..."
G    = custom_exponential_graph(networkx.barabasi_albert_graph(n=1e4, m=9), scale=100)


# Setup the simulation with given parameters:
print "Initializing simulation..."
model = SEIRSGraphModel(G, 
                        beta    = 0.155, 
                        sigma   = 1/5.2, 
                        gamma   = 1/12.39, 
                        xi      = 0.0, 
                        mu_I    = 0.0004, 
                        mu_0    = 0.0,   
                        nu      = 0.0, 
                        p       = 0.1,
                        Q       = G, 
                        beta_D  = 1.0, 
                        theta_E = 0.02, 
                        theta_I = 0.02, 
                        phi_E   = 0.2, 
                        phi_I   = 0.2, 
                        psi_E   = 1.0, 
                        psi_I   = 1.0, 
                        q       = 0.5,
                        initE   = 0, 
                        initI   = 100, 
                        initD_E = 0, 
                        initD_I = 0, 
                        initR   = 0, 
                        initF   = 0)


# Run the simulation:
print "Running simulation..."
model.run(300) 




# Visualize the simulation results:
# print "Visualizing simulation..."
pyplot.plot(model.tseries, model.numF, color='black', label='$F$')
pyplot.plot(model.tseries, model.numS, color='blue', label='$S$')
pyplot.plot(model.tseries, model.numR, color='green', label='$R$')
pyplot.plot(model.tseries, model.numE, color='orange', label='$E$')
pyplot.plot(model.tseries, model.numI, color='crimson', alpha=1.0, linestyle='--', label='$I$')
pyplot.plot(model.tseries, model.numD_E, color='mediumorchid', alpha=1.0, linestyle=':', label='$D_E$')
pyplot.plot(model.tseries, model.numD_I, color='mediumorchid', alpha=1.0, linestyle='--', label='$D_I$')
pyplot.plot(model.tseries, model.numI+model.numD_E+model.numD_I, color='crimson', label='$I+D_{all}$')
pyplot.ylabel('number of individuals')
pyplot.xlabel('days')
pyplot.legend(bbox_to_anchor=(1.0,0.95));

seaborn.set_style('ticks')
seaborn.despine()
pyplot.show()

