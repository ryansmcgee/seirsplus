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
                        p       = 1.0,
                        initE   = 0, 
                        initI   = 10, 
                        initR   = 0, 
                        initF   = 0)


# Run the simulation:
print "Running simulation..."
model.run(300) 


# Visualize the simulation results:
# print "Visualizing simulation..."
pyplot.plot(model.tseries, model.numF, color='black', label='F')
pyplot.plot(model.tseries, model.numS, color='blue', label='S')
pyplot.plot(model.tseries, model.numR, color='green', label='R')
pyplot.plot(model.tseries, model.numE, color='orange', label='E')
pyplot.plot(model.tseries, model.numI, color='red', label='I')
pyplot.ylabel('number of individuals')
pyplot.xlabel('days')
pyplot.legend(bbox_to_anchor=(1.0,0.95));

seaborn.set_style('ticks')
seaborn.despine()
pyplot.show()