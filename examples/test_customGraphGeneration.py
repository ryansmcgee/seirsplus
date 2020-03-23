from models import SEIRSGraphModel # Import the SEIRS graph model.
from models import custom_exponential_graph
import numpy
import networkx
import matplotlib.pyplot as pyplot
import seaborn

seaborn.set_style('ticks')
seaborn.despine()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define global params:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
N       = int(1e4)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generate the interaction graphs to be used:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print "Generating graphs..."
baseGraph   = networkx.barabasi_albert_graph(n=N, m=9)
# Baseline normal social interactions:
G_NORMAL    = custom_exponential_graph(baseGraph, scale=100)
# Moderate social distancing interactions:
G_MEDDIST   = custom_exponential_graph(baseGraph, scale=30)
# Major social distancing interactions:
G_BIGDIST   = custom_exponential_graph(baseGraph, scale=10)
# Quarantining interactions:
G_QRNTINE   = custom_exponential_graph(baseGraph, scale=5)

nodeDegrees_baseGraph    = [d[1] for d in baseGraph.degree()]
nodeDegrees_NORMAL       = [d[1] for d in G_NORMAL.degree()]
nodeDegrees_MEDDIST      = [d[1] for d in G_MEDDIST.degree()]
nodeDegrees_BIGDIST      = [d[1] for d in G_BIGDIST.degree()]
nodeDegrees_QRNTINE      = [d[1] for d in G_QRNTINE.degree()]

meanDegree_baseGraph= numpy.mean(nodeDegrees_baseGraph)
meanDegree_NORMAL 	= numpy.mean(nodeDegrees_NORMAL)
meanDegree_MEDDIST 	= numpy.mean(nodeDegrees_MEDDIST)
meanDegree_BIGDIST 	= numpy.mean(nodeDegrees_BIGDIST)
meanDegree_QRNTINE 	= numpy.mean(nodeDegrees_QRNTINE)

# get largest connected component
# unfortunately, the iterator over the components is not guaranteed to be sorted by size
components = sorted(networkx.connected_components(baseGraph), key=len, reverse=True)
numConnectedComps = len(components)
largestConnectedComp = baseGraph.subgraph(components[0])
print("baseGraph mean degree = "+str((meanDegree_baseGraph)))
print("baseGraph number of connected components = {0:d}".format(numConnectedComps))
print("baseGraph largest connected component = {0:d}".format(len(largestConnectedComp)))



components = sorted(networkx.connected_components(G_NORMAL), key=len, reverse=True)
numConnectedComps = len(components)
largestConnectedComp = G_NORMAL.subgraph(components[0])
print("G_NORMAL mean degree = "+str((meanDegree_NORMAL)))
print("G_NORMAL number of connected components = {0:d}".format(numConnectedComps))
print("G_NORMAL largest connected component = {0:d}".format(len(largestConnectedComp)))


components = sorted(networkx.connected_components(G_MEDDIST), key=len, reverse=True)
numConnectedComps = len(components)
largestConnectedComp = G_MEDDIST.subgraph(components[0])
print("G_MEDDIST mean degree = "+str((meanDegree_MEDDIST)))
print("G_MEDDIST number of connected components = {0:d}".format(numConnectedComps))
print("G_MEDDIST largest connected component = {0:d}".format(len(largestConnectedComp)))


components = sorted(networkx.connected_components(G_BIGDIST), key=len, reverse=True)
numConnectedComps = len(components)
largestConnectedComp = G_BIGDIST.subgraph(components[0])
print("G_BIGDIST mean degree = "+str((meanDegree_BIGDIST)))
print("G_BIGDIST number of connected components = {0:d}".format(numConnectedComps))
print("G_BIGDIST largest connected component = {0:d}".format(len(largestConnectedComp)))


components = sorted(networkx.connected_components(G_QRNTINE), key=len, reverse=True)
numConnectedComps = len(components)
largestConnectedComp = G_QRNTINE.subgraph(components[0])
print("G_QRNTINE mean degree = "+str((meanDegree_QRNTINE)))
print("G_QRNTINE number of connected components = {0:d}".format(numConnectedComps))
print("G_QRNTINE largest connected component = {0:d}".format(len(largestConnectedComp)))


pyplot.hist(nodeDegrees_baseGraph, bins=range(int(max(nodeDegrees_baseGraph))), alpha=0.5, color='black', label='BA graph')
pyplot.hist(nodeDegrees_NORMAL, bins=range(int(max(nodeDegrees_NORMAL))), alpha=0.5, color='tab:blue', label=('normal interactions (mean=%.1f)' % meanDegree_NORMAL))
pyplot.hist(nodeDegrees_MEDDIST, bins=range(int(max(nodeDegrees_MEDDIST))), alpha=0.5, color='tab:purple', label=('moderate distancing (mean=%.1f)' % meanDegree_MEDDIST))
pyplot.hist(nodeDegrees_BIGDIST, bins=range(int(max(nodeDegrees_BIGDIST))), alpha=0.5, color='tab:orange', label=('major distancing (mean=%.1f)' % meanDegree_BIGDIST))
pyplot.hist(nodeDegrees_QRNTINE, bins=range(int(max(nodeDegrees_QRNTINE))), alpha=0.5, color='tab:red', label=('quarantine (mean=%.1f)' % meanDegree_QRNTINE))
pyplot.xlim(0,40)
pyplot.xlabel('degree')
pyplot.ylabel('num nodes')
pyplot.legend(loc='upper right')
pyplot.show()