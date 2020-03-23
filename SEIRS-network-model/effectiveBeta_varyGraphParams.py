from models import SEIRSGraphModel # Import the model.
from models import custom_exponential_graph
import numpy
import pandas
import networkx
from scipy import stats
import matplotlib.pyplot as pyplot
import seaborn


GRAPH_SCALES = [100, 80, 50, 30, 10, 5]
GRAPH_P_VALS = [1.0, 0.75, 0.5, 0.25, 1.0e-1, 0.5e-1, 1.0e-2, 0.5e-2, 1.0e-3, 0.0]
GRAPH_REPS 	 = 10
N 			 = int(1e4)
TMAX         = 300

BETA 		 = 0.15

data = []



for rep in range(GRAPH_REPS):

	# Use Networkx to generate a random base graph:
	baseGraph   = networkx.barabasi_albert_graph(n=N, m=9)

	for i, GRAPH_SCALE in enumerate(GRAPH_SCALES):

		# Effective graph of interactions:
		G = custom_exponential_graph(baseGraph, scale=GRAPH_SCALE)

		meanDegree = numpy.mean([d[1] for d in G.degree()])
		print meanDegree

		for j, GRAPH_P in enumerate(GRAPH_P_VALS):

			# Setup the simulation with given parameters:
			print "Initializing simulation..."
			model = SEIRSGraphModel(G, 
									beta    = BETA, 
									sigma   = 1/5.2, 
									gamma   = 1/12.39, 
									xi      = 0.0, 
									mu_I    = 0.0004, 
									mu_0    = 0.0,   
									nu      = 0.0, 
									p       = GRAPH_P,
									initE   = 0, 
									initI   = int(N/100), 
									initR   = 0, 
									initF   = 0)

			# Run the simulation:
			print "Running simulation..."
			model.run(T=TMAX) 

			
			#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
			# Calculate effective transmission rate:
			#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

			logS = numpy.log(model.numS)

			meanDeltaT = numpy.mean(numpy.ediff1d(model.tseries))

			# Define effective beta as the min rate of decrease in number of susceptible individuals.
			# i.e., minimum slope of log(numS)
			maxSlope 	  = 0	# slope = dS/dt
			maxSlope_tidx = 0
			maxSlopeScore = 0.0

			regressionWindowSize = 30.0
			regressionWindowSize_indexSpan = int(regressionWindowSize/meanDeltaT)
			for i, windowStartTime in enumerate(model.tseries):
				tidx_start 		= i
				tidx_end 		= i+regressionWindowSize_indexSpan
				tidx_mid 		= int(tidx_start + (tidx_end - tidx_start)/2)
				if(tidx_end > len(model.tseries)):
					break

				tseries_window 	= model.tseries[tidx_start:min(len(model.tseries)-1, tidx_end)]
				logS_window 	= logS[tidx_start:min(len(logS)-1, tidx_end)]

				if(len(logS_window) == 0):
					break

				slope, intercept, r_value, p_value, std_err = stats.linregress(tseries_window, logS_window)

				if(-1*slope*r_value**2 >= maxSlopeScore):
					maxSlope 	  = -1*slope
					maxSlopeScore = -1*slope*r_value**2
					maxSlope_tidx = tidx_mid

			print maxSlope

			effectiveBeta = maxSlope * model.N[maxSlope_tidx] / (model.numI[maxSlope_tidx])

			print effectiveBeta


			data.append({ 'rep':rep, 'N':N, 'graph_scale':GRAPH_SCALE, 'meanDegree':meanDegree,
							'p':GRAPH_P, 'beta':BETA, 'effectiveBeta':effectiveBeta, 'effectiveBetaPct':effectiveBeta/BETA })

			dataframe = pandas.DataFrame(data)
			print dataframe
			dataframe.to_csv('effectiveBeta_varyingGraphParams.csv')
				
dataframe = pandas.DataFrame(data)
print dataframe
dataframe.to_csv('effectiveBeta_varyingGraphParams.csv')
