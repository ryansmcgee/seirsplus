from models import SEIRSGraphModel # Import the model.
import numpy
import networkx
import matplotlib.pyplot as pyplot
import seaborn


N_VALS       = [1e3, 1e4, 1e5, 1e6]
GRAPH_M_VALS = [6, 4, 2, 1]
GRAPH_P_VALS = [1.0, 0.5, 0.25, 0.0]
TMAX         = 300

for N in N_VALS:
        N = int(N)

        subplotRows = 2
        subplotCols = 2
        fig, ax = pyplot.subplots(subplotRows, subplotCols, figsize=(16,9))
        linestyles = {0:'solid', 1:'dashdot', 2:'dashed', 3:'dotted'}

        seaborn.set_style('ticks')
        seaborn.despine()


        for i, GRAPH_M in enumerate(GRAPH_M_VALS):

            # Use Networkx to generate a random graph:
            print "Generating graph..."
            G = networkx.barabasi_albert_graph(n=N, m=GRAPH_M)
            avgDegree = numpy.mean([d[1] for d in G.degree()])
            print avgDegree

            for j, GRAPH_P in enumerate(GRAPH_P_VALS):

                # Setup the simulation with given parameters:
                print "Initializing simulation..."
                model = SEIRSGraphModel(G, 
                                        beta    = 0.147, 
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
                                        initF   = 0, 
                                        tmax    = TMAX)


                # Run the simulation:
                print "Running simulation..."
                model.run() 

                print len(model.tseries)

                # exit()

                # Visualize the simulation results:
                # print "Visualizing simulation..."
                # ax[int(i/subplotRows),i%subplotRows].plot(model.tseries[::(int(N)/100)], model.numF[::(int(N)/100)], color='black', linestyle=linestyles[j], alpha=min(GRAPH_P+0.25, 1), label='F (p='+str(GRAPH_P)+')')
                # ax[int(i/subplotRows),i%subplotRows].plot(model.tseries[::(int(N)/100)], model.numS[::(int(N)/100)], color='blue', linestyle=linestyles[j], alpha=min(GRAPH_P+0.25, 1), label='S (p='+str(GRAPH_P)+')')
                # ax[int(i/subplotRows),i%subplotRows].plot(model.tseries[::(int(N)/100)], model.numR[::(int(N)/100)], color='green', linestyle=linestyles[j], alpha=min(GRAPH_P+0.25, 1), label='R (p='+str(GRAPH_P)+')')
                ax[int(i/subplotRows),i%subplotRows].plot(model.tseries[::(int(N)/100)], model.numE[::(int(N)/100)], color='orange', linestyle=linestyles[j], alpha=min(GRAPH_P+0.25, 1), label='E (p='+str(GRAPH_P)+')')
                ax[int(i/subplotRows),i%subplotRows].plot(model.tseries[::(int(N)/100)], model.numI[::(int(N)/100)], color='red', linestyle=linestyles[j], alpha=min(GRAPH_P+0.25, 1), label='I (p='+str(GRAPH_P)+')')

                ax[int(i/subplotRows),i%subplotRows].set_ylim(0, N*0.15)
                ax[int(i/subplotRows),i%subplotRows].set_xlim(0, TMAX)
                ax[int(i/subplotRows),i%subplotRows].set_ylabel('number of individuals')
                ax[int(i/subplotRows),i%subplotRows].set_xlabel('days')

                ax[int(i/subplotRows),i%subplotRows].legend(bbox_to_anchor=(1.0,0.95))
                ax[int(i/subplotRows),i%subplotRows].set_title("N = "+str(int(N))+", Average Degree = "+str(avgDegree))

                # pyplot.show()

                pyplot.tight_layout()

                pyplot.savefig('graphSEIR_N'+str(int(N))+'.png', dpi=200)



