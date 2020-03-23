from models import SEIRSGraphModel # Import the model.
import numpy
import networkx
import matplotlib.pyplot as pyplot
import seaborn


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Setup figure for later:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig, ax = pyplot.subplots(2, 2, figsize=(16,9))
seaborn.set_style('ticks')
seaborn.despine()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define global params:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
N       = int(1e5)
TMAX    = 300

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Use Networkx to generate a random graph:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print "Generating graph..."
G = networkx.barabasi_albert_graph(n=N, m=6)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generate a quarantine contacts graph
# by subsampling the edges of each node in G:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Q = G.copy()
for node in Q:
    neighbors = Q[node].keys()
    quarantineEdgeNum = min(int(numpy.random.exponential(scale=10, size=1)), len(neighbors))
    quarantineKeepNeighbors = numpy.random.choice(neighbors, size=quarantineEdgeNum, replace=False)
    for neighbor in neighbors:
        if(neighbor not in quarantineKeepNeighbors):
            Q.remove_edge(node, neighbor)
print numpy.mean([d[1] for d in G.degree()])
print numpy.mean([d[1] for d in Q.degree()])
# pyplot.hist([d[1] for d in G.degree()], bins=range(max([d[1] for d in G.degree()])))
# pyplot.hist([d[1] for d in Q.degree()], bins=range(max([d[1] for d in Q.degree()])))
# pyplot.ylim(0,20)
# pyplot.show()
# exit()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# NO TESTING SCENARIO
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print "Initializing simulation..."
model = SEIRSGraphModel(G, 
                        beta    = 0.165, 
                        sigma   = 1/5.2, 
                        gamma   = 1/12.39, 
                        xi      = 0.0, 
                        mu_I    = 0.0004, 
                        mu_0    = 0.0,   
                        nu      = 0.0, 
                        p       = 0.1,
                        Q       = G, 
                        beta_D  = 0.165, 
                        theta_E = 0.0, 
                        theta_I = 0.0, 
                        phi_E   = 0.0, 
                        phi_I   = 0.0, 
                        psi_E   = 1.0, 
                        psi_I   = 1.0, 
                        q       = 1.0,
                        initE   = 0, 
                        initI   = int(N/100),
                        initD   = 0, 
                        initR   = 0, 
                        initF   = 0, 
                        tmax    = TMAX)


# Run the simulation:
print "Running simulation..."
model.run() 

notesting_tseries = model.tseries
notesting_numI    = model.numI


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RANDOM TESTING SCENARIO
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print "Initializing simulation..."
model = SEIRSGraphModel(G, 
                        beta    = 0.165, 
                        sigma   = 1/5.2, 
                        gamma   = 1/12.39, 
                        xi      = 0.0, 
                        mu_I    = 0.0004, 
                        mu_0    = 0.0,   
                        nu      = 0.0, 
                        p       = 0.1,
                        Q       = G, 
                        beta_D  = 0.165, 
                        theta_E = 0.02, 
                        theta_I = 0.02, 
                        phi_E   = 0.0, 
                        phi_I   = 0.0, 
                        psi_E   = 1.0, 
                        psi_I   = 1.0, 
                        q       = 1.0,
                        initE   = 0, 
                        initI   = int(N/100),
                        initD   = 0, 
                        initR   = 0, 
                        initF   = 0, 
                        tmax    = TMAX)


# Run the simulation:
print "Running simulation..."
model.run() 


ax[0,0].fill_between(notesting_tseries[::(int(N)/100)], notesting_numI[::(int(N)/100)], 0, color='#EFEFEF')
ax[0,0].plot(notesting_tseries[::(int(N)/100)], notesting_numI[::(int(N)/100)], color='#E0E0E0', label='I (no testing)')
# ax[0,0].plot(model.tseries[::(int(N)/100)], model.numS[::(int(N)/100)], color='blue', label='S')
# ax[0,0].plot(model.tseries[::(int(N)/100)], model.numR[::(int(N)/100)], color='green', label='R')
ax[0,0].plot(model.tseries[::(int(N)/100)], model.numE[::(int(N)/100)], color='orange', label='E')
ax[0,0].plot(model.tseries[::(int(N)/100)], model.numI[::(int(N)/100)], color='crimson', alpha=0.5, linestyle=':', label='I')
ax[0,0].plot(model.tseries[::(int(N)/100)], model.numD[::(int(N)/100)], color='crimson', alpha=0.5, linestyle='--', label='D')
ax[0,0].plot(model.tseries[::(int(N)/100)], model.numI[::(int(N)/100)]+model.numD[::(int(N)/100)], color='crimson', label='I+D')
ax[0,0].plot(model.tseries[::(int(N)/100)], model.numF[::(int(N)/100)], color='black', label='F')

ax[0,0].set_ylim(0, N*0.15)
ax[0,0].set_xlim(0, TMAX)
ax[0,0].set_ylabel('number of individuals')
ax[0,0].set_xlabel('days')

ax[0,0].legend(bbox_to_anchor=(1.0,0.95))
ax[0,0].set_title("Random Testing (N="+str(N)+")")

# pyplot.show()
# exit()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RANDOM TESTING & CONTACT TRACING SCENARIO
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print "Initializing simulation..."
model = SEIRSGraphModel(G, 
                        beta    = 0.165, 
                        sigma   = 1/5.2, 
                        gamma   = 1/12.39, 
                        xi      = 0.0, 
                        mu_I    = 0.0004, 
                        mu_0    = 0.0,   
                        nu      = 0.0, 
                        p       = 0.1,
                        Q       = G, 
                        beta_D  = 0.165, 
                        theta_E = 0.02, 
                        theta_I = 0.02, 
                        phi_E   = 0.1, 
                        phi_I   = 0.1, 
                        psi_E   = 1.0, 
                        psi_I   = 1.0, 
                        q       = 1.0,
                        initE   = 0, 
                        initI   = int(N/100),
                        initD   = 0, 
                        initR   = 0, 
                        initF   = 0, 
                        tmax    = TMAX)


# Run the simulation:
print "Running simulation..."
model.run() 


ax[0,1].fill_between(notesting_tseries[::(int(N)/100)], notesting_numI[::(int(N)/100)], 0, color='#EFEFEF')
ax[0,1].plot(notesting_tseries[::(int(N)/100)], notesting_numI[::(int(N)/100)], color='#E0E0E0', label='I (no testing)')
# ax[0,1].plot(model.tseries[::(int(N)/100)], model.numS[::(int(N)/100)], color='blue', label='S')
# ax[0,1].plot(model.tseries[::(int(N)/100)], model.numR[::(int(N)/100)], color='green', label='R')
ax[0,1].plot(model.tseries[::(int(N)/100)], model.numE[::(int(N)/100)], color='orange', label='E')
ax[0,1].plot(model.tseries[::(int(N)/100)], model.numI[::(int(N)/100)], color='crimson', alpha=0.5, linestyle=':', label='I')
ax[0,1].plot(model.tseries[::(int(N)/100)], model.numD[::(int(N)/100)], color='crimson', alpha=0.5, linestyle='--', label='D')
ax[0,1].plot(model.tseries[::(int(N)/100)], model.numI[::(int(N)/100)]+model.numD[::(int(N)/100)], color='crimson', label='I+D')
ax[0,1].plot(model.tseries[::(int(N)/100)], model.numF[::(int(N)/100)], color='black', label='F')

ax[0,1].set_ylim(0, N*0.15)
ax[0,1].set_xlim(0, TMAX)
ax[0,1].set_ylabel('number of individuals')
ax[0,1].set_xlabel('days')

ax[0,1].legend(bbox_to_anchor=(1.0, 0.95))
ax[0,1].set_title("Random Testing & Contact Tracing (N="+str(N)+")")

# pyplot.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RANDOM TESTING & QUARANTINE
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print "Initializing simulation..."
model = SEIRSGraphModel(G, 
                        beta    = 0.165, 
                        sigma   = 1/5.2, 
                        gamma   = 1/12.39, 
                        xi      = 0.0, 
                        mu_I    = 0.0004, 
                        mu_0    = 0.0,   
                        nu      = 0.0, 
                        p       = 0.1,
                        Q       = Q, 
                        beta_D  = 0.165, 
                        theta_E = 0.02, 
                        theta_I = 0.02, 
                        phi_E   = 0.0, 
                        phi_I   = 0.0, 
                        psi_E   = 1.0, 
                        psi_I   = 1.0, 
                        q       = 0.5,
                        initE   = 0, 
                        initI   = int(N/100),
                        initD   = 0, 
                        initR   = 0, 
                        initF   = 0, 
                        tmax    = TMAX)


# Run the simulation:
print "Running simulation..."
model.run() 


ax[1,0].fill_between(notesting_tseries[::(int(N)/100)], notesting_numI[::(int(N)/100)], 0, color='#EFEFEF')
ax[1,0].plot(notesting_tseries[::(int(N)/100)], notesting_numI[::(int(N)/100)], color='#E0E0E0', label='I (no testing)')
# ax[1,0].plot(model.tseries[::(int(N)/100)], model.numS[::(int(N)/100)], color='blue', label='S')
# ax[1,0].plot(model.tseries[::(int(N)/100)], model.numR[::(int(N)/100)], color='green', label='R')
ax[1,0].plot(model.tseries[::(int(N)/100)], model.numE[::(int(N)/100)], color='orange', label='E')
ax[1,0].plot(model.tseries[::(int(N)/100)], model.numI[::(int(N)/100)], color='crimson', alpha=0.5, linestyle=':', label='I')
ax[1,0].plot(model.tseries[::(int(N)/100)], model.numD[::(int(N)/100)], color='crimson', alpha=0.5, linestyle='--', label='D')
ax[1,0].plot(model.tseries[::(int(N)/100)], model.numI[::(int(N)/100)]+model.numD[::(int(N)/100)], color='crimson', label='I+D')
ax[1,0].plot(model.tseries[::(int(N)/100)], model.numF[::(int(N)/100)], color='black', label='F')

ax[1,0].set_ylim(0, N*0.15)
ax[1,0].set_xlim(0, TMAX)
ax[1,0].set_ylabel('number of individuals')
ax[1,0].set_xlabel('days')

ax[1,0].legend(bbox_to_anchor=(1.0, 0.95))
ax[1,0].set_title("Random Testing & Quarantine (N="+str(N)+")")

# pyplot.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RANDOM TESTING, CONTACT TRACING, & QUARANTINE
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print "Initializing simulation..."
model = SEIRSGraphModel(G, 
                        beta    = 0.165, 
                        sigma   = 1/5.2, 
                        gamma   = 1/12.39, 
                        xi      = 0.0, 
                        mu_I    = 0.0004, 
                        mu_0    = 0.0,   
                        nu      = 0.0, 
                        p       = 0.1,
                        Q       = Q, 
                        beta_D  = 0.165, 
                        theta_E = 0.02, 
                        theta_I = 0.02, 
                        phi_E   = 0.1, 
                        phi_I   = 0.1, 
                        psi_E   = 1.0, 
                        psi_I   = 1.0, 
                        q       = 0.5,
                        initE   = 0, 
                        initI   = int(N/100),
                        initD   = 0, 
                        initR   = 0, 
                        initF   = 0, 
                        tmax    = TMAX)


# Run the simulation:
print "Running simulation..."
model.run() 


ax[1,1].fill_between(notesting_tseries[::(int(N)/100)], notesting_numI[::(int(N)/100)], 0, color='#EFEFEF')
ax[1,1].plot(notesting_tseries[::(int(N)/100)], notesting_numI[::(int(N)/100)], color='#E0E0E0', label='I (no testing)')
# ax[1,1].plot(model.tseries[::(int(N)/100)], model.numS[::(int(N)/100)], color='blue', label='S')
# ax[1,1].plot(model.tseries[::(int(N)/100)], model.numR[::(int(N)/100)], color='green', label='R')
ax[1,1].plot(model.tseries[::(int(N)/100)], model.numE[::(int(N)/100)], color='orange', label='E')
ax[1,1].plot(model.tseries[::(int(N)/100)], model.numI[::(int(N)/100)], color='crimson', alpha=0.5, linestyle=':', label='I')
ax[1,1].plot(model.tseries[::(int(N)/100)], model.numD[::(int(N)/100)], color='crimson', alpha=0.5, linestyle='--', label='D')
ax[1,1].plot(model.tseries[::(int(N)/100)], model.numI[::(int(N)/100)]+model.numD[::(int(N)/100)], color='crimson', label='I+D')
ax[1,1].plot(model.tseries[::(int(N)/100)], model.numF[::(int(N)/100)], color='black', label='F')

ax[1,1].set_ylim(0, N*0.15)
ax[1,1].set_xlim(0, TMAX)
ax[1,1].set_ylabel('number of individuals')
ax[1,1].set_xlabel('days')

ax[1,1].legend(bbox_to_anchor=(1.0, 0.95))
ax[1,1].set_title("Random Testing, Contact Tracing, & Quarantine (N="+str(N)+")")

# pyplot.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


pyplot.tight_layout()

# pyplot.show()

# exit()

pyplot.savefig('figs/try_SEIR_withTesting.png', dpi=200)
