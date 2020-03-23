from models import SEIRSGraphModel # Import the SEIRS graph model.
from models import custom_exponential_graph
import numpy
import networkx
import matplotlib.pyplot as pyplot
import seaborn


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define global params:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
N       = int(5e4)
T_MAX    = 300

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
G_QRNTINE   = custom_exponential_graph(baseGraph, scale=1)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define rate values to be used throughout (when non-zero):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Params with same val for all treatments:
initI       = int(N/100)
beta        = 0.155
sigma       = 1/5.2
gamma       = 1/12.39
mu_I        = 0.0004
beta_D      = 0.155
psi_E       = 1
psi_I       = 1
# Params that change vals between treatments:
P_NODIST    = 0.5
P_MEDDIST   = 0.25
P_BIGDIST   = 0.1
THETA_E     = 0.02
THETA_I     = 0.02
PHI_E       = 0.2
PHI_I       = 0.2
q_NORMAL    = 1
q_QRNTINE   = 0.5


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Setup figure for later:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# fig, ax = pyplot.subplots(3, 3, figsize=(16,9), # sharex='all', sharey='all',
#                               gridspec_kw={'left':0.05, 'right':0.97, 'top':0.97, 'bottom':0.07})
# seaborn.set_style('ticks')
# seaborn.despine()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Setup figure for later:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig, ax = pyplot.subplots(3, 3, figsize=(16,9), # sharex='all', sharey='all',
                              gridspec_kw={'left':0.06, 'right':0.98, 'top':0.96, 'bottom':0.06})
seaborn.set_style('ticks')
seaborn.despine()



#############################################################
#############################################################
# NO SOCIAL DISTANCING
#############################################################
#############################################################

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# NO TESTING 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print "Initializing simulation..."
model_NOTEST_NODIST = SEIRSGraphModel(G=G_NORMAL, 
                        beta=beta, sigma=sigma, gamma=gamma, mu_I=mu_I,  
                        initI=initI, initE=0, initD_E=0, initD_I=0, initR=0, initF=0, 
                        p       = P_NODIST, 
                        beta_D  = beta_D, 
                        theta_E = 0.0, 
                        theta_I = 0.0, 
                        phi_E   = 0.0, 
                        phi_I   = 0.0, 
                        psi_E   = psi_E, 
                        psi_I   = psi_I, 
                        Q       = G_NORMAL,
                        q       = q_NORMAL)

# Run the simulation:
print "Running simulation..."
model_NOTEST_NODIST.run(T=T_MAX) 

model_NOTEST_NODIST.plot(ax=ax[0,0],  
              dashed_reference_results=model_NOTEST_NODIST, dashed_reference_label='do nothing', 
              shaded_reference_results=None, shaded_reference_label=None,
              legend=True, title="No Testing", side_title="No Social Distancing", ylim=N*0.16)
pyplot.savefig('figs/try_varyingDistancingAndTesting_N'+str(N)+'.png', dpi=200)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RANDOM TESTING & QUARANTINE
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print "Initializing simulation..."
model = SEIRSGraphModel(G=G_NORMAL, 
                        beta=beta, sigma=sigma, gamma=gamma, mu_I=mu_I,  
                        initI=initI, initE=0, initD_E=0, initD_I=0, initR=0, initF=0, 
                        p       = P_NODIST, 
                        beta_D  = beta_D, 
                        theta_E = THETA_E, 
                        theta_I = THETA_I, 
                        phi_E   = 0.0, 
                        phi_I   = 0.0, 
                        psi_E   = psi_E, 
                        psi_I   = psi_I, 
                        Q       = G_QRNTINE,
                        q       = q_QRNTINE)

# Run the simulation:
print "Running simulation..."
model.run(T=T_MAX) 

model.plot(ax=ax[0,1],  
              dashed_reference_results=model_NOTEST_NODIST, dashed_reference_label='do nothing', 
              shaded_reference_results=None, shaded_reference_label=None,
              legend=True, title="Testing & Quarantine", side_title=None, ylim=N*0.16)
pyplot.savefig('figs/try_varyingDistancingAndTesting_N'+str(N)+'.png', dpi=200)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RANDOM TESTING CONTACT TRACING & QUARANTINE
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print "Initializing simulation..."
model = SEIRSGraphModel(G=G_NORMAL, 
                        beta=beta, sigma=sigma, gamma=gamma, mu_I=mu_I,  
                        initI=initI, initE=0, initD_E=0, initD_I=0, initR=0, initF=0, 
                        p       = P_NODIST, 
                        beta_D  = beta_D, 
                        theta_E = THETA_E, 
                        theta_I = THETA_I, 
                        phi_E   = PHI_E, 
                        phi_I   = PHI_I, 
                        psi_E   = psi_E, 
                        psi_I   = psi_I, 
                        Q       = G_QRNTINE,
                        q       = q_QRNTINE)

# Run the simulation:
print "Running simulation..."
model.run(T=T_MAX) 

model.plot(ax=ax[0,2],  
              dashed_reference_results=model_NOTEST_NODIST, dashed_reference_label='do nothing', 
              shaded_reference_results=None, shaded_reference_label=None,
              legend=True, title="Testing, Tracing, & Quarantine", side_title=None, ylim=N*0.16)
pyplot.savefig('figs/try_varyingDistancingAndTesting_N'+str(N)+'.png', dpi=200)


#############################################################
#############################################################
# MODERATE SOCIAL DISTANCING
#############################################################
#############################################################

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# NO TESTING 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print "Initializing simulation..."
model_NOTEST = SEIRSGraphModel(G=G_MEDDIST, 
                        beta=beta, sigma=sigma, gamma=gamma, mu_I=mu_I,  
                        initI=initI, initE=0, initD_E=0, initD_I=0, initR=0, initF=0, 
                        p       = P_MEDDIST, 
                        beta_D  = beta_D, 
                        theta_E = 0.0, 
                        theta_I = 0.0, 
                        phi_E   = 0.0, 
                        phi_I   = 0.0, 
                        psi_E   = psi_E, 
                        psi_I   = psi_I, 
                        Q       = G_MEDDIST,
                        q       = q_NORMAL)

# Run the simulation:
print "Running simulation..."
model_NOTEST.run(T=T_MAX) 

model_NOTEST.plot(ax=ax[1,0],  
              dashed_reference_results=model_NOTEST_NODIST, dashed_reference_label='do nothing', 
              shaded_reference_results=None, shaded_reference_label=None,
              legend=True, title=None, side_title="Moderate Social Distancing", ylim=N*0.16)
pyplot.savefig('figs/try_varyingDistancingAndTesting_N'+str(N)+'.png', dpi=200)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RANDOM TESTING & QUARANTINE
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print "Initializing simulation..."
model = SEIRSGraphModel(G=G_MEDDIST, 
                        beta=beta, sigma=sigma, gamma=gamma, mu_I=mu_I,  
                        initI=initI, initE=0, initD_E=0, initD_I=0, initR=0, initF=0, 
                        p       = P_MEDDIST, 
                        beta_D  = beta_D, 
                        theta_E = THETA_E, 
                        theta_I = THETA_I, 
                        phi_E   = 0.0, 
                        phi_I   = 0.0, 
                        psi_E   = psi_E, 
                        psi_I   = psi_I, 
                        Q       = G_QRNTINE,
                        q       = q_QRNTINE)

# Run the simulation:
print "Running simulation..."
model.run(T=T_MAX) 

model.plot(ax=ax[1,1],  
              dashed_reference_results=model_NOTEST_NODIST, dashed_reference_label='do nothing', 
              shaded_reference_results=model_NOTEST, shaded_reference_label='soc dist only', 
              legend=True, title=None, side_title=None, ylim=N*0.16)
pyplot.savefig('figs/try_varyingDistancingAndTesting_N'+str(N)+'.png', dpi=200)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RANDOM TESTING CONTACT TRACING & QUARANTINE
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print "Initializing simulation..."
model = SEIRSGraphModel(G=G_MEDDIST, 
                        beta=beta, sigma=sigma, gamma=gamma, mu_I=mu_I,  
                        initI=initI, initE=0, initD_E=0, initD_I=0, initR=0, initF=0, 
                        p       = P_MEDDIST, 
                        beta_D  = beta_D, 
                        theta_E = THETA_E, 
                        theta_I = THETA_I, 
                        phi_E   = PHI_E, 
                        phi_I   = PHI_I, 
                        psi_E   = psi_E, 
                        psi_I   = psi_I, 
                        Q       = G_QRNTINE,
                        q       = q_QRNTINE)

# Run the simulation:
print "Running simulation..."
model.run(T=T_MAX) 

model.plot(ax=ax[1,2],  
              dashed_reference_results=model_NOTEST_NODIST, dashed_reference_label='do nothing', 
              shaded_reference_results=model_NOTEST, shaded_reference_label='soc dist only', 
              legend=True, title=None, side_title=None, ylim=N*0.16)
pyplot.savefig('figs/try_varyingDistancingAndTesting_N'+str(N)+'.png', dpi=200)

#############################################################
#############################################################
# MAJOR SOCIAL DISTANCING
#############################################################
#############################################################

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# NO TESTING 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print "Initializing simulation..."
model_NOTEST = SEIRSGraphModel(G=G_BIGDIST, 
                        beta=beta, sigma=sigma, gamma=gamma, mu_I=mu_I,  
                        initI=initI, initE=0, initD_E=0, initD_I=0, initR=0, initF=0, 
                        p       = P_BIGDIST, 
                        beta_D  = beta_D, 
                        theta_E = 0.0, 
                        theta_I = 0.0, 
                        phi_E   = 0.0, 
                        phi_I   = 0.0, 
                        psi_E   = psi_E, 
                        psi_I   = psi_I, 
                        Q       = G_BIGDIST,
                        q       = q_NORMAL)

# Run the simulation:
print "Running simulation..."
model_NOTEST.run(T=T_MAX) 

model_NOTEST.plot(ax=ax[2,0],  
              dashed_reference_results=model_NOTEST_NODIST, dashed_reference_label='do nothing', 
              shaded_reference_results=None, shaded_reference_label=None,
              legend=True, title=None, side_title="Major Social Distancing", ylim=N*0.16)
pyplot.savefig('figs/try_varyingDistancingAndTesting_N'+str(N)+'.png', dpi=200)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RANDOM TESTING & QUARANTINE
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print "Initializing simulation..."
model = SEIRSGraphModel(G=G_BIGDIST, 
                        beta=beta, sigma=sigma, gamma=gamma, mu_I=mu_I,  
                        initI=initI, initE=0, initD_E=0, initD_I=0, initR=0, initF=0, 
                        p       = P_BIGDIST, 
                        beta_D  = beta_D, 
                        theta_E = THETA_E, 
                        theta_I = THETA_I, 
                        phi_E   = 0.0, 
                        phi_I   = 0.0, 
                        psi_E   = psi_E, 
                        psi_I   = psi_I, 
                        Q       = G_QRNTINE,
                        q       = q_QRNTINE)

# Run the simulation:
print "Running simulation..."
model.run(T=T_MAX) 

model.plot(ax=ax[2,1],  
              dashed_reference_results=model_NOTEST_NODIST, dashed_reference_label='do nothing', 
              shaded_reference_results=model_NOTEST, shaded_reference_label='soc dist only', 
              legend=True, title=None, side_title=None, ylim=N*0.16)
pyplot.savefig('figs/try_varyingDistancingAndTesting_N'+str(N)+'.png', dpi=200)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RANDOM TESTING CONTACT TRACING & QUARANTINE
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print "Initializing simulation..."
model = SEIRSGraphModel(G=G_BIGDIST, 
                        beta=beta, sigma=sigma, gamma=gamma, mu_I=mu_I,  
                        initI=initI, initE=0, initD_E=0, initD_I=0, initR=0, initF=0, 
                        p       = P_BIGDIST, 
                        beta_D  = beta_D, 
                        theta_E = THETA_E, 
                        theta_I = THETA_I, 
                        phi_E   = PHI_E, 
                        phi_I   = PHI_I, 
                        psi_E   = psi_E, 
                        psi_I   = psi_I, 
                        Q       = G_QRNTINE,
                        q       = q_QRNTINE)

# Run the simulation:
print "Running simulation..."
model.run(T=T_MAX) 

model.plot(ax=ax[2,2],  
              dashed_reference_results=model_NOTEST_NODIST, dashed_reference_label='do nothing', 
              shaded_reference_results=model_NOTEST, shaded_reference_label='soc dist only', 
              legend=True, title=None, side_title=None, ylim=N*0.16)
pyplot.savefig('figs/try_varyingDistancingAndTesting_N'+str(N)+'.png', dpi=200)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# fig.tight_layout()

# pyplot.show()

# pyplot.savefig('figs/try_varyingDistancingAndTesting.png', dpi=200)




