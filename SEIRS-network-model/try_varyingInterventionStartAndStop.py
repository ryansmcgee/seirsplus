from models import SEIRSGraphModel # Import the SEIRS graph model.
from models import custom_exponential_graph
import numpy
import networkx
import matplotlib.pyplot as pyplot
import seaborn


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define global params:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
N             = int(5e4)
T_MAX         = 300
T_START_IMMED = 1
T_START_EARLY = 20
T_START_LATE  = 50
T_START_NEVER = T_MAX
T_STOP_NEVER  = T_MAX
T_STOP_LATE   = 200
T_STOP_EARLY  = 150

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generate the interaction graphs to be used:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print "Generating graphs..."
baseGraph   = networkx.barabasi_albert_graph(n=N, m=9)
# Baseline normal social interactions:
G_NORMAL    = custom_exponential_graph(baseGraph, scale=100)
# Social distancing interactions:
G_DIST   = custom_exponential_graph(baseGraph, scale=10)
# Quarantining interactions:
G_QRNTINE   = custom_exponential_graph(baseGraph, scale=5)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define a function that will initialize and run
# the simulation for this analysis.
# Parameters that are shared between treatments have 
# their values defined as default arg values.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def run_simulation(T_START, T_STOP,
                        G_NORMAL    = G_NORMAL,
                        G_DIST      = G_DIST,
                        G_QRNTINE   = G_QRNTINE,
                        initI       = int(N/100),
                        initE       = 0,
                        beta        = 0.155,
                        sigma       = 1/5.2,
                        gamma       = 1/12.39,
                        mu_I        = 0.0004,
                        beta_D      = 0.155,
                        psi_E       = 1,
                        psi_I       = 1,
                        P_NORMAL    = 0.5,
                        P_DIST      = 0.1,
                        THETA_E     = 0.02,
                        THETA_I     = 0.02,
                        PHI_E       = 0.2,
                        PHI_I       = 0.2,
                        q_NORMAL    = 1,
                        q_QRNTINE   = 0.5):

      print "Initializing simulation..."
      model = SEIRSGraphModel(G=G_NORMAL, Q=G_QRNTINE, p=P_NORMAL, q=q_QRNTINE,
                              beta=beta, sigma=sigma, gamma=gamma, mu_I=mu_I,  
                              initI=initI, initE=initE, initD_E=0, initD_I=0, initR=0, initF=0, 
                              beta_D  = beta_D, theta_E=0, theta_I=0, phi_E=0, phi_I=0, psi_E=psi_E, psi_I=psi_I)


      model.run(T=300, checkpoints={'t':[T_START, T_STOP], 
                                  'G':[G_DIST, G_NORMAL], 
                                  'p':[P_DIST, P_NORMAL],
                                  'theta_E':[THETA_E, 0],
                                  'theta_I':[THETA_I, 0],
                                  'phi_E':[PHI_E, 0],
                                  'phi_I':[PHI_I, 0]})

      # Return the model object that holds results:
      return model

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Setup figure for later:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig, ax = pyplot.subplots(3, 3, figsize=(16,9), # sharex='all', sharey='all',
                              gridspec_kw={'left':0.06, 'right':0.98, 'top':0.96, 'bottom':0.06})
seaborn.set_style('ticks')
seaborn.despine()



#############################################################
#############################################################
# NEVER Start Intervention
#############################################################
#############################################################

T_START = T_START_NEVER

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# NEVER Stop Intervention
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_STOP = T_STOP_NEVER

results_NEVERSTART = run_simulation(T_START=T_START, T_STOP=T_STOP)



#############################################################
#############################################################
# IMMEDIATE Start Intervention
#############################################################
#############################################################

T_START = T_START_IMMED

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# NEVER Stop Intervention
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_STOP = T_STOP_NEVER

results_NEVERSTOP = run_simulation(T_START=T_START, T_STOP=T_STOP)

results_NEVERSTOP.plot(ax=ax[0,0],  
              dashed_reference_results=results_NEVERSTART, dashed_reference_label='do nothing', 
              shaded_reference_results=None, shaded_reference_label=None, 
              vlines=[T_START], vline_colors=['green'], vline_labels=['start SD+TTQ'],
              legend=True, title="Never Stop Intervention (SD+TTQ)", side_title="Immediate Start Intervention", ylim=N*0.16)
pyplot.savefig('figs/try_varyingInterventionStartAndStop_N'+str(N)+'.png', dpi=200)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# LATE Stop Intervention
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_STOP = T_STOP_LATE

results = run_simulation(T_START=T_START, T_STOP=T_STOP)

results.plot(ax=ax[0,1],  
              dashed_reference_results=results_NEVERSTART, dashed_reference_label='do nothing', 
              shaded_reference_results=results_NEVERSTOP, shaded_reference_label="never stop", 
              vlines=[T_START, T_STOP], vline_colors=['tab:green', 'tab:red'], vline_labels=['start SD+TTQ', 'stop SD+TTQ'],
              legend=True, title="Late Stop Intervention (SD+TTQ)", side_title=None, ylim=N*0.16)
pyplot.savefig('figs/try_varyingInterventionStartAndStop_N'+str(N)+'.png', dpi=200)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# EARLY Stop Intervention
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_STOP = T_STOP_EARLY

results = run_simulation(T_START=T_START, T_STOP=T_STOP)

results.plot(ax=ax[0,2],  
              dashed_reference_results=results_NEVERSTART, dashed_reference_label='do nothing', 
              shaded_reference_results=results_NEVERSTOP, shaded_reference_label="never stop", 
              vlines=[T_START, T_STOP], vline_colors=['tab:green', 'tab:red'], vline_labels=['start SD+TTQ', 'stop SD+TTQ'],
              legend=True, title="Early Stop Intervention (SD+TTQ)", side_title=None, ylim=N*0.16)
pyplot.savefig('figs/try_varyingInterventionStartAndStop_N'+str(N)+'.png', dpi=200)




#############################################################
#############################################################
# EARLY Start Intervention
#############################################################
#############################################################

T_START = T_START_EARLY

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# NEVER Stop Intervention
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_STOP = T_STOP_NEVER

results_NEVERSTOP = run_simulation(T_START=T_START, T_STOP=T_STOP)

results_NEVERSTOP.plot(ax=ax[1,0],  
              dashed_reference_results=results_NEVERSTART, dashed_reference_label='do nothing', 
              shaded_reference_results=None, shaded_reference_label=None, 
              vlines=[T_START], vline_colors=['green'], vline_labels=['start SD+TTQ'],
              legend=True, title=None, side_title="Early Start Intervention", ylim=N*0.16)
pyplot.savefig('figs/try_varyingInterventionStartAndStop_N'+str(N)+'.png', dpi=200)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# LATE Stop Intervention
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_STOP = T_STOP_LATE

results = run_simulation(T_START=T_START, T_STOP=T_STOP)

results.plot(ax=ax[1,1],  
              dashed_reference_results=results_NEVERSTART, dashed_reference_label='do nothing', 
              shaded_reference_results=results_NEVERSTOP, shaded_reference_label="never stop", 
              vlines=[T_START, T_STOP], vline_colors=['tab:green', 'tab:red'], vline_labels=['start SD+TTQ', 'stop SD+TTQ'],
              legend=True, title=None, side_title=None, ylim=N*0.16)
pyplot.savefig('figs/try_varyingInterventionStartAndStop_N'+str(N)+'.png', dpi=200)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# EARLY Stop Intervention
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_STOP = T_STOP_EARLY

results = run_simulation(T_START=T_START, T_STOP=T_STOP)

results.plot(ax=ax[1,2],  
              dashed_reference_results=results_NEVERSTART, dashed_reference_label='do nothing', 
              shaded_reference_results=results_NEVERSTOP, shaded_reference_label="never stop", 
              vlines=[T_START, T_STOP], vline_colors=['tab:green', 'tab:red'], vline_labels=['start SD+TTQ', 'stop SD+TTQ'],
              legend=True, title=None, side_title=None, ylim=N*0.16)
pyplot.savefig('figs/try_varyingInterventionStartAndStop_N'+str(N)+'.png', dpi=200)




#############################################################
#############################################################
# LATE Start Intervention
#############################################################
#############################################################

T_START = T_START_LATE

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# NEVER Stop Intervention
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_STOP = T_STOP_NEVER

results_NEVERSTOP = run_simulation(T_START=T_START, T_STOP=T_STOP)

results_NEVERSTOP.plot(ax=ax[2,0],  
              dashed_reference_results=results_NEVERSTART, dashed_reference_label='do nothing', 
              shaded_reference_results=None, shaded_reference_label=None, 
              vlines=[T_START], vline_colors=['green'], vline_labels=['start SD+TTQ'],
              legend=True, title=None, side_title="Late Start Intervention", ylim=N*0.16)
pyplot.savefig('figs/try_varyingInterventionStartAndStop_N'+str(N)+'.png', dpi=200)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# LATE Stop Intervention
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_STOP = T_STOP_LATE

results = run_simulation(T_START=T_START, T_STOP=T_STOP)

results.plot(ax=ax[2,1],  
              dashed_reference_results=results_NEVERSTART, dashed_reference_label='do nothing', 
              shaded_reference_results=results_NEVERSTOP, shaded_reference_label="never stop", 
              vlines=[T_START, T_STOP], vline_colors=['tab:green', 'tab:red'], vline_labels=['start SD+TTQ', 'stop SD+TTQ'],
              legend=True, title=None, side_title=None, ylim=N*0.16)
pyplot.savefig('figs/try_varyingInterventionStartAndStop_N'+str(N)+'.png', dpi=200)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# EARLY Stop Intervention
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_STOP = T_STOP_EARLY

results = run_simulation(T_START=T_START, T_STOP=T_STOP)

results.plot(ax=ax[2,2],  
              dashed_reference_results=results_NEVERSTART, dashed_reference_label='do nothing', 
              shaded_reference_results=results_NEVERSTOP, shaded_reference_label="never stop", 
              vlines=[T_START, T_STOP], vline_colors=['tab:green', 'tab:red'], vline_labels=['start SD+TTQ', 'stop SD+TTQ'],
              legend=True, title=None, side_title=None, ylim=N*0.16)
pyplot.savefig('figs/try_varyingInterventionStartAndStop_N'+str(N)+'.png', dpi=200)




