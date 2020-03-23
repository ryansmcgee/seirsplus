from models import SEIRSGraphModel # Import the SEIRS graph model.
from models import custom_exponential_graph
import numpy
import networkx
import matplotlib.pyplot as pyplot
import seaborn


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define global params:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
N        = int(5e4)
T_MAX    = 300
T_IMMED  = 1
T_EARLY  = 20
T_LATE   = 50

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
def run_simulation(T_TESTON, T_DISTON,
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
                        P_NODIST    = 0.5,
                        P_DIST      = 0.1,
                        THETA_E     = 0.02,
                        THETA_I     = 0.02,
                        PHI_E       = 0.2,
                        PHI_I       = 0.2,
                        q_NORMAL    = 1,
                        q_QRNTINE   = 0.5):

      print "Initializing simulation..."
      model = SEIRSGraphModel(G=G_NORMAL, Q=G_QRNTINE, p=P_NODIST, q=q_QRNTINE,
                              beta=beta, sigma=sigma, gamma=gamma, mu_I=mu_I,  
                              initI=initI, initE=initE, initD_E=0, initD_I=0, initR=0, initF=0, 
                              beta_D  = beta_D, theta_E=0, theta_I=0, phi_E=0, phi_I=0, psi_E=psi_E, psi_I=psi_I)

      if(T_TESTON is None):
            T_TESTON = T_MAX
      if(T_DISTON is None):
            T_DISTON = T_MAX

      print "Running simulation..."
      # Run the PRE-testing, PRE-distancing epoch of the sim:
      model.run(T=min(T_TESTON, T_DISTON))
      # Turn on testing and/or distancing if time to do so:
      if(T_TESTON==min(T_TESTON, T_DISTON)):
          model.theta_E = THETA_E
          model.theta_I = THETA_I
          model.phi_E   = PHI_E  
          model.phi_I   = PHI_I  
      if(T_DISTON==min(T_TESTON, T_DISTON)):
          model.update_G(G_DIST) 
          model.p = P_DIST
      # Run the middle, either testing OR distancing, epoch of the sim:
      model.run(T=max(T_TESTON, T_DISTON)-min(T_TESTON, T_DISTON))
      # Turn on testing and/or distancing if time to do so:
      if(T_TESTON==max(T_TESTON, T_DISTON)):
          model.theta_E = THETA_E
          model.theta_I = THETA_I
          model.phi_E   = PHI_E  
          model.phi_I   = PHI_I  
      if(T_DISTON==max(T_TESTON, T_DISTON)):
          model.update_G(G_DIST) 
          model.p = P_DIST
      # Run the testing AND distancing epoch of the sim:
      model.run(T=T_MAX-max(T_TESTON, T_DISTON))

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
# NEVER Social Distancing
#############################################################
#############################################################

T_DISTON = None

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# NEVER Testing
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_TESTON = None

results_NOTEST_NODIST = run_simulation(T_TESTON=T_TESTON, T_DISTON=T_DISTON)




#############################################################
#############################################################
# IMMEDIATE Social Distancing
#############################################################
#############################################################

T_DISTON = T_IMMED

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# NEVER Testing
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_TESTON = None

results_NOTEST = run_simulation(T_TESTON=T_TESTON, T_DISTON=T_DISTON)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# IMMEDIATE Testing, Tracing, & Quarantine
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_TESTON = T_IMMED

results = run_simulation(T_TESTON=T_TESTON, T_DISTON=T_DISTON)

results.plot(ax=ax[0,0],  
              dashed_reference_results=results_NOTEST_NODIST, dashed_reference_label='do nothing', 
              shaded_reference_results=results_NOTEST, shaded_reference_label='distancing only', 
              vlines=[T_DISTON, T_TESTON+1], vline_colors=['green', 'mediumorchid'], vline_labels=['start distancing', 'start testing'],
              legend=True, title="Immediate Test, Trace, & Quarantine", side_title="Immediate Social Distancing", ylim=N*0.16)
pyplot.savefig('figs/try_varyingInterventionOnset_N'+str(N)+'.png', dpi=200)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# EARLY Testing, Tracing, and Quarantine
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_TESTON = T_EARLY

results = run_simulation(T_TESTON=T_TESTON, T_DISTON=T_DISTON)

results.plot(ax=ax[0,1],  
              dashed_reference_results=results_NOTEST_NODIST, dashed_reference_label='do nothing', 
              shaded_reference_results=results_NOTEST, shaded_reference_label='distancing only', 
              vlines=[T_DISTON, T_TESTON+1], vline_colors=['green', 'mediumorchid'], vline_labels=['start distancing', 'start testing'],
              legend=True, title="Early Test, Trace, & Quarantine", side_title=None, ylim=N*0.16)
pyplot.savefig('figs/try_varyingInterventionOnset_N'+str(N)+'.png', dpi=200)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# LATE Testing, Tracing, and Quarantine
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_TESTON = T_LATE

results = run_simulation(T_TESTON=T_TESTON, T_DISTON=T_DISTON)

results.plot(ax=ax[0,2],  
              dashed_reference_results=results_NOTEST_NODIST, dashed_reference_label='do nothing', 
              shaded_reference_results=results_NOTEST, shaded_reference_label='distancing only', 
              vlines=[T_DISTON, T_TESTON+1], vline_colors=['green', 'mediumorchid'], vline_labels=['start distancing', 'start testing'],
              legend=True, title="Late Test, Trace, & Quarantine", side_title=None, ylim=N*0.16)
pyplot.savefig('figs/try_varyingInterventionOnset_N'+str(N)+'.png', dpi=200)




#############################################################
#############################################################
# EARLY Social Distancing
#############################################################
#############################################################

T_DISTON = T_EARLY

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# NEVER Testing
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_TESTON = None

results_NOTEST = run_simulation(T_TESTON=T_TESTON, T_DISTON=T_DISTON)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# IMMEDIATE Testing, Tracing, & Quarantine
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_TESTON = T_IMMED

results = run_simulation(T_TESTON=T_TESTON, T_DISTON=T_DISTON)

results.plot(ax=ax[1,0],  
              dashed_reference_results=results_NOTEST_NODIST, dashed_reference_label='do nothing', 
              shaded_reference_results=results_NOTEST, shaded_reference_label='distancing only', 
              vlines=[T_DISTON, T_TESTON+1], vline_colors=['green', 'mediumorchid'], vline_labels=['start distancing', 'start testing'],
              legend=True, title=None, side_title="Early Social Distancing", ylim=N*0.16)
pyplot.savefig('figs/try_varyingInterventionOnset_N'+str(N)+'.png', dpi=200)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# EARLY Testing, Tracing, and Quarantine
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_TESTON = T_EARLY

results = run_simulation(T_TESTON=T_TESTON, T_DISTON=T_DISTON)

results.plot(ax=ax[1,1],  
              dashed_reference_results=results_NOTEST_NODIST, dashed_reference_label='do nothing', 
              shaded_reference_results=results_NOTEST, shaded_reference_label='distancing only', 
              vlines=[T_DISTON, T_TESTON+1], vline_colors=['green', 'mediumorchid'], vline_labels=['start distancing', 'start testing'],
              legend=True, title=None, side_title=None, ylim=N*0.16)
pyplot.savefig('figs/try_varyingInterventionOnset_N'+str(N)+'.png', dpi=200)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# LATE Testing, Tracing, and Quarantine
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_TESTON = T_LATE

results = run_simulation(T_TESTON=T_TESTON, T_DISTON=T_DISTON)

results.plot(ax=ax[1,2],  
              dashed_reference_results=results_NOTEST_NODIST, dashed_reference_label='do nothing', 
              shaded_reference_results=results_NOTEST, shaded_reference_label='distancing only', 
              vlines=[T_DISTON, T_TESTON+1], vline_colors=['green', 'mediumorchid'], vline_labels=['start distancing', 'start testing'],
              legend=True, title=None, side_title=None, ylim=N*0.16)
pyplot.savefig('figs/try_varyingInterventionOnset_N'+str(N)+'.png', dpi=200)




#############################################################
#############################################################
# LATE Social Distancing
#############################################################
#############################################################

T_DISTON = T_LATE

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# NEVER Testing
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_TESTON = None

results_NOTEST = run_simulation(T_TESTON=T_TESTON, T_DISTON=T_DISTON)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# IMMEDIATE Testing, Tracing, & Quarantine
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_TESTON = T_IMMED

results = run_simulation(T_TESTON=T_TESTON, T_DISTON=T_DISTON)

results.plot(ax=ax[2,0],  
              dashed_reference_results=results_NOTEST_NODIST, dashed_reference_label='do nothing', 
              shaded_reference_results=results_NOTEST, shaded_reference_label='distancing only', 
              vlines=[T_DISTON, T_TESTON+1], vline_colors=['green', 'mediumorchid'], vline_labels=['start distancing', 'start testing'],
              legend=True, title=None, side_title="Late Social Distancing", ylim=N*0.16)
pyplot.savefig('figs/try_varyingInterventionOnset_N'+str(N)+'.png', dpi=200)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# EARLY Testing, Tracing, and Quarantine
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_TESTON = T_EARLY

results = run_simulation(T_TESTON=T_TESTON, T_DISTON=T_DISTON)

results.plot(ax=ax[2,1],  
              dashed_reference_results=results_NOTEST_NODIST, dashed_reference_label='do nothing', 
              shaded_reference_results=results_NOTEST, shaded_reference_label='distancing only', 
              vlines=[T_DISTON, T_TESTON+1], vline_colors=['green', 'mediumorchid'], vline_labels=['start distancing', 'start testing'],
              legend=True, title=None, side_title=None, ylim=N*0.16)
pyplot.savefig('figs/try_varyingInterventionOnset_N'+str(N)+'.png', dpi=200)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# LATE Testing, Tracing, and Quarantine
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_TESTON = T_LATE

results = run_simulation(T_TESTON=T_TESTON, T_DISTON=T_DISTON)

results.plot(ax=ax[2,2],  
              dashed_reference_results=results_NOTEST_NODIST, dashed_reference_label='do nothing', 
              shaded_reference_results=results_NOTEST, shaded_reference_label='distancing only', 
              vlines=[T_DISTON, T_TESTON+1], vline_colors=['green', 'mediumorchid'], vline_labels=['start distancing', 'start testing'],
              legend=True, title=None, side_title=None, ylim=N*0.16)
pyplot.savefig('figs/try_varyingInterventionOnset_N'+str(N)+'.png', dpi=200)




