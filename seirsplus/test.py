from models import *
import networkx

numNodes = 10000
baseGraph    = networkx.barabasi_albert_graph(n=numNodes, m=9)
G_normal     = custom_exponential_graph(baseGraph, scale=100)
# Social distancing interactions:
G_distancing = custom_exponential_graph(baseGraph, scale=10)
# Quarantine interactions:
G_quarantine = custom_exponential_graph(baseGraph, scale=5)



model1 = SEIRSNetworkModel(G       =G_normal, 
                          beta    =2.4/12.39, 
                          sigma   =1/4.58, 
                          gamma   =1/12.39, 
                          mu_I    =0.0004,
                          mu_0    =0, 
                          nu      =0, 
                          xi      =0,
                          p       =0.5,
                          Q       =G_quarantine, 
                          beta_D  =2.4/12.39, 
                          sigma_D =1/4.58, 
                          gamma_D =1/12.39, 
                          mu_D    =0.0004,
                          theta_E =0, 
                          theta_I =0, 
                          phi_E   =0, 
                          phi_I   =0, 
                          psi_E   =1.0, 
                          psi_I   =1.0,
                          q       =0.05,
                          initI   =numNodes/100, 
                          initE   =0, 
                          initD_E =0, 
                          initD_I =0, 
                          initR   =0, 
                          initF   =0)

model2 = SEIRSNetworkModel(G       =G_normal, 
                          beta    =2.4/12.39, 
                          sigma   =1/4.58, 
                          gamma   =1/12.39, 
                          mu_I    =0.0004,
                          mu_0    =0, 
                          nu      =0, 
                          xi      =0,
                          p       =0.5,
                          Q       =G_quarantine, 
                          beta_D  =2.4/12.39, 
                          sigma_D =1/4.58, 
                          gamma_D =1/12.39, 
                          mu_D    =0.0004,
                          theta_E =0, 
                          theta_I =0, 
                          phi_E   =0, 
                          phi_I   =0, 
                          psi_E   =1.0, 
                          psi_I   =1.0,
                          q       =0.05,
                          initI   =numNodes/100, 
                          initE   =0, 
                          initD_E =0, 
                          initD_I =0, 
                          initR   =0, 
                          initF   =0)

# checkpoints = {'t': [20, 100], 'G': [G_distancing, G_normal], 'p': [0.1, 0.5], 'theta_E': [0.02, 0.02], 'theta_I': [0.02, 0.02], 'phi_E':   [0.2, 0.2], 'phi_I':   [0.2, 0.2]}

checkpoints1 = {'t': [21, 42, 100], 'G': [G_distancing, G_distancing, G_normal], 'p': [0.1 , 0.1, 0.5 ], 'theta_E': [0.02, 0.1, 0.02], 'theta_I': [0.02, 0.1, 0.02], 'phi_E': [0.6 , 0.6, 0.6 ], 'phi_I': [0.6 , 0.6, 0.6]}

checkpoints2 = {'t': [21, 42, 100], 'G': [G_distancing, G_distancing, G_normal], 'p': [0.1 , 0.1, 0.5 ], 'theta_E': [0.02, 0.1, 0.02], 'theta_I': [0.02, 0.1, 0.02], 'phi_E': [0.6 , 0.85, 0.85 ], 'phi_I': [0.6 , 0.85, 0.85 ]}

model1.run(T=300, checkpoints=checkpoints1)

model2.run(T=300, checkpoints=checkpoints2)

model2.figure_infections(vlines=checkpoints2['t'], ylim=0.1, shaded_reference_results=model1)





