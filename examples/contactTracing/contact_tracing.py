from seirsplus.models import *
import networkx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

numNodes = 10000
baseGraph    = networkx.barabasi_albert_graph(n=numNodes, m=9)
G_normal     = custom_exponential_graph(baseGraph, scale=100)
# Social distancing interactions:
G_distancing = custom_exponential_graph(baseGraph, scale=10)
# Quarantine interactions:
G_quarantine = custom_exponential_graph(baseGraph, scale=5)


## first run with no contact-tracing

model = SEIRSNetworkModel(G=G_normal,beta=0.155, sigma=1/5.2,gamma=1/16.39,mu_I=0.01,p=0.5,
                          Q=G_quarantine,beta_D=0.155, sigma_D=1/5.2, gamma_D=1/12.39, mu_D=0.01,
                          theta_E=0,theta_I=0.02,psi_E=1.0, psi_I=1.0, q=0.5, initI = 10)

checkpoints = {'t': [20, 100], 'G': [G_distancing, G_normal], 'p': [0.1, 0.5]}

model.run(T=300, checkpoints=checkpoints)

## add contact tracing

results_I = dict()
results_F = dict()
results_Ts = dict()

results_I['no-contact-tracing'] = model.numI
results_F['no-contact-tracing'] = model.numF
results_Ts['no-contact-tracing'] = model.tseries

# contact tracing 

PhiS = list(np.linspace(0,1,11))

for phi in PhiS:
  
  #no lag
  model_ct = SEIRSNetworkModel(G=G_normal,beta=0.155, sigma=1/5.2,gamma=1/16.39,mu_I=0.01,p=0.5,
                          Q=G_quarantine,beta_D=0.155, sigma_D=1/5.2, gamma_D=1/12.39, mu_D=0.01,
                          theta_E=0,theta_I=0.02,psi_E=1.0, psi_I=1.0, q=0.5, phi_E = phi, phi_I=phi,initI = 10)

  checkpoints = {'t': [20, 100], 'G': [G_distancing, G_normal], 'p': [0.1, 0.5]}

  model_ct.run(T=300, checkpoints=checkpoints)
  results_I['contact-tracing-noLag-phi:'+str(phi)] = model_ct.numI
  results_F['contact-tracing-noLag-phi:'+str(phi)] = model_ct.numF
  results_Ts['contact-tracing-noLag-phi:'+str(phi)] = model_ct.tseries
  
  #lag
  model_ct = SEIRSNetworkModel(G=G_normal,beta=0.155, sigma=1/5.2,gamma=1/16.39,mu_I=0.01,p=0.5,
                          Q=G_quarantine,beta_D=0.155, sigma_D=1/5.2, gamma_D=1/12.39, mu_D=0.01,
                          theta_E=0,theta_I=0.02,psi_E=1.0, psi_I=1.0, q=0.5, phi_E = 0, phi_I=0,initI = 10)
  
  checkpoints = {'t': [20, 100], 'G': [G_distancing, G_normal], 'p': [0.1, 0.5], 'phi_E': [phi, phi], 'phi_I': [phi, phi]}
  model_ct.run(T=300, checkpoints=checkpoints)
  results_I['contact-tracing-Lag-phi:'+str(phi)] = model_ct.numI
  results_F['contact-tracing-Lag-phi:'+str(phi)] = model_ct.numF
  results_Ts['contact-tracing-Lag-phi:'+str(phi)] = model_ct.tseries
  
deaths_df = pd.DataFrame(list(zip(PhiS,
            [results_F['contact-tracing-Lag-phi:'+str(i)][-1] for i in PhiS],
            [results_F['contact-tracing-noLag-phi:'+str(i)][-1] for i in PhiS])),
            columns=['phi','Lag','NoLag'],index=PhiS)


fig, ax = plt.subplots()
deaths_df[['Lag','NoLag']].plot().axhline(y=results_F['no-contact-tracing'][-1],color='r')
plt.xlabel("phi")
plt.ylabel("# of deaths")
plt.annotate("No contact tracing",(0.45,results_F['no-contact-tracing'][-1]+10))

# Infections time-series


plt.plot(results_Ts['no-contact-tracing'],results_I['no-contact-tracing']/numNodes,marker='', 
         color='red', linewidth=2, linestyle='dashed', label="No Contact Tracing")
plt.plot(results_Ts['contact-tracing-noLag-phi:0.1'],results_I['contact-tracing-noLag-phi:0.1']/numNodes,marker='', 
         color='olive', linewidth=2, linestyle='dashed', label="Contact Tracing (no lag)")
plt.plot(results_Ts['contact-tracing-Lag-phi:0.1'],results_I['contact-tracing-Lag-phi:0.1']/numNodes,marker='', 
         color='blue', linewidth=2, linestyle='dashed', label="Contact Tracing (Lag)")

for phi in Phis[2:11]:
  plt.plot(results_Ts['contact-tracing-noLag-phi:'+str(phi)],results_I['contact-tracing-noLag-phi:'+str(phi)]/numNodes,marker='', 
         color='olive', linewidth=2, linestyle='dashed')
  plt.plot(results_Ts['contact-tracing-Lag-phi:'+str(phi)],results_I['contact-tracing-Lag-phi:'+str(phi)]/numNodes,marker='', 
         color='blue', linewidth=2, linestyle='dashed')
  
plt.legend()
plt.xlabel("Days")
plt.ylabel("Fraction of infected population")
