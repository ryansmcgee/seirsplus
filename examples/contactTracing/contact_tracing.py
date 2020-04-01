from seirsplus.models import *
import networkx
import numpy as np

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

model.figure_infections(title="No contact tracing")

## add contact tracing

results_I = dict()
results_F = dict()

results_I['no-contact-tracing'] = model.numI
results_F['no-contact-tracing'] = model.numF

#results.append((0,model.numF[-1],max(model.numI),model.numS[-1]))

# contact tracing 

for phi in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
  
  #no lag
  model_ct = SEIRSNetworkModel(G=G_normal,beta=0.155, sigma=1/5.2,gamma=1/16.39,mu_I=0.01,p=0.5,
                          Q=G_quarantine,beta_D=0.155, sigma_D=1/5.2, gamma_D=1/12.39, mu_D=0.01,
                          theta_E=0,theta_I=0.02,psi_E=1.0, psi_I=1.0, q=0.5, phi_E = phi, phi_I=phi,initI = 10)

  checkpoints = {'t': [20, 100], 'G': [G_distancing, G_normal], 'p': [0.1, 0.5]}

  model_ct.run(T=300, checkpoints=checkpoints)
  results_I['contact-tracing-noLag-phi:'+str(phi)] = model_ct.numI
  results_F['contact-tracing-noLag-phi:'+str(phi)] = model_ct.numF
  
  #lag
  model_ct = SEIRSNetworkModel(G=G_normal,beta=0.155, sigma=1/5.2,gamma=1/16.39,mu_I=0.01,p=0.5,
                          Q=G_quarantine,beta_D=0.155, sigma_D=1/5.2, gamma_D=1/12.39, mu_D=0.01,
                          theta_E=0,theta_I=0.02,psi_E=1.0, psi_I=1.0, q=0.5, phi_E = 0, phi_I=0,initI = 10)
  
  checkpoints = {'t': [20, 100], 'G': [G_distancing, G_normal], 'p': [0.1, 0.5], 'phi_E': [phi, phi], 'phi_I': [phi, phi]}
  model_ct.run(T=300, checkpoints=checkpoints)
  results_I['contact-tracing-Lag-phi:'+str(phi)] = model_ct.numI
  results_F['contact-tracing-Lag-phi:'+str(phi)] = model_ct.numF
  #results.append((phi,model_ct.numF[-1],max(model_ct.numI),model_ct.numS[-1]))
  #model_ct.figure_infections(title="Contact tracing")
  
deaths_df = pd.DataFrame(list(zip([0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            [results_F['contact-tracing-Lag-phi:'+str(i)][-1] for i in [0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]],
            [results_F['contact-tracing-noLag-phi:'+str(i)][-1] for i in [0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]])),
            columns=['phi','Lag','NoLag'])


fig, ax = plt.subplots()
deaths_df[['Lag','NoLag']].plot().axhline(y=results_F['no-contact-tracing'][-1],color='r')
plt.xlabel("phi")
plt.ylabel("# of deaths")
plt.annotate("No contact tracing",(3.5,results_F['no-contact-tracing'][-1]+10))
