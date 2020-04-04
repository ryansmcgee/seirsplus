from seirsplus.models import *
import random
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

results_I = dict()
results_F = dict()
results_Ts = dict()

PhiS = list(np.linspace(0,1,11))

for sims in range(100):
	print("++++++++++++++++++++++ SIM: "+str(sims)+"++++++++++++++++++++++")
	# no contact tracing for phi = 0
	for phi in PhiS:
		# no lag
		model_ct = SEIRSNetworkModel(G=G_normal,beta=0.155, sigma=1/5.2,gamma=1/16.39,mu_I=0.01,p=0.5,
                          Q=G_quarantine,beta_D=0.155, sigma_D=1/5.2, gamma_D=1/12.39, mu_D=0.01,
                          theta_E=0,theta_I=0.02,psi_E=1.0, psi_I=1.0, q=0.5, phi_E = phi, phi_I=phi,initI = 10)
		checkpoints = {'t': [20, 100], 'G': [G_distancing, G_normal], 'p': [0.1, 0.5]}
		model_ct.run(T=300, checkpoints=checkpoints)
		results_I['contact-tracing-noLag-phi:'+str(phi)+';'+str(sims)] = model_ct.numI
		results_F['contact-tracing-noLag-phi:'+str(phi)+';'+str(sims)] = model_ct.numF
		results_Ts['contact-tracing-noLag-phi:'+str(phi)+';'+str(sims)] = model_ct.tseries
		#lag
		model_ct = SEIRSNetworkModel(G=G_normal,beta=0.155, sigma=1/5.2,gamma=1/16.39,mu_I=0.01,p=0.5,
                          Q=G_quarantine,beta_D=0.155, sigma_D=1/5.2, gamma_D=1/12.39, mu_D=0.01,
                          theta_E=0,theta_I=0.02,psi_E=1.0, psi_I=1.0, q=0.5, phi_E = 0, phi_I=0,initI = 10)
		checkpoints = {'t': [20, 100], 'G': [G_distancing, G_normal], 'p': [0.1, 0.5], 'phi_E': [phi, phi], 'phi_I': [phi, phi]}
		model_ct.run(T=300, checkpoints=checkpoints)
		results_I['contact-tracing-Lag-phi:'+str(phi)+';'+str(sims)] = model_ct.numI
		results_F['contact-tracing-Lag-phi:'+str(phi)+';'+str(sims)] = model_ct.numF
		results_Ts['contact-tracing-Lag-phi:'+str(phi)+';'+str(sims)] = model_ct.tseries


deaths_df = pd.DataFrame(list(zip(PhiS,
            [np.mean([results_F[k][-1] for k in results_F.keys() if k.startswith('contact-tracing-Lag-phi:'+str(m))]) for m in PhiS],
            [np.mean([results_F[k][-1] for k in results_F.keys() if k.startswith('contact-tracing-noLag-phi:'+str(m))]) for m in PhiS],
            [np.std([results_F[k][-1] for k in results_F.keys() if k.startswith('contact-tracing-Lag-phi:'+str(m))]) for m in PhiS],
            [np.std([results_F[k][-1] for k in results_F.keys() if k.startswith('contact-tracing-noLag-phi:'+str(m))]) for m in PhiS])),
            columns=['phi','Lag','NoLag'],index=PhiS)


fig, ax = plt.subplots()
deaths_df[['Lag','NoLag']].plot() #.axhline(y=results_F['no-contact-tracing'][-1],color='r')
plt.xlabel("phi")
plt.ylabel("# of deaths")
plt.annotate("No contact tracing",(0.45,results_F['no-contact-tracing'][-1]+10))

# plot infection time-series

plt.plot(results_Ts['contact-tracing-Lag-phi:0.0;'+str(i)],results_I['contact-tracing-Lag-phi:0.0;'+str(i)]/numNodes,marker='',color='red', linewidth=2, linestyle='dashed', label="No Contact Tracing")
for i in range(1,100):
	plt.plot(results_Ts['contact-tracing-Lag-phi:0.0;'+str(i)],results_I['contact-tracing-Lag-phi:0.0;'+str(i)]/numNodes,marker='',color='red', linewidth=2, linestyle='dashed')

plt.plot(results_Ts['contact-tracing-Lag-phi:'+str(PhiS[0])+";0"], results_I['contact-tracing-Lag-phi:'+str(PhiS[0])+";0"]/numNodes,marker='', color='blue', linewidth=2, linestyle='dashed', label = "Contact Tracing (Lag)")
for i in range(100):
	for phi in PhiS:
		plt.plot(results_Ts['contact-tracing-Lag-phi:'+str(phi)+";"+str(i)], results_I['contact-tracing-Lag-phi:'+str(phi)+";"+str(i)]/numNodes,marker='', color='blue', linewidth=2, linestyle='dashed')

plt.plot(results_Ts['contact-tracing-noLag-phi:'+str(PhiS[0])+";0"], results_I['contact-tracing-noLag-phi:'+str(PhiS[0])+";0"]/numNodes,marker='', color='blue', linewidth=2, linestyle='dashed', label = "Contact Tracing (No Lag)") 
for i in range(100): 
	for phi in PhiS:
		plt.plot(results_Ts['contact-tracing-noLag-phi:'+str(phi)+";"+str(i)], results_I['contact-tracing-noLag-phi:'+str(phi)+";"+str(i)]/numNodes,marker='', color='blue', linewidth=2, linestyle='dashed') 

plt.legend()
plt.xlabel("Days")
plt.ylabel("Fraction of infected population")


###### Explore the fraction of people using contact tracing

pop_fraction = list(np.linspace(0,1,11))

for sims in range(100):
	print("++++++++++++++++++++++ SIM: "+str(sims)+"++++++++++++++++++++++")
	for pop in pop_fraction:
		phi = [1 if random.random() < pop else 0 for _ in range(10000)]
		model_ct = SEIRSNetworkModel(G=G_normal,beta=0.155, sigma=1/5.2,gamma=1/16.39,mu_I=0.01,p=0.5,
                          Q=G_quarantine,beta_D=0.155, sigma_D=1/5.2, gamma_D=1/12.39, mu_D=0.01,
                          theta_E=0,theta_I=0.02,psi_E=1.0, psi_I=1.0, q=0.5, phi_E = phi, phi_I=phi,initI = 10)
		checkpoints = {'t': [20, 100], 'G': [G_distancing, G_normal], 'p': [0.1, 0.5]}
		model_ct.run(T=300, checkpoints=checkpoints)
		results_I['contact-tracing-fraction:'+str(pop)+';'+str(sims)] = model_ct.numI
		results_F['contact-tracing-fraction:'+str(pop)+';'+str(sims)] = model_ct.numF
		results_Ts['contact-tracing-fraction:'+str(pop)+';'+str(sims)] = model_ct.tseries


deaths_df = pd.DataFrame(list(zip(pop_fraction,
            [np.mean([results_F[k][-1] for k in results_F.keys() if k.startswith('contact-tracing-fraction:'+str(m))]) for m in pop_fraction])),
            columns=['Population','Deaths'],index=pop_fraction)


fig, ax = plt.subplots()
deaths_df[['Deaths']].plot() #.axhline(y=results_F['no-contact-tracing'][-1],color='r')
plt.xlabel("Fraction of Tracking Population")
plt.ylabel("# of deaths")

# plot infections

fig, axs = plt.subplots(2,5)

for p in range(len(pop_fraction)-1):
	if p < 5:
		for i in range(100):
			axs[0,p].plot(results_Ts['contact-tracing-fraction:'+str(pop_fraction[p])+";"+str(i)],results_I['contact-tracing-fraction:'+str(pop_fraction[p])+";"+str(i)]/numNodes,marker='',color='red', linewidth=2, linestyle='dashed')
			axs[0,p].set(xlim=(0,300), ylim=(0, 0.1))
			if i == 0:
				axs[0,p].annotate("Tracing %: "+str(100*round(pop_fraction[p],1))+"%",(50,0.09))
	else:
		for i in range(100):
			axs[1,p-5].plot(results_Ts['contact-tracing-fraction:'+str(pop_fraction[p])+";"+str(i)],results_I['contact-tracing-fraction:'+str(pop_fraction[p])+";"+str(i)]/numNodes,marker='',color='red', linewidth=2, linestyle='dashed')
			axs[1,p-5].set(xlim=(0,300), ylim=(0, 0.1))
			if i == 0:
				axs[1,p-5].annotate("Tracing %: "+str(100*round(pop_fraction[p],1))+"%",(50,0.09))
