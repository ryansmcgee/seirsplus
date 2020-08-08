from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as networkx
import numpy as numpy
import scipy as scipy
import scipy.integrate


########################################################
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#@                                                    @#
#@  BASIC SEIRS MODELS                                @#
#@                                                    @#
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
########################################################

class SEIRSModel():
    """
    A class to simulate the Deterministic SEIRS Model
    ===================================================
    Params: beta    Rate of transmission (exposure) 
            sigma   Rate of infection (upon exposure) 
            gamma   Rate of recovery (upon infection) 
            xi      Rate of re-susceptibility (upon recovery)  
            mu_I    Rate of infection-related death  
            mu_0    Rate of baseline death   
            nu      Rate of baseline birth
            
            beta_D  Rate of transmission (exposure) for individuals with detected infections
            sigma_D Rate of infection (upon exposure) for individuals with detected infections
            gamma_D Rate of recovery (upon infection) for individuals with detected infections
            mu_D    Rate of infection-related death for individuals with detected infections
            theta_E Rate of baseline testing for exposed individuals
            theta_I Rate of baseline testing for infectious individuals
            psi_E   Probability of positive test results for exposed individuals
            psi_I   Probability of positive test results for exposed individuals
            q       Probability of quarantined individuals interacting with others
            
            initE   Init number of exposed individuals       
            initI   Init number of infectious individuals      
            initD_E Init number of detected infectious individuals
            initD_I Init number of detected infectious individuals   
            initR   Init number of recovered individuals     
            initF   Init number of infection-related fatalities
                    (all remaining nodes initialized susceptible)   
    """

    def __init__(self, initN, beta, sigma, gamma, xi=0, mu_I=0, mu_0=0, nu=0, p=0,
                        beta_D=None, sigma_D=None, gamma_D=None, mu_D=None, 
                        theta_E=0, theta_I=0, psi_E=0, psi_I=0, q=0,
                        initE=0, initI=10, initD_E=0, initD_I=0, initR=0, initF=0):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model Parameters:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.beta   = beta  
        self.sigma  = sigma 
        self.gamma  = gamma 
        self.xi     = xi    
        self.mu_I   = mu_I  
        self.mu_0   = mu_0  
        self.nu     = nu    
        self.p      = p     

        # Testing-related parameters:
        self.beta_D   = beta_D  if beta_D is not None else self.beta
        self.sigma_D  = sigma_D if sigma_D is not None else self.sigma
        self.gamma_D  = gamma_D if gamma_D is not None else self.gamma
        self.mu_D     = mu_D    if mu_D is not None else self.mu_I
        self.theta_E  = theta_E if theta_E is not None else self.theta_E
        self.theta_I  = theta_I if theta_I is not None else self.theta_I
        self.psi_E    = psi_E   if psi_E is not None else self.psi_E
        self.psi_I    = psi_I   if psi_I is not None else self.psi_I
        self.q        = q       if q is not None else self.q

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Timekeeping:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.t       = 0
        self.tmax    = 0 # will be set when run() is called
        self.tseries = numpy.array([0])
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Counts of inidividuals with each state:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.N          = numpy.array([int(initN)])
        self.numE       = numpy.array([int(initE)])
        self.numI       = numpy.array([int(initI)])
        self.numD_E     = numpy.array([int(initD_E)])
        self.numD_I     = numpy.array([int(initD_I)])
        self.numR       = numpy.array([int(initR)])
        self.numF       = numpy.array([int(initF)])
        self.numS       = numpy.array([self.N[-1] - self.numE[-1] - self.numI[-1] - self.numD_E[-1] - self.numD_I[-1] - self.numR[-1] - self.numF[-1]])
        assert(self.numS[0] >= 0), "The specified initial population size N must be greater than or equal to the initial compartment counts."


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    @staticmethod
    def system_dfes(t, variables, beta, sigma, gamma, xi, mu_I, mu_0, nu,
                                  beta_D, sigma_D, gamma_D, mu_D, theta_E, theta_I, psi_E, psi_I, q):

        S, E, I, D_E, D_I, R, F = variables    # varibles is a list with compartment counts as elements 

        N   = S + E + I + D_E + D_I + R

        dS  = - (beta*S*I)/N - q*(beta_D*S*D_I)/N + xi*R + nu*N - mu_0*S

        dE  = (beta*S*I)/N + q*(beta_D*S*D_I)/N - sigma*E - theta_E*psi_E*E - mu_0*E

        dI  = sigma*E - gamma*I - mu_I*I - theta_I*psi_I*I - mu_0*I

        dDE = theta_E*psi_E*E - sigma_D*D_E - mu_0*D_E

        dDI = theta_I*psi_I*I + sigma_D*D_E - gamma_D*D_I - mu_D*D_I - mu_0*D_I

        dR  = gamma*I + gamma_D*D_I - xi*R - mu_0*R

        dF  = mu_I*I + mu_D*D_I

        return [dS, dE, dI, dDE, dDI, dR, dF]


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def run_epoch(self, runtime, dt=0.1):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create a list of times at which the ODE solver should output system values.
        # Append this list of times as the model's timeseries
        t_eval    = numpy.arange(start=self.t, stop=self.t+runtime, step=dt)

        # Define the range of time values for the integration:
        t_span          = (self.t, self.t+runtime)

        # Define the initial conditions as the system's current state:
        # (which will be the t=0 condition if this is the first run of this model, 
        # else where the last sim left off)

        init_cond       = [self.numS[-1], self.numE[-1], self.numI[-1], self.numD_E[-1], self.numD_I[-1], self.numR[-1], self.numF[-1]]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Solve the system of differential eqns:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        solution        = scipy.integrate.solve_ivp(lambda t, X: SEIRSModel.system_dfes(t, X, self.beta, self.sigma, self.gamma, self.xi, self.mu_I, self.mu_0, self.nu,
                                                                                            self.beta_D, self.sigma_D, self.gamma_D, self.mu_D, self.theta_E, self.theta_I, self.psi_E, self.psi_I, self.q
                                                                                        ), 
                                                        t_span=[self.t, self.tmax], y0=init_cond, t_eval=t_eval
                                                   )

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store the solution output as the model's time series and data series:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.tseries    = numpy.append(self.tseries, solution['t'])
        self.numS       = numpy.append(self.numS, solution['y'][0])
        self.numE       = numpy.append(self.numE, solution['y'][1])
        self.numI       = numpy.append(self.numI, solution['y'][2])
        self.numD_E       = numpy.append(self.numD_E, solution['y'][3])
        self.numD_I       = numpy.append(self.numD_I, solution['y'][4])
        self.numR       = numpy.append(self.numR, solution['y'][5])
        self.numF       = numpy.append(self.numF, solution['y'][6])

        self.t = self.tseries[-1]


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def run(self, T, dt=0.1, checkpoints=None, verbose=False):

        if(T>0):
            self.tmax += T
        else:
            return False
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-process checkpoint values:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(checkpoints):
            numCheckpoints = len(checkpoints['t'])
            paramNames = ['beta', 'sigma', 'gamma', 'xi', 'mu_I', 'mu_0', 'nu',
                          'beta_D', 'sigma_D', 'gamma_D', 'mu_D',
                          'theta_E', 'theta_I', 'psi_E', 'psi_I', 'q']
            for param in paramNames:
                # For params that don't have given checkpoint values (or bad value given), 
                # set their checkpoint values to the value they have now for all checkpoints.
                if(param not in list(checkpoints.keys())
                    or not isinstance(checkpoints[param], (list, numpy.ndarray)) 
                    or len(checkpoints[param])!=numCheckpoints):
                    checkpoints[param] = [getattr(self, param)]*numCheckpoints

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Run the simulation loop:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if(not checkpoints):
            self.run_epoch(runtime=self.tmax, dt=dt)

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            print("t = %.2f" % self.t)
            if(verbose):
                print("\t S   = " + str(self.numS[-1]))
                print("\t E   = " + str(self.numE[-1]))
                print("\t I   = " + str(self.numI[-1]))
                print("\t D_E = " + str(self.numD_E[-1]))
                print("\t D_I = " + str(self.numD_I[-1]))
                print("\t R   = " + str(self.numR[-1]))
                print("\t F   = " + str(self.numF[-1]))
                    

        else: # checkpoints provided
            for checkpointIdx, checkpointTime in enumerate(checkpoints['t']):
                # Run the sim until the next checkpoint time:
                self.run_epoch(runtime=checkpointTime-self.t, dt=dt)
                # Having reached the checkpoint, update applicable parameters:
                print("[Checkpoint: Updating parameters]")
                for param in paramNames:
                    setattr(self, param, checkpoints[param][checkpointIdx])

                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                print("t = %.2f" % self.t)
                if(verbose):
                    print("\t S   = " + str(self.numS[-1]))
                    print("\t E   = " + str(self.numE[-1]))
                    print("\t I   = " + str(self.numI[-1]))
                    print("\t D_E = " + str(self.numD_E[-1]))
                    print("\t D_I = " + str(self.numD_I[-1]))
                    print("\t R   = " + str(self.numR[-1]))
                    print("\t F   = " + str(self.numF[-1]))

            if(self.t < self.tmax):
                self.run_epoch(runtime=self.tmax-self.t, dt=dt)

        return True

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def total_num_infections(self, t_idx=None):
        if(t_idx is None):
            return (self.numE[:] + self.numI[:] + self.numD_E[:] + self.numD_I[:])  
        else:
            return (self.numE[t_idx] + self.numI[t_idx] + self.numD_E[t_idx] + self.numD_I[t_idx])   


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def plot(self, ax=None,  plot_S='line', plot_E='line', plot_I='line',plot_R='line', plot_F='line',
                            plot_D_E='line', plot_D_I='line', combine_D=True,
                            color_S='tab:green', color_E='orange', color_I='crimson', color_R='tab:blue', color_F='black',
                            color_D_E='mediumorchid', color_D_I='mediumorchid', color_reference='#E0E0E0',
                            dashed_reference_results=None, dashed_reference_label='reference', 
                            shaded_reference_results=None, shaded_reference_label='reference', 
                            vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                            ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True):

        import matplotlib.pyplot as pyplot

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create an Axes object if None provided:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(not ax):
            fig, ax = pyplot.subplots()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prepare data series to be plotted:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Fseries     = self.numF/self.N if plot_percentages else self.numF
        Eseries     = self.numE/self.N if plot_percentages else self.numE
        Dseries     = (self.numD_E+self.numD_I)/self.N if plot_percentages else (self.numD_E+self.numD_I)
        D_Eseries   = self.numD_E/self.N if plot_percentages else self.numD_E
        D_Iseries   = self.numD_I/self.N if plot_percentages else self.numD_I
        Iseries     = self.numI/self.N if plot_percentages else self.numI
        Rseries     = self.numR/self.N if plot_percentages else self.numR
        Sseries     = self.numS/self.N if plot_percentages else self.numS 

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the reference data:      
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(dashed_reference_results):
            dashedReference_tseries  = dashed_reference_results.tseries[::int(self.N/100)]
            dashedReference_IDEstack = (dashed_reference_results.numI + dashed_reference_results.numD_I + dashed_reference_results.numD_E + dashed_reference_results.numE)[::int(self.N/100)] / (self.N if plot_percentages else 1)
            ax.plot(dashedReference_tseries, dashedReference_IDEstack, color='#E0E0E0', linestyle='--', label='$I+D+E$ ('+dashed_reference_label+')', zorder=0)
        if(shaded_reference_results):
            shadedReference_tseries  = shaded_reference_results.tseries
            shadedReference_IDEstack = (shaded_reference_results.numI + shaded_reference_results.numD_I + shaded_reference_results.numD_E + shaded_reference_results.numE) / (self.N if plot_percentages else 1)
            ax.fill_between(shaded_reference_results.tseries, shadedReference_IDEstack, 0, color='#EFEFEF', label='$I+D+E$ ('+shaded_reference_label+')', zorder=0)
            ax.plot(shaded_reference_results.tseries, shadedReference_IDEstack, color='#E0E0E0', zorder=1)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the stacked variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        topstack = numpy.zeros_like(self.tseries)
        if(any(Fseries) and plot_F=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, topstack+Fseries), topstack, color=color_F, alpha=0.5, label='$F$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, topstack+Fseries),           color=color_F, zorder=3)
            topstack = topstack+Fseries
        if(any(Eseries) and plot_E=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, topstack+Eseries), topstack, color=color_E, alpha=0.5, label='$E$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, topstack+Eseries),           color=color_E, zorder=3)
            topstack = topstack+Eseries
        if(combine_D and plot_D_E=='stacked' and plot_D_I=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, topstack+Dseries), topstack, color=color_D_E, alpha=0.5, label='$D_{all}$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, topstack+Dseries),           color=color_D_E, zorder=3)
            topstack = topstack+Dseries
        else:
            if(any(D_Eseries) and plot_D_E=='stacked'):
                ax.fill_between(numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, topstack+D_Eseries), topstack, color=color_D_E, alpha=0.5, label='$D_E$', zorder=2)
                ax.plot(        numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, topstack+D_Eseries),           color=color_D_E, zorder=3)
                topstack = topstack+D_Eseries
            if(any(D_Iseries) and plot_D_I=='stacked'):
                ax.fill_between(numpy.ma.masked_where(D_Iseries<=0, self.tseries), numpy.ma.masked_where(D_Iseries<=0, topstack+D_Iseries), topstack, color=color_D_I, alpha=0.5, label='$D_I$', zorder=2)
                ax.plot(        numpy.ma.masked_where(D_Iseries<=0, self.tseries), numpy.ma.masked_where(D_Iseries<=0, topstack+D_Iseries),           color=color_D_I, zorder=3)
                topstack = topstack+D_Iseries
        if(any(Iseries) and plot_I=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, topstack+Iseries), topstack, color=color_I, alpha=0.5, label='$I$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, topstack+Iseries),           color=color_I, zorder=3)
            topstack = topstack+Iseries
        if(any(Rseries) and plot_R=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, topstack+Rseries), topstack, color=color_R, alpha=0.5, label='$R$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, topstack+Rseries),           color=color_R, zorder=3)
            topstack = topstack+Rseries
        if(any(Sseries) and plot_S=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, topstack+Sseries), topstack, color=color_S, alpha=0.5, label='$S$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, topstack+Sseries),           color=color_S, zorder=3)
            topstack = topstack+Sseries
        

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the shaded variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(any(Fseries) and plot_F=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, Fseries), 0, color=color_F, alpha=0.5, label='$F$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, Fseries),    color=color_F, zorder=5)
        if(any(Eseries) and plot_E=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, Eseries), 0, color=color_E, alpha=0.5, label='$E$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, Eseries),    color=color_E, zorder=5)
        if(combine_D and (any(Dseries) and plot_D_E=='shaded' and plot_D_E=='shaded')):
            ax.fill_between(numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, Dseries), 0, color=color_D_E, alpha=0.5, label='$D_{all}$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, Dseries),    color=color_D_E, zorder=5)
        else:
            if(any(D_Eseries) and plot_D_E=='shaded'):
                ax.fill_between(numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, D_Eseries), 0, color=color_D_E, alpha=0.5, label='$D_E$', zorder=4)
                ax.plot(        numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, D_Eseries),    color=color_D_E, zorder=5)
            if(any(D_Iseries) and plot_D_I=='shaded'):
                ax.fill_between(numpy.ma.masked_where(D_Iseries<=0, self.tseries), numpy.ma.masked_where(D_Iseries<=0, D_Iseries), 0, color=color_D_I, alpha=0.5, label='$D_I$', zorder=4)
                ax.plot(        numpy.ma.masked_where(D_Iseries<=0, self.tseries), numpy.ma.masked_where(D_Iseries<=0, D_Iseries),    color=color_D_I, zorder=5)
        if(any(Iseries) and plot_I=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, Iseries), 0, color=color_I, alpha=0.5, label='$I$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, Iseries),    color=color_I, zorder=5)
        if(any(Sseries) and plot_S=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, Sseries), 0, color=color_S, alpha=0.5, label='$S$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, Sseries),    color=color_S, zorder=5)
        if(any(Rseries) and plot_R=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, Rseries), 0, color=color_R, alpha=0.5, label='$R$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, Rseries),    color=color_R, zorder=5)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the line variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
        if(any(Fseries) and plot_F=='line'):
            ax.plot(numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, Fseries), color=color_F, label='$F$', zorder=6)
        if(any(Eseries) and plot_E=='line'):
            ax.plot(numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, Eseries), color=color_E, label='$E$', zorder=6)
        if(combine_D and (any(Dseries) and plot_D_E=='line' and plot_D_E=='line')):
            ax.plot(numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, Dseries), color=color_D_E, label='$D_{all}$', zorder=6)
        else:
            if(any(D_Eseries) and plot_D_E=='line'):
                ax.plot(numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, D_Eseries), color=color_D_E, label='$D_E$', zorder=6)
            if(any(D_Iseries) and plot_D_I=='line'):
                ax.plot(numpy.ma.masked_where(D_Iseries<=0, self.tseries), numpy.ma.masked_where(D_Iseries<=0, D_Iseries), color=color_D_I, label='$D_I$', zorder=6)
        if(any(Iseries) and plot_I=='line'):
            ax.plot(numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, Iseries), color=color_I, label='$I$', zorder=6)
        if(any(Sseries) and plot_S=='line'):
            ax.plot(numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, Sseries), color=color_S, label='$S$', zorder=6)
        if(any(Rseries) and plot_R=='line'):
            ax.plot(numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, Rseries), color=color_R, label='$R$', zorder=6)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the vertical line annotations:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(len(vlines)>0 and len(vline_colors)==0):
            vline_colors = ['gray']*len(vlines)
        if(len(vlines)>0 and len(vline_labels)==0):
            vline_labels = [None]*len(vlines)
        if(len(vlines)>0 and len(vline_styles)==0):
            vline_styles = [':']*len(vlines)
        for vline_x, vline_color, vline_style, vline_label in zip(vlines, vline_colors, vline_styles, vline_labels):
            if(vline_x is not None):
                ax.axvline(x=vline_x, color=vline_color, linestyle=vline_style, alpha=1, label=vline_label)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the plot labels:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ax.set_xlabel('days')
        ax.set_ylabel('percent of population' if plot_percentages else 'number of individuals')
        ax.set_xlim(0, (max(self.tseries) if not xlim else xlim))
        ax.set_ylim(0, ylim)
        if(plot_percentages):
            ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])
        if(legend):
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend(legend_handles[::-1], legend_labels[::-1], loc='upper right', facecolor='white', edgecolor='none', framealpha=0.9, prop={'size': 8})
        if(title):
            ax.set_title(title, size=12)
        if(side_title):
            ax.annotate(side_title, (0, 0.5), xytext=(-45, 0), ha='right', va='center',
                size=12, rotation=90, xycoords='axes fraction', textcoords='offset points')
       
        return ax


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def figure_basic(self, plot_S='line', plot_E='line', plot_I='line',plot_R='line', plot_F='line',
                        plot_D_E='line', plot_D_I='line', combine_D=True,
                        color_S='tab:green', color_E='orange', color_I='crimson', color_R='tab:blue', color_F='black',
                        color_D_E='mediumorchid', color_D_I='mediumorchid', color_reference='#E0E0E0',
                        dashed_reference_results=None, dashed_reference_label='reference', 
                        shaded_reference_results=None, shaded_reference_label='reference', 
                        vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                        ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True,
                        figsize=(12,8), use_seaborn=True, show=True):

        import matplotlib.pyplot as pyplot

        fig, ax = pyplot.subplots(figsize=figsize)

        if(use_seaborn):
            import seaborn
            seaborn.set_style('ticks')
            seaborn.despine()

        self.plot(ax=ax, plot_S=plot_S, plot_E=plot_E, plot_I=plot_I,plot_R=plot_R, plot_F=plot_F,
                        plot_D_E=plot_D_E, plot_D_I=plot_D_I, combine_D=combine_D,
                        color_S=color_S, color_E=color_E, color_I=color_I, color_R=color_R, color_F=color_F,
                        color_D_E=color_D_E, color_D_I=color_D_I, color_reference=color_reference,
                        dashed_reference_results=dashed_reference_results, dashed_reference_label=dashed_reference_label, 
                        shaded_reference_results=shaded_reference_results, shaded_reference_label=shaded_reference_label, 
                        vlines=vlines, vline_colors=vline_colors, vline_styles=vline_styles, vline_labels=vline_labels,
                        ylim=ylim, xlim=xlim, legend=legend, title=title, side_title=side_title, plot_percentages=plot_percentages)

        if(show):
            pyplot.show()

        return fig, ax


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def figure_infections(self, plot_S=False, plot_E='stacked', plot_I='stacked',plot_R=False, plot_F=False,
                            plot_D_E='stacked', plot_D_I='stacked', combine_D=True,
                            color_S='tab:green', color_E='orange', color_I='crimson', color_R='tab:blue', color_F='black',
                            color_D_E='mediumorchid', color_D_I='mediumorchid', color_reference='#E0E0E0',
                            dashed_reference_results=None, dashed_reference_label='reference', 
                            shaded_reference_results=None, shaded_reference_label='reference', 
                            vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                            ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True,
                            figsize=(12,8), use_seaborn=True, show=True):

        import matplotlib.pyplot as pyplot

        fig, ax = pyplot.subplots(figsize=figsize)

        if(use_seaborn):
            import seaborn
            seaborn.set_style('ticks')
            seaborn.despine()

        self.plot(ax=ax, plot_S=plot_S, plot_E=plot_E, plot_I=plot_I,plot_R=plot_R, plot_F=plot_F,
                        plot_D_E=plot_D_E, plot_D_I=plot_D_I, combine_D=combine_D,
                        color_S=color_S, color_E=color_E, color_I=color_I, color_R=color_R, color_F=color_F,
                        color_D_E=color_D_E, color_D_I=color_D_I, color_reference=color_reference,
                        dashed_reference_results=dashed_reference_results, dashed_reference_label=dashed_reference_label, 
                        shaded_reference_results=shaded_reference_results, shaded_reference_label=shaded_reference_label, 
                        vlines=vlines, vline_colors=vline_colors, vline_styles=vline_styles, vline_labels=vline_labels, 
                        ylim=ylim, xlim=xlim, legend=legend, title=title, side_title=side_title, plot_percentages=plot_percentages)

        if(show):
            pyplot.show()

        return fig, ax


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class SEIRSNetworkModel():
    """
    A class to simulate the SEIRS Stochastic Network Model
    ===================================================
    Params: G       Network adjacency matrix (numpy array) or Networkx graph object.
            beta    Rate of transmission (exposure) (global)
            beta_local    Rate(s) of transmission (exposure) for adjacent individuals (optional)
            sigma   Rate of infection (upon exposure) 
            gamma   Rate of recovery (upon infection) 
            xi      Rate of re-susceptibility (upon recovery)  
            mu_I    Rate of infection-related death  
            mu_0    Rate of baseline death   
            nu      Rate of baseline birth
            p       Probability of interaction outside adjacent nodes
            
            Q       Quarantine adjacency matrix (numpy array) or Networkx graph object.
            beta_D  Rate of transmission (exposure) for individuals with detected infections (global)
            beta_local    Rate(s) of transmission (exposure) for adjacent individuals with detected infections (optional)
            sigma_D Rate of infection (upon exposure) for individuals with detected infections
            gamma_D Rate of recovery (upon infection) for individuals with detected infections
            mu_D    Rate of infection-related death for individuals with detected infections
            theta_E Rate of baseline testing for exposed individuals
            theta_I Rate of baseline testing for infectious individuals
            phi_E   Rate of contact tracing testing for exposed individuals
            phi_I   Rate of contact tracing testing for infectious individuals
            psi_E   Probability of positive test results for exposed individuals
            psi_I   Probability of positive test results for exposed individuals
            q       Probability of quarantined individuals interaction outside adjacent nodes
            
            initE   Init number of exposed individuals       
            initI   Init number of infectious individuals      
            initD_E Init number of detected infectious individuals
            initD_I Init number of detected infectious individuals   
            initR   Init number of recovered individuals     
            initF   Init number of infection-related fatalities
                    (all remaining nodes initialized susceptible)   
    """

    def __init__(self, G, beta, sigma, gamma, xi=0, mu_I=0, mu_0=0, nu=0, beta_local=None, p=0,
                    Q=None, beta_D=None, sigma_D=None, gamma_D=None, mu_D=None, beta_D_local=None,
                    theta_E=0, theta_I=0, phi_E=0, phi_I=0, psi_E=1, psi_I=1, q=0,
                    initE=0, initI=10, initD_E=0, initD_I=0, initR=0, initF=0,
                    node_groups=None, store_Xseries=False):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Setup Adjacency matrix:
        self.update_G(G)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Setup Quarantine Adjacency matrix:
        if(Q is None):
            Q = G # If no Q graph is provided, use G in its place
        self.update_Q(Q)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model Parameters:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.parameters = { 'beta':beta, 'sigma':sigma, 'gamma':gamma, 'xi':xi, 'mu_I':mu_I, 'mu_0':mu_0, 'nu':nu, 
                            'beta_D':beta_D, 'sigma_D':sigma_D, 'gamma_D':gamma_D, 'mu_D':mu_D, 
                            'beta_local':beta_local, 'beta_D_local':beta_D_local, 'p':p,'q':q,
                            'theta_E':theta_E, 'theta_I':theta_I, 'phi_E':phi_E, 'phi_I':phi_I, 'psi_E':psi_E, 'psi_I':psi_I }
        self.update_parameters()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Each node can undergo up to 4 transitions (sans vitality/re-susceptibility returns to S state),
        # so there are ~numNodes*4 events/timesteps expected; initialize numNodes*5 timestep slots to start 
        # (will be expanded during run if needed)
        self.tseries    = numpy.zeros(5*self.numNodes)
        self.numE       = numpy.zeros(5*self.numNodes)
        self.numI       = numpy.zeros(5*self.numNodes)
        self.numD_E     = numpy.zeros(5*self.numNodes)
        self.numD_I     = numpy.zeros(5*self.numNodes)
        self.numR       = numpy.zeros(5*self.numNodes)
        self.numF       = numpy.zeros(5*self.numNodes)
        self.numS       = numpy.zeros(5*self.numNodes)
        self.N          = numpy.zeros(5*self.numNodes)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Timekeeping:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.t          = 0
        self.tmax       = 0 # will be set when run() is called
        self.tidx       = 0
        self.tseries[0] = 0
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Counts of inidividuals with each state:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.numE[0]    = int(initE)
        self.numI[0]    = int(initI)
        self.numD_E[0]  = int(initD_E)
        self.numD_I[0]  = int(initD_I)
        self.numR[0]    = int(initR)
        self.numF[0]    = int(initF)
        self.numS[0]    = self.numNodes - self.numE[0] - self.numI[0] - self.numD_E[0] - self.numD_I[0] - self.numR[0] - self.numF[0]
        self.N[0]       = self.numS[0] + self.numE[0] + self.numI[0] + self.numD_E[0] + self.numD_I[0] + self.numR[0]
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Node states:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.S          = 1
        self.E          = 2
        self.I          = 3
        self.D_E        = 4
        self.D_I        = 5
        self.R          = 6
        self.F          = 7

        self.X = numpy.array([self.S]*int(self.numS[0]) + [self.E]*int(self.numE[0]) + [self.I]*int(self.numI[0]) + [self.D_E]*int(self.numD_E[0]) + [self.D_I]*int(self.numD_I[0]) + [self.R]*int(self.numR[0]) + [self.F]*int(self.numF[0])).reshape((self.numNodes,1))
        numpy.random.shuffle(self.X)

        self.store_Xseries = store_Xseries
        if(store_Xseries):
            self.Xseries        = numpy.zeros(shape=(5*self.numNodes, self.numNodes), dtype='uint8')
            self.Xseries[0,:]   = self.X.T

        self.transitions =  { 
                                'StoE': {'currentState':self.S, 'newState':self.E},
                                'EtoI': {'currentState':self.E, 'newState':self.I},
                                'ItoR': {'currentState':self.I, 'newState':self.R},
                                'ItoF': {'currentState':self.I, 'newState':self.F},
                                'RtoS': {'currentState':self.R, 'newState':self.S},
                                'EtoDE': {'currentState':self.E, 'newState':self.D_E},
                                'ItoDI': {'currentState':self.I, 'newState':self.D_I},
                                'DEtoDI': {'currentState':self.D_E, 'newState':self.D_I},
                                'DItoR': {'currentState':self.D_I, 'newState':self.R},
                                'DItoF': {'currentState':self.D_I, 'newState':self.F},
                                '_toS': {'currentState':True, 'newState':self.S},
                            }

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize node subgroup data series:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.nodeGroupData = None
        if(node_groups):
            self.nodeGroupData = {}
            for groupName, nodeList in node_groups.items():
                self.nodeGroupData[groupName] = {'nodes':   numpy.array(nodeList),
                                                 'mask':    numpy.isin(range(self.numNodes), nodeList).reshape((self.numNodes,1))}
                self.nodeGroupData[groupName]['numS']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numE']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numI']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numD_E']     = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numD_I']     = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numR']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numF']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['N']          = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numS'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.S)
                self.nodeGroupData[groupName]['numE'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.E)
                self.nodeGroupData[groupName]['numI'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I)
                self.nodeGroupData[groupName]['numD_E'][0]  = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.D_E)
                self.nodeGroupData[groupName]['numD_I'][0]  = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.D_I)
                self.nodeGroupData[groupName]['numR'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.R)
                self.nodeGroupData[groupName]['numF'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.F)
                self.nodeGroupData[groupName]['N'][0]       = self.nodeGroupData[groupName]['numS'][0] + self.nodeGroupData[groupName]['numE'][0] + self.nodeGroupData[groupName]['numI'][0] + self.nodeGroupData[groupName]['numD_E'][0] + self.nodeGroupData[groupName]['numD_I'][0] + self.nodeGroupData[groupName]['numR'][0]

         

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def update_parameters(self):
        import time
        updatestart = time.time()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model parameters:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.beta           = numpy.array(self.parameters['beta']).reshape((self.numNodes, 1))  if isinstance(self.parameters['beta'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['beta'], shape=(self.numNodes,1))
        self.sigma          = numpy.array(self.parameters['sigma']).reshape((self.numNodes, 1)) if isinstance(self.parameters['sigma'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['sigma'], shape=(self.numNodes,1))
        self.gamma          = numpy.array(self.parameters['gamma']).reshape((self.numNodes, 1)) if isinstance(self.parameters['gamma'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['gamma'], shape=(self.numNodes,1))
        self.xi             = numpy.array(self.parameters['xi']).reshape((self.numNodes, 1))    if isinstance(self.parameters['xi'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['xi'], shape=(self.numNodes,1))
        self.mu_I           = numpy.array(self.parameters['mu_I']).reshape((self.numNodes, 1))  if isinstance(self.parameters['mu_I'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['mu_I'], shape=(self.numNodes,1))
        self.mu_0           = numpy.array(self.parameters['mu_0']).reshape((self.numNodes, 1))  if isinstance(self.parameters['mu_0'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['mu_0'], shape=(self.numNodes,1))
        self.nu             = numpy.array(self.parameters['nu']).reshape((self.numNodes, 1))    if isinstance(self.parameters['nu'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['nu'], shape=(self.numNodes,1))
        self.p              = numpy.array(self.parameters['p']).reshape((self.numNodes, 1))     if isinstance(self.parameters['p'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['p'], shape=(self.numNodes,1))
        
        # Testing-related parameters:
        self.beta_D         = (numpy.array(self.parameters['beta_D']).reshape((self.numNodes, 1))  if isinstance(self.parameters['beta_D'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['beta_D'], shape=(self.numNodes,1))) if self.parameters['beta_D'] is not None else self.beta
        self.sigma_D        = (numpy.array(self.parameters['sigma_D']).reshape((self.numNodes, 1)) if isinstance(self.parameters['sigma_D'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['sigma_D'], shape=(self.numNodes,1))) if self.parameters['sigma_D'] is not None else self.sigma
        self.gamma_D        = (numpy.array(self.parameters['gamma_D']).reshape((self.numNodes, 1)) if isinstance(self.parameters['gamma_D'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['gamma_D'], shape=(self.numNodes,1))) if self.parameters['gamma_D'] is not None else self.gamma
        self.mu_D           = (numpy.array(self.parameters['mu_D']).reshape((self.numNodes, 1))    if isinstance(self.parameters['mu_D'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['mu_D'], shape=(self.numNodes,1))) if self.parameters['mu_D'] is not None else self.mu_I
        self.theta_E        = numpy.array(self.parameters['theta_E']).reshape((self.numNodes, 1))  if isinstance(self.parameters['theta_E'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['theta_E'], shape=(self.numNodes,1))
        self.theta_I        = numpy.array(self.parameters['theta_I']).reshape((self.numNodes, 1))  if isinstance(self.parameters['theta_I'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['theta_I'], shape=(self.numNodes,1))
        self.phi_E          = numpy.array(self.parameters['phi_E']).reshape((self.numNodes, 1))    if isinstance(self.parameters['phi_E'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['phi_E'], shape=(self.numNodes,1))
        self.phi_I          = numpy.array(self.parameters['phi_I']).reshape((self.numNodes, 1))    if isinstance(self.parameters['phi_I'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['phi_I'], shape=(self.numNodes,1))
        self.psi_E          = numpy.array(self.parameters['psi_E']).reshape((self.numNodes, 1))    if isinstance(self.parameters['psi_E'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['psi_E'], shape=(self.numNodes,1))
        self.psi_I          = numpy.array(self.parameters['psi_I']).reshape((self.numNodes, 1))    if isinstance(self.parameters['psi_I'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['psi_I'], shape=(self.numNodes,1))
        self.q              = numpy.array(self.parameters['q']).reshape((self.numNodes, 1))        if isinstance(self.parameters['q'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['q'], shape=(self.numNodes,1))

        #Local transmission parameters:
        if(self.parameters['beta_local'] is not None):
            if(isinstance(self.parameters['beta_local'], (list, numpy.ndarray))):
                if(isinstance(self.parameters['beta_local'], list)):
                    self.beta_local = numpy.array(self.parameters['beta_local'])
                else: # is numpy.ndarray
                    self.beta_local = self.parameters['beta_local']
                if(self.beta_local.ndim == 1):
                    self.beta_local.reshape((self.numNodes, 1))
                elif(self.beta_local.ndim == 2):
                    self.beta_local.reshape((self.numNodes, self.numNodes))
            else:
                self.beta_local = numpy.full_like(self.beta, fill_value=self.parameters['beta_local'])
        else:
            self.beta_local = self.beta
        #----------------------------------------
        if(self.parameters['beta_D_local'] is not None):
            if(isinstance(self.parameters['beta_D_local'], (list, numpy.ndarray))):
                if(isinstance(self.parameters['beta_D_local'], list)):
                    self.beta_D_local = numpy.array(self.parameters['beta_D_local'])
                else: # is numpy.ndarray
                    self.beta_D_local = self.parameters['beta_D_local']
                if(self.beta_D_local.ndim == 1):
                    self.beta_D_local.reshape((self.numNodes, 1))
                elif(self.beta_D_local.ndim == 2):
                    self.beta_D_local.reshape((self.numNodes, self.numNodes))
            else:
                self.beta_D_local = numpy.full_like(self.beta_D, fill_value=self.parameters['beta_D_local'])
        else:
            self.beta_D_local = self.beta_D
        
        # Pre-multiply beta values by the adjacency matrix ("transmission weight connections")
        if(self.beta_local.ndim == 1):
            self.A_beta     = scipy.sparse.csr_matrix.multiply(self.A, numpy.tile(self.beta_local, (1,self.numNodes))).tocsr()
        elif(self.beta_local.ndim == 2):
            self.A_beta     = scipy.sparse.csr_matrix.multiply(self.A, self.beta_local).tocsr()
        # Pre-multiply beta_D values by the quarantine adjacency matrix ("transmission weight connections")
        if(self.beta_D_local.ndim == 1):
            self.A_Q_beta_D = scipy.sparse.csr_matrix.multiply(self.A_Q, numpy.tile(self.beta_D_local, (1,self.numNodes))).tocsr()
        elif(self.beta_D_local.ndim == 2):
            self.A_Q_beta_D = scipy.sparse.csr_matrix.multiply(self.A_Q, self.beta_D_local).tocsr()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update scenario flags:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.update_scenario_flags()

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def node_degrees(self, Amat):
        return Amat.sum(axis=0).reshape(self.numNodes,1)   # sums of adj matrix cols

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def update_G(self, new_G):
        self.G = new_G
        # Adjacency matrix:
        if type(new_G)==numpy.ndarray:
            self.A = scipy.sparse.csr_matrix(new_G)
        elif type(new_G)==networkx.classes.graph.Graph:
            self.A = networkx.adj_matrix(new_G) # adj_matrix gives scipy.sparse csr_matrix
        else:
            raise BaseException("Input an adjacency matrix or networkx object only.")

        self.numNodes   = int(self.A.shape[1])
        self.degree     = numpy.asarray(self.node_degrees(self.A)).astype(float)

        return

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def update_Q(self, new_Q):
        self.Q = new_Q
        # Quarantine Adjacency matrix:
        if type(new_Q)==numpy.ndarray:
            self.A_Q = scipy.sparse.csr_matrix(new_Q)
        elif type(new_Q)==networkx.classes.graph.Graph:
            self.A_Q = networkx.adj_matrix(new_Q) # adj_matrix gives scipy.sparse csr_matrix
        else:
            raise BaseException("Input an adjacency matrix or networkx object only.")

        self.numNodes_Q   = int(self.A_Q.shape[1])
        self.degree_Q     = numpy.asarray(self.node_degrees(self.A_Q)).astype(float)

        assert(self.numNodes == self.numNodes_Q), "The normal and quarantine adjacency graphs must be of the same size."

        return

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def update_scenario_flags(self):
        self.testing_scenario   = ( (numpy.any(self.psi_I) and (numpy.any(self.theta_I) or numpy.any(self.phi_I)))  
                                    or (numpy.any(self.psi_E) and (numpy.any(self.theta_E) or numpy.any(self.phi_E))) )
        self.tracing_scenario   = ( (numpy.any(self.psi_E) and numpy.any(self.phi_E)) 
                                    or (numpy.any(self.psi_I) and numpy.any(self.phi_I)) )
        self.vitality_scenario  = (numpy.any(self.mu_0) and numpy.any(self.nu))
        self.resusceptibility_scenario  = (numpy.any(self.xi))

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def total_num_infections(self, t_idx=None):
        if(t_idx is None):
            return (self.numE[:] + self.numI[:] + self.numD_E[:] + self.numD_I[:])            
        else:
            return (self.numE[t_idx] + self.numI[t_idx] + self.numD_E[t_idx] + self.numD_I[t_idx])          


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
    def calc_propensities(self):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-calculate matrix multiplication terms that may be used in multiple propensity calculations,
        # and check to see if their computation is necessary before doing the multiplication
        transmissionTerms_I = numpy.zeros(shape=(self.numNodes,1))
        if(numpy.any(self.numI[self.tidx]) 
            and numpy.any(self.beta!=0)):
            transmissionTerms_I = numpy.asarray( scipy.sparse.csr_matrix.dot(self.A_beta, self.X==self.I) )

        transmissionTerms_DI = numpy.zeros(shape=(self.numNodes,1))
        if(self.testing_scenario 
            and numpy.any(self.numD_I[self.tidx])
            and numpy.any(self.beta_D)):
            transmissionTerms_DI = numpy.asarray( scipy.sparse.csr_matrix.dot(self.A_Q_beta_D, self.X==self.D_I) )

        numContacts_D = numpy.zeros(shape=(self.numNodes,1))
        if(self.tracing_scenario 
            and (numpy.any(self.numD_E[self.tidx]) or numpy.any(self.numD_I[self.tidx]))):
            numContacts_D = numpy.asarray( scipy.sparse.csr_matrix.dot( self.A, ((self.X==self.D_E)|(self.X==self.D_I)) ) )
                                            

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        propensities_StoE   = ( self.p*((self.beta*self.numI[self.tidx] + self.q*self.beta_D*self.numD_I[self.tidx])/self.N[self.tidx])
                                + (1-self.p)*numpy.divide((transmissionTerms_I + transmissionTerms_DI), self.degree, out=numpy.zeros_like(self.degree), where=self.degree!=0)
                              )*(self.X==self.S)

        propensities_EtoI   = self.sigma*(self.X==self.E)

        propensities_ItoR   = self.gamma*(self.X==self.I)

        propensities_ItoF   = self.mu_I*(self.X==self.I)

        # propensities_EtoDE  = ( self.theta_E + numpy.divide((self.phi_E*numContacts_D), self.degree, out=numpy.zeros_like(self.degree), where=self.degree!=0) )*self.psi_E*(self.X==self.E)
        propensities_EtoDE  = (self.theta_E + self.phi_E*numContacts_D)*self.psi_E*(self.X==self.E)

        # propensities_ItoDI  = ( self.theta_I + numpy.divide((self.phi_I*numContacts_D), self.degree, out=numpy.zeros_like(self.degree), where=self.degree!=0) )*self.psi_I*(self.X==self.I)
        propensities_ItoDI  = (self.theta_I + self.phi_I*numContacts_D)*self.psi_I*(self.X==self.I)

        propensities_DEtoDI = self.sigma_D*(self.X==self.D_E)

        propensities_DItoR  = self.gamma_D*(self.X==self.D_I)

        propensities_DItoF  = self.mu_D*(self.X==self.D_I)

        propensities_RtoS   = self.xi*(self.X==self.R)

        propensities__toS   = self.nu*(self.X!=self.F)

        propensities = numpy.hstack([propensities_StoE, propensities_EtoI, 
                                     propensities_ItoR, propensities_ItoF, 
                                     propensities_EtoDE, propensities_ItoDI, propensities_DEtoDI, 
                                     propensities_DItoR, propensities_DItoF,
                                     propensities_RtoS, propensities__toS])

        columns = ['StoE', 'EtoI', 'ItoR', 'ItoF', 'EtoDE', 'ItoDI', 'DEtoDI', 'DItoR', 'DItoF', 'RtoS', '_toS']

        return propensities, columns


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    

    def increase_data_series_length(self):
        self.tseries= numpy.pad(self.tseries, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numS   = numpy.pad(self.numS, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numE   = numpy.pad(self.numE, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numI   = numpy.pad(self.numI, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numD_E = numpy.pad(self.numD_E, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numD_I = numpy.pad(self.numD_I, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numR   = numpy.pad(self.numR, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numF   = numpy.pad(self.numF, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.N      = numpy.pad(self.N, [(0, 5*self.numNodes)], mode='constant', constant_values=0)

        if(self.store_Xseries):
            self.Xseries = numpy.pad(self.Xseries, [(0, 5*self.numNodes), (0,0)], mode='constant', constant_values=0)

        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                self.nodeGroupData[groupName]['numS']     = numpy.pad(self.nodeGroupData[groupName]['numS'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numE']     = numpy.pad(self.nodeGroupData[groupName]['numE'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numI']     = numpy.pad(self.nodeGroupData[groupName]['numI'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numD_E']   = numpy.pad(self.nodeGroupData[groupName]['numD_E'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numD_I']   = numpy.pad(self.nodeGroupData[groupName]['numD_I'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numR']     = numpy.pad(self.nodeGroupData[groupName]['numR'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numF']     = numpy.pad(self.nodeGroupData[groupName]['numF'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['N']        = numpy.pad(self.nodeGroupData[groupName]['N'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)

        return None

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 

    def finalize_data_series(self):
        self.tseries= numpy.array(self.tseries, dtype=float)[:self.tidx+1]
        self.numS   = numpy.array(self.numS, dtype=float)[:self.tidx+1]
        self.numE   = numpy.array(self.numE, dtype=float)[:self.tidx+1]
        self.numI   = numpy.array(self.numI, dtype=float)[:self.tidx+1]
        self.numD_E = numpy.array(self.numD_E, dtype=float)[:self.tidx+1]
        self.numD_I = numpy.array(self.numD_I, dtype=float)[:self.tidx+1]
        self.numR   = numpy.array(self.numR, dtype=float)[:self.tidx+1]
        self.numF   = numpy.array(self.numF, dtype=float)[:self.tidx+1]
        self.N      = numpy.array(self.N, dtype=float)[:self.tidx+1]

        if(self.store_Xseries):
            self.Xseries = self.Xseries[:self.tidx+1, :]

        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                self.nodeGroupData[groupName]['numS']    = numpy.array(self.nodeGroupData[groupName]['numS'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numE']    = numpy.array(self.nodeGroupData[groupName]['numE'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numI']    = numpy.array(self.nodeGroupData[groupName]['numI'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numD_E']  = numpy.array(self.nodeGroupData[groupName]['numD_E'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numD_I']  = numpy.array(self.nodeGroupData[groupName]['numD_I'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numR']    = numpy.array(self.nodeGroupData[groupName]['numR'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numF']    = numpy.array(self.nodeGroupData[groupName]['numF'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['N']       = numpy.array(self.nodeGroupData[groupName]['N'], dtype=float)[:self.tidx+1]

        return None

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^     

    def run_iteration(self):

        if(self.tidx >= len(self.tseries)-1):
            # Room has run out in the timeseries storage arrays; double the size of these arrays:
            self.increase_data_series_length()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1. Generate 2 random numbers uniformly distributed in (0,1)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        r1 = numpy.random.rand()
        r2 = numpy.random.rand()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2. Calculate propensities
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        propensities, transitionTypes = self.calc_propensities()

        # Terminate when probability of all events is 0:
        if(propensities.sum() <= 0.0):            
            self.finalize_data_series()
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3. Calculate alpha
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        propensities_flat   = propensities.ravel(order='F')
        cumsum              = propensities_flat.cumsum()
        alpha               = propensities_flat.sum()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 4. Compute the time until the next event takes place
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        tau = (1/alpha)*numpy.log(float(1/r1))
        self.t += tau

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 5. Compute which event takes place
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        transitionIdx   = numpy.searchsorted(cumsum,r2*alpha)
        transitionNode  = transitionIdx % self.numNodes
        transitionType  = transitionTypes[ int(transitionIdx/self.numNodes) ]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 6. Update node states and data series
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        assert(self.X[transitionNode] == self.transitions[transitionType]['currentState'] and self.X[transitionNode]!=self.F), "Assertion error: Node "+str(transitionNode)+" has unexpected current state "+str(self.X[transitionNode])+" given the intended transition of "+str(transitionType)+"."
        self.X[transitionNode] = self.transitions[transitionType]['newState']

        self.tidx += 1
        
        self.tseries[self.tidx]  = self.t
        self.numS[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.S), a_min=0, a_max=self.numNodes)
        self.numE[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.E), a_min=0, a_max=self.numNodes)
        self.numI[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.I), a_min=0, a_max=self.numNodes)
        self.numD_E[self.tidx]   = numpy.clip(numpy.count_nonzero(self.X==self.D_E), a_min=0, a_max=self.numNodes)
        self.numD_I[self.tidx]   = numpy.clip(numpy.count_nonzero(self.X==self.D_I), a_min=0, a_max=self.numNodes)
        self.numR[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.R), a_min=0, a_max=self.numNodes)
        self.numF[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.F), a_min=0, a_max=self.numNodes)
        self.N[self.tidx]        = numpy.clip((self.numS[self.tidx] + self.numE[self.tidx] + self.numI[self.tidx] + self.numD_E[self.tidx] + self.numD_I[self.tidx] + self.numR[self.tidx]), a_min=0, a_max=self.numNodes)

        if(self.store_Xseries):
            self.Xseries[self.tidx,:] = self.X.T

        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                self.nodeGroupData[groupName]['numS'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.S)
                self.nodeGroupData[groupName]['numE'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.E)
                self.nodeGroupData[groupName]['numI'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I)
                self.nodeGroupData[groupName]['numD_E'][self.tidx]  = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.D_E)
                self.nodeGroupData[groupName]['numD_I'][self.tidx]  = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.D_I)
                self.nodeGroupData[groupName]['numR'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.R)
                self.nodeGroupData[groupName]['numF'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.F)
                self.nodeGroupData[groupName]['N'][self.tidx]       = numpy.clip((self.nodeGroupData[groupName]['numS'][0] + self.nodeGroupData[groupName]['numE'][0] + self.nodeGroupData[groupName]['numI'][0] + self.nodeGroupData[groupName]['numD_E'][0] + self.nodeGroupData[groupName]['numD_I'][0] + self.nodeGroupData[groupName]['numR'][0]), a_min=0, a_max=self.numNodes)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Terminate if tmax reached or num infectious and num exposed is 0:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(self.t >= self.tmax or (self.numI[self.tidx]<1 and self.numE[self.tidx]<1 and self.numD_E[self.tidx]<1 and self.numD_I[self.tidx]<1)):
            self.finalize_data_series()
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        return True


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def run(self, T, checkpoints=None, print_interval=10, verbose='t'):
        if(T>0):
            self.tmax += T
        else:
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-process checkpoint values:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(checkpoints):
            numCheckpoints = len(checkpoints['t'])
            for chkpt_param, chkpt_values in checkpoints.items():
                assert(isinstance(chkpt_values, (list, numpy.ndarray)) and len(chkpt_values)==numCheckpoints), "Expecting a list of values with length equal to number of checkpoint times ("+str(numCheckpoints)+") for each checkpoint parameter."
            checkpointIdx  = numpy.searchsorted(checkpoints['t'], self.t) # Finds 1st index in list greater than given val
            if(checkpointIdx >= numCheckpoints):
                # We are out of checkpoints, stop checking them:
                checkpoints = None 
            else:
                checkpointTime = checkpoints['t'][checkpointIdx]

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Run the simulation loop:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        print_reset = True
        running     = True
        while running:

            running = self.run_iteration()

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Handle checkpoints if applicable:
            if(checkpoints):
                if(self.t >= checkpointTime):
                    if(verbose is not False):
                        print("[Checkpoint: Updating parameters]")
                    # A checkpoint has been reached, update param values:
                    if('G' in list(checkpoints.keys())):
                        self.update_G(checkpoints['G'][checkpointIdx])
                    if('Q' in list(checkpoints.keys())):
                        self.update_Q(checkpoints['Q'][checkpointIdx])
                    for param in list(self.parameters.keys()):
                        if(param in list(checkpoints.keys())):
                            self.parameters.update({param: checkpoints[param][checkpointIdx]})
                    # Update parameter data structures and scenario flags:
                    self.update_parameters()
                    # Update the next checkpoint time:
                    checkpointIdx  = numpy.searchsorted(checkpoints['t'], self.t) # Finds 1st index in list greater than given val
                    if(checkpointIdx >= numCheckpoints):
                        # We are out of checkpoints, stop checking them:
                        checkpoints = None 
                    else:
                        checkpointTime = checkpoints['t'][checkpointIdx]
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if(print_interval):
                if(print_reset and (int(self.t) % print_interval == 0)):
                    if(verbose=="t"):
                        print("t = %.2f" % self.t)
                    if(verbose==True):
                        print("t = %.2f" % self.t)
                        print("\t S   = " + str(self.numS[self.tidx]))
                        print("\t E   = " + str(self.numE[self.tidx]))
                        print("\t I   = " + str(self.numI[self.tidx]))
                        print("\t D_E = " + str(self.numD_E[self.tidx]))
                        print("\t D_I = " + str(self.numD_I[self.tidx]))
                        print("\t R   = " + str(self.numR[self.tidx]))
                        print("\t F   = " + str(self.numF[self.tidx]))
                    print_reset = False
                elif(not print_reset and (int(self.t) % 10 != 0)):
                    print_reset = True

        return True


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def plot(self, ax=None,  plot_S='line', plot_E='line', plot_I='line',plot_R='line', plot_F='line',
                            plot_D_E='line', plot_D_I='line', combine_D=True,
                            color_S='tab:green', color_E='orange', color_I='crimson', color_R='tab:blue', color_F='black',
                            color_D_E='mediumorchid', color_D_I='mediumorchid', color_reference='#E0E0E0',
                            dashed_reference_results=None, dashed_reference_label='reference', 
                            shaded_reference_results=None, shaded_reference_label='reference', 
                            vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                            ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True):

        import matplotlib.pyplot as pyplot

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create an Axes object if None provided:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(not ax):
            fig, ax = pyplot.subplots()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prepare data series to be plotted:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Fseries     = self.numF/self.numNodes if plot_percentages else self.numF
        Eseries     = self.numE/self.numNodes if plot_percentages else self.numE
        Dseries     = (self.numD_E+self.numD_I)/self.numNodes if plot_percentages else (self.numD_E+self.numD_I)
        D_Eseries   = self.numD_E/self.numNodes if plot_percentages else self.numD_E
        D_Iseries   = self.numD_I/self.numNodes if plot_percentages else self.numD_I
        Iseries     = self.numI/self.numNodes if plot_percentages else self.numI
        Rseries     = self.numR/self.numNodes if plot_percentages else self.numR
        Sseries     = self.numS/self.numNodes if plot_percentages else self.numS 

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the reference data:      
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(dashed_reference_results):
            dashedReference_tseries  = dashed_reference_results.tseries[::int(self.numNodes/100)]
            dashedReference_IDEstack = (dashed_reference_results.numI + dashed_reference_results.numD_I + dashed_reference_results.numD_E + dashed_reference_results.numE)[::int(self.numNodes/100)] / (self.numNodes if plot_percentages else 1)
            ax.plot(dashedReference_tseries, dashedReference_IDEstack, color='#E0E0E0', linestyle='--', label='$I+D+E$ ('+dashed_reference_label+')', zorder=0)
        if(shaded_reference_results):
            shadedReference_tseries  = shaded_reference_results.tseries
            shadedReference_IDEstack = (shaded_reference_results.numI + shaded_reference_results.numD_I + shaded_reference_results.numD_E + shaded_reference_results.numE) / (self.numNodes if plot_percentages else 1)
            ax.fill_between(shaded_reference_results.tseries, shadedReference_IDEstack, 0, color='#EFEFEF', label='$I+D+E$ ('+shaded_reference_label+')', zorder=0)
            ax.plot(shaded_reference_results.tseries, shadedReference_IDEstack, color='#E0E0E0', zorder=1)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the stacked variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        topstack = numpy.zeros_like(self.tseries)
        if(any(Fseries) and plot_F=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, topstack+Fseries), topstack, color=color_F, alpha=0.5, label='$F$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, topstack+Fseries),           color=color_F, zorder=3)
            topstack = topstack+Fseries
        if(any(Eseries) and plot_E=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, topstack+Eseries), topstack, color=color_E, alpha=0.5, label='$E$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, topstack+Eseries),           color=color_E, zorder=3)
            topstack = topstack+Eseries
        if(combine_D and plot_D_E=='stacked' and plot_D_I=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, topstack+Dseries), topstack, color=color_D_E, alpha=0.5, label='$D_{all}$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, topstack+Dseries),           color=color_D_E, zorder=3)
            topstack = topstack+Dseries
        else:
            if(any(D_Eseries) and plot_D_E=='stacked'):
                ax.fill_between(numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, topstack+D_Eseries), topstack, color=color_D_E, alpha=0.5, label='$D_E$', zorder=2)
                ax.plot(        numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, topstack+D_Eseries),           color=color_D_E, zorder=3)
                topstack = topstack+D_Eseries
            if(any(D_Iseries) and plot_D_I=='stacked'):
                ax.fill_between(numpy.ma.masked_where(D_Iseries<=0, self.tseries), numpy.ma.masked_where(D_Iseries<=0, topstack+D_Iseries), topstack, color=color_D_I, alpha=0.5, label='$D_I$', zorder=2)
                ax.plot(        numpy.ma.masked_where(D_Iseries<=0, self.tseries), numpy.ma.masked_where(D_Iseries<=0, topstack+D_Iseries),           color=color_D_I, zorder=3)
                topstack = topstack+D_Iseries
        if(any(Iseries) and plot_I=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, topstack+Iseries), topstack, color=color_I, alpha=0.5, label='$I$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, topstack+Iseries),           color=color_I, zorder=3)
            topstack = topstack+Iseries
        if(any(Rseries) and plot_R=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, topstack+Rseries), topstack, color=color_R, alpha=0.5, label='$R$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, topstack+Rseries),           color=color_R, zorder=3)
            topstack = topstack+Rseries
        if(any(Sseries) and plot_S=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, topstack+Sseries), topstack, color=color_S, alpha=0.5, label='$S$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, topstack+Sseries),           color=color_S, zorder=3)
            topstack = topstack+Sseries
        

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the shaded variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(any(Fseries) and plot_F=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, Fseries), 0, color=color_F, alpha=0.5, label='$F$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, Fseries),    color=color_F, zorder=5)
        if(any(Eseries) and plot_E=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, Eseries), 0, color=color_E, alpha=0.5, label='$E$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, Eseries),    color=color_E, zorder=5)
        if(combine_D and (any(Dseries) and plot_D_E=='shaded' and plot_D_I=='shaded')):
            ax.fill_between(numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, Dseries), 0, color=color_D_E, alpha=0.5, label='$D_{all}$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, Dseries),    color=color_D_E, zorder=5)
        else:
            if(any(D_Eseries) and plot_D_E=='shaded'):
                ax.fill_between(numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, D_Eseries), 0, color=color_D_E, alpha=0.5, label='$D_E$', zorder=4)
                ax.plot(        numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, D_Eseries),    color=color_D_E, zorder=5)
            if(any(D_Iseries) and plot_D_I=='shaded'):
                ax.fill_between(numpy.ma.masked_where(D_Iseries<=0, self.tseries), numpy.ma.masked_where(D_Iseries<=0, D_Iseries), 0, color=color_D_I, alpha=0.5, label='$D_I$', zorder=4)
                ax.plot(        numpy.ma.masked_where(D_Iseries<=0, self.tseries), numpy.ma.masked_where(D_Iseries<=0, D_Iseries),    color=color_D_I, zorder=5)
        if(any(Iseries) and plot_I=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, Iseries), 0, color=color_I, alpha=0.5, label='$I$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, Iseries),    color=color_I, zorder=5)
        if(any(Sseries) and plot_S=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, Sseries), 0, color=color_S, alpha=0.5, label='$S$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, Sseries),    color=color_S, zorder=5)
        if(any(Rseries) and plot_R=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, Rseries), 0, color=color_R, alpha=0.5, label='$R$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, Rseries),    color=color_R, zorder=5)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the line variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
        if(any(Fseries) and plot_F=='line'):
            ax.plot(numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, Fseries), color=color_F, label='$F$', zorder=6)
        if(any(Eseries) and plot_E=='line'):
            ax.plot(numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, Eseries), color=color_E, label='$E$', zorder=6)
        if(combine_D and (any(Dseries) and plot_D_E=='line' and plot_D_I=='line')):
            ax.plot(numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, Dseries), color=color_D_E, label='$D_{all}$', zorder=6)
        else:
            if(any(D_Eseries) and plot_D_E=='line'):
                ax.plot(numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, D_Eseries), color=color_D_E, label='$D_E$', zorder=6)
            if(any(D_Iseries) and plot_D_I=='line'):
                ax.plot(numpy.ma.masked_where(D_Iseries<=0, self.tseries), numpy.ma.masked_where(D_Iseries<=0, D_Iseries), color=color_D_I, label='$D_I$', zorder=6)
        if(any(Iseries) and plot_I=='line'):
            ax.plot(numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, Iseries), color=color_I, label='$I$', zorder=6)
        if(any(Sseries) and plot_S=='line'):
            ax.plot(numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, Sseries), color=color_S, label='$S$', zorder=6)
        if(any(Rseries) and plot_R=='line'):
            ax.plot(numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, Rseries), color=color_R, label='$R$', zorder=6)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the vertical line annotations:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(len(vlines)>0 and len(vline_colors)==0):
            vline_colors = ['gray']*len(vlines)
        if(len(vlines)>0 and len(vline_labels)==0):
            vline_labels = [None]*len(vlines)
        if(len(vlines)>0 and len(vline_styles)==0):
            vline_styles = [':']*len(vlines)
        for vline_x, vline_color, vline_style, vline_label in zip(vlines, vline_colors, vline_styles, vline_labels):
            if(vline_x is not None):
                ax.axvline(x=vline_x, color=vline_color, linestyle=vline_style, alpha=1, label=vline_label)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the plot labels:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ax.set_xlabel('days')
        ax.set_ylabel('percent of population' if plot_percentages else 'number of individuals')
        ax.set_xlim(0, (max(self.tseries) if not xlim else xlim))
        ax.set_ylim(0, ylim)
        if(plot_percentages):
            ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])
        if(legend):
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend(legend_handles[::-1], legend_labels[::-1], loc='upper right', facecolor='white', edgecolor='none', framealpha=0.9, prop={'size': 8})
        if(title):
            ax.set_title(title, size=12)
        if(side_title):
            ax.annotate(side_title, (0, 0.5), xytext=(-45, 0), ha='right', va='center',
                size=12, rotation=90, xycoords='axes fraction', textcoords='offset points')
       
        return ax


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def figure_basic(self, plot_S='line', plot_E='line', plot_I='line',plot_R='line', plot_F='line',
                        plot_D_E='line', plot_D_I='line', combine_D=True,
                        color_S='tab:green', color_E='orange', color_I='crimson', color_R='tab:blue', color_F='black',
                        color_D_E='mediumorchid', color_D_I='mediumorchid', color_reference='#E0E0E0',
                        dashed_reference_results=None, dashed_reference_label='reference', 
                        shaded_reference_results=None, shaded_reference_label='reference', 
                        vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                        ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True,
                        figsize=(12,8), use_seaborn=True, show=True):

        import matplotlib.pyplot as pyplot

        fig, ax = pyplot.subplots(figsize=figsize)

        if(use_seaborn):
            import seaborn
            seaborn.set_style('ticks')
            seaborn.despine()

        self.plot(ax=ax, plot_S=plot_S, plot_E=plot_E, plot_I=plot_I,plot_R=plot_R, plot_F=plot_F,
                        plot_D_E=plot_D_E, plot_D_I=plot_D_I, combine_D=combine_D,
                        color_S=color_S, color_E=color_E, color_I=color_I, color_R=color_R, color_F=color_F,
                        color_D_E=color_D_E, color_D_I=color_D_I, color_reference=color_reference,
                        dashed_reference_results=dashed_reference_results, dashed_reference_label=dashed_reference_label, 
                        shaded_reference_results=shaded_reference_results, shaded_reference_label=shaded_reference_label, 
                        vlines=vlines, vline_colors=vline_colors, vline_styles=vline_styles, vline_labels=vline_labels,
                        ylim=ylim, xlim=xlim, legend=legend, title=title, side_title=side_title, plot_percentages=plot_percentages)

        if(show):
            pyplot.show()

        return fig, ax


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def figure_infections(self, plot_S=False, plot_E='stacked', plot_I='stacked',plot_R=False, plot_F=False,
                            plot_D_E='stacked', plot_D_I='stacked', combine_D=True,
                            color_S='tab:green', color_E='orange', color_I='crimson', color_R='tab:blue', color_F='black',
                            color_D_E='mediumorchid', color_D_I='mediumorchid', color_reference='#E0E0E0',
                            dashed_reference_results=None, dashed_reference_label='reference', 
                            shaded_reference_results=None, shaded_reference_label='reference', 
                            vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                            ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True,
                            figsize=(12,8), use_seaborn=True, show=True):

        import matplotlib.pyplot as pyplot

        fig, ax = pyplot.subplots(figsize=figsize)

        if(use_seaborn):
            import seaborn
            seaborn.set_style('ticks')
            seaborn.despine()

        self.plot(ax=ax, plot_S=plot_S, plot_E=plot_E, plot_I=plot_I,plot_R=plot_R, plot_F=plot_F,
                        plot_D_E=plot_D_E, plot_D_I=plot_D_I, combine_D=combine_D,
                        color_S=color_S, color_E=color_E, color_I=color_I, color_R=color_R, color_F=color_F,
                        color_D_E=color_D_E, color_D_I=color_D_I, color_reference=color_reference,
                        dashed_reference_results=dashed_reference_results, dashed_reference_label=dashed_reference_label, 
                        shaded_reference_results=shaded_reference_results, shaded_reference_label=shaded_reference_label, 
                        vlines=vlines, vline_colors=vline_colors, vline_styles=vline_styles, vline_labels=vline_labels, 
                        ylim=ylim, xlim=xlim, legend=legend, title=title, side_title=side_title, plot_percentages=plot_percentages)

        if(show):
            pyplot.show()

        return fig, ax


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




########################################################
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#@                                                    @#
#@  EXTENDED SEIRS MODELS                             @#
#@                                                    @#
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
########################################################

class SymptomaticSEIRSNetworkModel():
    """
    A class to simulate the SEIRS Stochastic Network Model
    with Symptom Presentation Compartments
    ===================================================
    Params: 
            G               Network adjacency matrix (numpy array) or Networkx graph object.
            beta            Rate of transmission (global interactions)
            beta_local      Rate(s) of transmission between adjacent individuals (optional)
            beta_A          Rate of transmission (global interactions)
            beta_A_local    Rate(s) of transmission between adjacent individuals (optional)
            sigma           Rate of progression to infectious state (inverse of latent period)             
            lamda           Rate of progression to infectious (a)symptomatic state (inverse of prodromal period)               
            eta             Rate of progression to hospitalized state (inverse of onset-to-admission period)           
            gamma           Rate of recovery for non-hospitalized symptomatic individuals (inverse of symptomatic infectious period)           
            gamma_A         Rate of recovery for asymptomatic individuals (inverse of asymptomatic infectious period)              
            gamma_H         Rate of recovery for hospitalized symptomatic individuals (inverse of hospitalized infectious period)              
            mu_H            Rate of death for hospitalized individuals (inverse of admission-to-death period)              
            xi              Rate of re-susceptibility (upon recovery)
            mu_0            Rate of baseline death
            nu              Rate of baseline birth
            a               Probability of an infected individual remaining asymptomatic
            h               Probability of a symptomatic individual being hospitalized  
            f               Probability of death for hospitalized individuals (case fatality rate)                         
            p               Probability of individuals interacting with global population              
            
            Q               Quarantine adjacency matrix (numpy array) or Networkx graph object.
            beta_D          Rate of transmission for individuals with detected infections (global interactions)
            beta_D_local    Rate(s) of transmission (exposure) for adjacent individuals with detected infections (optional)              
            sigma_D         Rate of progression to infectious state for individuals with detected infections           
            lamda_D        Rate of progression to infectious (a)symptomatic state for individuals with detected infections            
            eta_D           Rate of progression to hospitalized state for individuals with detected infections             
            gamma_D_S       Rate of recovery for non-hospitalized symptomatic individuals for individuals with detected infections             
            gamma_D_A       Rate of recovery for asymptomatic individuals for individuals with detected infections             
            theta_E         Rate of random testing for exposed individuals             
            theta_pre       Rate of random testing for infectious pre-symptomatic individuals              
            theta_S         Rate of random testing for infectious symptomatic individuals              
            theta_A         Rate of random testing for infectious asymptomatic individuals             
            phi_E           Rate of testing when a close contact has tested positive for exposed individuals               
            phi_pre         Rate of testing when a close contact has tested positive for infectious pre-symptomatic individuals                
            phi_S           Rate of testing when a close contact has tested positive for infectious symptomatic individuals                
            phi_A           Rate of testing when a close contact has tested positive for infectious asymptomatic individuals               
            d_E             Probability of positive test for exposed individuals               
            d_pre           Probability of positive test for infectious pre-symptomatic individuals                
            d_S             Probability of positive test for infectious symptomatic individuals                
            d_A             Probability of positive test for infectious asymptomatic individuals               
            q               Probability of individuals with detected infection interacting with global population              
            
            initE           Initial number of exposed individuals
            initI_pre       Initial number of infectious pre-symptomatic individuals
            initI_S         Initial number of infectious symptomatic individuals
            initI_A         Initial number of infectious asymptomatic individuals
            initH           Initial number of hospitalized individuals
            initR           Initial number of recovered individuals     
            initF           Initial number of infection-related fatalities
            initD_E         Initial number of detected exposed individuals
            initD_pre     Initial number of detected infectious pre-symptomatic individuals
            initD_S       Initial number of detected infectious symptomatic individuals
            initD_A       Initial number of detected infectious asymptomatic individuals
                            (all remaining nodes initialized susceptible)   
    """
    def __init__(self, G, beta, sigma, lamda, gamma, 
                    eta=0, gamma_A=None, gamma_H=None, mu_H=0, xi=0, mu_0=0, nu=0, a=0, h=0, f=0, p=0,             
                    beta_local=None, beta_A=None, beta_A_local=None,
                    Q=None, lamda_D=None, beta_D=None, beta_D_local=None, sigma_D=None, eta_D=None, gamma_D_S=None, gamma_D_A=None, 
                    theta_E=0, theta_pre=0, theta_S=0, theta_A=0, phi_E=0, phi_pre=0, phi_S=0, phi_A=0,    
                    d_E=1, d_pre=1, d_S=1, d_A=1, q=0,
                    initE=0, initI_pre=0, initI_S=0, initI_A=0, initH=0, initR=0, initF=0,        
                    initD_E=0, initD_pre=0, initD_S=0, initD_A=0,
                    node_groups=None, store_Xseries=False):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Setup Adjacency matrix:
        self.update_G(G)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Setup Quarantine Adjacency matrix:
        if(Q is None):
            Q = G # If no Q graph is provided, use G in its place
        self.update_Q(Q)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model Parameters:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        


        self.parameters = { 'beta':beta, 'sigma':sigma, 'lamda':lamda, 'gamma':gamma, 
                            'eta':eta, 'gamma_A':gamma_A, 'gamma_H':gamma_H, 'mu_H':mu_H, 
                            'xi':xi, 'mu_0':mu_0, 'nu':nu, 'a':a, 'h':h, 'f':f, 'p':p, 
                            'beta_local':beta_local, 'beta_A':beta_A, 'beta_A_local':beta_A_local, 
                            'lamda_D':lamda_D, 'beta_D':beta_D, 'beta_D_local':beta_D_local, 'sigma_D':sigma_D, 
                            'eta_D':eta_D, 'gamma_D_S':gamma_D_S, 'gamma_D_A':gamma_D_A, 
                            'theta_E':theta_E, 'theta_pre':theta_pre, 'theta_S':theta_S, 'theta_A':theta_A, 
                            'phi_E':phi_E, 'phi_pre':phi_pre, 'phi_S':phi_S, 'phi_A':phi_A, 
                            'd_E':d_E, 'd_pre':d_pre, 'd_S':d_S, 'd_A':d_A, 'q':q, 
                            'initE':initE, 'initI_pre':initI_pre, 'initI_S':initI_S, 'initI_A':initI_A, 
                            'initH':initH, 'initR':initR, 'initF':initF, 
                            'initD_E':initD_E, 'initD_pre':initD_pre, 'initD_S':initD_S, 'initD_A':initD_A }
        self.update_parameters()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Each node can undergo 4-6 transitions (sans vitality/re-susceptibility returns to S state),
        # so there are ~numNodes*6 events/timesteps expected; initialize numNodes*6 timestep slots to start 
        # (will be expanded during run if needed for some reason)
        self.tseries    = numpy.zeros(5*self.numNodes)
        self.numS       = numpy.zeros(5*self.numNodes)
        self.numE       = numpy.zeros(5*self.numNodes)
        self.numI_pre   = numpy.zeros(5*self.numNodes)
        self.numI_S     = numpy.zeros(5*self.numNodes)
        self.numI_A     = numpy.zeros(5*self.numNodes)
        self.numH       = numpy.zeros(5*self.numNodes)
        self.numR       = numpy.zeros(5*self.numNodes)
        self.numF       = numpy.zeros(5*self.numNodes)
        self.numD_E     = numpy.zeros(5*self.numNodes)
        self.numD_pre   = numpy.zeros(5*self.numNodes)
        self.numD_S     = numpy.zeros(5*self.numNodes)
        self.numD_A     = numpy.zeros(5*self.numNodes)
        self.N          = numpy.zeros(5*self.numNodes)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Timekeeping:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.t          = 0
        self.tmax       = 0 # will be set when run() is called
        self.tidx       = 0
        self.tseries[0] = 0
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Counts of inidividuals with each state:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.numE[0]        = int(initE)
        self.numI_pre[0]    = int(initI_pre)
        self.numI_S[0]      = int(initI_S)
        self.numI_A[0]      = int(initI_A)
        self.numH[0]        = int(initH)
        self.numR[0]        = int(initR)
        self.numF[0]        = int(initF)
        self.numD_E[0]      = int(initD_E)
        self.numD_pre[0]    = int(initD_pre)
        self.numD_S[0]      = int(initD_S)
        self.numD_A[0]      = int(initD_A)
        self.numS[0]        = (self.numNodes - self.numE[0] - self.numI_pre[0] - self.numI_S[0] - self.numI_A[0] - self.numH[0] - self.numR[0] 
                                             - self.numD_E[0] - self.numD_pre[0] - self.numD_S[0] - self.numD_A[0] - self.numF[0])
        self.N[0]           = self.numNodes - self.numF[0]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Node states:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.S          = 1
        self.E          = 2
        self.I_pre      = 3
        self.I_S        = 4
        self.I_A        = 5
        self.H          = 6
        self.R          = 7
        self.F          = 8
        self.D_E        = 9
        self.D_pre      = 10
        self.D_S        = 11
        self.D_A        = 12
        
        self.X = numpy.array( [self.S]*int(self.numS[0]) + [self.E]*int(self.numE[0]) 
                               + [self.I_pre]*int(self.numI_pre[0]) + [self.I_S]*int(self.numI_S[0]) + [self.I_A]*int(self.numI_A[0]) 
                               + [self.H]*int(self.numH[0]) + [self.R]*int(self.numR[0]) + [self.F]*int(self.numF[0])
                               + [self.D_E]*int(self.numD_E[0]) + [self.D_pre]*int(self.numD_pre[0]) + [self.D_S]*int(self.numD_S[0]) + [self.D_A]*int(self.numD_A[0]) 
                            ).reshape((self.numNodes,1))
        numpy.random.shuffle(self.X)

        self.store_Xseries = store_Xseries
        if(store_Xseries):
            self.Xseries        = numpy.zeros(shape=(5*self.numNodes, self.numNodes), dtype='uint8')
            self.Xseries[0,:]   = self.X.T

        self.transitions =  { 
                                'StoE':         {'currentState':self.S,     'newState':self.E},
                                'EtoIPRE':      {'currentState':self.E,     'newState':self.I_pre},
                                'EtoDE':        {'currentState':self.E,     'newState':self.D_E},
                                'IPREtoIS':     {'currentState':self.I_pre, 'newState':self.I_S},
                                'IPREtoIA':     {'currentState':self.I_pre, 'newState':self.I_A},
                                'IPREtoDPRE':   {'currentState':self.I_pre, 'newState':self.D_pre},
                                'IStoH':        {'currentState':self.I_S,   'newState':self.H},
                                'IStoR':        {'currentState':self.I_S,   'newState':self.R},
                                'IStoDS':       {'currentState':self.I_S,   'newState':self.D_S},
                                'IAtoR':        {'currentState':self.I_A,   'newState':self.R},
                                'IAtoDA':       {'currentState':self.I_A,   'newState':self.D_A},
                                'HtoR':         {'currentState':self.H,     'newState':self.R},
                                'HtoF':         {'currentState':self.H,     'newState':self.F},
                                'RtoS':         {'currentState':self.R,     'newState':self.S},
                                'DEtoDPRE':     {'currentState':self.D_E,   'newState':self.D_pre},
                                'DPREtoDS':     {'currentState':self.D_pre, 'newState':self.D_S},
                                'DPREtoDA':     {'currentState':self.D_pre, 'newState':self.D_A},
                                'DStoH':        {'currentState':self.D_S,   'newState':self.H},
                                'DStoR':        {'currentState':self.D_S,   'newState':self.R},
                                'DAtoR':        {'currentState':self.D_A,   'newState':self.R},
                                '_toS':         {'currentState':True,       'newState':self.S},
                            }

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize node subgroup data series:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.nodeGroupData = None
        if(node_groups):
            self.nodeGroupData = {}
            for groupName, nodeList in node_groups.items():
                self.nodeGroupData[groupName] = {'nodes':   numpy.array(nodeList),
                                                 'mask':    numpy.isin(range(self.numNodes), nodeList).reshape((self.numNodes,1))}
                self.nodeGroupData[groupName]['numS']           = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numE']           = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numI_pre']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numI_S']         = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numI_A']         = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numH']           = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numR']           = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numF']           = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numD_E']         = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numD_pre']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numD_S']         = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numD_A']         = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['N']              = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numS'][0]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.S)
                self.nodeGroupData[groupName]['numE'][0]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.E)
                self.nodeGroupData[groupName]['numI_pre'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I_pre)
                self.nodeGroupData[groupName]['numI_S'][0]      = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I_S)
                self.nodeGroupData[groupName]['numI_A'][0]      = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I_A)
                self.nodeGroupData[groupName]['numH'][0]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.H)
                self.nodeGroupData[groupName]['numR'][0]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.R)
                self.nodeGroupData[groupName]['numF'][0]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.F)
                self.nodeGroupData[groupName]['numD_E'][0]      = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.D_E)
                self.nodeGroupData[groupName]['numD_pre'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.D_pre)
                self.nodeGroupData[groupName]['numD_I_S'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.D_I_S)
                self.nodeGroupData[groupName]['numD_I_A'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.D_I_A)
                self.nodeGroupData[groupName]['N'][0]           = self.numNodes - self.numF[0]

         
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def update_parameters(self):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model parameters:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.beta           = numpy.array(self.parameters['beta']).reshape((self.numNodes, 1))      if isinstance(self.parameters['beta'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['beta'], shape=(self.numNodes,1))
        self.beta_A         = (numpy.array(self.parameters['beta_A']).reshape((self.numNodes, 1))   if isinstance(self.parameters['beta_A'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['beta_A'], shape=(self.numNodes,1))) if self.parameters['beta_A'] is not None else self.beta
        self.sigma          = numpy.array(self.parameters['sigma']).reshape((self.numNodes, 1))     if isinstance(self.parameters['sigma'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['sigma'], shape=(self.numNodes,1))
        self.lamda          = numpy.array(self.parameters['lamda']).reshape((self.numNodes, 1))     if isinstance(self.parameters['lamda'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['lamda'], shape=(self.numNodes,1))
        self.gamma          = numpy.array(self.parameters['gamma']).reshape((self.numNodes, 1))     if isinstance(self.parameters['gamma'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['gamma'], shape=(self.numNodes,1))
        self.eta            = numpy.array(self.parameters['eta']).reshape((self.numNodes, 1))       if isinstance(self.parameters['eta'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['eta'], shape=(self.numNodes,1))
        self.gamma_A      = (numpy.array(self.parameters['gamma_A']).reshape((self.numNodes, 1))if isinstance(self.parameters['gamma_A'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['gamma_A'], shape=(self.numNodes,1))) if self.parameters['gamma_A'] is not None else self.gamma
        self.gamma_H      = (numpy.array(self.parameters['gamma_H']).reshape((self.numNodes, 1))if isinstance(self.parameters['gamma_H'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['gamma_H'], shape=(self.numNodes,1))) if self.parameters['gamma_H'] is not None else self.gamma
        self.mu_H           = numpy.array(self.parameters['mu_H']).reshape((self.numNodes, 1))      if isinstance(self.parameters['mu_H'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['mu_H'], shape=(self.numNodes,1))
        self.xi             = numpy.array(self.parameters['xi']).reshape((self.numNodes, 1))        if isinstance(self.parameters['xi'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['xi'], shape=(self.numNodes,1))
        self.mu_0           = numpy.array(self.parameters['mu_0']).reshape((self.numNodes, 1))      if isinstance(self.parameters['mu_0'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['mu_0'], shape=(self.numNodes,1))
        self.nu             = numpy.array(self.parameters['nu']).reshape((self.numNodes, 1))        if isinstance(self.parameters['nu'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['nu'], shape=(self.numNodes,1))
        self.a              = numpy.array(self.parameters['a']).reshape((self.numNodes, 1))         if isinstance(self.parameters['a'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['a'], shape=(self.numNodes,1))
        self.h              = numpy.array(self.parameters['h']).reshape((self.numNodes, 1))         if isinstance(self.parameters['h'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['h'], shape=(self.numNodes,1))
        self.f              = numpy.array(self.parameters['f']).reshape((self.numNodes, 1))         if isinstance(self.parameters['f'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['f'], shape=(self.numNodes,1))
        self.p              = numpy.array(self.parameters['p']).reshape((self.numNodes, 1))         if isinstance(self.parameters['p'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['p'], shape=(self.numNodes,1))
       
        # Testing-related parameters:
        self.beta_D         = (numpy.array(self.parameters['beta_D']).reshape((self.numNodes, 1))   if isinstance(self.parameters['beta_D'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['beta_D'], shape=(self.numNodes,1))) if self.parameters['beta_D'] is not None else self.beta
        self.sigma_D        = (numpy.array(self.parameters['sigma_D']).reshape((self.numNodes, 1))  if isinstance(self.parameters['sigma_D'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['sigma_D'], shape=(self.numNodes,1))) if self.parameters['sigma_D'] is not None else self.sigma
        self.lamda_D        = (numpy.array(self.parameters['lamda_D']).reshape((self.numNodes, 1))  if isinstance(self.parameters['lamda_D'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['lamda_D'], shape=(self.numNodes,1))) if self.parameters['lamda_D'] is not None else self.lamda
        self.gamma_D_S      = (numpy.array(self.parameters['gamma_D_S']).reshape((self.numNodes, 1))if isinstance(self.parameters['gamma_D_S'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['gamma_D_S'], shape=(self.numNodes,1))) if self.parameters['gamma_D_S'] is not None else self.gamma
        self.gamma_D_A      = (numpy.array(self.parameters['gamma_D_A']).reshape((self.numNodes, 1))if isinstance(self.parameters['gamma_D_A'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['gamma_D_A'], shape=(self.numNodes,1))) if self.parameters['gamma_D_A'] is not None else self.gamma
        self.eta_D          = (numpy.array(self.parameters['eta_D']).reshape((self.numNodes, 1))    if isinstance(self.parameters['eta_D'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['eta_D'], shape=(self.numNodes,1))) if self.parameters['eta_D'] is not None else self.eta
        self.theta_E        = numpy.array(self.parameters['theta_E']).reshape((self.numNodes, 1))   if isinstance(self.parameters['theta_E'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['theta_E'], shape=(self.numNodes,1))
        self.theta_pre      = numpy.array(self.parameters['theta_pre']).reshape((self.numNodes, 1)) if isinstance(self.parameters['theta_pre'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['theta_pre'], shape=(self.numNodes,1))
        self.theta_S        = numpy.array(self.parameters['theta_S']).reshape((self.numNodes, 1))   if isinstance(self.parameters['theta_S'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['theta_S'], shape=(self.numNodes,1))
        self.theta_A        = numpy.array(self.parameters['theta_A']).reshape((self.numNodes, 1))   if isinstance(self.parameters['theta_A'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['theta_A'], shape=(self.numNodes,1))
        self.phi_E          = numpy.array(self.parameters['phi_E']).reshape((self.numNodes, 1))     if isinstance(self.parameters['phi_E'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['phi_E'], shape=(self.numNodes,1))
        self.phi_pre        = numpy.array(self.parameters['phi_pre']).reshape((self.numNodes, 1))   if isinstance(self.parameters['phi_pre'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['phi_pre'], shape=(self.numNodes,1))
        self.phi_S          = numpy.array(self.parameters['phi_S']).reshape((self.numNodes, 1))     if isinstance(self.parameters['phi_S'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['phi_S'], shape=(self.numNodes,1))
        self.phi_A          = numpy.array(self.parameters['phi_A']).reshape((self.numNodes, 1))     if isinstance(self.parameters['phi_A'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['phi_A'], shape=(self.numNodes,1))
        self.d_E            = numpy.array(self.parameters['d_E']).reshape((self.numNodes, 1))       if isinstance(self.parameters['d_E'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['d_E'], shape=(self.numNodes,1))
        self.d_pre          = numpy.array(self.parameters['d_pre']).reshape((self.numNodes, 1))     if isinstance(self.parameters['d_pre'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['d_pre'], shape=(self.numNodes,1))
        self.d_S            = numpy.array(self.parameters['d_S']).reshape((self.numNodes, 1))       if isinstance(self.parameters['d_S'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['d_S'], shape=(self.numNodes,1))
        self.d_A            = numpy.array(self.parameters['d_A']).reshape((self.numNodes, 1))       if isinstance(self.parameters['d_A'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['d_A'], shape=(self.numNodes,1))
        self.q              = numpy.array(self.parameters['q']).reshape((self.numNodes, 1))         if isinstance(self.parameters['q'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['q'], shape=(self.numNodes,1))

        #Local transmission parameters:
        if(self.parameters['beta_local'] is not None):
            if(isinstance(self.parameters['beta_local'], (list, numpy.ndarray))):
                if(isinstance(self.parameters['beta_local'], list)):
                    self.beta_local = numpy.array(self.parameters['beta_local'])
                else: # is numpy.ndarray
                    self.beta_local = self.parameters['beta_local']
                if(self.beta_local.ndim == 1):
                    self.beta_local.reshape((self.numNodes, 1))
                elif(self.beta_local.ndim == 2):
                    self.beta_local.reshape((self.numNodes, self.numNodes))
            else:
                self.beta_local = numpy.full_like(self.beta, fill_value=self.parameters['beta_local'])
        else:
            self.beta_local = self.beta
        #----------------------------------------
        if(self.parameters['beta_A_local'] is not None):
            if(isinstance(self.parameters['beta_A_local'], (list, numpy.ndarray))):
                if(isinstance(self.parameters['beta_A_local'], list)):
                    self.beta_A_local = numpy.array(self.parameters['beta_A_local'])
                else: # is numpy.ndarray
                    self.beta_A_local = self.parameters['beta_A_local']
                if(self.beta_A_local.ndim == 1):
                    self.beta_A_local.reshape((self.numNodes, 1))
                elif(self.beta_A_local.ndim == 2):
                    self.beta_A_local.reshape((self.numNodes, self.numNodes))
            else:
                self.beta_A_local = numpy.full_like(self.beta_A, fill_value=self.parameters['beta_A_local'])
        else:
            self.beta_A_local = self.beta_A
        #----------------------------------------
        if(self.parameters['beta_D_local'] is not None):
            if(isinstance(self.parameters['beta_D_local'], (list, numpy.ndarray))):
                if(isinstance(self.parameters['beta_D_local'], list)):
                    self.beta_D_local = numpy.array(self.parameters['beta_D_local'])
                else: # is numpy.ndarray
                    self.beta_D_local = self.parameters['beta_D_local']
                if(self.beta_D_local.ndim == 1):
                    self.beta_D_local.reshape((self.numNodes, 1))
                elif(self.beta_D_local.ndim == 2):
                    self.beta_D_local.reshape((self.numNodes, self.numNodes))
            else:
                self.beta_D_local = numpy.full_like(self.beta_D, fill_value=self.parameters['beta_D_local'])
        else:
            self.beta_D_local = self.beta_D
        
        # Pre-multiply beta values by the adjacency matrix ("transmission weight connections")
        if(self.beta_local.ndim == 1):
            self.A_beta     = scipy.sparse.csr_matrix.multiply(self.A, numpy.tile(self.beta_local, (1,self.numNodes))).tocsr()
        elif(self.beta_local.ndim == 2):
            self.A_beta     = scipy.sparse.csr_matrix.multiply(self.A, self.beta_local).tocsr()
        # Pre-multiply beta_A values by the adjacency matrix ("transmission weight connections")
        if(self.beta_A_local.ndim == 1):
            self.A_beta_A     = scipy.sparse.csr_matrix.multiply(self.A, numpy.tile(self.beta_A_local, (1,self.numNodes))).tocsr()
        elif(self.beta_A_local.ndim == 2):
            self.A_beta_A     = scipy.sparse.csr_matrix.multiply(self.A, self.beta_A_local).tocsr()
        # Pre-multiply beta_D values by the quarantine adjacency matrix ("transmission weight connections")
        if(self.beta_D_local.ndim == 1):
            self.A_Q_beta_D = scipy.sparse.csr_matrix.multiply(self.A_Q, numpy.tile(self.beta_D_local, (1,self.numNodes))).tocsr()
        elif(self.beta_D_local.ndim == 2):
            self.A_Q_beta_D = scipy.sparse.csr_matrix.multiply(self.A_Q, self.beta_D_local).tocsr()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update scenario flags:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.update_scenario_flags()

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def node_degrees(self, Amat):
        return Amat.sum(axis=0).reshape(self.numNodes,1)   # sums of adj matrix cols

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def update_G(self, new_G):
        self.G = new_G
        # Adjacency matrix:
        if type(new_G)==numpy.ndarray:
            self.A = scipy.sparse.csr_matrix(new_G)
        elif type(new_G)==networkx.classes.graph.Graph:
            self.A = networkx.adj_matrix(new_G) # adj_matrix gives scipy.sparse csr_matrix
        else:
            raise BaseException("Input an adjacency matrix or networkx object only.")

        self.numNodes   = int(self.A.shape[1])
        self.degree     = numpy.asarray(self.node_degrees(self.A)).astype(float)

        return

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def update_Q(self, new_Q):
        self.Q = new_Q
        # Quarantine Adjacency matrix:
        if type(new_Q)==numpy.ndarray:
            self.A_Q = scipy.sparse.csr_matrix(new_Q)
        elif type(new_Q)==networkx.classes.graph.Graph:
            self.A_Q = networkx.adj_matrix(new_Q) # adj_matrix gives scipy.sparse csr_matrix
        else:
            raise BaseException("Input an adjacency matrix or networkx object only.")

        self.numNodes_Q   = int(self.A_Q.shape[1])
        self.degree_Q     = numpy.asarray(self.node_degrees(self.A_Q)).astype(float)

        assert(self.numNodes == self.numNodes_Q), "The normal and quarantine adjacency graphs must be of the same size."

        return

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def update_scenario_flags(self):
        self.testing_scenario   = ( (numpy.any(self.d_E) and (numpy.any(self.theta_E) or numpy.any(self.phi_E)))
                                    or (numpy.any(self.d_pre) and (numpy.any(self.theta_pre) or numpy.any(self.phi_pre)))
                                    or (numpy.any(self.d_S) and (numpy.any(self.theta_S) or numpy.any(self.phi_S)))
                                    or (numpy.any(self.d_A) and (numpy.any(self.theta_A) or numpy.any(self.phi_A))) )
        self.tracing_scenario   = ( (numpy.any(self.d_E) and numpy.any(self.phi_E)) 
                                    or (numpy.any(self.d_pre) and numpy.any(self.phi_pre))
                                    or (numpy.any(self.d_S) and numpy.any(self.phi_S))
                                    or (numpy.any(self.d_A) and numpy.any(self.phi_A)) )
        self.vitality_scenario  = (numpy.any(self.mu_0) and numpy.any(self.nu))
        self.resusceptibility_scenario  = (numpy.any(self.xi))


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def total_num_infections(self, t_idx=None):
        if(t_idx is None):
            return (self.numE[:] + self.numI_pre[:] + self.numI_S[:] + self.numI_A[:] 
                    + self.numD_E[:] + self.numD_pre[:] + self.numD_S[:] + self.numD_A[:] + self.numH[:])            
        else:
            return (self.numE[t_idx] + self.numI_pre[t_idx] + self.numI_S[t_idx] + self.numI_A[t_idx] 
                    + self.numD_E[t_idx] + self.numD_pre[t_idx] + self.numD_S[t_idx] + self.numD_A[t_idx] + self.numH[t_idx])

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def total_num_detected(self, t_idx=None):
        if(t_idx is None):
            return (self.numD_E[:] + self.numD_pre[:] + self.numD_S[:] + self.numD_A[:])            
        else:
            return (self.numD_E[t_idx] + self.numD_pre[t_idx] + self.numD_S[t_idx] + self.numD_A[t_idx])


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
    def calc_propensities(self):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-calculate matrix multiplication terms that may be used in multiple propensity calculations,
        # and check to see if their computation is necessary before doing the multiplication
        transmissionTerms_I = numpy.zeros(shape=(self.numNodes,1))
        if( (numpy.any(self.numI_S[self.tidx]) and self.A_beta.count_nonzero()>0)
            or ((numpy.any(self.numI_pre[self.tidx]) or numpy.any(self.numI_A[self.tidx])) and self.A_beta_A.count_nonzero()>0) ):
            transmissionTerms_I = numpy.asarray( scipy.sparse.csr_matrix.dot( self.A_beta, self.X==self.I_S )
                                                 + scipy.sparse.csr_matrix.dot( self.A_beta_A, ((self.X==self.I_pre)|(self.X==self.I_A)) ) )

        transmissionTerms_D = numpy.zeros(shape=(self.numNodes,1))
        if(self.testing_scenario 
            and (numpy.any(self.numD_pre[self.tidx]) or numpy.any(self.numD_S[self.tidx]) or numpy.any(self.numD_A[self.tidx]) or numpy.any(self.numH[self.tidx]))
            and self.A_Q_beta_D.count_nonzero()>0 ):
            transmissionTerms_D = numpy.asarray( scipy.sparse.csr_matrix.dot( self.A_Q_beta_D, ((self.X==self.D_pre)|(self.X==self.D_S)|(self.X==self.D_A)|(self.X==self.H)) ) )

        numContacts_D = numpy.zeros(shape=(self.numNodes,1))
        if(self.tracing_scenario 
            and (numpy.any(self.numD_E[self.tidx]) or numpy.any(self.numD_pre[self.tidx]) or numpy.any(self.numD_S[self.tidx]) or numpy.any(self.numD_A[self.tidx]) or numpy.any(self.numH[self.tidx])) ):
            numContacts_D = numpy.asarray( scipy.sparse.csr_matrix.dot( self.A, ((self.X==self.D_E)|(self.X==self.D_pre)|(self.X==self.D_S)|(self.X==self.D_A)|(self.X==self.H)) ) )

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        propensities_StoE       = ( self.p*((self.beta*self.numI_S[self.tidx] + self.beta_A*(self.numI_pre[self.tidx] + self.numI_A[self.tidx])  + self.q*self.beta_D*(self.numD_pre[self.tidx] + self.numD_S[self.tidx] + self.numD_A[self.tidx]))/self.N[self.tidx])
                                    + (1-self.p)*numpy.divide((transmissionTerms_I + transmissionTerms_D), self.degree, out=numpy.zeros_like(self.degree), where=self.degree!=0)
                                  )*(self.X==self.S)

        propensities_EtoIPRE    = self.sigma*(self.X==self.E)

        propensities_IPREtoIS   = (1-self.a)*self.lamda*(self.X==self.I_pre)

        propensities_IPREtoIA   = self.a*self.lamda*(self.X==self.I_pre)

        propensities_IStoR      = (1-self.h)*self.gamma*(self.X==self.I_S)

        propensities_IStoH      = self.h*self.eta*(self.X==self.I_S)

        propensities_IAtoR      = self.gamma_A*(self.X==self.I_A)

        propensities_HtoR       = (1-self.f)*self.gamma_H*(self.X==self.H)

        propensities_HtoF       = self.f*self.mu_H*(self.X==self.H)

        propensities_EtoDE      = (self.theta_E + self.phi_E*numContacts_D)*self.d_E*(self.X==self.E)

        propensities_IPREtoDPRE = (self.theta_pre + self.phi_pre*numContacts_D)*self.d_pre*(self.X==self.I_pre)

        propensities_IStoDS     = (self.theta_S + self.phi_S*numContacts_D)*self.d_S*(self.X==self.I_S)

        propensities_IAtoDA     = (self.theta_A + self.phi_A*numContacts_D)*self.d_A*(self.X==self.I_A)

        propensities_DEtoDPRE   = self.sigma_D*(self.X==self.D_E)

        propensities_DPREtoDS   = (1-self.a)*self.lamda_D*(self.X==self.D_pre)

        propensities_DPREtoDA   = self.a*self.lamda_D*(self.X==self.D_pre)

        propensities_DStoR      = (1-self.h)*self.gamma_D_S*(self.X==self.D_S)

        propensities_DStoH      = self.h*self.eta_D*(self.X==self.D_S)

        propensities_DAtoR      = self.gamma_D_A*(self.X==self.D_A)

        propensities_RtoS       = self.xi*(self.X==self.R)

        propensities__toS       = self.nu*(self.X!=self.F)

        propensities = numpy.hstack([propensities_StoE, propensities_EtoIPRE, propensities_IPREtoIS, propensities_IPREtoIA,
                                     propensities_IStoR, propensities_IStoH, propensities_IAtoR, propensities_HtoR, propensities_HtoF, 
                                     propensities_EtoDE, propensities_IPREtoDPRE, propensities_IStoDS, propensities_IAtoDA, 
                                     propensities_DEtoDPRE, propensities_DPREtoDS, propensities_DPREtoDA, propensities_DStoR, propensities_DStoH, 
                                     propensities_DAtoR, propensities_RtoS, propensities__toS])

        columns = ['StoE', 'EtoIPRE', 'IPREtoIS', 'IPREtoIA', 'IStoR', 'IStoH', 'IAtoR', 'HtoR', 'HtoF', 
                   'EtoDE', 'IPREtoDPRE', 'IStoDS', 'IAtoDA', 'DEtoDPRE', 'DPREtoDS', 'DPREtoDA', 'DStoR', 'DStoH', 'DAtoR', 
                   'RtoS', '_toS']

        return propensities, columns


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    

    def increase_data_series_length(self):
        self.tseries    = numpy.pad(self.tseries, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numS       = numpy.pad(self.numS, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numE       = numpy.pad(self.numE, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numI_pre   = numpy.pad(self.numI_pre, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numI_S     = numpy.pad(self.numI_S, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numI_A     = numpy.pad(self.numI_A, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numH       = numpy.pad(self.numH, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numR       = numpy.pad(self.numR, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numF       = numpy.pad(self.numF, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numD_E     = numpy.pad(self.numD_E, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numD_pre   = numpy.pad(self.numD_pre, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numD_S     = numpy.pad(self.numD_S, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numD_A     = numpy.pad(self.numD_A, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.N          = numpy.pad(self.N, [(0, 5*self.numNodes)], mode='constant', constant_values=0)

        if(self.store_Xseries):
            self.Xseries = numpy.pad(self.Xseries, [(0, 5*self.numNodes), (0,0)], mode='constant', constant_values=0)

        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                self.nodeGroupData[groupName]['numS']       = numpy.pad(self.nodeGroupData[groupName]['numS'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numE']       = numpy.pad(self.nodeGroupData[groupName]['numE'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numI_pre']   = numpy.pad(self.nodeGroupData[groupName]['numI_pre'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numI_S']     = numpy.pad(self.nodeGroupData[groupName]['numI_S'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numI_A']     = numpy.pad(self.nodeGroupData[groupName]['numI_A'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numH']       = numpy.pad(self.nodeGroupData[groupName]['numH'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numR']       = numpy.pad(self.nodeGroupData[groupName]['numR'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numF']       = numpy.pad(self.nodeGroupData[groupName]['numF'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numD_E']     = numpy.pad(self.nodeGroupData[groupName]['numD_E'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numD_pre']   = numpy.pad(self.nodeGroupData[groupName]['numD_pre'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numD_S']     = numpy.pad(self.nodeGroupData[groupName]['numD_S'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numD_A']     = numpy.pad(self.nodeGroupData[groupName]['numD_A'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['N']          = numpy.pad(self.nodeGroupData[groupName]['N'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)

        return None

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 

    def finalize_data_series(self):
        self.tseries    = numpy.array(self.tseries, dtype=float)[:self.tidx+1]
        self.numS       = numpy.array(self.numS, dtype=float)[:self.tidx+1]
        self.numE       = numpy.array(self.numE, dtype=float)[:self.tidx+1]
        self.numI_pre   = numpy.array(self.numI_pre, dtype=float)[:self.tidx+1]
        self.numI_S     = numpy.array(self.numI_S, dtype=float)[:self.tidx+1]
        self.numI_A     = numpy.array(self.numI_A, dtype=float)[:self.tidx+1]
        self.numH       = numpy.array(self.numH, dtype=float)[:self.tidx+1]
        self.numR       = numpy.array(self.numR, dtype=float)[:self.tidx+1]
        self.numF       = numpy.array(self.numF, dtype=float)[:self.tidx+1]
        self.numD_E     = numpy.array(self.numD_E, dtype=float)[:self.tidx+1]
        self.numD_pre   = numpy.array(self.numD_pre, dtype=float)[:self.tidx+1]
        self.numD_S     = numpy.array(self.numD_S, dtype=float)[:self.tidx+1]
        self.numD_A     = numpy.array(self.numD_A, dtype=float)[:self.tidx+1]
        self.N          = numpy.array(self.N, dtype=float)[:self.tidx+1]

        if(self.store_Xseries):
            self.Xseries = self.Xseries[:self.tidx+1, :]

        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                self.nodeGroupData[groupName]['numS']       = numpy.array(self.nodeGroupData[groupName]['numS'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numE']       = numpy.array(self.nodeGroupData[groupName]['numE'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numI_pre']   = numpy.array(self.nodeGroupData[groupName]['numI_pre'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numI_S']     = numpy.array(self.nodeGroupData[groupName]['numI_S'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numI_A']     = numpy.array(self.nodeGroupData[groupName]['numI_A'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numR']       = numpy.array(self.nodeGroupData[groupName]['numR'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numF']       = numpy.array(self.nodeGroupData[groupName]['numF'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numD_E']     = numpy.array(self.nodeGroupData[groupName]['numD_E'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numD_pre']   = numpy.array(self.nodeGroupData[groupName]['numD_pre'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numD_S']     = numpy.array(self.nodeGroupData[groupName]['numD_S'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numD_A']     = numpy.array(self.nodeGroupData[groupName]['numD_A'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['N']          = numpy.array(self.nodeGroupData[groupName]['N'], dtype=float)[:self.tidx+1]

        return None

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^     

    def run_iteration(self):

        if(self.tidx >= len(self.tseries)-1):
            # Room has run out in the timeseries storage arrays; double the size of these arrays:
            self.increase_data_series_length()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1. Generate 2 random numbers uniformly distributed in (0,1)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        r1 = numpy.random.rand()
        r2 = numpy.random.rand()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2. Calculate propensities
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        propensities, transitionTypes = self.calc_propensities()

        # Terminate when probability of all events is 0:
        if(propensities.sum() <= 0.0):            
            self.finalize_data_series()
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3. Calculate alpha
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        propensities_flat   = propensities.ravel(order='F')
        cumsum              = propensities_flat.cumsum()
        alpha               = propensities_flat.sum()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 4. Compute the time until the next event takes place
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        tau = (1/alpha)*numpy.log(float(1/r1))
        self.t += tau

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 5. Compute which event takes place
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        transitionIdx   = numpy.searchsorted(cumsum,r2*alpha)
        transitionNode  = transitionIdx % self.numNodes
        transitionType  = transitionTypes[ int(transitionIdx/self.numNodes) ]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 6. Update node states and data series
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        assert(self.X[transitionNode] == self.transitions[transitionType]['currentState'] and self.X[transitionNode]!=self.F), "Assertion error: Node "+str(transitionNode)+" has unexpected current state "+str(self.X[transitionNode])+" given the intended transition of "+str(transitionType)+"."
        self.X[transitionNode] = self.transitions[transitionType]['newState']

        self.tidx += 1
        
        self.tseries[self.tidx]  = self.t
        self.numS[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.S), a_min=0, a_max=self.numNodes)
        self.numE[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.E), a_min=0, a_max=self.numNodes)
        self.numI_pre[self.tidx] = numpy.clip(numpy.count_nonzero(self.X==self.I_pre), a_min=0, a_max=self.numNodes)
        self.numI_S[self.tidx]   = numpy.clip(numpy.count_nonzero(self.X==self.I_S), a_min=0, a_max=self.numNodes)
        self.numI_A[self.tidx]   = numpy.clip(numpy.count_nonzero(self.X==self.I_A), a_min=0, a_max=self.numNodes)
        self.numH[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.H), a_min=0, a_max=self.numNodes)
        self.numR[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.R), a_min=0, a_max=self.numNodes)
        self.numF[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.F), a_min=0, a_max=self.numNodes)
        self.numD_E[self.tidx]   = numpy.clip(numpy.count_nonzero(self.X==self.D_E), a_min=0, a_max=self.numNodes)
        self.numD_pre[self.tidx] = numpy.clip(numpy.count_nonzero(self.X==self.D_pre), a_min=0, a_max=self.numNodes)
        self.numD_S[self.tidx]   = numpy.clip(numpy.count_nonzero(self.X==self.D_S), a_min=0, a_max=self.numNodes)
        self.numD_A[self.tidx]   = numpy.clip(numpy.count_nonzero(self.X==self.D_A), a_min=0, a_max=self.numNodes)
        
        self.N[self.tidx]        = numpy.clip((self.numNodes - self.numF[self.tidx]), a_min=0, a_max=self.numNodes)

        if(self.store_Xseries):
            self.Xseries[self.tidx,:] = self.X.T

        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                self.nodeGroupData[groupName]['numS'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.S)
                self.nodeGroupData[groupName]['numE'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.E)
                self.nodeGroupData[groupName]['numI_pre'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I_pre)
                self.nodeGroupData[groupName]['numI_S'][self.tidx]      = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I_S)
                self.nodeGroupData[groupName]['numI_A'][self.tidx]      = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I_A)
                self.nodeGroupData[groupName]['numH'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.H)
                self.nodeGroupData[groupName]['numR'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.R)
                self.nodeGroupData[groupName]['numF'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.F)
                self.nodeGroupData[groupName]['numD_E'][self.tidx]      = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.D_E)
                self.nodeGroupData[groupName]['numD_pre'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.D_pre)
                self.nodeGroupData[groupName]['numD_S'][self.tidx]      = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.D_S)
                self.nodeGroupData[groupName]['numD_A'][self.tidx]      = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.D_A)
                self.nodeGroupData[groupName]['N'][self.tidx]           = numpy.clip((self.nodeGroupData[groupName]['numS'][0] + self.nodeGroupData[groupName]['numE'][0] + self.nodeGroupData[groupName]['numI'][0] + self.nodeGroupData[groupName]['numD_E'][0] + self.nodeGroupData[groupName]['numD_I'][0] + self.nodeGroupData[groupName]['numR'][0]), a_min=0, a_max=self.numNodes)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Terminate if tmax reached or num infections is 0:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(self.t >= self.tmax or self.total_num_infections(self.tidx) < 1):
            self.finalize_data_series()
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        return True


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def run(self, T, checkpoints=None, print_interval=10, verbose='t'):
        if(T>0):
            self.tmax += T
        else:
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-process checkpoint values:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(checkpoints):
            numCheckpoints = len(checkpoints['t'])
            for chkpt_param, chkpt_values in checkpoints.items():
                assert(isinstance(chkpt_values, (list, numpy.ndarray)) and len(chkpt_values)==numCheckpoints), "Expecting a list of values with length equal to number of checkpoint times ("+str(numCheckpoints)+") for each checkpoint parameter."
            checkpointIdx  = numpy.searchsorted(checkpoints['t'], self.t) # Finds 1st index in list greater than given val
            if(checkpointIdx >= numCheckpoints):
                # We are out of checkpoints, stop checking them:
                checkpoints = None 
            else:
                checkpointTime = checkpoints['t'][checkpointIdx]

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Run the simulation loop:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        print_reset = True
        running     = True
        while running:

            running = self.run_iteration()

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Handle checkpoints if applicable:
            if(checkpoints):
                if(self.t >= checkpointTime):
                    if(verbose is not False):
                        print("[Checkpoint: Updating parameters]")
                    # A checkpoint has been reached, update param values:
                    if('G' in list(checkpoints.keys())):
                        self.update_G(checkpoints['G'][checkpointIdx])
                    if('Q' in list(checkpoints.keys())):
                        self.update_Q(checkpoints['Q'][checkpointIdx])
                    for param in list(self.parameters.keys()):
                        if(param in list(checkpoints.keys())):
                            self.parameters.update({param: checkpoints[param][checkpointIdx]})
                    # Update parameter data structures and scenario flags:
                    self.update_parameters()
                    # Update the next checkpoint time:
                    checkpointIdx  = numpy.searchsorted(checkpoints['t'], self.t) # Finds 1st index in list greater than given val
                    if(checkpointIdx >= numCheckpoints):
                        # We are out of checkpoints, stop checking them:
                        checkpoints = None 
                    else:
                        checkpointTime = checkpoints['t'][checkpointIdx]
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if(print_interval):
                if(print_reset and (int(self.t) % print_interval == 0)):
                    if(verbose=="t"):
                        print("t = %.2f" % self.t)
                    if(verbose==True):
                        print("t = %.2f" % self.t)
                        print("\t S     = " + str(self.numS[self.tidx]))
                        print("\t E     = " + str(self.numE[self.tidx]))
                        print("\t I_pre = " + str(self.numI_pre[self.tidx]))
                        print("\t I_S   = " + str(self.numI_S[self.tidx]))
                        print("\t I_A   = " + str(self.numI_A[self.tidx]))
                        print("\t H     = " + str(self.numH[self.tidx]))
                        print("\t R     = " + str(self.numR[self.tidx]))
                        print("\t F     = " + str(self.numF[self.tidx]))
                        print("\t D_E   = " + str(self.numD_E[self.tidx]))
                        print("\t D_pre = " + str(self.numD_pre[self.tidx]))
                        print("\t D_S   = " + str(self.numD_S[self.tidx]))
                        print("\t D_A   = " + str(self.numD_A[self.tidx]))
                        
                    print_reset = False
                elif(not print_reset and (int(self.t) % 10 != 0)):
                    print_reset = True

        return True


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def plot(self, ax=None, plot_S='line', plot_E='line', plot_I_pre='line', plot_I_S='line', plot_I_A='line',
                            plot_H='line', plot_R='line', plot_F='line',
                            plot_D_E='line', plot_D_pre='line', plot_D_S='line', plot_D_A='line', combine_D=True,
                            color_S='tab:green', color_E='orange', color_I_pre='tomato', color_I_S='crimson', color_I_A='crimson', 
                            color_H='violet', color_R='tab:blue', color_F='black',
                            color_D_E='mediumorchid', color_D_pre='mediumorchid', color_D_S='mediumorchid', color_D_A='mediumorchid', 
                            color_reference='#E0E0E0',
                            dashed_reference_results=None, dashed_reference_label='reference', 
                            shaded_reference_results=None, shaded_reference_label='reference', 
                            vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                            ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True):

        import matplotlib.pyplot as pyplot

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create an Axes object if None provided:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(not ax):
            fig, ax = pyplot.subplots()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prepare data series to be plotted:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Fseries     = self.numF/self.numNodes if plot_percentages else self.numF
        Dseries     = self.total_num_detected()/self.numNodes if plot_percentages else self.total_num_detected()
        D_Eseries   = self.numD_E/self.numNodes if plot_percentages else self.numD_E 
        D_preseries = self.numD_pre/self.numNodes if plot_percentages else self.numD_pre 
        D_Aseries   = self.numD_A/self.numNodes if plot_percentages else self.numD_A 
        D_Sseries   = self.numD_S/self.numNodes if plot_percentages else self.numD_S 
        Hseries     = self.numH/self.numNodes if plot_percentages else self.numH 
        Eseries     = self.numE/self.numNodes if plot_percentages else self.numE
        I_preseries = self.numI_pre/self.numNodes if plot_percentages else self.numI_pre 
        I_Sseries   = self.numI_S/self.numNodes if plot_percentages else self.numI_S
        I_Aseries   = self.numI_A/self.numNodes if plot_percentages else self.numI_A
        Rseries     = self.numR/self.numNodes if plot_percentages else self.numR 
        Sseries     = self.numS/self.numNodes if plot_percentages else self.numS

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the reference data:      
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(dashed_reference_results):
            dashedReference_tseries       = dashed_reference_results.tseries[::int(self.numNodes/100)]
            dashedReference_infectedStack = dashed_reference_results.total_num_infections()[::int(self.numNodes/100)] / (self.numNodes if plot_percentages else 1)
            ax.plot(dashedReference_tseries, dashedReference_infectedStack, color='#E0E0E0', linestyle='--', label='Total infections ('+dashed_reference_label+')', zorder=0)
        if(shaded_reference_results):
            shadedReference_tseries       = shaded_reference_results.tseries
            shadedReference_infectedStack = shaded_reference_results.total_num_infections() / (self.numNodes if plot_percentages else 1)
            ax.fill_between(shaded_reference_results.tseries, shadedReference_infectedStack, 0, color='#EFEFEF', label='Total infections ('+shaded_reference_label+')', zorder=0)
            ax.plot(shaded_reference_results.tseries, shadedReference_infectedStack, color='#E0E0E0', zorder=1)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the stacked variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        topstack = numpy.zeros_like(self.tseries)
        if(any(Fseries) and plot_F=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, topstack+Fseries), topstack, color=color_F, alpha=0.5, label='$F$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, topstack+Fseries),           color=color_F, zorder=3)
            topstack = topstack+Fseries
        if(any(Hseries) and plot_H=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Hseries<=0, self.tseries), numpy.ma.masked_where(Hseries<=0, topstack+Hseries), topstack, color=color_H, alpha=0.5, label='$H$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Hseries<=0, self.tseries), numpy.ma.masked_where(Hseries<=0, topstack+Hseries),           color=color_H, zorder=3)
            topstack = topstack+Hseries
        if(combine_D and any(Dseries) and plot_D_E=='stacked' and plot_D_pre=='stacked' and plot_D_S=='stacked' and plot_D_A=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, topstack+Dseries), topstack, color=color_D_S, alpha=0.5, label='$D_{all}$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, topstack+Dseries),           color=color_D_S, zorder=3)
            topstack = topstack+Dseries
        else:
            if(any(D_Eseries) and plot_D_E=='stacked'):
                ax.fill_between(numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, topstack+D_Eseries), topstack, color=color_D_E, alpha=0.5, label='$D_E$', zorder=2)
                ax.plot(        numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, topstack+D_Eseries),           color=color_D_E, zorder=3)
                topstack = topstack+D_Eseries
            if(any(D_preseries) and plot_D_pre=='stacked'):
                ax.fill_between(numpy.ma.masked_where(D_preseries<=0, self.tseries), numpy.ma.masked_where(D_preseries<=0, topstack+D_preseries), topstack, color=color_D_pre, alpha=0.5, label='$D_{pre}$', zorder=2)
                ax.plot(        numpy.ma.masked_where(D_preseries<=0, self.tseries), numpy.ma.masked_where(D_preseries<=0, topstack+D_preseries),           color=color_D_pre, zorder=3)
                topstack = topstack+D_preseries
            if(any(D_Sseries) and plot_D_S=='stacked'):
                ax.fill_between(numpy.ma.masked_where(D_Sseries<=0, self.tseries), numpy.ma.masked_where(D_Sseries<=0, topstack+D_Sseries), topstack, color=color_D_S, alpha=0.5, label='$D_S$', zorder=2)
                ax.plot(        numpy.ma.masked_where(D_Sseries<=0, self.tseries), numpy.ma.masked_where(D_Sseries<=0, topstack+D_Sseries),           color=color_D_S, zorder=3)
                topstack = topstack+D_Sseries
            if(any(D_Aseries) and plot_D_A=='stacked'):
                ax.fill_between(numpy.ma.masked_where(D_Aseries<=0, self.tseries), numpy.ma.masked_where(D_Aseries<=0, topstack+D_Aseries), topstack, color=color_D_A, alpha=0.5, label='$D_A$', zorder=2)
                ax.plot(        numpy.ma.masked_where(D_Aseries<=0, self.tseries), numpy.ma.masked_where(D_Aseries<=0, topstack+D_Aseries),           color=color_D_A, zorder=3)
                topstack = topstack+D_Aseries
        if(any(Eseries) and plot_E=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, topstack+Eseries), topstack, color=color_E, alpha=0.5, label='$E$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, topstack+Eseries),           color=color_E, zorder=3)
            topstack = topstack+Eseries
        if(any(I_preseries) and plot_I_pre=='stacked'):
            ax.fill_between(numpy.ma.masked_where(I_preseries<=0, self.tseries), numpy.ma.masked_where(I_preseries<=0, topstack+I_preseries), topstack, color=color_I_pre, alpha=0.5, label='$I_{pre}$', zorder=2)
            ax.plot(        numpy.ma.masked_where(I_preseries<=0, self.tseries), numpy.ma.masked_where(I_preseries<=0, topstack+I_preseries),           color=color_I_pre, zorder=3)
            topstack = topstack+I_preseries
        if(any(I_Sseries) and plot_I_S=='stacked'):
            ax.fill_between(numpy.ma.masked_where(I_Sseries<=0, self.tseries), numpy.ma.masked_where(I_Sseries<=0, topstack+I_Sseries), topstack, color=color_I_S, alpha=0.5, label='$I_S$', zorder=2)
            ax.plot(        numpy.ma.masked_where(I_Sseries<=0, self.tseries), numpy.ma.masked_where(I_Sseries<=0, topstack+I_Sseries),           color=color_I_S, zorder=3)
            topstack = topstack+I_Sseries
        if(any(I_Aseries) and plot_I_A=='stacked'):
            ax.fill_between(numpy.ma.masked_where(I_Aseries<=0, self.tseries), numpy.ma.masked_where(I_Aseries<=0, topstack+I_Aseries), topstack, color=color_I_A, alpha=0.25, label='$I_A$', zorder=2)
            ax.plot(        numpy.ma.masked_where(I_Aseries<=0, self.tseries), numpy.ma.masked_where(I_Aseries<=0, topstack+I_Aseries),           color=color_I_A, zorder=3)
            topstack = topstack+I_Aseries
        if(any(Rseries) and plot_R=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, topstack+Rseries), topstack, color=color_R, alpha=0.5, label='$R$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, topstack+Rseries),           color=color_R, zorder=3)
            topstack = topstack+Rseries
        if(any(Sseries) and plot_S=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, topstack+Sseries), topstack, color=color_S, alpha=0.5, label='$S$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, topstack+Sseries),           color=color_S, zorder=3)
            topstack = topstack+Sseries
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the shaded variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(any(Fseries) and plot_F=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, Fseries), 0, color=color_F, alpha=0.5, label='$F$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, Fseries),    color=color_F, zorder=5)
        if(any(Hseries) and plot_H=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Hseries<=0, self.tseries), numpy.ma.masked_where(Hseries<=0, Hseries), 0, color=color_H, alpha=0.5, label='$H$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Hseries<=0, self.tseries), numpy.ma.masked_where(Hseries<=0, Hseries),    color=color_H, zorder=5)
        if(combine_D and (any(Dseries) and plot_D_E=='shaded' and plot_D_pre=='shaded') and plot_D_S=='shaded' and plot_D_A=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, Dseries), 0, color=color_D_S, alpha=0.5, label='$D_{all}$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, Dseries),    color=color_D_S, zorder=5)
        else:
            if(any(D_Eseries) and plot_D_E=='shaded'):
                ax.fill_between(numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, D_Eseries), 0, color=color_D_E, alpha=0.5, label='$D_E$', zorder=4)
                ax.plot(        numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, D_Eseries),    color=color_D_E, zorder=5)
            if(any(D_preseries) and plot_D_pre=='shaded'):
                ax.fill_between(numpy.ma.masked_where(D_preseries<=0, self.tseries), numpy.ma.masked_where(D_preseries<=0, D_preseries), 0, color=color_D_pre, alpha=0.5, label='$D_{pre}$', zorder=4)
                ax.plot(        numpy.ma.masked_where(D_preseries<=0, self.tseries), numpy.ma.masked_where(D_preseries<=0, D_preseries),    color=color_D_pre, zorder=5)
            if(any(D_Sseries) and plot_D_S=='shaded'):
                ax.fill_between(numpy.ma.masked_where(D_Sseries<=0, self.tseries), numpy.ma.masked_where(D_Sseries<=0, D_Sseries), 0, color=color_D_S, alpha=0.5, label='$D_S$', zorder=4)
                ax.plot(        numpy.ma.masked_where(D_Sseries<=0, self.tseries), numpy.ma.masked_where(D_Sseries<=0, D_Sseries),    color=color_D_S, zorder=5)
            if(any(D_Aseries) and plot_D_A=='shaded'):
                ax.fill_between(numpy.ma.masked_where(D_Aseries<=0, self.tseries), numpy.ma.masked_where(D_Aseries<=0, D_Aseries), 0, color=color_D_A, alpha=0.5, label='$D_A$', zorder=4)
                ax.plot(        numpy.ma.masked_where(D_Aseries<=0, self.tseries), numpy.ma.masked_where(D_Aseries<=0, D_Aseries),    color=color_D_A, zorder=5)
        if(any(Eseries) and plot_E=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, Eseries), 0, color=color_E, alpha=0.5, label='$E$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, Eseries),    color=color_E, zorder=5)
        if(any(I_preseries) and plot_I_pre=='shaded'):
            ax.fill_between(numpy.ma.masked_where(I_preseries<=0, self.tseries), numpy.ma.masked_where(I_preseries<=0, I_preseries), 0, color=color_I_pre, alpha=0.5, label='$I_{pre}$', zorder=4)
            ax.plot(        numpy.ma.masked_where(I_preseries<=0, self.tseries), numpy.ma.masked_where(I_preseries<=0, I_preseries),    color=color_I_pre, zorder=5)
        if(any(I_Sseries) and plot_I_S=='shaded'):
            ax.fill_between(numpy.ma.masked_where(I_Sseries<=0, self.tseries), numpy.ma.masked_where(I_Sseries<=0, I_Sseries), 0, color=color_I_S, alpha=0.5, label='$I_S$', zorder=4)
            ax.plot(        numpy.ma.masked_where(I_Sseries<=0, self.tseries), numpy.ma.masked_where(I_Sseries<=0, I_Sseries),    color=color_I_S, zorder=5)
        if(any(I_Aseries) and plot_I_A=='shaded'):
            ax.fill_between(numpy.ma.masked_where(I_Aseries<=0, self.tseries), numpy.ma.masked_where(I_Aseries<=0, I_Aseries), 0, color=color_I_A, alpha=0.5, label='$I_A$', zorder=4)
            ax.plot(        numpy.ma.masked_where(I_Aseries<=0, self.tseries), numpy.ma.masked_where(I_Aseries<=0, I_Aseries),    color=color_I_A, zorder=5)
        if(any(Sseries) and plot_S=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, Sseries), 0, color=color_S, alpha=0.5, label='$S$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, Sseries),    color=color_S, zorder=5)
        if(any(Rseries) and plot_R=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, Rseries), 0, color=color_R, alpha=0.5, label='$R$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, Rseries),    color=color_R, zorder=5)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the line variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
        if(any(Fseries) and plot_F=='line'):
            ax.plot(numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, Fseries), color=color_F, label='$F$', zorder=6)
        if(any(Hseries) and plot_H=='line'):
            ax.plot(numpy.ma.masked_where(Hseries<=0, self.tseries), numpy.ma.masked_where(Hseries<=0, Hseries), color=color_H, label='$H$', zorder=6)
        if(combine_D and (any(Dseries) and plot_D_E=='line' and plot_D_pre=='line' and plot_D_S=='line' and plot_D_A=='line')):
            ax.plot(numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, Dseries), color=color_D_S, label='$D_{all}$', zorder=6)
        else:
            if(any(D_Eseries) and plot_D_E=='line'):
                ax.plot(numpy.ma.masked_where(D_Eseries<=0, self.tseries), numpy.ma.masked_where(D_Eseries<=0, D_Eseries), color=color_D_E, label='$D_E$', zorder=6)
            if(any(D_preseries) and plot_D_pre=='line'):
                ax.plot(numpy.ma.masked_where(D_preseries<=0, self.tseries), numpy.ma.masked_where(D_preseries<=0, D_preseries), color=color_D_pre, label='$D_{pre}$', zorder=6)
            if(any(D_Sseries) and plot_D_S=='line'):
                ax.plot(numpy.ma.masked_where(D_Sseries<=0, self.tseries), numpy.ma.masked_where(D_Sseries<=0, D_Sseries), color=color_D_S, label='$D_S$', zorder=6)
            if(any(D_Aseries) and plot_D_A=='line'):
                ax.plot(numpy.ma.masked_where(D_Aseries<=0, self.tseries), numpy.ma.masked_where(D_Aseries<=0, D_Aseries), color=color_D_A, label='$D_A$', zorder=6)
        if(any(Eseries) and plot_E=='line'):
            ax.plot(numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, Eseries), color=color_E, label='$E$', zorder=6)
        if(any(I_preseries) and plot_I_pre=='line'):
            ax.plot(numpy.ma.masked_where(I_preseries<=0, self.tseries), numpy.ma.masked_where(I_preseries<=0, I_preseries), color=color_I_pre, label='$I_{pre}$', zorder=6)
        if(any(I_Sseries) and plot_I_S=='line'):
            ax.plot(numpy.ma.masked_where(I_Sseries<=0, self.tseries), numpy.ma.masked_where(I_Sseries<=0, I_Sseries), color=color_I_S, label='$I_S$', zorder=6)
        if(any(I_Aseries) and plot_I_A=='line'):
            ax.plot(numpy.ma.masked_where(I_Aseries<=0, self.tseries), numpy.ma.masked_where(I_Aseries<=0, I_Aseries), color=color_I_A, label='$I_A$', zorder=6)
        if(any(Sseries) and plot_S=='line'):
            ax.plot(numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, Sseries), color=color_S, label='$S$', zorder=6)
        if(any(Rseries) and plot_R=='line'):
            ax.plot(numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, Rseries), color=color_R, label='$R$', zorder=6)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the vertical line annotations:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(len(vlines)>0 and len(vline_colors)==0):
            vline_colors = ['gray']*len(vlines)
        if(len(vlines)>0 and len(vline_labels)==0):
            vline_labels = [None]*len(vlines)
        if(len(vlines)>0 and len(vline_styles)==0):
            vline_styles = [':']*len(vlines)
        for vline_x, vline_color, vline_style, vline_label in zip(vlines, vline_colors, vline_styles, vline_labels):
            if(vline_x is not None):
                ax.axvline(x=vline_x, color=vline_color, linestyle=vline_style, alpha=1, label=vline_label)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the plot labels:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ax.set_xlabel('days')
        ax.set_ylabel('percent of population' if plot_percentages else 'number of individuals')
        ax.set_xlim(0, (max(self.tseries) if not xlim else xlim))
        ax.set_ylim(0, ylim)
        if(plot_percentages):
            ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])
        if(legend):
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend(legend_handles[::-1], legend_labels[::-1], loc='upper right', facecolor='white', edgecolor='none', framealpha=0.9, prop={'size': 8})
        if(title):
            ax.set_title(title, size=12)
        if(side_title):
            ax.annotate(side_title, (0, 0.5), xytext=(-45, 0), ha='right', va='center',
                size=12, rotation=90, xycoords='axes fraction', textcoords='offset points')
       
        return ax


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def figure_basic(self, plot_S='line', plot_E='line', plot_I_pre='line', plot_I_S='line', plot_I_A='line',
                        plot_H='line', plot_R='line', plot_F='line',
                        plot_D_E='line', plot_D_pre='line', plot_D_S='line', plot_D_A='line', combine_D=True,
                        color_S='tab:green', color_E='orange', color_I_pre='tomato', color_I_S='crimson', color_I_A='crimson', 
                        color_H='violet', color_R='tab:blue', color_F='black',
                        color_D_E='mediumorchid', color_D_pre='mediumorchid', color_D_S='mediumorchid', color_D_A='mediumorchid', 
                        color_reference='#E0E0E0',
                        dashed_reference_results=None, dashed_reference_label='reference', 
                        shaded_reference_results=None, shaded_reference_label='reference', 
                        vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                        ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True,
                        figsize=(12,8), use_seaborn=True, show=True):

        import matplotlib.pyplot as pyplot

        fig, ax = pyplot.subplots(figsize=figsize)

        if(use_seaborn):
            import seaborn
            seaborn.set_style('ticks')
            seaborn.despine()

        self.plot(ax=ax, plot_S=plot_S, plot_E=plot_E, plot_I_pre=plot_I_pre, plot_I_S=plot_I_S, plot_I_A=plot_I_A,
                        plot_H=plot_H, plot_R=plot_R, plot_F=plot_F,
                        plot_D_E=plot_D_E, plot_D_pre=plot_D_pre, plot_D_S=plot_D_S, plot_D_A=plot_D_A, combine_D=True,
                        color_S=color_S, color_E=color_E, color_I_pre=color_I_pre, color_I_S=color_I_S, color_I_A=color_I_A, 
                        color_H=color_H, color_R=color_R, color_F=color_F,
                        color_D_E=color_D_E, color_D_pre=color_D_pre, color_D_S=color_D_S, color_D_A=color_D_A, 
                        color_reference=color_reference,
                        dashed_reference_results=dashed_reference_results, dashed_reference_label=dashed_reference_label, 
                        shaded_reference_results=shaded_reference_results, shaded_reference_label=shaded_reference_label, 
                        vlines=vlines, vline_colors=vline_colors, vline_styles=vline_styles, vline_labels=vline_labels,
                        ylim=ylim, xlim=xlim, legend=legend, title=title, side_title=side_title, plot_percentages=plot_percentages)

        if(show):
            pyplot.show()

        return fig, ax


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def figure_infections(self, plot_S=False, plot_E='stacked', plot_I_pre='stacked', plot_I_S='stacked', plot_I_A='stacked',
                            plot_H='stacked', plot_R=False, plot_F='stacked',
                            plot_D_E='stacked', plot_D_pre='stacked', plot_D_S='stacked', plot_D_A='stacked', combine_D=True,
                            color_S='tab:green', color_E='orange', color_I_pre='tomato', color_I_S='crimson', color_I_A='crimson', 
                            color_H='violet', color_R='tab:blue', color_F='black',
                            color_D_E='mediumorchid', color_D_pre='mediumorchid', color_D_S='mediumorchid', color_D_A='mediumorchid', 
                            color_reference='#E0E0E0',
                            dashed_reference_results=None, dashed_reference_label='reference', 
                            shaded_reference_results=None, shaded_reference_label='reference', 
                            vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                            ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True,
                            figsize=(12,8), use_seaborn=True, show=True):

        import matplotlib.pyplot as pyplot

        fig, ax = pyplot.subplots(figsize=figsize)

        if(use_seaborn):
            import seaborn
            seaborn.set_style('ticks')
            seaborn.despine()

        self.plot(ax=ax, plot_S=plot_S, plot_E=plot_E, plot_I_pre=plot_I_pre, plot_I_S=plot_I_S, plot_I_A=plot_I_A,
                        plot_H=plot_H, plot_R=plot_R, plot_F=plot_F,
                        plot_D_E=plot_D_E, plot_D_pre=plot_D_pre, plot_D_S=plot_D_S, plot_D_A=plot_D_A, combine_D=True,
                        color_S=color_S, color_E=color_E, color_I_pre=color_I_pre, color_I_S=color_I_S, color_I_A=color_I_A, 
                        color_H=color_H, color_R=color_R, color_F=color_F,
                        color_D_E=color_D_E, color_D_pre=color_D_pre, color_D_S=color_D_S, color_D_A=color_D_A, 
                        color_reference=color_reference,
                        dashed_reference_results=dashed_reference_results, dashed_reference_label=dashed_reference_label, 
                        shaded_reference_results=shaded_reference_results, shaded_reference_label=shaded_reference_label, 
                        vlines=vlines, vline_colors=vline_colors, vline_styles=vline_styles, vline_labels=vline_labels, 
                        ylim=ylim, xlim=xlim, legend=legend, title=title, side_title=side_title, plot_percentages=plot_percentages)

        if(show):
            pyplot.show()

        return fig, ax
        



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define a custom method for generating 
# power-law-like graphs with exponential tails 
# both above and below the degree mean and  
# where the mean degree be easily down-shifted
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def custom_exponential_graph(base_graph=None, scale=100, min_num_edges=0, m=9, n=None):
    # Generate a random preferential attachment power law graph as a starting point.
    # By the way this graph is constructed, it is expected to have 1 connected component.
    # Every node is added along with m=8 edges, so the min degree is m=8.
    if(base_graph):
        graph = base_graph.copy()
    else:
        assert(n is not None), "Argument n (number of nodes) must be provided when no base graph is given."
        graph = networkx.barabasi_albert_graph(n=n, m=m)

    # To get a graph with power-law-esque properties but without the fixed minimum degree,
    # We modify the graph by probabilistically dropping some edges from each node. 
    for node in graph:
        neighbors = list(graph[node].keys())
        quarantineEdgeNum = int( max(min(numpy.random.exponential(scale=scale, size=1), len(neighbors)), min_num_edges) )
        quarantineKeepNeighbors = numpy.random.choice(neighbors, size=quarantineEdgeNum, replace=False)
        for neighbor in neighbors:
            if(neighbor not in quarantineKeepNeighbors):
                graph.remove_edge(node, neighbor)
    
    return graph

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def plot_degree_distn(graph, max_degree=None, show=True, use_seaborn=True):
    import matplotlib.pyplot as pyplot
    if(use_seaborn):
        import seaborn
        seaborn.set_style('ticks')
        seaborn.despine()
    # Get a list of the node degrees:
    if type(graph)==numpy.ndarray:
        nodeDegrees = graph.sum(axis=0).reshape((graph.shape[0],1))   # sums of adj matrix cols
    elif type(graph)==networkx.classes.graph.Graph:
        nodeDegrees = [d[1] for d in graph.degree()]
    else:
        raise BaseException("Input an adjacency matrix or networkx object only.")
    # Calculate the mean degree:
    meanDegree = numpy.mean(nodeDegrees)
    # Generate a histogram of the node degrees:
    pyplot.hist(nodeDegrees, bins=range(max(nodeDegrees)), alpha=0.5, color='tab:blue', label=('mean degree = %.1f' % meanDegree))
    pyplot.xlim(0, max(nodeDegrees) if not max_degree else max_degree)
    pyplot.xlabel('degree')
    pyplot.ylabel('num nodes')
    pyplot.legend(loc='upper right')
    if(show):
        pyplot.show()




class ExtSEIRSNetworkModel():
    """
    A class to simulate the Extended SEIRS Stochastic Network Model
    ===================================================
    Params: 
            G               Network adjacency matrix (numpy array) or Networkx graph object.
            beta            Rate of transmission (global interactions)
            beta_local      Rate(s) of transmission between adjacent individuals (optional)
            beta_asym       Rate of transmission (global interactions)
            beta_asym_local Rate(s) of transmission between adjacent individuals (optional)
            sigma           Rate of progression to infectious state (inverse of latent period)             
            lamda           Rate of progression to infectious (a)symptomatic state (inverse of prodromal period)               
            eta             Rate of progression to hospitalized state (inverse of onset-to-admission period)           
            gamma           Rate of recovery for non-hospitalized symptomatic individuals (inverse of symptomatic infectious period)           
            gamma_asym      Rate of recovery for asymptomatic individuals (inverse of asymptomatic infectious period)              
            gamma_H         Rate of recovery for hospitalized symptomatic individuals (inverse of hospitalized infectious period)              
            mu_H            Rate of death for hospitalized individuals (inverse of admission-to-death period)              
            xi              Rate of re-susceptibility (upon recovery)
            mu_0            Rate of baseline death
            nu              Rate of baseline birth
            a               Probability of an infected individual remaining asymptomatic
            h               Probability of a symptomatic individual being hospitalized  
            f               Probability of death for hospitalized individuals (case fatality rate)                         
            p               Probability of individuals interacting with global population              
            
            G_Q             Quarantine adjacency matrix (numpy array) or Networkx graph object.
            beta_Q          Rate of transmission for isolated individuals (global interactions)
            beta_Q_local    Rate(s) of transmission (exposure) for adjacent isolated individuals (optional)              
            sigma_Q         Rate of progression to infectious state for isolated individuals           
            lamda_Q         Rate of progression to infectious (a)symptomatic state for isolated individuals            
            eta_Q           Rate of progression to hospitalized state for isolated individuals             
            gamma_Q_sym     Rate of recovery for non-hospitalized symptomatic individuals for isolated individuals             
            gamma_Q_asym    Rate of recovery for asymptomatic individuals for isolated individuals             
            theta_E         Rate of random testing for exposed individuals             
            theta_pre       Rate of random testing for infectious pre-symptomatic individuals              
            theta_sym       Rate of random testing for infectious symptomatic individuals              
            theta_asym      Rate of random testing for infectious asymptomatic individuals             
            phi_E           Rate of testing when a close contact has tested positive for exposed individuals               
            phi_pre         Rate of testing when a close contact has tested positive for infectious pre-symptomatic individuals                
            phi_sym         Rate of testing when a close contact has tested positive for infectious symptomatic individuals                
            phi_asym        Rate of testing when a close contact has tested positive for infectious asymptomatic individuals               
            psi_E           Probability of positive test for exposed individuals               
            psi_pre         Probability of positive test for infectious pre-symptomatic individuals                
            psi_sym         Probability of positive test for infectious symptomatic individuals                
            psi_asym        Probability of positive test for infectious asymptomatic individuals               
            q               Probability of isolated individuals interacting with global population
            isolation_time  Time to remain in isolation upon positive test, self-isolation, etc.            
            
            initE           Initial number of exposed individuals
            initI_pre       Initial number of infectious pre-symptomatic individuals
            initI_sym       Initial number of infectious symptomatic individuals
            initI_asym      Initial number of infectious asymptomatic individuals
            initH           Initial number of hospitalized individuals
            initR           Initial number of recovered individuals     
            initF           Initial number of infection-related fatalities
            initQ_S         Initial number of isolated susceptible individuals
            initQ_E         Initial number of isolated exposed individuals
            initQ_pre       Initial number of isolated infectious pre-symptomatic individuals
            initQ_sym       Initial number of isolated infectious symptomatic individuals
            initQ_asym      Initial number of isolated infectious asymptomatic individuals
            initQ_R         Initial number of isolated recovered individuals
                            (all remaining nodes initialized susceptible)   
    """
    def __init__(self, G, beta, sigma, lamda, gamma, 
                    gamma_asym=None, eta=0, gamma_H=None, mu_H=0, alpha=1.0, xi=0, mu_0=0, nu=0, a=0, h=0, f=0, p=0,             
                    beta_local=None, beta_asym=None, beta_asym_local=None, beta_pairwise_mode='infected', delta=None, delta_pairwise_mode=None,
                    G_Q=None, beta_Q=None, beta_Q_local=None, sigma_Q=None, lamda_Q=None, eta_Q=None, gamma_Q_sym=None, gamma_Q_asym=None, alpha_Q=None, delta_Q=None,
                    theta_S=0, theta_E=0, theta_pre=0, theta_sym=0, theta_asym=0, phi_S=0, phi_E=0, phi_pre=0, phi_sym=0, phi_asym=0,    
                    psi_S=0, psi_E=1, psi_pre=1, psi_sym=1, psi_asym=1, q=0, isolation_time=14,
                    initE=0, initI_pre=0, initI_sym=0, initI_asym=0, initH=0, initR=0, initF=0,        
                    initQ_S=0, initQ_E=0, initQ_pre=0, initQ_sym=0, initQ_asym=0, initQ_R=0,
                    o=0, prevalence_ext=0,
                    transition_mode='exponential_rates', node_groups=None, store_Xseries=False, seed=None):

        if(seed is not None):
            numpy.random.seed(seed)
            self.seed = seed

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model Parameters:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.parameters = { 'G':G, 'G_Q':G_Q,
                            'beta':beta, 'sigma':sigma, 'lamda':lamda, 'gamma':gamma, 
                            'eta':eta, 'gamma_asym':gamma_asym, 'gamma_H':gamma_H, 'mu_H':mu_H, 
                            'xi':xi, 'mu_0':mu_0, 'nu':nu, 'a':a, 'h':h, 'f':f, 'p':p, 
                            'beta_local':beta_local, 'beta_asym':beta_asym, 'beta_asym_local':beta_asym_local, 'beta_pairwise_mode':beta_pairwise_mode,
                            'alpha':alpha, 'delta':delta, 'delta_pairwise_mode':delta_pairwise_mode,
                            'lamda_Q':lamda_Q, 'beta_Q':beta_Q, 'beta_Q_local':beta_Q_local, 'alpha_Q':alpha_Q, 'sigma_Q':sigma_Q, 
                            'eta_Q':eta_Q, 'gamma_Q_sym':gamma_Q_sym, 'gamma_Q_asym':gamma_Q_asym, 'delta_Q':delta_Q,
                            'theta_S':theta_S, 'theta_E':theta_E, 'theta_pre':theta_pre, 'theta_sym':theta_sym, 'theta_asym':theta_asym, 
                            'phi_S':phi_S, 'phi_E':phi_E, 'phi_pre':phi_pre, 'phi_sym':phi_sym, 'phi_asym':phi_asym, 
                            'psi_S':psi_S, 'psi_E':psi_E, 'psi_pre':psi_pre, 'psi_sym':psi_sym, 'psi_asym':psi_asym, 'q':q, 'isolation_time':isolation_time,
                            'initE':initE, 'initI_pre':initI_pre, 'initI_sym':initI_sym, 'initI_asym':initI_asym, 
                            'initH':initH, 'initR':initR, 'initF':initF, 
                            'initQ_S':initQ_S, 'initQ_E':initQ_E, 'initQ_pre':initQ_pre, 
                            'initQ_sym':initQ_sym, 'initQ_asym':initQ_asym, 'initQ_R':initQ_R,
                            'o':o, 'prevalence_ext':prevalence_ext}
        self.update_parameters()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Each node can undergo 4-6 transitions (sans vitality/re-susceptibility returns to S state),
        # so there are ~numNodes*6 events/timesteps expected; initialize numNodes*6 timestep slots to start 
        # (will be expanded during run if needed for some reason)
        self.tseries    = numpy.zeros(6*self.numNodes)
        self.numS       = numpy.zeros(6*self.numNodes)
        self.numE       = numpy.zeros(6*self.numNodes)
        self.numI_pre   = numpy.zeros(6*self.numNodes)
        self.numI_sym   = numpy.zeros(6*self.numNodes)
        self.numI_asym  = numpy.zeros(6*self.numNodes)
        self.numH       = numpy.zeros(6*self.numNodes)
        self.numR       = numpy.zeros(6*self.numNodes)
        self.numF       = numpy.zeros(6*self.numNodes)
        self.numQ_S     = numpy.zeros(6*self.numNodes)
        self.numQ_E     = numpy.zeros(6*self.numNodes)
        self.numQ_pre   = numpy.zeros(6*self.numNodes)
        self.numQ_sym   = numpy.zeros(6*self.numNodes)
        self.numQ_asym  = numpy.zeros(6*self.numNodes)
        self.numQ_R     = numpy.zeros(6*self.numNodes)
        self.N          = numpy.zeros(6*self.numNodes)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Timekeeping:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.t          = 0
        self.tmax       = 0 # will be set when run() is called
        self.tidx       = 0
        self.tseries[0] = 0

        # Vectors holding the time that each node has been in a given state or in isolation:
        self.timer_state     = numpy.zeros((self.numNodes,1))
        self.timer_isolation = numpy.zeros(self.numNodes)
        self.isolationTime   = isolation_time
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Counts of inidividuals with each state:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.numE[0]        = int(initE)
        self.numI_pre[0]    = int(initI_pre)
        self.numI_sym[0]    = int(initI_sym)
        self.numI_asym[0]   = int(initI_asym)
        self.numH[0]        = int(initH)
        self.numR[0]        = int(initR)
        self.numF[0]        = int(initF)
        self.numQ_S[0]      = int(initQ_S)
        self.numQ_E[0]      = int(initQ_E)
        self.numQ_pre[0]    = int(initQ_pre)
        self.numQ_sym[0]    = int(initQ_sym)
        self.numQ_asym[0]   = int(initQ_asym)
        self.numQ_R[0]      = int(initQ_R)
        self.numS[0]        = (self.numNodes - self.numE[0] - self.numI_pre[0] - self.numI_sym[0] - self.numI_asym[0] - self.numH[0] - self.numR[0] 
                                             - self.numQ_S[0] - self.numQ_E[0] - self.numQ_pre[0] - self.numQ_sym[0] - self.numQ_asym[0] - self.numQ_R[0] - self.numF[0])
        self.N[0]           = self.numNodes - self.numF[0]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Node states:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.S          = 1
        self.E          = 2
        self.I_pre      = 3
        self.I_sym      = 4
        self.I_asym     = 5
        self.H          = 6
        self.R          = 7
        self.F          = 8
        self.Q_S        = 11
        self.Q_E        = 12
        self.Q_pre      = 13
        self.Q_sym      = 14
        self.Q_asym     = 15
        self.Q_R        = 17
        
        self.X = numpy.array( [self.S]*int(self.numS[0]) + [self.E]*int(self.numE[0]) 
                               + [self.I_pre]*int(self.numI_pre[0]) + [self.I_sym]*int(self.numI_sym[0]) + [self.I_asym]*int(self.numI_asym[0]) 
                               + [self.H]*int(self.numH[0]) + [self.R]*int(self.numR[0]) + [self.F]*int(self.numF[0])
                               + [self.Q_S]*int(self.numQ_S[0]) + [self.Q_E]*int(self.numQ_E[0]) 
                               + [self.Q_pre]*int(self.numQ_pre[0]) + [self.Q_sym]*int(self.numQ_sym[0]) + [self.Q_asym]*int(self.numQ_asym[0]) 
                               + [self.Q_R]*int(self.numQ_R[0])
                            ).reshape((self.numNodes,1))
        numpy.random.shuffle(self.X)

        self.store_Xseries = store_Xseries
        if(store_Xseries):
            self.Xseries        = numpy.zeros(shape=(6*self.numNodes, self.numNodes), dtype='uint8')
            self.Xseries[0,:]   = self.X.T

        self.transitions =  { 
                                'StoE':         {'currentState':self.S,       'newState':self.E},
                                'StoQS':        {'currentState':self.S,       'newState':self.Q_S},
                                'EtoIPRE':      {'currentState':self.E,       'newState':self.I_pre},
                                'EtoQE':        {'currentState':self.E,       'newState':self.Q_E},
                                'IPREtoISYM':   {'currentState':self.I_pre,   'newState':self.I_sym},
                                'IPREtoIASYM':  {'currentState':self.I_pre,   'newState':self.I_asym},
                                'IPREtoQPRE':   {'currentState':self.I_pre,   'newState':self.Q_pre},
                                'ISYMtoH':      {'currentState':self.I_sym,   'newState':self.H},
                                'ISYMtoR':      {'currentState':self.I_sym,   'newState':self.R},
                                'ISYMtoQSYM':   {'currentState':self.I_sym,   'newState':self.Q_sym},
                                'IASYMtoR':     {'currentState':self.I_asym,  'newState':self.R},
                                'IASYMtoQASYM': {'currentState':self.I_asym,  'newState':self.Q_asym},
                                'HtoR':         {'currentState':self.H,       'newState':self.R},
                                'HtoF':         {'currentState':self.H,       'newState':self.F},
                                'RtoS':         {'currentState':self.R,       'newState':self.S},
                                'QStoQE':       {'currentState':self.Q_S,     'newState':self.Q_E},
                                'QEtoQPRE':     {'currentState':self.Q_E,     'newState':self.Q_pre},
                                'QPREtoQSYM':   {'currentState':self.Q_pre,   'newState':self.Q_sym},
                                'QPREtoQASYM':  {'currentState':self.Q_pre,   'newState':self.Q_asym},
                                'QSYMtoH':      {'currentState':self.Q_sym,   'newState':self.H},
                                'QSYMtoQR':     {'currentState':self.Q_sym,   'newState':self.Q_R},
                                'QASYMtoQR':    {'currentState':self.Q_asym,  'newState':self.Q_R},
                                '_toS':         {'currentState':True,         'newState':self.S},
                            }

        self.transition_mode = transition_mode

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize other node metadata:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.tested      = numpy.array([False]*self.numNodes).reshape((self.numNodes,1))
        self.positive    = numpy.array([False]*self.numNodes).reshape((self.numNodes,1))
        self.numTested   = numpy.zeros(6*self.numNodes) 
        self.numPositive = numpy.zeros(6*self.numNodes) 

        self.testedInCurrentState = numpy.array([False]*self.numNodes).reshape((self.numNodes,1))

        self.infectionsLog = []

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize node subgroup data series:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.nodeGroupData = None
        if(node_groups):
            self.nodeGroupData = {}
            for groupName, nodeList in node_groups.items():
                self.nodeGroupData[groupName] = {'nodes':   numpy.array(nodeList),
                                                 'mask':    numpy.isin(range(self.numNodes), nodeList).reshape((self.numNodes,1))}
                self.nodeGroupData[groupName]['numS']           = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numE']           = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numI_pre']       = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numI_sym']       = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numI_asym']      = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numH']           = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numR']           = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numF']           = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numQ_S']         = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numQ_E']         = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numQ_pre']       = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numQ_sym']       = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numQ_asym']      = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numQ_R']         = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['N']              = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numPositive']    = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numTested']      = numpy.zeros(6*self.numNodes)
                self.nodeGroupData[groupName]['numS'][0]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.S)
                self.nodeGroupData[groupName]['numE'][0]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.E)
                self.nodeGroupData[groupName]['numI_pre'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I_pre)
                self.nodeGroupData[groupName]['numI_sym'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I_sym)
                self.nodeGroupData[groupName]['numI_asym'][0]   = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I_asym)
                self.nodeGroupData[groupName]['numH'][0]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.H)
                self.nodeGroupData[groupName]['numR'][0]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.R)
                self.nodeGroupData[groupName]['numF'][0]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.F)
                self.nodeGroupData[groupName]['numQ_S'][0]      = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_E)
                self.nodeGroupData[groupName]['numQ_E'][0]      = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_E)
                self.nodeGroupData[groupName]['numQ_pre'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_pre)
                self.nodeGroupData[groupName]['numQ_I_sym'][0]  = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_I_sym)
                self.nodeGroupData[groupName]['numQ_I_asym'][0] = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_I_asym)
                self.nodeGroupData[groupName]['numQ_R'][0]      = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_E)
                self.nodeGroupData[groupName]['N'][0]           = self.numNodes - self.numF[0]

         
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def update_parameters(self):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model graphs:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.G = self.parameters['G']
        # Adjacency matrix:
        if type(self.G)==numpy.ndarray:
            self.A = scipy.sparse.csr_matrix(self.G)
        elif type(self.G)==networkx.classes.graph.Graph:
            self.A = networkx.adj_matrix(self.G) # adj_matrix gives scipy.sparse csr_matrix
        else:
            raise BaseException("Input an adjacency matrix or networkx object only.")
        self.numNodes   = int(self.A.shape[1])
        self.degree     = numpy.asarray(self.node_degrees(self.A)).astype(float)
        #----------------------------------------
        if(self.parameters['G_Q'] is None):
            self.G_Q = self.G # If no Q graph is provided, use G in its place
        else:
            self.G_Q = self.parameters['G_Q']
        # Quarantine Adjacency matrix:
        if type(self.G_Q)==numpy.ndarray:
            self.A_Q = scipy.sparse.csr_matrix(self.G_Q)
        elif type(self.G_Q)==networkx.classes.graph.Graph:
            self.A_Q = networkx.adj_matrix(self.G_Q) # adj_matrix gives scipy.sparse csr_matrix
        else:
            raise BaseException("Input an adjacency matrix or networkx object only.")
        self.numNodes_Q   = int(self.A_Q.shape[1])
        self.degree_Q     = numpy.asarray(self.node_degrees(self.A_Q)).astype(float)
        #----------------------------------------
        assert(self.numNodes == self.numNodes_Q), "The normal and quarantine adjacency graphs must be of the same size."

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model parameters:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.beta           = numpy.array(self.parameters['beta']).reshape((self.numNodes, 1))          if isinstance(self.parameters['beta'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['beta'], shape=(self.numNodes,1))
        self.beta_asym      = (numpy.array(self.parameters['beta_asym']).reshape((self.numNodes, 1))    if isinstance(self.parameters['beta_asym'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['beta_asym'], shape=(self.numNodes,1))) if self.parameters['beta_asym'] is not None else self.beta
        self.sigma          = numpy.array(self.parameters['sigma']).reshape((self.numNodes, 1))         if isinstance(self.parameters['sigma'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['sigma'], shape=(self.numNodes,1))
        self.lamda          = numpy.array(self.parameters['lamda']).reshape((self.numNodes, 1))         if isinstance(self.parameters['lamda'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['lamda'], shape=(self.numNodes,1))
        self.gamma          = numpy.array(self.parameters['gamma']).reshape((self.numNodes, 1))         if isinstance(self.parameters['gamma'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['gamma'], shape=(self.numNodes,1))
        self.eta            = numpy.array(self.parameters['eta']).reshape((self.numNodes, 1))           if isinstance(self.parameters['eta'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['eta'], shape=(self.numNodes,1))
        self.gamma_asym     = (numpy.array(self.parameters['gamma_asym']).reshape((self.numNodes, 1))   if isinstance(self.parameters['gamma_asym'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['gamma_asym'], shape=(self.numNodes,1))) if self.parameters['gamma_asym'] is not None else self.gamma
        self.gamma_H        = (numpy.array(self.parameters['gamma_H']).reshape((self.numNodes, 1))      if isinstance(self.parameters['gamma_H'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['gamma_H'], shape=(self.numNodes,1))) if self.parameters['gamma_H'] is not None else self.gamma
        self.mu_H           = numpy.array(self.parameters['mu_H']).reshape((self.numNodes, 1))          if isinstance(self.parameters['mu_H'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['mu_H'], shape=(self.numNodes,1))
        self.alpha          = numpy.array(self.parameters['alpha']).reshape((self.numNodes, 1))         if isinstance(self.parameters['alpha'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['alpha'], shape=(self.numNodes,1))
        self.xi             = numpy.array(self.parameters['xi']).reshape((self.numNodes, 1))            if isinstance(self.parameters['xi'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['xi'], shape=(self.numNodes,1))
        self.mu_0           = numpy.array(self.parameters['mu_0']).reshape((self.numNodes, 1))          if isinstance(self.parameters['mu_0'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['mu_0'], shape=(self.numNodes,1))
        self.nu             = numpy.array(self.parameters['nu']).reshape((self.numNodes, 1))            if isinstance(self.parameters['nu'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['nu'], shape=(self.numNodes,1))
        self.a              = numpy.array(self.parameters['a']).reshape((self.numNodes, 1))             if isinstance(self.parameters['a'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['a'], shape=(self.numNodes,1))
        self.h              = numpy.array(self.parameters['h']).reshape((self.numNodes, 1))             if isinstance(self.parameters['h'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['h'], shape=(self.numNodes,1))
        self.f              = numpy.array(self.parameters['f']).reshape((self.numNodes, 1))             if isinstance(self.parameters['f'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['f'], shape=(self.numNodes,1))
        self.p              = numpy.array(self.parameters['p']).reshape((self.numNodes, 1))             if isinstance(self.parameters['p'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['p'], shape=(self.numNodes,1))
        self.o              = numpy.array(self.parameters['o']).reshape((self.numNodes, 1))             if isinstance(self.parameters['o'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['o'], shape=(self.numNodes,1))

        self.rand_a = numpy.random.rand(self.a.shape[0], self.a.shape[1])
        self.rand_h = numpy.random.rand(self.h.shape[0], self.h.shape[1])
        self.rand_f = numpy.random.rand(self.f.shape[0], self.f.shape[1])

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # External infection introduction variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.prevalence_ext     = numpy.array(self.parameters['prevalence_ext']).reshape((self.numNodes, 1)) if isinstance(self.parameters['prevalence_ext'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['prevalence_ext'], shape=(self.numNodes,1))
       
        #----------------------------------------
        # Testing-related parameters:
        #----------------------------------------
        self.beta_Q         = (numpy.array(self.parameters['beta_Q']).reshape((self.numNodes, 1))       if isinstance(self.parameters['beta_Q'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['beta_Q'], shape=(self.numNodes,1))) if self.parameters['beta_Q'] is not None else self.beta
        self.sigma_Q        = (numpy.array(self.parameters['sigma_Q']).reshape((self.numNodes, 1))      if isinstance(self.parameters['sigma_Q'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['sigma_Q'], shape=(self.numNodes,1))) if self.parameters['sigma_Q'] is not None else self.sigma
        self.lamda_Q        = (numpy.array(self.parameters['lamda_Q']).reshape((self.numNodes, 1))      if isinstance(self.parameters['lamda_Q'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['lamda_Q'], shape=(self.numNodes,1))) if self.parameters['lamda_Q'] is not None else self.lamda
        self.gamma_Q_sym    = (numpy.array(self.parameters['gamma_Q_sym']).reshape((self.numNodes, 1))  if isinstance(self.parameters['gamma_Q_sym'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['gamma_Q_sym'], shape=(self.numNodes,1))) if self.parameters['gamma_Q_sym'] is not None else self.gamma
        self.gamma_Q_asym   = (numpy.array(self.parameters['gamma_Q_asym']).reshape((self.numNodes, 1)) if isinstance(self.parameters['gamma_Q_asym'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['gamma_Q_asym'], shape=(self.numNodes,1))) if self.parameters['gamma_Q_asym'] is not None else self.gamma
        self.eta_Q          = (numpy.array(self.parameters['eta_Q']).reshape((self.numNodes, 1))        if isinstance(self.parameters['eta_Q'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['eta_Q'], shape=(self.numNodes,1))) if self.parameters['eta_Q'] is not None else self.eta
        self.alpha_Q        = (numpy.array(self.parameters['alpha_Q']).reshape((self.numNodes, 1))      if isinstance(self.parameters['alpha_Q'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['alpha_Q'], shape=(self.numNodes,1))) if self.parameters['alpha_Q'] is not None else self.alpha
        self.theta_S        = numpy.array(self.parameters['theta_S']).reshape((self.numNodes, 1))       if isinstance(self.parameters['theta_S'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['theta_S'], shape=(self.numNodes,1))
        self.theta_E        = numpy.array(self.parameters['theta_E']).reshape((self.numNodes, 1))       if isinstance(self.parameters['theta_E'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['theta_E'], shape=(self.numNodes,1))
        self.theta_pre      = numpy.array(self.parameters['theta_pre']).reshape((self.numNodes, 1))     if isinstance(self.parameters['theta_pre'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['theta_pre'], shape=(self.numNodes,1))
        self.theta_sym      = numpy.array(self.parameters['theta_sym']).reshape((self.numNodes, 1))     if isinstance(self.parameters['theta_sym'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['theta_sym'], shape=(self.numNodes,1))
        self.theta_asym     = numpy.array(self.parameters['theta_asym']).reshape((self.numNodes, 1))    if isinstance(self.parameters['theta_asym'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['theta_asym'], shape=(self.numNodes,1))
        self.phi_S          = numpy.array(self.parameters['phi_S']).reshape((self.numNodes, 1))         if isinstance(self.parameters['phi_S'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['phi_S'], shape=(self.numNodes,1))
        self.phi_E          = numpy.array(self.parameters['phi_E']).reshape((self.numNodes, 1))         if isinstance(self.parameters['phi_E'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['phi_E'], shape=(self.numNodes,1))
        self.phi_pre        = numpy.array(self.parameters['phi_pre']).reshape((self.numNodes, 1))       if isinstance(self.parameters['phi_pre'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['phi_pre'], shape=(self.numNodes,1))
        self.phi_sym        = numpy.array(self.parameters['phi_sym']).reshape((self.numNodes, 1))       if isinstance(self.parameters['phi_sym'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['phi_sym'], shape=(self.numNodes,1))
        self.phi_asym       = numpy.array(self.parameters['phi_asym']).reshape((self.numNodes, 1))      if isinstance(self.parameters['phi_asym'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['phi_asym'], shape=(self.numNodes,1))
        self.psi_S            = numpy.array(self.parameters['psi_S']).reshape((self.numNodes, 1))           if isinstance(self.parameters['psi_S'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['psi_S'], shape=(self.numNodes,1))
        self.psi_E            = numpy.array(self.parameters['psi_E']).reshape((self.numNodes, 1))           if isinstance(self.parameters['psi_E'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['psi_E'], shape=(self.numNodes,1))
        self.psi_pre          = numpy.array(self.parameters['psi_pre']).reshape((self.numNodes, 1))         if isinstance(self.parameters['psi_pre'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['psi_pre'], shape=(self.numNodes,1))
        self.psi_sym          = numpy.array(self.parameters['psi_sym']).reshape((self.numNodes, 1))         if isinstance(self.parameters['psi_sym'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['psi_sym'], shape=(self.numNodes,1))
        self.psi_asym         = numpy.array(self.parameters['psi_asym']).reshape((self.numNodes, 1))        if isinstance(self.parameters['psi_asym'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['psi_asym'], shape=(self.numNodes,1))
        self.q              = numpy.array(self.parameters['q']).reshape((self.numNodes, 1))             if isinstance(self.parameters['q'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['q'], shape=(self.numNodes,1))

        #----------------------------------------

        self.beta_pairwise_mode = self.parameters['beta_pairwise_mode']

        #----------------------------------------
        # Global transmission parameters:
        #----------------------------------------
        if(self.beta_pairwise_mode == 'infected' or self.beta_pairwise_mode is None):
            self.beta_global         = numpy.full_like(self.beta, fill_value=numpy.mean(self.beta))
            self.beta_Q_global       = numpy.full_like(self.beta_Q, fill_value=numpy.mean(self.beta_Q))
            self.beta_asym_global    = numpy.full_like(self.beta_asym, fill_value=numpy.mean(self.beta_asym))
        elif(self.beta_pairwise_mode == 'infectee'):
            self.beta_global         = self.beta      
            self.beta_Q_global       = self.beta_Q    
            self.beta_asym_global    = self.beta_asym
        elif(self.beta_pairwise_mode == 'min'):
            self.beta_global         = numpy.minimum(self.beta, numpy.mean(beta)) 
            self.beta_Q_global       = numpy.minimum(self.beta_Q, numpy.mean(beta_Q)) 
            self.beta_asym_global    = numpy.minimum(self.beta_asym, numpy.mean(beta_asym))
        elif(self.beta_pairwise_mode == 'max'):
            self.beta_global         = numpy.maximum(self.beta, numpy.mean(beta)) 
            self.beta_Q_global       = numpy.maximum(self.beta_Q, numpy.mean(beta_Q)) 
            self.beta_asym_global    = numpy.maximum(self.beta_asym, numpy.mean(beta_asym))
        elif(self.beta_pairwise_mode == 'mean'):
            self.beta_global         = (self.beta + numpy.full_like(self.beta, fill_value=numpy.mean(self.beta)))/2
            self.beta_Q_global       = (self.beta_Q + numpy.full_like(self.beta_Q, fill_value=numpy.mean(self.beta_Q)))/2
            self.beta_asym_global    = (self.beta_asym + numpy.full_like(self.beta_asym, fill_value=numpy.mean(self.beta_asym)))/2
            
        #----------------------------------------
        # Local transmission parameters:
        #----------------------------------------
        self.beta_local         = self.beta      if self.parameters['beta_local'] is None      else numpy.array(self.parameters['beta_local'])      if isinstance(self.parameters['beta_local'], (list, numpy.ndarray))      else numpy.full(fill_value=self.parameters['beta_local'], shape=(self.numNodes,1))
        self.beta_Q_local       = self.beta_Q    if self.parameters['beta_Q_local'] is None    else numpy.array(self.parameters['beta_Q_local'])    if isinstance(self.parameters['beta_Q_local'], (list, numpy.ndarray))    else numpy.full(fill_value=self.parameters['beta_Q_local'], shape=(self.numNodes,1))
        self.beta_asym_local    = None           if self.parameters['beta_asym_local'] is None else numpy.array(self.parameters['beta_asym_local']) if isinstance(self.parameters['beta_asym_local'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['beta_asym_local'], shape=(self.numNodes,1))
        #----------------------------------------
        if(self.beta_local.ndim == 2 and self.beta_local.shape[0] == self.numNodes and self.beta_local.shape[1] == self.numNodes):
            self.A_beta_pairwise = self.beta_local
        elif((self.beta_local.ndim == 1 and self.beta_local.shape[0] == self.numNodes) or (self.beta_local.ndim == 2 and (self.beta_local.shape[0] == self.numNodes or self.beta_local.shape[1] == self.numNodes))):
            self.beta_local = self.beta_local.reshape((self.numNodes,1))
            # Pre-multiply beta values by the adjacency matrix ("transmission weight connections")
            A_beta_pairwise_byInfected = scipy.sparse.csr_matrix.multiply(self.A, self.beta_local.T).tocsr()
            A_beta_pairwise_byInfectee = scipy.sparse.csr_matrix.multiply(self.A, self.beta_local).tocsr()    
            #------------------------------
            # Compute the effective pairwise beta values as a function of the infected/infectee pair:
            if(self.beta_pairwise_mode == 'infected'):
                self.A_beta_pairwise = A_beta_pairwise_byInfected
            elif(self.beta_pairwise_mode == 'infectee'):
                self.A_beta_pairwise = A_beta_pairwise_byInfectee
            elif(self.beta_pairwise_mode == 'min'):
                self.A_beta_pairwise = scipy.sparse.csr_matrix.minimum(A_beta_pairwise_byInfected, A_beta_pairwise_byInfectee)
            elif(self.beta_pairwise_mode == 'max'):
                self.A_beta_pairwise = scipy.sparse.csr_matrix.maximum(A_beta_pairwise_byInfected, A_beta_pairwise_byInfectee)
            elif(self.beta_pairwise_mode == 'mean' or self.beta_pairwise_mode is None):
                self.A_beta_pairwise = (A_beta_pairwise_byInfected + A_beta_pairwise_byInfectee)/2
            else:
                print("Unrecognized beta_pairwise_mode value (support for 'infected', 'infectee', 'min', 'max', and 'mean').")
        else:
            print("Invalid values given for beta_local (expected 1xN list/array or NxN 2d array)")
        #----------------------------------------
        if(self.beta_Q_local.ndim == 2 and self.beta_Q_local.shape[0] == self.numNodes and self.beta_Q_local.shape[1] == self.numNodes):
            self.A_Q_beta_Q_pairwise = self.beta_Q_local
        elif((self.beta_Q_local.ndim == 1 and self.beta_Q_local.shape[0] == self.numNodes) or (self.beta_Q_local.ndim == 2 and (self.beta_Q_local.shape[0] == self.numNodes or self.beta_Q_local.shape[1] == self.numNodes))):
            self.beta_Q_local = self.beta_Q_local.reshape((self.numNodes,1))
            # Pre-multiply beta_Q values by the isolation adjacency matrix ("transmission weight connections")
            A_Q_beta_Q_pairwise_byInfected      = scipy.sparse.csr_matrix.multiply(self.A_Q, self.beta_Q_local.T).tocsr()
            A_Q_beta_Q_pairwise_byInfectee      = scipy.sparse.csr_matrix.multiply(self.A_Q, self.beta_Q_local).tocsr()
            #------------------------------
            # Compute the effective pairwise beta values as a function of the infected/infectee pair:
            if(self.beta_pairwise_mode == 'infected'):
                self.A_Q_beta_Q_pairwise = A_Q_beta_Q_pairwise_byInfected
            elif(self.beta_pairwise_mode == 'infectee'):
                self.A_Q_beta_Q_pairwise = A_Q_beta_Q_pairwise_byInfectee
            elif(self.beta_pairwise_mode == 'min'):
                self.A_Q_beta_Q_pairwise = scipy.sparse.csr_matrix.minimum(A_Q_beta_Q_pairwise_byInfected, A_Q_beta_Q_pairwise_byInfectee)
            elif(self.beta_pairwise_mode == 'max'):
                self.A_Q_beta_Q_pairwise = scipy.sparse.csr_matrix.maximum(A_Q_beta_Q_pairwise_byInfected, A_Q_beta_Q_pairwise_byInfectee)
            elif(self.beta_pairwise_mode == 'mean' or self.beta_pairwise_mode is None):
                self.A_Q_beta_Q_pairwise = (A_Q_beta_Q_pairwise_byInfected + A_Q_beta_Q_pairwise_byInfectee)/2
            else:
                print("Unrecognized beta_pairwise_mode value (support for 'infected', 'infectee', 'min', 'max', and 'mean').")
        else:
            print("Invalid values given for beta_Q_local (expected 1xN list/array or NxN 2d array)")
        #----------------------------------------
        if(self.beta_asym_local is None):
            self.A_beta_asym_pairwise = None
        elif(self.beta_asym_local.ndim == 2 and self.beta_asym_local.shape[0] == self.numNodes and self.beta_asym_local.shape[1] == self.numNodes):
            self.A_beta_asym_pairwise = self.beta_asym_local
        elif((self.beta_asym_local.ndim == 1 and self.beta_asym_local.shape[0] == self.numNodes) or (self.beta_asym_local.ndim == 2 and (self.beta_asym_local.shape[0] == self.numNodes or self.beta_asym_local.shape[1] == self.numNodes))):
            self.beta_asym_local = self.beta_asym_local.reshape((self.numNodes,1))
            # Pre-multiply beta_asym values by the adjacency matrix ("transmission weight connections")
            A_beta_asym_pairwise_byInfected      = scipy.sparse.csr_matrix.multiply(self.A, self.beta_asym_local.T).tocsr()
            A_beta_asym_pairwise_byInfectee      = scipy.sparse.csr_matrix.multiply(self.A, self.beta_asym_local).tocsr()
            #------------------------------
            # Compute the effective pairwise beta values as a function of the infected/infectee pair:
            if(self.beta_pairwise_mode == 'infected'):
                self.A_beta_asym_pairwise = A_beta_asym_pairwise_byInfected
            elif(self.beta_pairwise_mode == 'infectee'):
                self.A_beta_asym_pairwise = A_beta_asym_pairwise_byInfectee
            elif(self.beta_pairwise_mode == 'min'):
                self.A_beta_asym_pairwise = scipy.sparse.csr_matrix.minimum(A_beta_asym_pairwise_byInfected, A_beta_asym_pairwise_byInfectee)
            elif(self.beta_pairwise_mode == 'max'):
                self.A_beta_asym_pairwise = scipy.sparse.csr_matrix.maximum(A_beta_asym_pairwise_byInfected, A_beta_asym_pairwise_byInfectee)
            elif(self.beta_pairwise_mode == 'mean' or self.beta_pairwise_mode is None):
                self.A_beta_asym_pairwise = (A_beta_asym_pairwise_byInfected + A_beta_asym_pairwise_byInfectee)/2
            else:
                print("Unrecognized beta_pairwise_mode value (support for 'infected', 'infectee', 'min', 'max', and 'mean').")
        else:
            print("Invalid values given for beta_asym_local (expected 1xN list/array or NxN 2d array)")

        #----------------------------------------
        # Degree-based transmission scaling parameters:
        #----------------------------------------
        self.delta_pairwise_mode = self.parameters['delta_pairwise_mode']
        self.delta               = numpy.log(self.degree)/numpy.log(numpy.mean(self.degree))     if self.parameters['delta'] is None   else numpy.array(self.parameters['delta'])   if isinstance(self.parameters['delta'], (list, numpy.ndarray))   else numpy.full(fill_value=self.parameters['delta'], shape=(self.numNodes,1))
        self.delta_Q             = numpy.log(self.degree_Q)/numpy.log(numpy.mean(self.degree_Q)) if self.parameters['delta_Q'] is None else numpy.array(self.parameters['delta_Q']) if isinstance(self.parameters['delta_Q'], (list, numpy.ndarray)) else numpy.full(fill_value=self.parameters['delta_Q'], shape=(self.numNodes,1))
        #----------------------------------------
        if(self.delta.ndim == 2 and self.delta.shape[0] == self.numNodes and self.delta.shape[1] == self.numNodes):
            self.A_delta_pairwise = self.delta
        elif((self.delta.ndim == 1 and self.delta.shape[0] == self.numNodes) or (self.delta.ndim == 2 and (self.delta.shape[0] == self.numNodes or self.delta.shape[1] == self.numNodes))):
            self.delta = self.delta.reshape((self.numNodes,1))
            # Pre-multiply delta values by the adjacency matrix ("transmission weight connections")
            A_delta_pairwise_byInfected = scipy.sparse.csr_matrix.multiply(self.A, self.delta.T).tocsr()
            A_delta_pairwise_byInfectee = scipy.sparse.csr_matrix.multiply(self.A, self.delta).tocsr()    
            #------------------------------
            # Compute the effective pairwise delta values as a function of the infected/infectee pair:
            if(self.delta_pairwise_mode == 'infected'):
                self.A_delta_pairwise = A_delta_pairwise_byInfected
            elif(self.delta_pairwise_mode == 'infectee'):
                self.A_delta_pairwise = A_delta_pairwise_byInfectee
            elif(self.delta_pairwise_mode == 'min'):
                self.A_delta_pairwise = scipy.sparse.csr_matrix.minimum(A_delta_pairwise_byInfected, A_delta_pairwise_byInfectee)
            elif(self.delta_pairwise_mode == 'max'):
                self.A_delta_pairwise = scipy.sparse.csr_matrix.maximum(A_delta_pairwise_byInfected, A_delta_pairwise_byInfectee)
            elif(self.delta_pairwise_mode == 'mean'):
                self.A_delta_pairwise = (A_delta_pairwise_byInfected + A_delta_pairwise_byInfectee)/2
            elif(self.delta_pairwise_mode is None):
                self.A_delta_pairwise = self.A
            else:
                print("Unrecognized delta_pairwise_mode value (support for 'infected', 'infectee', 'min', 'max', and 'mean').")
        else:
            print("Invalid values given for delta (expected 1xN list/array or NxN 2d array)")
        #----------------------------------------
        if(self.delta_Q.ndim == 2 and self.delta_Q.shape[0] == self.numNodes and self.delta_Q.shape[1] == self.numNodes):
            self.A_Q_delta_Q_pairwise = self.delta_Q
        elif((self.delta_Q.ndim == 1 and self.delta_Q.shape[0] == self.numNodes) or (self.delta_Q.ndim == 2 and (self.delta_Q.shape[0] == self.numNodes or self.delta_Q.shape[1] == self.numNodes))):
            self.delta_Q = self.delta_Q.reshape((self.numNodes,1))
            # Pre-multiply delta_Q values by the isolation adjacency matrix ("transmission weight connections")
            A_Q_delta_Q_pairwise_byInfected      = scipy.sparse.csr_matrix.multiply(self.A_Q, self.delta_Q).tocsr()
            A_Q_delta_Q_pairwise_byInfectee      = scipy.sparse.csr_matrix.multiply(self.A_Q, self.delta_Q.T).tocsr()
            #------------------------------
            # Compute the effective pairwise delta values as a function of the infected/infectee pair:
            if(self.delta_pairwise_mode == 'infected'):
                self.A_Q_delta_Q_pairwise = A_Q_delta_Q_pairwise_byInfected
            elif(self.delta_pairwise_mode == 'infectee'):
                self.A_Q_delta_Q_pairwise = A_Q_delta_Q_pairwise_byInfectee
            elif(self.delta_pairwise_mode == 'min'):
                self.A_Q_delta_Q_pairwise = scipy.sparse.csr_matrix.minimum(A_Q_delta_Q_pairwise_byInfected, A_Q_delta_Q_pairwise_byInfectee)
            elif(self.delta_pairwise_mode == 'max'):
                self.A_Q_delta_Q_pairwise = scipy.sparse.csr_matrix.maximum(A_Q_delta_Q_pairwise_byInfected, A_Q_delta_Q_pairwise_byInfectee)
            elif(self.delta_pairwise_mode == 'mean'):
                self.A_Q_delta_Q_pairwise = (A_Q_delta_Q_pairwise_byInfected + A_Q_delta_Q_pairwise_byInfectee)/2
            elif(self.delta_pairwise_mode is None):
                self.A_Q_delta_Q_pairwise = self.A
            else:
                print("Unrecognized delta_pairwise_mode value (support for 'infected', 'infectee', 'min', 'max', and 'mean').")
        else:
            print("Invalid values given for delta_Q (expected 1xN list/array or NxN 2d array)")

        #----------------------------------------
        # Pre-calculate the pairwise delta*beta values:
        #----------------------------------------
        self.A_deltabeta          = scipy.sparse.csr_matrix.multiply(self.A_delta_pairwise, self.A_beta_pairwise)
        self.A_Q_deltabeta_Q      = scipy.sparse.csr_matrix.multiply(self.A_Q_delta_Q_pairwise, self.A_Q_beta_Q_pairwise)
        if(self.A_beta_asym_pairwise is not None):
            self.A_deltabeta_asym = scipy.sparse.csr_matrix.multiply(self.A_delta_pairwise, self.A_beta_asym_pairwise)
        else:
            self.A_deltabeta_asym = None
    

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def node_degrees(self, Amat):
        return Amat.sum(axis=0).reshape(self.numNodes,1)   # sums of adj matrix cols


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def total_num_susceptible(self, t_idx=None):
        if(t_idx is None):
            return (self.numS[:] + self.numQ_S[:])            
        else:
            return (self.numS[t_idx] + self.numQ_S[t_idx])

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def total_num_infected(self, t_idx=None):
        if(t_idx is None):
            return (self.numE[:] + self.numI_pre[:] + self.numI_sym[:] + self.numI_asym[:] + self.numH[:]
                    + self.numQ_E[:] + self.numQ_pre[:] + self.numQ_sym[:] + self.numQ_asym[:])            
        else:
            return (self.numE[t_idx] + self.numI_pre[t_idx] + self.numI_sym[t_idx] + self.numI_asym[t_idx] + self.numH[t_idx]
                    + self.numQ_E[t_idx] + self.numQ_pre[t_idx] + self.numQ_sym[t_idx] + self.numQ_asym[t_idx])

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def total_num_isolated(self, t_idx=None):
        if(t_idx is None):
            return (self.numQ_S[:] + self.numQ_E[:] + self.numQ_pre[:] + self.numQ_sym[:] + self.numQ_asym[:] + self.numQ_R[:])            
        else:
            return (self.numQ_S[t_idx] + self.numQ_E[t_idx] + self.numQ_pre[t_idx] + self.numQ_sym[t_idx] + self.numQ_asym[t_idx] + self.numQ_R[t_idx])            

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def total_num_tested(self, t_idx=None):
        if(t_idx is None):
            return (self.numTested[:])            
        else:
            return (self.numTested[t_idx])

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def total_num_positive(self, t_idx=None):
        if(t_idx is None):
            return (self.numPositive[:])            
        else:
            return (self.numPositive[t_idx])

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def total_num_recovered(self, t_idx=None):
        if(t_idx is None):
            return (self.numR[:] + self.numQ_R[:])            
        else:
            return (self.numR[t_idx] + self.numQ_R[t_idx])


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
    def calc_propensities(self):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-calculate matrix multiplication terms that may be used in multiple propensity calculations,
        # and check to see if their computation is necessary before doing the multiplication
        #------------------------------------

        self.transmissionTerms_I = numpy.zeros(shape=(self.numNodes,1))      
        if(numpy.any(self.numI_sym[self.tidx]) or numpy.any(self.numI_asym[self.tidx]) or numpy.any(self.numI_pre[self.tidx])):
            if(self.A_deltabeta_asym is not None):
                self.transmissionTerms_sym  = numpy.asarray(scipy.sparse.csr_matrix.dot(self.A_deltabeta, self.X==self.I_sym))
                self.transmissionTerms_asym = numpy.asarray(scipy.sparse.csr_matrix.dot(self.A_deltabeta_asym, ((self.X==self.I_pre)|(self.X==self.I_asym))))
                self.transmissionTerms_I    = self.transmissionTerms_sym+self.transmissionTerms_asym
            else:
                self.transmissionTerms_I    = numpy.asarray(scipy.sparse.csr_matrix.dot(self.A_deltabeta, ((self.X==self.I_sym)|(self.X==self.I_pre)|(self.X==self.I_asym))))

        #------------------------------------

        self.transmissionTerms_Q = numpy.zeros(shape=(self.numNodes,1))
        if(numpy.any(self.numQ_pre[self.tidx]) or numpy.any(self.numQ_sym[self.tidx]) or numpy.any(self.numQ_asym[self.tidx])):
            self.transmissionTerms_Q = numpy.asarray(scipy.sparse.csr_matrix.dot(self.A_Q_deltabeta_Q, ((self.X==self.Q_pre)|(self.X==self.Q_sym)|(self.X==self.Q_asym))))

        #------------------------------------

        self.transmissionTerms_IQ = numpy.zeros(shape=(self.numNodes,1))      
        if(numpy.any(self.numQ_S[self.tidx]) and (numpy.any(self.numI_sym[self.tidx]) or numpy.any(self.numI_asym[self.tidx]) or numpy.any(self.numI_pre[self.tidx]))):
            self.transmissionTerms_IQ = numpy.asarray(scipy.sparse.csr_matrix.dot(self.A_Q_deltabeta_Q, ((self.X==self.I_sym)|(self.X==self.I_pre)|(self.X==self.I_asym))))

        #------------------------------------

        numContacts_Q = numpy.zeros(shape=(self.numNodes,1))
        if(numpy.any(self.positive) and (numpy.any(self.phi_S) or numpy.any(self.phi_E) or numpy.any(self.phi_pre) or numpy.any(self.phi_sym) or numpy.any(self.phi_asym))):
            numContacts_Q = numpy.asarray(scipy.sparse.csr_matrix.dot(self.A, ((self.positive)&(self.X!=self.R)&(self.X!=self.Q_R)&(self.X!=self.F))))

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        propensities_StoE       = ( self.alpha *
                                        (self.o*(self.beta_global*self.prevalence_ext)
                                        + (1-self.o)*(
                                            self.p*((self.beta_global*self.numI_sym[self.tidx] + self.beta_asym_global*(self.numI_pre[self.tidx] + self.numI_asym[self.tidx])
                                            + self.q*self.beta_Q_global*(self.numQ_pre[self.tidx] + self.numQ_sym[self.tidx] + self.numQ_asym[self.tidx]))/self.N[self.tidx])
                                            + (1-self.p)*(numpy.divide(self.transmissionTerms_I, self.degree, out=numpy.zeros_like(self.degree), where=self.degree!=0)
                                                          + numpy.divide(self.transmissionTerms_Q, self.degree_Q, out=numpy.zeros_like(self.degree_Q), where=self.degree_Q!=0))))
                                  )*(self.X==self.S)

        propensities_QStoQE = numpy.zeros_like(propensities_StoE)
        if(numpy.any(self.X==self.Q_S)):
            propensities_QStoQE = ( self.alpha_Q * 
                                        (self.o*(self.q*self.beta_global*self.prevalence_ext)
                                        + (1-self.o)*(
                                            self.p*(self.q*(self.beta_global*self.numI_sym[self.tidx] + self.beta_asym_global*(self.numI_pre[self.tidx] + self.numI_asym[self.tidx])
                                            + self.beta_Q_global*(self.numQ_pre[self.tidx] + self.numQ_sym[self.tidx] + self.numQ_asym[self.tidx]))/self.N[self.tidx])
                                            + (1-self.p)*(numpy.divide(self.transmissionTerms_IQ+self.transmissionTerms_Q, self.degree_Q, out=numpy.zeros_like(self.degree_Q), where=self.degree_Q!=0))))
                                   )*(self.X==self.Q_S)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if(self.transition_mode == 'time_in_state'):

            propensities_EtoIPRE     = 1e5 * ((self.X==self.E) & numpy.greater(self.timer_state, 1/self.sigma))

            propensities_IPREtoISYM  = 1e5 * ((self.X==self.I_pre) & numpy.greater(self.timer_state, 1/self.lamda) & numpy.greater_equal(self.rand_a, self.a))

            propensities_IPREtoIASYM = 1e5 * ((self.X==self.I_pre) & numpy.greater(self.timer_state, 1/self.lamda) & numpy.less(self.rand_a, self.a))

            propensities_ISYMtoR     = 1e5 * ((self.X==self.I_sym) & numpy.greater(self.timer_state, 1/self.gamma) & numpy.greater_equal(self.rand_h, self.h))

            propensities_ISYMtoH     = 1e5 * ((self.X==self.I_sym) & numpy.greater(self.timer_state, 1/self.eta) & numpy.less(self.rand_h, self.h))

            propensities_IASYMtoR    = 1e5 * ((self.X==self.I_asym) & numpy.greater(self.timer_state, 1/self.gamma))

            propensities_HtoR        = 1e5 * ((self.X==self.H) & numpy.greater(self.timer_state, 1/self.gamma_H) & numpy.greater_equal(self.rand_f, self.f))

            propensities_HtoF        = 1e5 * ((self.X==self.H) & numpy.greater(self.timer_state, 1/self.mu_H) & numpy.less(self.rand_f, self.f))

            propensities_StoQS       = numpy.zeros_like(propensities_StoE)

            propensities_EtoQE       = numpy.zeros_like(propensities_StoE)

            propensities_IPREtoQPRE  = numpy.zeros_like(propensities_StoE)

            propensities_ISYMtoQSYM  = numpy.zeros_like(propensities_StoE)

            propensities_IASYMtoQASYM = numpy.zeros_like(propensities_StoE)

            propensities_QEtoQPRE    = 1e5 * ((self.X==self.Q_E) & numpy.greater(self.timer_state, 1/self.sigma_Q))

            propensities_QPREtoQSYM  = 1e5 * ((self.X==self.Q_pre) & numpy.greater(self.timer_state, 1/self.lamda_Q) & numpy.greater_equal(self.rand_a, self.a))

            propensities_QPREtoQASYM = 1e5 * ((self.X==self.Q_pre) & numpy.greater(self.timer_state, 1/self.lamda_Q) & numpy.less(self.rand_a, self.a))

            propensities_QSYMtoQR    = 1e5 * ((self.X==self.Q_sym) & numpy.greater(self.timer_state, 1/self.gamma_Q_sym) & numpy.greater_equal(self.rand_h, self.h))

            propensities_QSYMtoH     = 1e5 * ((self.X==self.Q_sym) & numpy.greater(self.timer_state, 1/self.eta_Q) & numpy.less(self.rand_h, self.h))

            propensities_QASYMtoQR   = 1e5 * ((self.X==self.Q_asym) & numpy.greater(self.timer_state, 1/self.gamma_Q_asym))

            propensities_RtoS        = 1e5 * ((self.X==self.R) & numpy.greater(self.timer_state, 1/self.xi))

            propensities__toS        = 1e5 * ((self.X!=self.F) & numpy.greater(self.timer_state, 1/self.nu))

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        else: # exponential_rates

            propensities_EtoIPRE     = self.sigma * (self.X==self.E)

            propensities_IPREtoISYM  = self.lamda * ((self.X==self.I_pre) & (numpy.greater_equal(self.rand_a, self.a)))

            propensities_IPREtoIASYM = self.lamda * ((self.X==self.I_pre) & (numpy.less(self.rand_a, self.a)))

            propensities_ISYMtoR     = self.gamma * ((self.X==self.I_sym) & (numpy.greater_equal(self.rand_h, self.h)))

            propensities_ISYMtoH     = self.eta * ((self.X==self.I_sym) & (numpy.less(self.rand_h, self.h)))

            propensities_IASYMtoR    = self.gamma_asym * (self.X==self.I_asym)

            propensities_HtoR        = self.gamma_H * ((self.X==self.H) & (numpy.greater_equal(self.rand_f, self.f)))

            propensities_HtoF        = self.mu_H * ((self.X==self.H) & (numpy.less(self.rand_f, self.f)))

            propensities_StoQS       = (self.theta_S + self.phi_S*numContacts_Q)*self.psi_S * (self.X==self.S)

            propensities_EtoQE       = (self.theta_E + self.phi_E*numContacts_Q)*self.psi_E * (self.X==self.E)

            propensities_IPREtoQPRE  = (self.theta_pre + self.phi_pre*numContacts_Q)*self.psi_pre * (self.X==self.I_pre)

            propensities_ISYMtoQSYM  = (self.theta_sym + self.phi_sym*numContacts_Q)*self.psi_sym * (self.X==self.I_sym)

            propensities_IASYMtoQASYM = (self.theta_asym + self.phi_asym*numContacts_Q)*self.psi_asym * (self.X==self.I_asym)

            propensities_QEtoQPRE    = self.sigma_Q * (self.X==self.Q_E)

            propensities_QPREtoQSYM  = self.lamda_Q * ((self.X==self.Q_pre) & (numpy.greater_equal(self.rand_a, self.a)))

            propensities_QPREtoQASYM = self.lamda_Q * ((self.X==self.Q_pre) & (numpy.less(self.rand_a, self.a)))

            propensities_QSYMtoQR    = self.gamma_Q_sym * ((self.X==self.Q_sym) & (numpy.greater_equal(self.rand_h, self.h)))

            propensities_QSYMtoH     = self.eta_Q * ((self.X==self.Q_sym) & (numpy.less(self.rand_h, self.h)))

            propensities_QASYMtoQR   = self.gamma_Q_asym * (self.X==self.Q_asym)

            propensities_RtoS        = self.xi * (self.X==self.R)

            propensities__toS        = self.nu * (self.X!=self.F)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        propensities = numpy.hstack([propensities_StoE, propensities_EtoIPRE, propensities_IPREtoISYM, propensities_IPREtoIASYM,
                                     propensities_ISYMtoR, propensities_ISYMtoH, propensities_IASYMtoR, propensities_HtoR, propensities_HtoF, 
                                     propensities_StoQS, propensities_EtoQE, propensities_IPREtoQPRE, propensities_ISYMtoQSYM, propensities_IASYMtoQASYM, 
                                     propensities_QStoQE, propensities_QEtoQPRE, propensities_QPREtoQSYM, propensities_QPREtoQASYM, 
                                     propensities_QSYMtoQR, propensities_QSYMtoH, propensities_QASYMtoQR, propensities_RtoS, propensities__toS])

        columns = [ 'StoE', 'EtoIPRE', 'IPREtoISYM', 'IPREtoIASYM',
                    'ISYMtoR', 'ISYMtoH', 'IASYMtoR', 'HtoR', 'HtoF', 
                    'StoQS', 'EtoQE', 'IPREtoQPRE', 'ISYMtoQSYM', 'IASYMtoQASYM', 
                    'QStoQE', 'QEtoQPRE', 'QPREtoQSYM', 'QPREtoQASYM', 
                    'QSYMtoQR', 'QSYMtoH', 'QASYMtoQR', 'RtoS', '_toS' ]

        return propensities, columns


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def set_isolation(self, node, isolate):
        # Move this node in/out of the appropriate isolation state:
        if(isolate == True):
            if(self.X[node] == self.S):
                self.X[node] = self.Q_S
            elif(self.X[node] == self.E):
                self.X[node] = self.Q_E
            elif(self.X[node] == self.I_pre):
                self.X[node] = self.Q_pre
            elif(self.X[node] == self.I_sym):
                self.X[node] = self.Q_sym
            elif(self.X[node] == self.I_asym):
                self.X[node] = self.Q_asym
            elif(self.X[node] == self.R):
                self.X[node] = self.Q_R
        elif(isolate == False):
            if(self.X[node] == self.Q_S):
                self.X[node] = self.S
            elif(self.X[node] == self.Q_E):
                self.X[node] = self.E
            elif(self.X[node] == self.Q_pre):
                self.X[node] = self.I_pre
            elif(self.X[node] == self.Q_sym):
                self.X[node] = self.I_sym
            elif(self.X[node] == self.Q_asym):
                self.X[node] = self.I_asym
            elif(self.X[node] == self.Q_R):
                self.X[node] = self.R
        # Reset the isolation timer:
        self.timer_isolation[node] = 0

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def set_tested(self, node, tested):
        self.tested[node] = tested
        self.testedInCurrentState[node] = tested

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def set_positive(self, node, positive):
        self.positive[node] = positive

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def introduce_exposures(self, num_new_exposures):
        exposedNodes = numpy.random.choice(range(self.numNodes), size=num_new_exposures, replace=False)
        for exposedNode in exposedNodes:
            if(self.X[exposedNode]==self.S):
                self.X[exposedNode] = self.E
            elif(self.X[exposedNode]==self.Q_S):
                self.X[exposedNode] = self.Q_E


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   

    def increase_data_series_length(self):
        self.tseries     = numpy.pad(self.tseries, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numS        = numpy.pad(self.numS, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numE        = numpy.pad(self.numE, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numI_pre    = numpy.pad(self.numI_pre, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numI_sym    = numpy.pad(self.numI_sym, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numI_asym   = numpy.pad(self.numI_asym, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numH        = numpy.pad(self.numH, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numR        = numpy.pad(self.numR, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numF        = numpy.pad(self.numF, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numQ_S      = numpy.pad(self.numQ_S, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numQ_E      = numpy.pad(self.numQ_E, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numQ_pre    = numpy.pad(self.numQ_pre, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numQ_sym    = numpy.pad(self.numQ_sym, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numQ_asym   = numpy.pad(self.numQ_asym, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numQ_R      = numpy.pad(self.numQ_R, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.N           = numpy.pad(self.N, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numTested   = numpy.pad(self.numTested, [(0, 6*self.numNodes)], mode='constant', constant_values=0)
        self.numPositive = numpy.pad(self.numPositive, [(0, 6*self.numNodes)], mode='constant', constant_values=0)

        if(self.store_Xseries):
            self.Xseries = numpy.pad(self.Xseries, [(0, 6*self.numNodes), (0,0)], mode='constant', constant_values=0)

        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                self.nodeGroupData[groupName]['numS']        = numpy.pad(self.nodeGroupData[groupName]['numS'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numE']        = numpy.pad(self.nodeGroupData[groupName]['numE'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numI_pre']    = numpy.pad(self.nodeGroupData[groupName]['numI_pre'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numI_sym']    = numpy.pad(self.nodeGroupData[groupName]['numI_sym'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numI_asym']   = numpy.pad(self.nodeGroupData[groupName]['numI_asym'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numH']        = numpy.pad(self.nodeGroupData[groupName]['numH'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numR']        = numpy.pad(self.nodeGroupData[groupName]['numR'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numF']        = numpy.pad(self.nodeGroupData[groupName]['numF'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numQ_S']      = numpy.pad(self.nodeGroupData[groupName]['numQ_S'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numQ_E']      = numpy.pad(self.nodeGroupData[groupName]['numQ_E'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numQ_pre']    = numpy.pad(self.nodeGroupData[groupName]['numQ_pre'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numQ_sym']    = numpy.pad(self.nodeGroupData[groupName]['numQ_sym'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numQ_asym']   = numpy.pad(self.nodeGroupData[groupName]['numQ_asym'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numQ_R']      = numpy.pad(self.nodeGroupData[groupName]['numQ_R'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['N']           = numpy.pad(self.nodeGroupData[groupName]['N'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numTested']   = numpy.pad(self.nodeGroupData[groupName]['numTested'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numPositive'] = numpy.pad(self.nodeGroupData[groupName]['numPositive'], [(0, 6*self.numNodes)], mode='constant', constant_values=0)

        return None

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 

    def finalize_data_series(self):
        self.tseries     = numpy.array(self.tseries, dtype=float)[:self.tidx+1]
        self.numS        = numpy.array(self.numS, dtype=float)[:self.tidx+1]
        self.numE        = numpy.array(self.numE, dtype=float)[:self.tidx+1]
        self.numI_pre    = numpy.array(self.numI_pre, dtype=float)[:self.tidx+1]
        self.numI_sym    = numpy.array(self.numI_sym, dtype=float)[:self.tidx+1]
        self.numI_asym   = numpy.array(self.numI_asym, dtype=float)[:self.tidx+1]
        self.numH        = numpy.array(self.numH, dtype=float)[:self.tidx+1]
        self.numR        = numpy.array(self.numR, dtype=float)[:self.tidx+1]
        self.numF        = numpy.array(self.numF, dtype=float)[:self.tidx+1]
        self.numQ_S      = numpy.array(self.numQ_S, dtype=float)[:self.tidx+1]
        self.numQ_E      = numpy.array(self.numQ_E, dtype=float)[:self.tidx+1]
        self.numQ_pre    = numpy.array(self.numQ_pre, dtype=float)[:self.tidx+1]
        self.numQ_sym    = numpy.array(self.numQ_sym, dtype=float)[:self.tidx+1]
        self.numQ_asym   = numpy.array(self.numQ_asym, dtype=float)[:self.tidx+1]
        self.numQ_R      = numpy.array(self.numQ_R, dtype=float)[:self.tidx+1]
        self.N           = numpy.array(self.N, dtype=float)[:self.tidx+1]
        self.numTested   = numpy.array(self.numTested, dtype=float)[:self.tidx+1]
        self.numPositive = numpy.array(self.numPositive, dtype=float)[:self.tidx+1]

        if(self.store_Xseries):
            self.Xseries = self.Xseries[:self.tidx+1, :]

        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                self.nodeGroupData[groupName]['numS']        = numpy.array(self.nodeGroupData[groupName]['numS'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numE']        = numpy.array(self.nodeGroupData[groupName]['numE'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numI_pre']    = numpy.array(self.nodeGroupData[groupName]['numI_pre'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numI_sym']    = numpy.array(self.nodeGroupData[groupName]['numI_sym'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numI_asym']   = numpy.array(self.nodeGroupData[groupName]['numI_asym'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numR']        = numpy.array(self.nodeGroupData[groupName]['numR'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numF']        = numpy.array(self.nodeGroupData[groupName]['numF'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numQ_S']      = numpy.array(self.nodeGroupData[groupName]['numQ_S'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numQ_E']      = numpy.array(self.nodeGroupData[groupName]['numQ_E'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numQ_pre']    = numpy.array(self.nodeGroupData[groupName]['numQ_pre'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numQ_sym']    = numpy.array(self.nodeGroupData[groupName]['numQ_sym'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numQ_asym']   = numpy.array(self.nodeGroupData[groupName]['numQ_asym'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numQ_R']      = numpy.array(self.nodeGroupData[groupName]['numQ_R'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['N']           = numpy.array(self.nodeGroupData[groupName]['N'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numTested']   = numpy.array(self.nodeGroupData[groupName]['numTested'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numPositive'] = numpy.array(self.nodeGroupData[groupName]['numPositive'], dtype=float)[:self.tidx+1]

        return None

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^     

    def run_iteration(self):

        if(self.tidx >= len(self.tseries)-1):
            # Room has run out in the timeseries storage arrays; double the size of these arrays:
            self.increase_data_series_length()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate 2 random numbers uniformly distributed in (0,1)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        r1 = numpy.random.rand()
        r2 = numpy.random.rand()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Calculate propensities
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        propensities, transitionTypes = self.calc_propensities()

        if(propensities.sum() > 0):

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Calculate alpha
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            propensities_flat   = propensities.ravel(order='F')
            cumsum              = propensities_flat.cumsum()
            alpha               = propensities_flat.sum()

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute the time until the next event takes place
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            tau = (1/alpha)*numpy.log(float(1/r1))
            self.t += tau
            self.timer_state += tau

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute which event takes place
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            transitionIdx   = numpy.searchsorted(cumsum,r2*alpha)
            transitionNode  = transitionIdx % self.numNodes
            transitionType  = transitionTypes[ int(transitionIdx/self.numNodes) ]

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Perform updates triggered by rate propensities:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            assert(self.X[transitionNode] == self.transitions[transitionType]['currentState'] and self.X[transitionNode]!=self.F), "Assertion error: Node "+str(transitionNode)+" has unexpected current state "+str(self.X[transitionNode])+" given the intended transition of "+str(transitionType)+"."
            self.X[transitionNode] = self.transitions[transitionType]['newState']

            self.testedInCurrentState[transitionNode] = False

            self.timer_state[transitionNode] = 0.0

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # Save information about infection events when they occur:
            if(transitionType == 'StoE' or transitionType == 'QStoQE'):
                transitionNode_GNbrs  = list(self.G[transitionNode].keys())
                transitionNode_GQNbrs = list(self.G_Q[transitionNode].keys())
                self.infectionsLog.append({ 't':                            self.t,
                                            'infected_node':                transitionNode,
                                            'infection_type':               transitionType,
                                            'infected_node_degree':         self.degree[transitionNode],
                                            'local_contact_nodes':          transitionNode_GNbrs,
                                            'local_contact_node_states':    self.X[transitionNode_GNbrs].flatten(),
                                            'isolation_contact_nodes':      transitionNode_GQNbrs,
                                            'isolation_contact_node_states':self.X[transitionNode_GQNbrs].flatten() })

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if(transitionType in ['EtoQE', 'IPREtoQPRE', 'ISYMtoQSYM', 'IASYMtoQASYM', 'ISYMtoH']):
                self.set_positive(node=transitionNode, positive=True)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        else:

            tau = 0.01
            self.t += tau
            self.timer_state += tau

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.tidx += 1
        
        self.tseries[self.tidx]     = self.t
        self.numS[self.tidx]        = numpy.clip(numpy.count_nonzero(self.X==self.S), a_min=0, a_max=self.numNodes)
        self.numE[self.tidx]        = numpy.clip(numpy.count_nonzero(self.X==self.E), a_min=0, a_max=self.numNodes)
        self.numI_pre[self.tidx]    = numpy.clip(numpy.count_nonzero(self.X==self.I_pre), a_min=0, a_max=self.numNodes)
        self.numI_sym[self.tidx]    = numpy.clip(numpy.count_nonzero(self.X==self.I_sym), a_min=0, a_max=self.numNodes)
        self.numI_asym[self.tidx]   = numpy.clip(numpy.count_nonzero(self.X==self.I_asym), a_min=0, a_max=self.numNodes)
        self.numH[self.tidx]        = numpy.clip(numpy.count_nonzero(self.X==self.H), a_min=0, a_max=self.numNodes)
        self.numR[self.tidx]        = numpy.clip(numpy.count_nonzero(self.X==self.R), a_min=0, a_max=self.numNodes)
        self.numF[self.tidx]        = numpy.clip(numpy.count_nonzero(self.X==self.F), a_min=0, a_max=self.numNodes)
        self.numQ_S[self.tidx]      = numpy.clip(numpy.count_nonzero(self.X==self.Q_S), a_min=0, a_max=self.numNodes)
        self.numQ_E[self.tidx]      = numpy.clip(numpy.count_nonzero(self.X==self.Q_E), a_min=0, a_max=self.numNodes)
        self.numQ_pre[self.tidx]    = numpy.clip(numpy.count_nonzero(self.X==self.Q_pre), a_min=0, a_max=self.numNodes)
        self.numQ_sym[self.tidx]    = numpy.clip(numpy.count_nonzero(self.X==self.Q_sym), a_min=0, a_max=self.numNodes)
        self.numQ_asym[self.tidx]   = numpy.clip(numpy.count_nonzero(self.X==self.Q_asym), a_min=0, a_max=self.numNodes)
        self.numQ_R[self.tidx]      = numpy.clip(numpy.count_nonzero(self.X==self.Q_R), a_min=0, a_max=self.numNodes)
        self.numTested[self.tidx]   = numpy.clip(numpy.count_nonzero(self.tested), a_min=0, a_max=self.numNodes)
        self.numPositive[self.tidx] = numpy.clip(numpy.count_nonzero(self.positive), a_min=0, a_max=self.numNodes)
        
        self.N[self.tidx]           = numpy.clip((self.numNodes - self.numF[self.tidx]), a_min=0, a_max=self.numNodes)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update testing and isolation statuses
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        isolatedNodes = numpy.argwhere((self.X==self.Q_S)|(self.X==self.Q_E)|(self.X==self.Q_pre)|(self.X==self.Q_sym)|(self.X==self.Q_asym)|(self.X==self.Q_R))[:,0].flatten()
        self.timer_isolation[isolatedNodes] = self.timer_isolation[isolatedNodes] + tau

        nodesExitingIsolation = numpy.argwhere(self.timer_isolation >= self.isolationTime)
        for isoNode in nodesExitingIsolation:
            self.set_isolation(node=isoNode, isolate=False)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store system states
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(self.store_Xseries):
            self.Xseries[self.tidx,:] = self.X.T

        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                self.nodeGroupData[groupName]['numS'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.S)
                self.nodeGroupData[groupName]['numE'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.E)
                self.nodeGroupData[groupName]['numI_pre'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I_pre)
                self.nodeGroupData[groupName]['numI_sym'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I_sym)
                self.nodeGroupData[groupName]['numI_asym'][self.tidx]   = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I_asym)
                self.nodeGroupData[groupName]['numH'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.H)
                self.nodeGroupData[groupName]['numR'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.R)
                self.nodeGroupData[groupName]['numF'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.F)
                self.nodeGroupData[groupName]['numQ_S'][self.tidx]      = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_S)
                self.nodeGroupData[groupName]['numQ_E'][self.tidx]      = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_E)
                self.nodeGroupData[groupName]['numQ_pre'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_pre)
                self.nodeGroupData[groupName]['numQ_sym'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_sym)
                self.nodeGroupData[groupName]['numQ_asym'][self.tidx]   = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_asym)
                self.nodeGroupData[groupName]['numQ_R'][self.tidx]      = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_R)
                self.nodeGroupData[groupName]['N'][self.tidx]           = numpy.clip((self.nodeGroupData[groupName]['numS'][0] + self.nodeGroupData[groupName]['numE'][0] + self.nodeGroupData[groupName]['numI'][0] + self.nodeGroupData[groupName]['numQ_E'][0] + self.nodeGroupData[groupName]['numQ_I'][0] + self.nodeGroupData[groupName]['numR'][0]), a_min=0, a_max=self.numNodes)
                self.nodeGroupData[groupName]['numTested'][self.tidx]   = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.tested)
                self.nodeGroupData[groupName]['numPositive'][self.tidx] = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.positive)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Terminate if tmax reached or num infections is 0:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(self.t >= self.tmax or (self.total_num_infected(self.tidx) < 1 and self.total_num_isolated(self.tidx) < 1)):
            self.finalize_data_series()
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        return True


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def run(self, T, checkpoints=None, print_interval=10, verbose='t'):
        if(T>0):
            self.tmax += T
        else:
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-process checkpoint values:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(checkpoints):
            numCheckpoints = len(checkpoints['t'])
            for chkpt_param, chkpt_values in checkpoints.items():
                assert(isinstance(chkpt_values, (list, numpy.ndarray)) and len(chkpt_values)==numCheckpoints), "Expecting a list of values with length equal to number of checkpoint times ("+str(numCheckpoints)+") for each checkpoint parameter."
            checkpointIdx  = numpy.searchsorted(checkpoints['t'], self.t) # Finds 1st index in list greater than given val
            if(checkpointIdx >= numCheckpoints):
                # We are out of checkpoints, stop checking them:
                checkpoints = None 
            else:
                checkpointTime = checkpoints['t'][checkpointIdx]

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Run the simulation loop:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        print_reset = True
        running     = True
        while running:

            running = self.run_iteration()

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Handle checkpoints if applicable:
            if(checkpoints):
                if(self.t >= checkpointTime):
                    if(verbose is not False):
                        print("[Checkpoint: Updating parameters]")
                    # A checkpoint has been reached, update param values:
                    for param in list(self.parameters.keys()):
                        if(param in list(checkpoints.keys())):
                            self.parameters.update({param: checkpoints[param][checkpointIdx]})
                    # Update parameter data structures and scenario flags:
                    self.update_parameters()
                    # Update the next checkpoint time:
                    checkpointIdx  = numpy.searchsorted(checkpoints['t'], self.t) # Finds 1st index in list greater than given val
                    if(checkpointIdx >= numCheckpoints):
                        # We are out of checkpoints, stop checking them:
                        checkpoints = None 
                    else:
                        checkpointTime = checkpoints['t'][checkpointIdx]
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if(print_interval):
                if(print_reset and (int(self.t) % print_interval == 0)):
                    if(verbose=="t"):
                        print("t = %.2f" % self.t)
                    if(verbose==True):
                        print("t = %.2f" % self.t)
                        print("\t S      = " + str(self.numS[self.tidx]))
                        print("\t E      = " + str(self.numE[self.tidx]))
                        print("\t I_pre  = " + str(self.numI_pre[self.tidx]))
                        print("\t I_sym  = " + str(self.numI_sym[self.tidx]))
                        print("\t I_asym = " + str(self.numI_asym[self.tidx]))
                        print("\t H      = " + str(self.numH[self.tidx]))
                        print("\t R      = " + str(self.numR[self.tidx]))
                        print("\t F      = " + str(self.numF[self.tidx]))
                        print("\t Q_S    = " + str(self.numQ_S[self.tidx]))
                        print("\t Q_E    = " + str(self.numQ_E[self.tidx]))
                        print("\t Q_pre  = " + str(self.numQ_pre[self.tidx]))
                        print("\t Q_sym  = " + str(self.numQ_sym[self.tidx]))
                        print("\t Q_asym = " + str(self.numQ_asym[self.tidx]))
                        print("\t Q_R    = " + str(self.numQ_R[self.tidx]))
                        
                    print_reset = False
                elif(not print_reset and (int(self.t) % 10 != 0)):
                    print_reset = True

        return True


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def plot(self, ax=None, plot_S='line', plot_E='line', plot_I_pre='line', plot_I_sym='line', plot_I_asym='line',
                            plot_H='line', plot_R='line', plot_F='line',
                            plot_Q_E='line', plot_Q_pre='line', plot_Q_sym='line', plot_Q_asym='line', 
                            plot_Q_S='line', plot_Q_R='line', combine_Q_infected=True,
                            color_S='tab:green', color_E='orange', color_I_pre='tomato', color_I_sym='crimson', color_I_asym='#F0909B', 
                            color_H='violet', color_R='tab:blue', color_F='black',
                            color_Q_E='orange', color_Q_pre='tomato', color_Q_sym='crimson', color_Q_asym='#F0909B', 
                            color_Q_S='tab:green',  color_Q_R='tab:blue', color_Q_infected='tab:purple',
                            color_reference='#E0E0E0',
                            dashed_reference_results=None, dashed_reference_label='reference', 
                            shaded_reference_results=None, shaded_reference_label='reference', 
                            vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                            ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True):

        import matplotlib.pyplot as pyplot

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create an Axes object if None provided:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(not ax):
            fig, ax = pyplot.subplots()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prepare data series to be plotted:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Sseries      = self.numS/self.numNodes if plot_percentages else self.numS
        Eseries      = self.numE/self.numNodes if plot_percentages else self.numE
        I_preseries  = self.numI_pre/self.numNodes if plot_percentages else self.numI_pre 
        I_symseries  = self.numI_sym/self.numNodes if plot_percentages else self.numI_sym
        I_asymseries = self.numI_asym/self.numNodes if plot_percentages else self.numI_asym
        Rseries      = self.numR/self.numNodes if plot_percentages else self.numR
        Hseries      = self.numH/self.numNodes if plot_percentages else self.numH
        Fseries      = self.numF/self.numNodes if plot_percentages else self.numF
        Q_Sseries    = self.numQ_S/self.numNodes if plot_percentages else self.numQ_S 
        Q_Eseries    = self.numQ_E/self.numNodes if plot_percentages else self.numQ_E 
        Q_preseries  = self.numQ_pre/self.numNodes if plot_percentages else self.numQ_pre 
        Q_asymseries = self.numQ_asym/self.numNodes if plot_percentages else self.numQ_asym 
        Q_symseries  = self.numQ_sym/self.numNodes if plot_percentages else self.numQ_sym 
        Q_Rseries    = self.numQ_R/self.numNodes if plot_percentages else self.numQ_R 
        Q_infectedseries = (Q_Eseries + Q_preseries + Q_asymseries + Q_symseries) 

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the reference data:      
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(dashed_reference_results):
            dashedReference_tseries       = dashed_reference_results.tseries[::int(self.numNodes/100)]
            dashedReference_infectedStack = dashed_reference_results.total_num_infected()[::int(self.numNodes/100)] / (self.numNodes if plot_percentages else 1)
            ax.plot(dashedReference_tseries, dashedReference_infectedStack, color='#E0E0E0', linestyle='--', label='Total infections ('+dashed_reference_label+')', zorder=0)
        if(shaded_reference_results):
            shadedReference_tseries       = shaded_reference_results.tseries
            shadedReference_infectedStack = shaded_reference_results.total_num_infected() / (self.numNodes if plot_percentages else 1)
            ax.fill_between(shaded_reference_results.tseries, shadedReference_infectedStack, 0, color='#EFEFEF', label='Total infections ('+shaded_reference_label+')', zorder=0)
            ax.plot(shaded_reference_results.tseries, shadedReference_infectedStack, color='#E0E0E0', zorder=1)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the stacked variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        topstack = numpy.zeros_like(self.tseries)
        if(any(Fseries) and plot_F=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, topstack+Fseries), topstack, color=color_F, alpha=0.75, label='$F$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, topstack+Fseries),           color=color_F, zorder=3)
            topstack = topstack+Fseries
        
        if(any(Hseries) and plot_H=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Hseries<=0, self.tseries), numpy.ma.masked_where(Hseries<=0, topstack+Hseries), topstack, color=color_H, alpha=0.75, label='$H$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Hseries<=0, self.tseries), numpy.ma.masked_where(Hseries<=0, topstack+Hseries),           color=color_H, zorder=3)
            topstack = topstack+Hseries
        
        if(combine_Q_infected and any(Q_infectedseries) and plot_Q_E=='stacked' and plot_Q_pre=='stacked' and plot_Q_sym=='stacked' and plot_Q_asym=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Q_infectedseries<=0, self.tseries), numpy.ma.masked_where(Q_infectedseries<=0, topstack+Q_infectedseries), topstack, facecolor=color_Q_infected, alpha=0.75, hatch='//////', edgecolor='white', linewidth=0.0, label='$Q_{infected}$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Q_infectedseries<=0, self.tseries), numpy.ma.masked_where(Q_infectedseries<=0, topstack+Q_infectedseries),           color=color_Q_infected, zorder=3)
            topstack = topstack+Q_infectedseries

        if(not combine_Q_infected and any(Q_Eseries) and plot_Q_E=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Q_Eseries<=0, self.tseries), numpy.ma.masked_where(Q_Eseries<=0, topstack+Q_Eseries), topstack, facecolor=color_Q_E, alpha=0.75, hatch='//////', edgecolor='white', linewidth=0.0, label='$Q_E$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Q_Eseries<=0, self.tseries), numpy.ma.masked_where(Q_Eseries<=0, topstack+Q_Eseries),           color=color_Q_E, zorder=3)
            topstack = topstack+Q_Eseries
        
        if(any(Eseries) and plot_E=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, topstack+Eseries), topstack, color=color_E, alpha=0.75, label='$E$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, topstack+Eseries),           color=color_E, zorder=3)
            topstack = topstack+Eseries
        
        if(not combine_Q_infected and any(Q_preseries) and plot_Q_pre=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Q_preseries<=0, self.tseries), numpy.ma.masked_where(Q_preseries<=0, topstack+Q_preseries), topstack, facecolor=color_Q_pre, alpha=0.75,  hatch='//////', edgecolor='white', linewidth=0.0, label='$Q_{pre}$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Q_preseries<=0, self.tseries), numpy.ma.masked_where(Q_preseries<=0, topstack+Q_preseries),           color=color_Q_pre, zorder=3)
            topstack = topstack+Q_preseries

        if(any(I_preseries) and plot_I_pre=='stacked'):
            ax.fill_between(numpy.ma.masked_where(I_preseries<=0, self.tseries), numpy.ma.masked_where(I_preseries<=0, topstack+I_preseries), topstack, color=color_I_pre, alpha=0.75, label='$I_{pre}$', zorder=2)
            ax.plot(        numpy.ma.masked_where(I_preseries<=0, self.tseries), numpy.ma.masked_where(I_preseries<=0, topstack+I_preseries),           color=color_I_pre, zorder=3)
            topstack = topstack+I_preseries
        
        if(not combine_Q_infected and any(Q_symseries) and plot_Q_sym=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Q_symseries<=0, self.tseries), numpy.ma.masked_where(Q_symseries<=0, topstack+Q_symseries), topstack, facecolor=color_Q_sym, alpha=0.75,  hatch='//////', edgecolor='white', linewidth=0.0, label='$Q_{sym}$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Q_symseries<=0, self.tseries), numpy.ma.masked_where(Q_symseries<=0, topstack+Q_symseries),           color=color_Q_sym, zorder=3)
            topstack = topstack+Q_symseries

        if(any(I_symseries) and plot_I_sym=='stacked'):
            ax.fill_between(numpy.ma.masked_where(I_symseries<=0, self.tseries), numpy.ma.masked_where(I_symseries<=0, topstack+I_symseries), topstack, color=color_I_sym, alpha=0.75, label='$I_{sym}$', zorder=2)
            ax.plot(        numpy.ma.masked_where(I_symseries<=0, self.tseries), numpy.ma.masked_where(I_symseries<=0, topstack+I_symseries),           color=color_I_sym, zorder=3)
            topstack = topstack+I_symseries
        
        if(not combine_Q_infected and any(Q_asymseries) and plot_Q_asym=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Q_asymseries<=0, self.tseries), numpy.ma.masked_where(Q_asymseries<=0, topstack+Q_asymseries), topstack, facecolor=color_Q_asym, alpha=0.75,  hatch='//////', edgecolor='white', linewidth=0.0, label='$Q_{asym}$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Q_asymseries<=0, self.tseries), numpy.ma.masked_where(Q_asymseries<=0, topstack+Q_asymseries),           color=color_Q_asym, zorder=3)
            topstack = topstack+Q_asymseries

        if(any(I_asymseries) and plot_I_asym=='stacked'):
            ax.fill_between(numpy.ma.masked_where(I_asymseries<=0, self.tseries), numpy.ma.masked_where(I_asymseries<=0, topstack+I_asymseries), topstack, color=color_I_asym, alpha=0.75, label='$I_{asym}$', zorder=2)
            ax.plot(        numpy.ma.masked_where(I_asymseries<=0, self.tseries), numpy.ma.masked_where(I_asymseries<=0, topstack+I_asymseries),           color=color_I_asym, zorder=3)
            topstack = topstack+I_asymseries
        
        if(any(Q_Rseries) and plot_Q_R=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Q_Rseries<=0, self.tseries), numpy.ma.masked_where(Q_Rseries<=0, topstack+Q_Rseries), topstack, facecolor=color_Q_R, alpha=0.75, hatch='//////', edgecolor='white', linewidth=0.0, label='$Q_R$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Q_Rseries<=0, self.tseries), numpy.ma.masked_where(Q_Rseries<=0, topstack+Q_Rseries),           color=color_Q_R, zorder=3)
            topstack = topstack+Q_Rseries

        if(any(Rseries) and plot_R=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, topstack+Rseries), topstack, color=color_R, alpha=0.75, label='$R$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, topstack+Rseries),           color=color_R, zorder=3)
            topstack = topstack+Rseries
        
        if(any(Q_Sseries) and plot_Q_S=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Q_Sseries<=0, self.tseries), numpy.ma.masked_where(Q_Sseries<=0, topstack+Q_Sseries), topstack, facecolor=color_Q_S, alpha=0.75, hatch='//////', edgecolor='white', linewidth=0.0, label='$Q_S$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Q_Sseries<=0, self.tseries), numpy.ma.masked_where(Q_Sseries<=0, topstack+Q_Sseries),           color=color_Q_S, zorder=3)
            topstack = topstack+Q_Sseries

        if(any(Sseries) and plot_S=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, topstack+Sseries), topstack, color=color_S, alpha=0.75, label='$S$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, topstack+Sseries),           color=color_S, zorder=3)
            topstack = topstack+Sseries
        
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the shaded variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(any(Fseries) and plot_F=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, Fseries), 0, color=color_F, alpha=0.75, label='$F$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, Fseries),    color=color_F, zorder=5)
        
        if(any(Hseries) and plot_H=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Hseries<=0, self.tseries), numpy.ma.masked_where(Hseries<=0, Hseries), 0, color=color_H, alpha=0.75, label='$H$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Hseries<=0, self.tseries), numpy.ma.masked_where(Hseries<=0, Hseries),    color=color_H, zorder=5)
        
        if(combine_Q_infected and any(Q_infectedseries) and plot_Q_E=='shaded' and plot_Q_pre=='shaded' and plot_Q_sym=='shaded' and plot_Q_asym=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Q_infectedseries<=0, self.tseries), numpy.ma.masked_where(Q_infectedseries<=0, Q_infectedseries), 0, color=color_Q_infected, alpha=0.75, hatch='//////', edgecolor='white', linewidth=0.0, label='$Q_{infected}$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Q_infectedseries<=0, self.tseries), numpy.ma.masked_where(Q_infectedseries<=0, Q_infectedseries),    color=color_Q_infected, zorder=5)

        if(not combine_Q_infected and any(Q_Eseries) and plot_Q_E=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Q_Eseries<=0, self.tseries), numpy.ma.masked_where(Q_Eseries<=0, Q_Eseries), 0, facecolor=color_Q_E, alpha=0.75, hatch='//////', edgecolor='white', linewidth=0.0, label='$Q_E$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Q_Eseries<=0, self.tseries), numpy.ma.masked_where(Q_Eseries<=0, Q_Eseries),    color=color_Q_E, zorder=5)
        
        if(any(Eseries) and plot_E=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, Eseries), 0, color=color_E, alpha=0.75, label='$E$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, Eseries),    color=color_E, zorder=5)
        
        if(not combine_Q_infected and any(Q_preseries) and plot_Q_pre=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Q_preseries<=0, self.tseries), numpy.ma.masked_where(Q_preseries<=0, Q_preseries), 0, facecolor=color_Q_pre, alpha=0.75,  hatch='//////', edgecolor='white', linewidth=0.0, label='$Q_{pre}$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Q_preseries<=0, self.tseries), numpy.ma.masked_where(Q_preseries<=0, Q_preseries),    color=color_Q_pre, zorder=5)

        if(any(I_preseries) and plot_I_pre=='shaded'):
            ax.fill_between(numpy.ma.masked_where(I_preseries<=0, self.tseries), numpy.ma.masked_where(I_preseries<=0, I_preseries), 0, color=color_I_pre, alpha=0.75, label='$I_{pre}$', zorder=4)
            ax.plot(        numpy.ma.masked_where(I_preseries<=0, self.tseries), numpy.ma.masked_where(I_preseries<=0, I_preseries),    color=color_I_pre, zorder=5)
        
        if(not combine_Q_infected and any(Q_symseries) and plot_Q_sym=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Q_symseries<=0, self.tseries), numpy.ma.masked_where(Q_symseries<=0, Q_symseries), 0, facecolor=color_Q_sym, alpha=0.75,  hatch='//////', edgecolor='white', linewidth=0.0, label='$Q_{sym}$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Q_symseries<=0, self.tseries), numpy.ma.masked_where(Q_symseries<=0, Q_symseries),    color=color_Q_sym, zorder=5)

        if(any(I_symseries) and plot_I_sym=='shaded'):
            ax.fill_between(numpy.ma.masked_where(I_symseries<=0, self.tseries), numpy.ma.masked_where(I_symseries<=0, I_symseries), 0, color=color_I_sym, alpha=0.75, label='$I_{sym}$', zorder=4)
            ax.plot(        numpy.ma.masked_where(I_symseries<=0, self.tseries), numpy.ma.masked_where(I_symseries<=0, I_symseries),    color=color_I_sym, zorder=5)
        
        if(not combine_Q_infected and any(Q_asymseries) and plot_Q_asym=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Q_asymseries<=0, self.tseries), numpy.ma.masked_where(Q_asymseries<=0, Q_asymseries), 0, facecolor=color_Q_asym, alpha=0.75,  hatch='//////', edgecolor='white', linewidth=0.0, label='$Q_{asym}$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Q_asymseries<=0, self.tseries), numpy.ma.masked_where(Q_asymseries<=0, Q_asymseries),    color=color_Q_asym, zorder=5)

        if(any(I_asymseries) and plot_I_asym=='shaded'):
            ax.fill_between(numpy.ma.masked_where(I_asymseries<=0, self.tseries), numpy.ma.masked_where(I_asymseries<=0, I_asymseries), 0, color=color_I_asym, alpha=0.75, label='$I_{asym}$', zorder=4)
            ax.plot(        numpy.ma.masked_where(I_asymseries<=0, self.tseries), numpy.ma.masked_where(I_asymseries<=0, I_asymseries),    color=color_I_asym, zorder=5)
        
        if(any(Q_Rseries) and plot_Q_R=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Q_Rseries<=0, self.tseries), numpy.ma.masked_where(Q_Rseries<=0, Q_Rseries), 0, facecolor=color_Q_R, alpha=0.75, hatch='//////', edgecolor='white', linewidth=0.0, label='$Q_R$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Q_Rseries<=0, self.tseries), numpy.ma.masked_where(Q_Rseries<=0, Q_Rseries),    color=color_Q_R, zorder=5)

        if(any(Rseries) and plot_R=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, Rseries), 0, color=color_R, alpha=0.75, label='$R$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, Rseries),    color=color_R, zorder=5)

        if(any(Q_Sseries) and plot_Q_S=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Q_Sseries<=0, self.tseries), numpy.ma.masked_where(Q_Sseries<=0, Q_Sseries), 0, facecolor=color_Q_S, alpha=0.75, hatch='//////', edgecolor='white', linewidth=0.0, label='$Q_S$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Q_Sseries<=0, self.tseries), numpy.ma.masked_where(Q_Sseries<=0, Q_Sseries),    color=color_Q_S, zorder=5)
        
        if(any(Sseries) and plot_S=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, Sseries), 0, color=color_S, alpha=0.75, label='$S$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, Sseries),    color=color_S, zorder=5)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the line variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(any(Fseries) and plot_F=='line'):
            ax.plot(numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, Fseries), color=color_F, label='$F$', zorder=6)
        
        if(any(Hseries) and plot_H=='line'):
            ax.plot(numpy.ma.masked_where(Hseries<=0, self.tseries), numpy.ma.masked_where(Hseries<=0, Hseries), color=color_H, label='$H$', zorder=6)
        
        if(combine_Q_infected and any(Q_infectedseries) and plot_Q_E=='line' and plot_Q_pre=='line' and plot_Q_sym=='line' and plot_Q_asym=='line'):
            ax.plot(numpy.ma.masked_where(Q_infectedseries<=0, self.tseries), numpy.ma.masked_where(Q_infectedseries<=0, Q_infectedseries), color=color_Q_infected, label='$Q_{infected}$', zorder=6)

        if(not combine_Q_infected and any(Q_Eseries) and plot_Q_E=='line'):
            ax.plot(numpy.ma.masked_where(Q_Eseries<=0, self.tseries), numpy.ma.masked_where(Q_Eseries<=0, Q_Eseries), color=color_Q_E, label='$Q_E$', zorder=6)
        
        if(any(Eseries) and plot_E=='line'):
            ax.plot(numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, Eseries), color=color_E, label='$E$', zorder=6)
        
        if(not combine_Q_infected and any(Q_preseries) and plot_Q_pre=='line'):
            ax.plot(numpy.ma.masked_where(Q_preseries<=0, self.tseries), numpy.ma.masked_where(Q_preseries<=0, Q_preseries), color=color_Q_pre, label='$Q_{pre}$', zorder=6)

        if(any(I_preseries) and plot_I_pre=='line'):
            ax.plot(numpy.ma.masked_where(I_preseries<=0, self.tseries), numpy.ma.masked_where(I_preseries<=0, I_preseries), color=color_I_pre, label='$I_{pre}$', zorder=6)
        
        if(not combine_Q_infected and any(Q_symseries) and plot_Q_sym=='line'):
            ax.plot(numpy.ma.masked_where(Q_symseries<=0, self.tseries), numpy.ma.masked_where(Q_symseries<=0, Q_symseries), color=color_Q_sym, label='$Q_{sym}$', zorder=6)

        if(any(I_symseries) and plot_I_sym=='line'):
            ax.plot(numpy.ma.masked_where(I_symseries<=0, self.tseries), numpy.ma.masked_where(I_symseries<=0, I_symseries), color=color_I_sym, label='$I_{sym}$', zorder=6)
        
        if(not combine_Q_infected and any(Q_asymseries) and plot_Q_asym=='line'):
            ax.plot(numpy.ma.masked_where(Q_asymseries<=0, self.tseries), numpy.ma.masked_where(Q_asymseries<=0, Q_asymseries), color=color_Q_asym, label='$Q_{asym}$', zorder=6)

        if(any(I_asymseries) and plot_I_asym=='line'):
            ax.plot(numpy.ma.masked_where(I_asymseries<=0, self.tseries), numpy.ma.masked_where(I_asymseries<=0, I_asymseries), color=color_I_asym, label='$I_{asym}$', zorder=6)
        
        if(any(Q_Rseries) and plot_Q_R=='line'):
            ax.plot(numpy.ma.masked_where(Q_Rseries<=0, self.tseries), numpy.ma.masked_where(Q_Rseries<=0, Q_Rseries), color=color_Q_R, linestyle='--', label='$Q_R$', zorder=6)

        if(any(Rseries) and plot_R=='line'):
            ax.plot(numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, Rseries), color=color_R, label='$R$', zorder=6)

        if(any(Q_Sseries) and plot_Q_S=='line'):
            ax.plot(numpy.ma.masked_where(Q_Sseries<=0, self.tseries), numpy.ma.masked_where(Q_Sseries<=0, Q_Sseries), color=color_Q_S, linestyle='--', label='$Q_S$', zorder=6)
        
        if(any(Sseries) and plot_S=='line'):
            ax.plot(numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, Sseries), color=color_S, label='$S$', zorder=6)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the vertical line annotations:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(len(vlines)>0 and len(vline_colors)==0):
            vline_colors = ['gray']*len(vlines)
        if(len(vlines)>0 and len(vline_labels)==0):
            vline_labels = [None]*len(vlines)
        if(len(vlines)>0 and len(vline_styles)==0):
            vline_styles = [':']*len(vlines)
        for vline_x, vline_color, vline_style, vline_label in zip(vlines, vline_colors, vline_styles, vline_labels):
            if(vline_x is not None):
                ax.axvline(x=vline_x, color=vline_color, linestyle=vline_style, alpha=1, label=vline_label)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the plot labels:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ax.set_xlabel('days')
        ax.set_ylabel('percent of population' if plot_percentages else 'number of individuals')
        ax.set_xlim(0, (max(self.tseries) if not xlim else xlim))
        ax.set_ylim(0, ylim)
        if(plot_percentages):
            ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])
        if(legend):
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend(legend_handles[::-1], legend_labels[::-1], loc='upper right', facecolor='white', edgecolor='none', framealpha=0.9, prop={'size': 8})
        if(title):
            ax.set_title(title, size=12)
        if(side_title):
            ax.annotate(side_title, (0, 0.5), xytext=(-45, 0), ha='right', va='center',
                size=12, rotation=90, xycoords='axes fraction', textcoords='offset points')
       
        return ax


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def figure_basic(self,  plot_S='line', plot_E='line', plot_I_pre='line', plot_I_sym='line', plot_I_asym='line',
                            plot_H='line', plot_R='line', plot_F='line',
                            plot_Q_E='line', plot_Q_pre='line', plot_Q_sym='line', plot_Q_asym='line', 
                            plot_Q_S=False, plot_Q_R=False, combine_Q_infected=True,
                            color_S='tab:green', color_E='orange', color_I_pre='tomato', color_I_sym='crimson', color_I_asym='#F0909B', 
                            color_H='violet', color_R='tab:blue', color_F='black',
                            color_Q_E='orange', color_Q_pre='tomato', color_Q_sym='crimson', color_Q_asym='#F0909B', 
                            color_Q_S='tab:green',  color_Q_R='tab:blue', color_Q_infected='tab:purple',
                            color_reference='#E0E0E0',
                            dashed_reference_results=None, dashed_reference_label='reference', 
                            shaded_reference_results=None, shaded_reference_label='reference', 
                            vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                            ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True,
                            figsize=(12,8), use_seaborn=True, show=True):

        import matplotlib.pyplot as pyplot

        fig, ax = pyplot.subplots(figsize=figsize)

        if(use_seaborn):
            import seaborn
            seaborn.set_style('ticks')
            seaborn.despine()

        self.plot(ax=ax, plot_S=plot_S, plot_E=plot_E, plot_I_pre=plot_I_pre, plot_I_sym=plot_I_sym, plot_I_asym=plot_I_asym,
                        plot_H=plot_H, plot_R=plot_R, plot_F=plot_F,
                        plot_Q_E=plot_Q_E, plot_Q_pre=plot_Q_pre, plot_Q_sym=plot_Q_sym, plot_Q_asym=plot_Q_asym, 
                        plot_Q_S=plot_Q_S, plot_Q_R=plot_Q_R, combine_Q_infected=combine_Q_infected,
                        color_S=color_S, color_E=color_E, color_I_pre=color_I_pre, color_I_sym=color_I_sym, color_I_asym=color_I_asym, 
                        color_H=color_H, color_R=color_R, color_F=color_F,
                        color_Q_E=color_Q_E, color_Q_pre=color_Q_pre, color_Q_sym=color_Q_sym, color_Q_asym=color_Q_asym, 
                        color_Q_S=color_Q_S,  color_Q_R=color_Q_R, color_Q_infected=color_Q_infected,
                        color_reference=color_reference,
                        dashed_reference_results=dashed_reference_results, dashed_reference_label=dashed_reference_label, 
                        shaded_reference_results=shaded_reference_results, shaded_reference_label=shaded_reference_label, 
                        vlines=vlines, vline_colors=vline_colors, vline_styles=vline_styles, vline_labels=vline_labels,
                        ylim=ylim, xlim=xlim, legend=legend, title=title, side_title=side_title, plot_percentages=plot_percentages)

        if(show):
            pyplot.show()

        return fig, ax


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def figure_infections(self, plot_S=False, plot_E='stacked', plot_I_pre='stacked', plot_I_sym='stacked', plot_I_asym='stacked',
                                plot_H='stacked', plot_R=False, plot_F='stacked',
                                plot_Q_E='stacked', plot_Q_pre='stacked', plot_Q_sym='stacked', plot_Q_asym='stacked', 
                                plot_Q_S=False, plot_Q_R=False, combine_Q_infected=True,
                                color_S='tab:green', color_E='orange', color_I_pre='tomato', color_I_sym='crimson', color_I_asym='#F0909B', 
                                color_H='violet', color_R='tab:blue', color_F='black',
                                color_Q_E='orange', color_Q_pre='tomato', color_Q_sym='crimson', color_Q_asym='#F0909B', 
                                color_Q_S='tab:green',  color_Q_R='tab:blue', color_Q_infected='tab:purple',
                                color_reference='#E0E0E0',
                                dashed_reference_results=None, dashed_reference_label='reference', 
                                shaded_reference_results=None, shaded_reference_label='reference', 
                                vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                                ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True,
                                figsize=(12,8), use_seaborn=True, show=True):

        import matplotlib.pyplot as pyplot

        fig, ax = pyplot.subplots(figsize=figsize)

        if(use_seaborn):
            import seaborn
            seaborn.set_style('ticks')
            seaborn.despine()

        self.plot(ax=ax, plot_S=plot_S, plot_E=plot_E, plot_I_pre=plot_I_pre, plot_I_sym=plot_I_sym, plot_I_asym=plot_I_asym,
                        plot_H=plot_H, plot_R=plot_R, plot_F=plot_F,
                        plot_Q_E=plot_Q_E, plot_Q_pre=plot_Q_pre, plot_Q_sym=plot_Q_sym, plot_Q_asym=plot_Q_asym, 
                        plot_Q_S=plot_Q_S, plot_Q_R=plot_Q_R, combine_Q_infected=combine_Q_infected,
                        color_S=color_S, color_E=color_E, color_I_pre=color_I_pre, color_I_sym=color_I_sym, color_I_asym=color_I_asym, 
                        color_H=color_H, color_R=color_R, color_F=color_F,
                        color_Q_E=color_Q_E, color_Q_pre=color_Q_pre, color_Q_sym=color_Q_sym, color_Q_asym=color_Q_asym, 
                        color_Q_S=color_Q_S,  color_Q_R=color_Q_R, color_Q_infected=color_Q_infected,
                        color_reference=color_reference,
                        dashed_reference_results=dashed_reference_results, dashed_reference_label=dashed_reference_label, 
                        shaded_reference_results=shaded_reference_results, shaded_reference_label=shaded_reference_label, 
                        vlines=vlines, vline_colors=vline_colors, vline_styles=vline_styles, vline_labels=vline_labels, 
                        ylim=ylim, xlim=xlim, legend=legend, title=title, side_title=side_title, plot_percentages=plot_percentages)

        if(show):
            pyplot.show()

        return fig, ax
        

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







