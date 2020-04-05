from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as networkx
import numpy as numpy
import scipy as scipy
import scipy.integrate


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
                    theta_E=0, theta_I=0, phi_E=0, phi_I=0, psi_E=0, psi_I=0, q=0,
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
                            'theta_E':theta_E, 'theta_I':theta_I, 'phi_E':phi_E, 'phi_I':phi_I, 'psi_E':phi_E, 'psi_I':psi_I }
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
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
    def calc_propensities(self):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-calculate matrix multiplication terms that may be used in multiple propensity calculations,
        # and check to see if their computation is necessary before doing the multiplication
        numContacts_I       = numpy.zeros(shape=(self.numNodes,1))
        transmissionTerms_I = numpy.zeros(shape=(self.numNodes,1))
        if(numpy.any(self.numI[self.tidx]) 
            and numpy.any(self.beta!=0)):
            transmissionTerms_I = numpy.asarray( scipy.sparse.csr_matrix.dot(self.A_beta, self.X==self.I) )

        numQuarantineContacts_DI = numpy.zeros(shape=(self.numNodes,1))
        transmissionTerms_DI = numpy.zeros(shape=(self.numNodes,1))
        if(self.testing_scenario 
            and numpy.any(self.numD_I[self.tidx])
            and numpy.any(self.beta_D)):
            transmissionTerms_DI = numpy.asarray( scipy.sparse.csr_matrix.dot(self.A_Q_beta_D, self.X==self.D_I) )

        numContacts_D = numpy.zeros(shape=(self.numNodes,1))
        if(self.tracing_scenario 
            and (numpy.any(self.numD_E[self.tidx]) or numpy.any(self.numD_I[self.tidx]))):
            numContacts_D = numpy.asarray( scipy.sparse.csr_matrix.dot(self.A, self.X==self.D_E)
                                            + scipy.sparse.csr_matrix.dot(self.A, self.X==self.D_I) )

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
            self.Xseries = numpy.pad(self.Xseries, [(0, 5*self.numNodes), (0,0)], mode=constant, constant_values=0)

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




