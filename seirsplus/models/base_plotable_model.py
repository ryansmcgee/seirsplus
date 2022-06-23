"""Common model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

class BasePlotableModel:
    """
    Base model to abstract common plotting features.
    """

    numE = None
    numF = None
    numI = None
    numQ_E = None
    numQ_I = None
    numR = None
    numS = None
    tseries = None

    plotting_number_property = None
    """Property to access the number to base plotting on."""

    def plot(self, ax=None,  plot_S='line', plot_E='line', plot_I='line',plot_R='line', plot_F='line',
                            plot_Q_E='line', plot_Q_I='line', combine_Q=True,
                            color_S='tab:green', color_E='orange', color_I='crimson', color_R='tab:blue', color_F='black',
                            color_Q_E='mediumorchid', color_Q_I='mediumorchid', color_reference='#E0E0E0',
                            dashed_reference_results=None, dashed_reference_label='reference',
                            shaded_reference_results=None, shaded_reference_label='reference',
                            vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                            ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True):

        import matplotlib.pyplot as pyplot

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create an Axes object if None provided:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if not ax:
            fig, ax = pyplot.subplots()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prepare data series to be plotted:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Fseries     = self.numF/getattr(self, self.plotting_number_property) if plot_percentages else self.numF
        Eseries     = self.numE/getattr(self, self.plotting_number_property) if plot_percentages else self.numE
        Dseries     = (self.numQ_E+self.numQ_I)/getattr(self, self.plotting_number_property) if plot_percentages else (self.numQ_E+self.numQ_I)
        Q_Eseries   = self.numQ_E/getattr(self, self.plotting_number_property) if plot_percentages else self.numQ_E
        Q_Iseries   = self.numQ_I/getattr(self, self.plotting_number_property) if plot_percentages else self.numQ_I
        Iseries     = self.numI/getattr(self, self.plotting_number_property) if plot_percentages else self.numI
        Rseries     = self.numR/getattr(self, self.plotting_number_property) if plot_percentages else self.numR
        Sseries     = self.numS/getattr(self, self.plotting_number_property) if plot_percentages else self.numS

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the reference data:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if dashed_reference_results:
            dashedReference_tseries  = dashed_reference_results.tseries[::int(getattr(self, self.plotting_number_property)/100)]
            dashedReference_IDEstack = (dashed_reference_results.numI + dashed_reference_results.numQ_I + dashed_reference_results.numQ_E + dashed_reference_results.numE)[::int(getattr(self, self.plotting_number_property)/100)] / (getattr(self, self.plotting_number_property) if plot_percentages else 1)
            ax.plot(dashedReference_tseries, dashedReference_IDEstack, color='#E0E0E0', linestyle='--', label='$I+D+E$ ('+dashed_reference_label+')', zorder=0)
        if shaded_reference_results:
            shadedReference_tseries  = shaded_reference_results.tseries
            shadedReference_IDEstack = (shaded_reference_results.numI + shaded_reference_results.numQ_I + shaded_reference_results.numQ_E + shaded_reference_results.numE) / (getattr(self, self.plotting_number_property) if plot_percentages else 1)
            ax.fill_between(shaded_reference_results.tseries, shadedReference_IDEstack, 0, color='#EFEFEF', label='$I+D+E$ ('+shaded_reference_label+')', zorder=0)
            ax.plot(shaded_reference_results.tseries, shadedReference_IDEstack, color='#E0E0E0', zorder=1)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the stacked variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        topstack = numpy.zeros_like(self.tseries)
        if (any(Fseries) and plot_F=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, topstack+Fseries), topstack, color=color_F, alpha=0.5, label='$F$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, topstack+Fseries),           color=color_F, zorder=3)
            topstack = topstack+Fseries
        if (any(Eseries) and plot_E=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, topstack+Eseries), topstack, color=color_E, alpha=0.5, label='$E$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, topstack+Eseries),           color=color_E, zorder=3)
            topstack = topstack+Eseries
        if (combine_Q and plot_Q_E=='stacked' and plot_Q_I=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, topstack+Dseries), topstack, color=color_Q_E, alpha=0.5, label='$Q_{all}$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, topstack+Dseries),           color=color_Q_E, zorder=3)
            topstack = topstack+Dseries
        else:
            if (any(Q_Eseries) and plot_Q_E=='stacked'):
                ax.fill_between(numpy.ma.masked_where(Q_Eseries<=0, self.tseries), numpy.ma.masked_where(Q_Eseries<=0, topstack+Q_Eseries), topstack, color=color_Q_E, alpha=0.5, label='$Q_E$', zorder=2)
                ax.plot(        numpy.ma.masked_where(Q_Eseries<=0, self.tseries), numpy.ma.masked_where(Q_Eseries<=0, topstack+Q_Eseries),           color=color_Q_E, zorder=3)
                topstack = topstack+Q_Eseries
            if (any(Q_Iseries) and plot_Q_I=='stacked'):
                ax.fill_between(numpy.ma.masked_where(Q_Iseries<=0, self.tseries), numpy.ma.masked_where(Q_Iseries<=0, topstack+Q_Iseries), topstack, color=color_Q_I, alpha=0.5, label='$Q_I$', zorder=2)
                ax.plot(        numpy.ma.masked_where(Q_Iseries<=0, self.tseries), numpy.ma.masked_where(Q_Iseries<=0, topstack+Q_Iseries),           color=color_Q_I, zorder=3)
                topstack = topstack+Q_Iseries
        if (any(Iseries) and plot_I=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, topstack+Iseries), topstack, color=color_I, alpha=0.5, label='$I$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, topstack+Iseries),           color=color_I, zorder=3)
            topstack = topstack+Iseries
        if (any(Rseries) and plot_R=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, topstack+Rseries), topstack, color=color_R, alpha=0.5, label='$R$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, topstack+Rseries),           color=color_R, zorder=3)
            topstack = topstack+Rseries
        if (any(Sseries) and plot_S=='stacked'):
            ax.fill_between(numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, topstack+Sseries), topstack, color=color_S, alpha=0.5, label='$S$', zorder=2)
            ax.plot(        numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, topstack+Sseries),           color=color_S, zorder=3)
            topstack = topstack+Sseries


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the shaded variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if (any(Fseries) and plot_F=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, Fseries), 0, color=color_F, alpha=0.5, label='$F$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, Fseries),    color=color_F, zorder=5)
        if (any(Eseries) and plot_E=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, Eseries), 0, color=color_E, alpha=0.5, label='$E$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, Eseries),    color=color_E, zorder=5)
        if (combine_Q and (any(Dseries) and plot_Q_E=='shaded' and plot_Q_I=='shaded')):
            ax.fill_between(numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, Dseries), 0, color=color_Q_E, alpha=0.5, label='$Q_{all}$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, Dseries),    color=color_Q_E, zorder=5)
        else:
            if (any(Q_Eseries) and plot_Q_E=='shaded'):
                ax.fill_between(numpy.ma.masked_where(Q_Eseries<=0, self.tseries), numpy.ma.masked_where(Q_Eseries<=0, Q_Eseries), 0, color=color_Q_E, alpha=0.5, label='$Q_E$', zorder=4)
                ax.plot(        numpy.ma.masked_where(Q_Eseries<=0, self.tseries), numpy.ma.masked_where(Q_Eseries<=0, Q_Eseries),    color=color_Q_E, zorder=5)
            if (any(Q_Iseries) and plot_Q_I=='shaded'):
                ax.fill_between(numpy.ma.masked_where(Q_Iseries<=0, self.tseries), numpy.ma.masked_where(Q_Iseries<=0, Q_Iseries), 0, color=color_Q_I, alpha=0.5, label='$Q_I$', zorder=4)
                ax.plot(        numpy.ma.masked_where(Q_Iseries<=0, self.tseries), numpy.ma.masked_where(Q_Iseries<=0, Q_Iseries),    color=color_Q_I, zorder=5)
        if (any(Iseries) and plot_I=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, Iseries), 0, color=color_I, alpha=0.5, label='$I$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, Iseries),    color=color_I, zorder=5)
        if (any(Sseries) and plot_S=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, Sseries), 0, color=color_S, alpha=0.5, label='$S$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, Sseries),    color=color_S, zorder=5)
        if (any(Rseries) and plot_R=='shaded'):
            ax.fill_between(numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, Rseries), 0, color=color_R, alpha=0.5, label='$R$', zorder=4)
            ax.plot(        numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, Rseries),    color=color_R, zorder=5)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the line variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if (any(Fseries) and plot_F=='line'):
            ax.plot(numpy.ma.masked_where(Fseries<=0, self.tseries), numpy.ma.masked_where(Fseries<=0, Fseries), color=color_F, label='$F$', zorder=6)
        if (any(Eseries) and plot_E=='line'):
            ax.plot(numpy.ma.masked_where(Eseries<=0, self.tseries), numpy.ma.masked_where(Eseries<=0, Eseries), color=color_E, label='$E$', zorder=6)
        if (combine_Q and (any(Dseries) and plot_Q_E=='line' and plot_Q_I == 'line')):
            ax.plot(numpy.ma.masked_where(Dseries<=0, self.tseries), numpy.ma.masked_where(Dseries<=0, Dseries), color=color_Q_E, label='$Q_{all}$', zorder=6)
        else:
            if (any(Q_Eseries) and plot_Q_E=='line'):
                ax.plot(numpy.ma.masked_where(Q_Eseries<=0, self.tseries), numpy.ma.masked_where(Q_Eseries<=0, Q_Eseries), color=color_Q_E, label='$Q_E$', zorder=6)
            if (any(Q_Iseries) and plot_Q_I=='line'):
                ax.plot(numpy.ma.masked_where(Q_Iseries<=0, self.tseries), numpy.ma.masked_where(Q_Iseries<=0, Q_Iseries), color=color_Q_I, label='$Q_I$', zorder=6)
        if (any(Iseries) and plot_I=='line'):
            ax.plot(numpy.ma.masked_where(Iseries<=0, self.tseries), numpy.ma.masked_where(Iseries<=0, Iseries), color=color_I, label='$I$', zorder=6)
        if (any(Sseries) and plot_S=='line'):
            ax.plot(numpy.ma.masked_where(Sseries<=0, self.tseries), numpy.ma.masked_where(Sseries<=0, Sseries), color=color_S, label='$S$', zorder=6)
        if (any(Rseries) and plot_R=='line'):
            ax.plot(numpy.ma.masked_where(Rseries<=0, self.tseries), numpy.ma.masked_where(Rseries<=0, Rseries), color=color_R, label='$R$', zorder=6)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the vertical line annotations:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if (len(vlines)>0 and len(vline_colors)==0):
            vline_colors = ['gray']*len(vlines)
        if (len(vlines)>0 and len(vline_labels)==0):
            vline_labels = [None]*len(vlines)
        if (len(vlines)>0 and len(vline_styles)==0):
            vline_styles = [':']*len(vlines)
        for vline_x, vline_color, vline_style, vline_label in zip(vlines, vline_colors, vline_styles, vline_labels):
            if vline_x is not None:
                ax.axvline(x=vline_x, color=vline_color, linestyle=vline_style, alpha=1, label=vline_label)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the plot labels:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ax.set_xlabel('days')
        ax.set_ylabel('percent of population' if plot_percentages else 'number of individuals')
        ax.set_xlim(0, (max(self.tseries) if not xlim else xlim))
        ax.set_ylim(0, ylim)
        if plot_percentages:
            ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])
        if legend:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend(legend_handles[::-1], legend_labels[::-1], loc='upper right', facecolor='white', edgecolor='none', framealpha=0.9, prop={'size': 8})
        if title:
            ax.set_title(title, size=12)
        if side_title:
            ax.annotate(side_title, (0, 0.5), xytext=(-45, 0), ha='right', va='center',
                size=12, rotation=90, xycoords='axes fraction', textcoords='offset points')

        return ax


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def figure_basic(self, plot_S='line', plot_E='line', plot_I='line',plot_R='line', plot_F='line',
                        plot_Q_E='line', plot_Q_I='line', combine_Q=True,
                        color_S='tab:green', color_E='orange', color_I='crimson', color_R='tab:blue', color_F='black',
                        color_Q_E='mediumorchid', color_Q_I='mediumorchid', color_reference='#E0E0E0',
                        dashed_reference_results=None, dashed_reference_label='reference',
                        shaded_reference_results=None, shaded_reference_label='reference',
                        vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                        ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True,
                        figsize=(12,8), use_seaborn=True, show=True):

        import matplotlib.pyplot as pyplot

        fig, ax = pyplot.subplots(figsize=figsize)

        if use_seaborn:
            import seaborn
            seaborn.set_style('ticks')
            seaborn.despine()

        self.plot(ax=ax, plot_S=plot_S, plot_E=plot_E, plot_I=plot_I,plot_R=plot_R, plot_F=plot_F,
                        plot_Q_E=plot_Q_E, plot_Q_I=plot_Q_I, combine_Q=combine_Q,
                        color_S=color_S, color_E=color_E, color_I=color_I, color_R=color_R, color_F=color_F,
                        color_Q_E=color_Q_E, color_Q_I=color_Q_I, color_reference=color_reference,
                        dashed_reference_results=dashed_reference_results, dashed_reference_label=dashed_reference_label,
                        shaded_reference_results=shaded_reference_results, shaded_reference_label=shaded_reference_label,
                        vlines=vlines, vline_colors=vline_colors, vline_styles=vline_styles, vline_labels=vline_labels,
                        ylim=ylim, xlim=xlim, legend=legend, title=title, side_title=side_title, plot_percentages=plot_percentages)

        if show:
            pyplot.show()

        return fig, ax


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def figure_infections(self, plot_S=False, plot_E='stacked', plot_I='stacked',plot_R=False, plot_F=False,
                            plot_Q_E='stacked', plot_Q_I='stacked', combine_Q=True,
                            color_S='tab:green', color_E='orange', color_I='crimson', color_R='tab:blue', color_F='black',
                            color_Q_E='mediumorchid', color_Q_I='mediumorchid', color_reference='#E0E0E0',
                            dashed_reference_results=None, dashed_reference_label='reference',
                            shaded_reference_results=None, shaded_reference_label='reference',
                            vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                            ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True,
                            figsize=(12,8), use_seaborn=True, show=True):

        import matplotlib.pyplot as pyplot

        fig, ax = pyplot.subplots(figsize=figsize)

        if use_seaborn:
            import seaborn
            seaborn.set_style('ticks')
            seaborn.despine()

        self.plot(ax=ax, plot_S=plot_S, plot_E=plot_E, plot_I=plot_I,plot_R=plot_R, plot_F=plot_F,
                        plot_Q_E=plot_Q_E, plot_Q_I=plot_Q_I, combine_Q=combine_Q,
                        color_S=color_S, color_E=color_E, color_I=color_I, color_R=color_R, color_F=color_F,
                        color_Q_E=color_Q_E, color_Q_I=color_Q_I, color_reference=color_reference,
                        dashed_reference_results=dashed_reference_results, dashed_reference_label=dashed_reference_label,
                        shaded_reference_results=shaded_reference_results, shaded_reference_label=shaded_reference_label,
                        vlines=vlines, vline_colors=vline_colors, vline_styles=vline_styles, vline_labels=vline_labels,
                        ylim=ylim, xlim=xlim, legend=legend, title=title, side_title=side_title, plot_percentages=plot_percentages)

        if show:
            pyplot.show()

        return fig, ax
