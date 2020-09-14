import sys
import numpy
import matplotlib.pyplot as pyplot

def gamma_dist(mean, coeffvar, N):
	scale = mean*coeffvar**2
	shape = mean/scale
	return numpy.random.gamma(scale=scale, shape=shape, size=N)


def dist_info(dists, names=None, plot=False, bin_size=1, colors=None, reverse_plot=False):
    dists  = [dists] if not isinstance(dists, list) else dists
    names  = [names] if(names is not None and not isinstance(names, list)) else (names if names is not None else [None]*len(dists))
    colors = [colors] if(colors is not None and not isinstance(colors, list)) else (colors if colors is not None else pyplot.rcParams['axes.prop_cycle'].by_key()['color'])
    
    for i, (dist, name) in enumerate(zip(dists, names)):
        print((name+": " if name else "")+" mean = %.2f, std = %.2f, 95%% CI = (%.2f, %.2f)" % (numpy.mean(dist), numpy.std(dist), numpy.percentile(dist, 2.5), numpy.percentile(dist, 97.5)))
        print()
    
        if(plot):
            pyplot.hist(dist, bins=numpy.arange(0, int(max(dist)+1), step=bin_size), label=(name if name else False), color=colors[i], edgecolor='white', alpha=0.6, zorder=(-1*i if reverse_plot else i))
            
    if(plot):
        pyplot.ylabel('num nodes')
        pyplot.legend(loc='upper right')
        pyplot.show()


def network_info(networks, names=None, plot=False, bin_size=1, colors=None, reverse_plot=False):
    import networkx
    networks = [networks] if not isinstance(networks, list) else networks
    names    = [names] if not isinstance(names, list) else names
    colors = [colors] if(colors is not None and not isinstance(colors, list)) else (colors if colors is not None else pyplot.rcParams['axes.prop_cycle'].by_key()['color'])
    
    for i, (network, name) in enumerate(zip(networks, names)):
    
        degree        = [d[1] for d in network.degree()]

        if(name):
            print(name+":")
        print("Degree: mean = %.2f, std = %.2f, 95%% CI = (%.2f, %.2f)\n        coeff var = %.2f" 
              % (numpy.mean(degree), numpy.std(degree), numpy.percentile(degree, 2.5), numpy.percentile(degree, 97.5), 
                 numpy.std(degree)/numpy.mean(degree)))
        r = networkx.degree_assortativity_coefficient(network)
        print("Assortativity:    %.2f" % (r))
        c = networkx.average_clustering(network)
        print("Clustering coeff: %.2f" % (c))
        print()
    
        if(plot):
            pyplot.hist(degree, bins=numpy.arange(0, int(max(degree)+1), step=bin_size), label=(name+" degree" if name else False), color=colors[i], edgecolor='white', alpha=0.6, zorder=(-1*i if reverse_plot else i))
    
    if(plot):
        pyplot.ylabel('num nodes')
        pyplot.legend(loc='upper right')
        pyplot.show()


def results_summary(model):
    print("total percent infected: %0.2f%%" % ((model.total_num_infected()[-1]+model.total_num_recovered()[-1])/model.numNodes * 100) )
    print("total percent fatality: %0.2f%%" % (model.numF[-1]/model.numNodes * 100) )
    print("peak  pct hospitalized: %0.2f%%" % (numpy.max(model.numH)/model.numNodes * 100) )


#########################################################################################################################################


try:
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt


    def last(x):
        """Return last element of a pandas Series"""
        return x.iloc[-1]

    def summarize(df):
        """Return a Series with last value, sum of values, and weighted average of values"""
        temp = df.copy()
        orig_cols = list(df.columns)
        todrop = []
        for col in orig_cols:
            if col == 'time': continue
            tcol = temp[['time',col]].dropna()
            lengths = (tcol['time'] - tcol['time'].shift(1)).fillna(0)
            total  = sum(lengths)
            temp = temp.assign(**{col+"/scaled": tcol[col] * lengths / total if total else 0})
            todrop.append(col+"/scaled/last")
        temp = temp.fillna(0)
        summary = temp.agg([last, numpy.sum])
        summary = summary.stack()
        summary.index = ['/'.join(reversed(col)).strip() for col in summary.index.values]
        summary.drop(todrop,inplace=True)
        summary.rename({col+"/scaled/sum": col+"/average" for col in orig_cols},inplace=True)
        return summary



    def make_compact(val):
        """Take a potentially object and reduce it to smaller for logging.
        If the object is number - return it
        If the object is a long list or ndarray of numbers - return its average
        Otherwise, stringify it
        If the object is a string - return its first 30 characters
        """
        if isinstance(val, (int, float, numpy.number)):
            return val
        if isinstance(val, numpy.ndarray):
            return val.flatten().mean()
        if isinstance(val, (list, tuple)):
            if len(val)<5:
                return str(val)
            if isinstance(val[0], (int, float, numpy.number)):
                return sum(val) / len(val)
        return str(val)[:64]



    def hist2df(history , **kwargs):
        """Take history dictionary and return:
        pandas DataFrame of all history
        pandas Series of the summary of history, taking the last value and the sum, as well average over time (sum of scaled)
        Optional kwargs argument - if given then add them to the dataFrame and DataSeries - helpful when merging many logs from different runs.
        """
        L = [{'time': t, **d} for t, d in history.items()]
        n = len(L)
        df = pd.DataFrame(L)
        if 'numPositive' in df.columns:
            df['overallPositive'] = df['numPositive'].fillna(0).cumsum()
        summary = summarize(df)

        # add to summary statistics up to first detection

        test_lag = 0
        if 'test_lag' in kwargs:
            test_lag = kwargs['test_lag']
        else:
            for t,d in history.items():
                if 'isolation_lag_positive' in d:
                    test_lag = d['isolation_lag_positive']
                    break
        detectionTime = -1
        firstPositiveTestTime = -1
        temp = df[df.numPositive>0]
        row = None
        if len(temp)>0:
            firstPositiveTestTime = temp['time'].iloc[0]
            detectionTime = firstPositiveTestTime + test_lag
        temp = df[df['time']<= detectionTime]
        if len(temp):
            summary2 = summarize(temp)
            summary2.rename({col: col+"/1st" for col in summary2.index }, inplace=True)
            summary = summary.append(summary2)
        summary = summary.append(pd.Series([firstPositiveTestTime, test_lag, detectionTime],
                                  index= ['firstPositiveTestTime', 'test_lag', 'detectionTime']))

        if kwargs:
            for key,val in kwargs.items():
                val = make_compact(val)
                df[key] = val
                summary[key] = val
        return df, summary


    def violin_plot(lists, labels, title="", ylabel="", xlabel=""):
        sns.set()
        fig, ax = plt.subplots(figsize=(16, 8))

        vp = ax.violinplot(lists, showmeans=True)
        i = 1
        for pc in vp['bodies']:
            pc.set_color(f'C{i}')

        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
            pc = vp[partname]
            pc.set_edgecolor("black")
            pc.set_linewidth(1)

        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(numpy.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_ylabel(ylabel)
        if xlabel:
            ax.set_xlabel(xlabel)
        if title:
            ax.set_title(title)
        plt.show()


    def show_violins(data, field, groupby = 'variant', ylabel=None,title="", key = None):
        """Show 'violin graphs' of a certain field according to different variants"""
        plots = []
        labels = []
        if ylabel is None:
            ylabel = field
        for v in sorted(data[groupby].unique(), key=key):
            plots.append(data[data[groupby] == v][field])
            labels.append(v)
        violin_plot(plots, labels, ylabel=ylabel, title=title)


except ImportError:
    print("Warning: pandas missing - some logging functions will not work", file=sys.stderr)
    def last(x):
        raise NotImplementedError("This function requires pandas to work")

    def hist2df(history):
        raise NotImplementedError("This function requires pandas to work")









