import numpy
import matplotlib.pyplot as pyplot

def gamma_dist(mean, coeffvar, N):
	scale = mean*coeffvar**2
	shape = mean/scale
	return numpy.random.gamma(scale=scale, shape=shape, size=N)


def dist_stats(dists, names=None, plot=False, bin_size=1, colors=None, reverse_plot=False):
    dists  = [dists] if not isinstance(dists, list) else dists
    names  = [names] if(names is not None and not isinstance(names, list)) else (names if names is not None else [None]*len(dists))
    colors = [colors] if(colors is not None and not isinstance(colors, list)) else (colors if colors is not None else pyplot.rcParams['axes.prop_cycle'].by_key()['color'])
    
    stats = {}

    for i, (dist, name) in enumerate(zip(dists, names)):
        
        stats.update( { name+'_mean':  numpy.mean(dist),
                        name+'_stdev': numpy.std(dist),
                        name+'_95CI':  (numpy.percentile(dist, 2.5), numpy.percentile(dist, 97.5)) } )

        print((name+": " if name else "")+" mean = %.2f, std = %.2f, 95%% CI = (%.2f, %.2f)" % (numpy.mean(dist), numpy.std(dist), numpy.percentile(dist, 2.5), numpy.percentile(dist, 97.5)))
        print()
    
        if(plot):
            pyplot.hist(dist, bins=numpy.arange(0, int(max(dist)+1), step=bin_size), label=(name if name else False), color=colors[i], edgecolor='white', alpha=0.6, zorder=(-1*i if reverse_plot else i))
            
    if(plot):
        pyplot.ylabel('num nodes')
        pyplot.legend(loc='upper right')
        pyplot.show()

    return stats