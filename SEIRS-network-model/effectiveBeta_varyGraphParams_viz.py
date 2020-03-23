from models import SEIRSGraphModel # Import the model.
import numpy
import pandas
import networkx
from scipy import stats
import matplotlib.pyplot as pyplot
import seaborn


dataframe = pandas.read_csv('effectiveBeta_varyingGraphParams.csv')
print dataframe

dataframe = dataframe[dataframe['effectiveBetaPct'] < 1.3]
print dataframe['meanDegree'].unique()
print dataframe['effectiveBetaPct'].max()

dataframe=dataframe.sort_values(by=['meanDegree'])
dataframe['meanDegreeStr'] = ["$%s$" % int(x) for x in dataframe['meanDegree']]
print dataframe['meanDegreeStr'].unique()


fig, ax = pyplot.subplots(1, 1, figsize=(16,9))
seaborn.lineplot(ax=ax, data=dataframe, x='p', y='effectiveBetaPct', hue='meanDegreeStr', markers=True, err_style="bars",
				palette=seaborn.cubehelix_palette(len(dataframe['meanDegreeStr'].unique())))
ax.set_xscale('log')
seaborn.set_style('ticks')
seaborn.despine()
pyplot.tight_layout()
# pyplot.show()
pyplot.savefig('figs/effectiveBeta_varyingGraphParams_prelim.png')