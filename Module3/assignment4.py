import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from pandas.tools.plotting import parallel_coordinates

# Look pretty...
matplotlib.style.use('ggplot')

import os
os.chdir('/Users/torenunez/PycharmProjects/DAT210x/Module3')
#
# Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
wheat = pd.read_csv('Datasets/wheat.data')



#
# Drop the 'id', 'area', and 'perimeter' feature
# 
wheat = wheat.drop(labels=['id', 'area', 'perimeter'], axis = 1)



#
# Plot a parallel coordinates chart grouped by
# the 'wheat_type' feature. Be sure to set the optional
# display parameter alpha to 0.4
# 
plt.figure()
parallel_coordinates(wheat,'wheat_type', alpha=0.4)



plt.show()


