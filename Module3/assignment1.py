import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Look pretty...
matplotlib.style.use('ggplot')

import os
os.chdir('/Users/torenunez/PycharmProjects/DAT210x/Module3')

#
# Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
wheat = pd.read_csv('./Datasets/wheat.data')


#
# Create a slice of your dataframe (call it s1)
# that only includes the 'area' and 'perimeter' features
# 
s1 = pd.concat([wheat.area, wheat.perimeter], axis=1)


#
# Create another slice of your dataframe (call it s2)
# that only includes the 'groove' and 'asymmetry' features
# 
s2 = pd.concat([wheat.groove, wheat.asymmetry], axis = 1)


#
# TODO: Create a histogram plot using the first slice,
# and another histogram plot using the second slice.
# Be sure to set alpha=0.75
# 
s1.plot.hist(alpha=0.75)
s2.plot.hist(alpha=0.75)


plt.show()

