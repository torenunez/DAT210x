import pandas as pd
import matplotlib.pyplot as plt

import os
os.chdir('/Users/torenunez/PycharmProjects/DAT210x/Module3')

#
# Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
wheat = pd.read_csv('Datasets/wheat.data')


#
# Drop the 'id' feature
# 
wheat = wheat.drop (labels = ['id'], axis = 1)


#
# Compute the correlation matrix of your dataframe
# 
wheat.corr ()


#
# Graph the correlation matrix using imshow or matshow
# 
plt.imshow(wheat.corr(), cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range (len(wheat.columns))]
plt.xticks(tick_marks, wheat.columns, rotation='vertical')
plt.yticks(tick_marks, wheat.columns)

plt.show()



