import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

# Look pretty...
matplotlib.style.use('ggplot')


import os
os.chdir('/Users/torenunez/PycharmProjects/DAT210x/Module3')

# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
wheat = pd.read_csv('Datasets/wheat.data')



fig = plt.figure()
#
# Create a new 3D subplot using fig. Then use the
# subplot to graph a 3D scatter plot using the area,
# perimeter and asymmetry features. Be sure to use the
# optional display parameter c='red', and also label your
# axes
# 
ax1 = fig.add_subplot(111,projection='3d')
ax1.set_xlabel('area')
ax1.set_ylabel('perimter')
ax1.set_zlabel('asymmetry')
ax1.scatter(wheat['area'], wheat['perimeter'], wheat['asymmetry'], c='red', marker='.')


fig = plt.figure()
#
# TODO: Create a new 3D subplot using fig. Then use the
# subplot to graph a 3D scatter plot using the width,
# groove and length features. Be sure to use the
# optional display parameter c='green', and also label your
# axes
# 
ax1 = fig.add_subplot(111,projection='3d')
ax1.set_xlabel('width')
ax1.set_ylabel('groove')
ax1.set_zlabel('length')
ax1.scatter(wheat['width'], wheat['groove'], wheat['length'], c='green', marker='.')

plt.show()


