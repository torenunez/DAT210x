import pandas as pd

from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt

# Look pretty...
matplotlib.style.use('ggplot')

import os
os.chdir('/Users/torenunez/PycharmProjects/DAT210x/Module4')


#
# Start by creating a regular old, plain, "vanilla"
# python list. You can call it 'samples'.
#
samples = []

#
# Write a for-loop that iterates over the images in the
# Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a
# temporary NDArray, just as shown in the Feature
# Representation reading.
#
for file in os.listdir('Datasets/ALOI/32'):
	a = os.path.join('Datasets/ALOI/32', file)
	img = misc.imread(a).reshape(-1)
	samples.append(img)
print len(samples) # 72, as expected since there 72 files in the folder
#
#
# Optional: Resample the image down by a factor of two if you
# have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no
# effect on the algorithm's results.
#
for filei in os.listdir('Datasets/ALOI/32i'):	# Also append the 32i images to the list/dataframe
	b = os.path.join('Datasets/ALOI/32i', filei)
	imgi = misc.imread(b).reshape(-1)
	samples.append(imgi)


#
# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#
# .. your code here .. 


#
# Convert the list to a dataframe
#
df = pd.DataFrame(samples)

colors = ['b']*72+['r']*12



#
# Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
from sklearn import manifold
iso = manifold.Isomap(n_neighbors = 1, n_components = 3)
Z = iso.fit_transform(df)



#
# Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
def Plot2D(T, title, x, y):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title(title)
  ax.set_xlabel('Component: {0}'.format(x))
  ax.set_ylabel('Component: {0}'.format(y))
  x_size = (max(T[:,x]) - min(T[:,x])) * 0.08
  y_size = (max(T[:,y]) - min(T[:,y])) * 0.08
  # It also plots the full scatter:
  ax.scatter(T[:,x],T[:,y], marker='.', c = colors, alpha=0.7)




#
# Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#
def Plot3D(T, title, x, y, z):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection = '3d')
  ax.set_title(title)
  ax.set_xlabel('Component: {0}'.format(x))
  ax.set_ylabel('Component: {0}'.format(y))
  ax.set_zlabel('Component: {0}'.format(z))
  x_size = (max(T[:,x]) - min(T[:,x])) * 0.08
  y_size = (max(T[:,y]) - min(T[:,y])) * 0.08
  z_size = (max(T[:,z]) - min(T[:,z])) * 0.08
  # It also plots the full scatter:
  ax.scatter(T[:,x],T[:,y],T[:,z], marker='.', c = colors, alpha=0.65)

Plot2D(Z, "Isomap transformed data, 2D", 0, 1)
Plot3D(Z, "Isomap transformed data 3D", 0, 1, 2)


plt.show()

