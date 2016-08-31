import pandas as pd


# Load up the table, and extract the dataset
# out of it. If you're having issues with this, look
# carefully at the sample code provided in the reading
#
df = pd.read_html('http://espn.go.com/nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2')[0]


# Rename the columns so that they match the
# column definitions provided to you on the website
#
df.columns = ['RK', 'PLAYER', 'TEAM', 'GP', 'G', 'A', 'PTS', '+/-', 'PIM', 'PTS/G', 'SOG', 'PCT', 'GWG', 'PPG', 'PPA', 'SHG', 'SHA']


# Get rid of any row that has at least 4 NANs in it
#
df = df.dropna(axis=0, thresh=4)


# At this point, look through your dataset by printing
# it. There probably still are some erroneous rows in there.
# What indexing command(s) can you use to select all rows
# EXCEPT those rows?
#
df.drop([1,13,25,37], axis = 0, inplace = True)


# Get rid of the 'RK' column
#
df.drop('RK', axis = 1, inplace = True)


# Ensure there are no holes in your index by resetting
# it. By the way, don't store the original index
#
df.reset_index(inplace = True, drop = True)



# Check the data type of all columns, and ensure those
print (df.dtypes)
df = df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
print (df.dtypes)


# Your dataframe is now ready! Use the appropriate
# commands to answer the questions on the course lab page.
print (df.shape[0])
print (df.PCT.nunique())
print (df.GP[15:17].sum())

