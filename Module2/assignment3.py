import pandas as pd

# Load up the dataset
# Ensuring you set the appropriate header column names
#
df = pd.read_csv("Module2/Datasets/servo.data",
                 header = None,
                 names = ['motor', 'screw', 'pgain', 'vgain', 'class'])


# Create a slice that contains all entries
# having a vgain equal to 5. Then print the 
# length of (# of samples in) that slice:
#
sum(df.vgain==5)


# Create a slice that contains all entries
# having a motor equal to E and screw equal
# to E. Then print the length of (# of
# samples in) that slice:
#
df.loc[(df["motor"] == "E") & (df["screw"] == "E"), ]



# Create a slice that contains all entries
# having a pgain equal to 4. Use one of the
# various methods of finding the mean vgain
# value for the samples in that slice. Once
# you've found it, print it:
#
df.loc[(df["pgain"] == 4) , "vgain"].mean()



#(Bonus) See what happens when you run
# the .dtypes method on your dataframe!
df.dtypes


