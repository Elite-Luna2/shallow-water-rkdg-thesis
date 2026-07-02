# Code to obtain the analytical values from the SWASHES library and save them to a CSV file

import pyswashes

s = pyswashes.TwoDimensional(1, 1, 1, 100, 100)
#s = pyswashes.PseudoTwoDimensional(1, 1, 1, 400)

print("Domain parameters:")
print(s.dom_params)

print("\nAvailable columns:")
print(s.cols())

df = s.dataframe().reset_index()
print("\nFirst rows:")
print(df.head())

df.to_csv("swashes_2d_thacker_radial.csv", index=False)
print("\nSaved: swashes_2d_thacker_radial.csv")

#df.to_csv("swashes_pseudo2d_subcritical.csv", index=False)
#print("\nSaved: swashes_pseudo2d_subcritical.csv")