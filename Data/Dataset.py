"""
Remove features to improve accuracy and efficiency by dropping features that are add error, etc.

Copyright (C) 2024 Daniel Nelson

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
import numpy as np
import pandas as pd
from os import scandir

from pandas import isnull

input_path = "D:/Wildfire_Compiled_v8/"
datasets_path = "probability_wildfire_dataset.csv"
combined_ds = []

# Merge all the csv files into one
with scandir(input_path) as pth:
    for file in pth:
        ds = pd.read_csv(file)
        combined_ds.append(ds)
df = pd.concat(combined_ds)
# df.to_csv(datasets_path, index=False)

# Print column headers
print(df.columns)

# Remove columns that are not needed
error_columns = ['Unnamed: 0', 'spatial_ref']
df.drop(error_columns, axis=1, inplace=True)

# Drop irrelevant features, i.e. lake temperature
irrelevant_columns = ['8_clm', '9_clm', '10_clm', '11_clm', '12_clm', '13_clm', '14_clm', '15_clm', '16_clm', '17_clm',
                      '18_clm', '19_clm', '20_clm', '21_clm', '22_clm', '41_clm']
df.drop(irrelevant_columns, axis=1, inplace=True)

df.dropna(inplace=True)

for column in df:
    df = df[df[column] != np.nan]
    df = df[df[column] != np.nan]

# Remove no data rows
df = df[df.LC != 255]
df = df[df.LC != 0]

# Remove Unclassified data rows
df = df[df.LC != 0]

# Remove water data rows
df = df[df.LC != 20]

# Remove glacier/snow data rows
df = df[df.LC != 31]

# Remove rock data rows
df = df[df.LC != 32]

# Remove unburned rows
df.dNBR = df.dNBR.multiply(-1)

df['prob'] = np.where(
    df['dNBR'] < 100, 0, np.where(
        df['dNBR'] > 100, 1, -1))

print(df['dNBR'].mean())
"""
# reduced with xy
todrop = ['x', 'y', '24_clm', '25_clm', '26_clm', '27_clm', '29_clm', '2_clm', '32_clm', '37_clm', 'twi',
          '3_clm', '40_clm', '42_clm', '44_clm', '45_clm', '4_clm', '5_clm', '6_clm', '7_clm', '49_clm']
perm = df.drop(todrop, axis=1)

permutated = perm.sample(frac=1, random_state=32).reset_index(drop=True)
# print(permutated.columns)

tdrop = ['1_clm', '23_clm', '28_clm', '30_clm', '31_clm', '33_clm', '34_clm', '35_clm', '36_clm', '38_clm', '39_clm',
         '43_clm', '46_clm', '47_clm', '48_clm', '50_clm', 'Aspect', 'dNBR', 'Elevation', 'LC', 'mNBR', 'pre', 'Slope',
         'tpi']
dfr = df.drop(tdrop, axis=1).reset_index(drop=True)

# print(dfr.columns)
# print(permutated.columns)

dfc = pd.concat([dfr, permutated], axis=1)
# dfc.to_csv(perm_path, index=False)
"""
# big no xy
todrop = ['x', 'y', 'dNBR']
perm = df.drop(todrop, axis=1)
permutated = perm.sample(frac=1, random_state=32).reset_index(drop=True)
permutated.to_csv(datasets_path, index=False)

# Print remaining columns and rows
print(permutated.columns)
print(permutated.eq(0).sum().sum())
print(permutated.eq(np.nan).sum().sum())
print(permutated.isnull().sum().sum())
print(permutated.shape)
