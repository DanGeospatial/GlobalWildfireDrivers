"""
The Wildfire dataset has many features that are correlated with each other
This script tries to reduce the collinearity within the dataset

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
from collinearity import SelectNonCollinear
from sklearn.feature_selection import f_regression, mutual_info_regression
import pandas as pd
import numpy as np

checked_data = "D:/Wildfire_Datasets_v8/perm_full_wildfire_dataset.csv"
df = pd.read_csv(checked_data)

left_upper_rh = df['1_clm'].mul(17.269)
left_bottom_rh = df['1_clm'].add(273.3)
right_upper_rh = df['2_clm'].mul(17.269)
right_bottom_rh = df['2_clm'].add(237.3)

left_bracket_rh = left_upper_rh.div(left_bottom_rh)
right_bracket_rh = right_upper_rh.div(right_bottom_rh)

right_rh = left_bracket_rh.sub(right_bracket_rh)
rh = np.exp(right_rh)
rh_f = rh.to_frame('rh')

dfc = pd.concat([df, rh_f], axis=1)

df_clean = dfc.dropna()
print(dfc.shape)
print(dfc.columns)

y_data = df_clean['prob'].to_numpy()
x_data = df_clean.drop('prob', axis=1).to_numpy()

select_function = SelectNonCollinear(0.5, scoring=f_regression)
select_function.fit(X=x_data, y=y_data)

print(select_function.get_support())

select_function = SelectNonCollinear(0.55, scoring=mutual_info_regression)
select_function.fit(X=x_data, y=y_data)

print(select_function.get_support())
