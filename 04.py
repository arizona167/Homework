import numpy as np
import pandas as pd

#04

# a. Generate synthetic data
np.random.seed(42)  
X = np.random.randn(1000, 5)
Y = np.random.randint(0, 3, 1000)

#b. Convert to Pandas DataFrame
xy_df = pd.DataFrame(data=X, columns=['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5'])
xy_df['Target'] = Y

# Filtering conditions
condition_0 = (xy_df['Target'] == 0) & (xy_df['Feature_2'] < 0)
condition_1 = (xy_df['Target'] == 1) & (xy_df['Feature_3'] > 0)
condition_2 = (xy_df['Target'] == 2) & (xy_df['Feature_4'].between(-1, 1))

xy_df_filtered = xy_df[condition_0 | condition_1 | condition_2]

# c. Create separate NumPy arrays based on filtering conditions
x_arr_filtered_0 = X[condition_0]
y_arr_filtered_0 = Y[condition_0]
#cii
x_arr_filtered_1 = X[condition_1]
y_arr_filtered_1 = Y[condition_1]
#Ciii
x_arr_filtered_2 = X[condition_2]
y_arr_filtered_2 = Y[condition_2]

# Display some rows of xy_df and xy_df_filtered
print("a. Synthetic Data (xy_df):")
print(xy_df.head())

print("\nb. Filtered Data (xy_df_filtered):")
print(xy_df_filtered.head())
