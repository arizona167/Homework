import pandas as pd
from category_encoders import TargetEncoder

#03

# the dataset into a DataFrame named diabetes_with_race
diabetes_with_race = pd.read_csv("diabetes_with_race.csv")

# a. One-Hot Encoding
one_hot_encoded_df = pd.get_dummies(diabetes_with_race, columns=['race'], prefix='race', drop_first=True)
# Drop the original 'race' feature
one_hot_encoded_df.drop('race', axis=1, inplace=True)

# b. Target Encoding
target_variable = 'target_variable'

# Creating a TargetEncoder object
target_encoder = TargetEncoder(cols=['race'])

# Fit and transform the data, handling missing values
diabetes_with_race['race_encoded'] = target_encoder.fit_transform(diabetes_with_race['race'], diabetes_with_race[target_variable])

# Drop the original 'race' feature
diabetes_with_race.drop('race', axis=1, inplace=True)



