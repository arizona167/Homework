import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#01

# Load the breast cancer dataset
df = pd.read_csv('Breast_cancer_data.csv')


target_variable = 'diagnosis'

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Identify the feature with the highest correlation with the target variable
highest_corr_feature = correlation_matrix[target_variable].idxmax()
highest_corr_value = correlation_matrix[target_variable].max()

# Print the result
print(f"The feature with the highest correlation with '{target_variable}' is '{highest_corr_feature}' with a correlation of {highest_corr_value:.2f}")

# Visualize the correlation matrix
plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png') 
plt.show()

