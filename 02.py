import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns


#02

#a Load the Iris dataset
iris_data = load_iris()
iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
iris_df['species'] = iris_data.target_names[iris_data.target]

#b Display the first 10 rows of the DataFrame
print("First 10 rows of the DataFrame:")
print(iris_df.head(10))


#d Group by 'species' and calculate statistics for sepal length
print(" Grouping by 'species' and calculating statistics for sepal length:")
grouped_species = iris_df.groupby('species')['sepal length (cm)'].agg(['mean', 'median', 'min', 'max'])
print(grouped_species)

#e Filter the dataset for petal width greater than 1.5 units
print(".Filtering flowers with petal width greater than 1.5 units:")
filtered_data = iris_df[iris_df['petal width (cm)'] > 1.5]
print(filtered_data)

#f Create a new column 'sepal_area'
iris_df['sepal_area'] = iris_df['sepal length (cm)'] * iris_df['sepal width (cm)']

#g Create a scatter plot for petal length vs petal width for each species
print("Scatter plot for petal length vs petal width:")
sns.scatterplot(data=iris_df, x='petal length (cm)', y='petal width (cm)', hue='species')
plt.title('Scatter plot: Petal Length vs Petal Width')
plt.show()


#h Export the modified DataFrame to a new CSV file
iris_df.to_csv('modified_iris_dataset.csv', index=False)

