
''''Task 2 Script - The penguins dataset
Author: Cecilia Patore
Main porpuse of this section is explore the Penguins Dataset, understand what type of variables it is composed 
and for what analysis they can be used.
'''

# impport needed libraries 
import pandas as pd
import seaborn as sns

# import the Penguins Dataset from the seaborn libraries 
df=sns.load_dataset("penguins")
# change columns name 
df.columns = ["Species", "Island", "Bill_Length(mm)", "Bill_Depth(mm)", "Flipper_Length(mm)", "Body_Mass(g)", "Sex"]

# Display a random sample of 5 rows from the DataFrame
df.sample(5)
# Output the shape (number of rows and columns) of the DataFrame
df.shape
# Output information about the DataFrame, including data types and non-null values
df.info()
# Generate summary statistics of the DataFrame
df.describe()

# checking missing value 
#https://dzone.com/articles/pandas-dataframe-functionsplaying-with-multiple-da
# count the number of missing value
missing_value_count = df.isnull().sum()
# calculate the percentage on the total
#https://stackoverflow.com/questions/67922276/convert-pandas-dataframe-values-to-percentage#comment120055288_67922371
percentage_missing_value = (100 * df.isnull().sum() / len(df)).round(0).astype(int).astype(str) + '%'
#Create the table 
missing_values_table = pd.concat([missing_value_count, percentage_missing_value], axis=1)
# assign columns name 
missing_values_table.columns = ["Missing Values", "Percentage Missing"]
print(missing_values_table)