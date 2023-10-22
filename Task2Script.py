

# impport needed libraries 
import pandas as pd
import seaborn as sns

# import the Penguins Dataset from the seaborn libraries 
df=sns.load_dataset("penguins")
# change columns name 
df.columns = ["Species", "Island", "Bill_Length(mm)", "Bill_Depth(mm)", "Flipper_Length(mm)", "Body_Mass(g)", "Sex"]

print (df.sample(5))
print(df.shape)
print(df.info())
print(df.describe())

# checking missing value 
#https://dzone.com/articles/pandas-dataframe-functionsplaying-with-multiple-da
# count the number of missing value
missing_value_count = df.isnull().sum()
# calculate the percentage on the total
percentage_missing_value = (100 * df.isnull().sum() / len(df)).round(0).astype(int).astype(str) + '%'
# show the data type
data_types = df.dtypes
#Create the table 
missing_values_table = pd.concat([missing_value_count, percentage_missing_value, data_types], axis=1)
# assign columns name 
missing_values_table.columns = ["Missing Values", "Percentage Missing", "Data Type"]
print(missing_values_table)