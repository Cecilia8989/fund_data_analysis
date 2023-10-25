
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

# Define a function to check missing value 
#https://dzone.com/articles/pandas-dataframe-functionsplaying-with-multiple-da
def missing_value_summary(df):
    # count the number of missing value
    missing_value_count = df.isnull().sum()
    # calculate the percentage on the total
    #https://stackoverflow.com/questions/67922276/convert-pandas-dataframe-values-to-percentage#comment120055288_67922371
    percentage_missing_value = (100 * df.isnull().sum() / len(df)).round(0).astype(int).astype(str) + '%'
    #Create the table 
    missing_values_table = pd.concat([missing_value_count, percentage_missing_value], axis=1)
    # assign columns name 
    missing_values_table.columns = ["Missing Values", "Percentage Missing"]
    # return the missing value table 
    return missing_values_table
#print(df[df.isnull().any(axis=1)])

# handling missing value 
# Import the SimpleImputer class from scikit-learn's impute module
from sklearn.impute import SimpleImputer
# Create an instance of SimpleImputer with the 'most_frequent' strategy
imputer = SimpleImputer(strategy='most_frequent')
# Use the imputer to fill missing values in the DataFrame 'df'
df.iloc[:,:] = imputer.fit_transform(df)


# Count the occurrences of each unique value in the 'Species' column
#https://www.datacamp.com/tutorial/categorical-data
count_species = df[['Species']].value_counts()
# Calculate the normalized counts, round them to two decimal places, convert to integers,
# and add '%' to represent them as percentages
#https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html
normalize_species = (df[['Species']].value_counts(normalize=True).round(2)*100).astype(int).astype(str) + '%'
# Concatenate the 'count_species' and 'normalize_species' Series into a DataFrame
species_table = pd.concat([count_species, normalize_species], axis=1)
# Rename the columns to 'Count' and '%'
species_table.columns = ['Count', '%']

# Count the occurrences of each unique value in the 'Island' column
count_islands = df[['Island']].value_counts()
# Calculate the normalized counts, round them to two decimal places, convert to integers,
# and add '%' to represent them as percentages
normalize_islands = (df[['Island']].value_counts(normalize=True).round(2)*100).astype(int).astype(str) + '%'
# Concatenate the 'count_species' and 'normalize_species' Series into a DataFrame
islands_table = pd.concat([count_islands, normalize_islands], axis=1)
# Rename the columns to 'Count' and '%'
islands_table.columns = ['Count', '%']


# Count the occurrences of each unique value in the 'Island' column
count_sex = df[['Sex']].value_counts()
# Calculate the normalized counts, round them to two decimal places, convert to integers,
# and add '%' to represent them as percentages
normalize_sex = (df[['Sex']].value_counts(normalize=True).round(2)*100).astype(int).astype(str) + '%'
# Concatenate the 'count_species' and 'normalize_species' Series into a DataFrame
sex_table = pd.concat([count_sex, normalize_sex], axis=1)
# Rename the columns to 'Count' and '%'
sex_table.columns = ['Count', '%']

# Generate descriptive statistics on Numerical - continuos - ratio variables 
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html
df.describe().round(2)