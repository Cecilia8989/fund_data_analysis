
''''Task 2 Script - The penguins dataset
Author: Cecilia Patore
Main porpuse of this section is explore the Penguins Dataset, understand what type of variables it is composed 
and for what analysis they can be used.
'''

# impport needed libraries 
import pandas as pd
import seaborn as sns
import numpy as np

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

''''Task 2 Script - The penguins dataset
Author: Cecilia Patore
Main porpuse of this section is explore the Penguins Dataset, understand what type of variables it is composed 
and for what analysis they can be used.
'''



# sex - categorical - nominal variable with 2 possible outputs: male and female
# The Numpy randon binominal distribution is hte one that better apply
#https://www.kaggle.com/code/abdelrhmaneltawagny/apply-probability-distributions-in-real-data
#https://benhay.es/posts/exploring-distributions/
#https://www.w3schools.com/python/numpy/numpy_random_binomial.asp
#https://medium.com/@aifakhri/a-practical-approach-to-solve-probability-and-statistical-problems-with-python-part-1-755c8a12f916
#https://www.euanrussano.com/post/probability/uni_bernoulli_binomial/
#https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html
#https://www.tutorialspoint.com/python_data_science/python_bernoulli_distribution.htm



#Count the total number of observations in the 'Sex' column of the dataframe
tot_observation = df['Sex'].count()
# Count the number of male observations in the dataset using value_counts and store it in a variable
# https://www.width.ai/pandas/count-specific-value-in-column-with-pandas
count_male = (df["Sex"] == "Male").sum()
# Calculate the probability that the observation is male and round it to 2 decimal places
prob_male = (count_male / tot_observation).round(2)
print(prob_male)

# Import the required librarie
import matplotlib.pyplot as plt


# Set the parameters for the binomial distribution as a bernoulli distribution with trial 1
#https://benhay.es/posts/exploring-distributions/ 
n = 1 # Number of trials
p = prob_male # Probability of success
size = 1000
# Generate a random binomial distribution with the specified parameters
sex_distribution = np.random.binomial(n=n, p=prob_male, size=size)
# Set Seaborn style to 'whitegrid' for a specific background style
#https://www.codecademy.com/article/seaborn-design-ii
sns.set(style="whitegrid")
# Create a figure for the plot with a specified size
plt.figure(figsize=(7,5))
# Create a countplot of the sex_distribution
ax =sns.countplot(x=sex_distribution, palette="Set2")
# Set the title and labels for the plot
plt.title(f'Binomial / Bernoulli distribution, Trail ({n}) Prob. ({p*100}%), Size ({size})')
plt.xlabel('Gender (0:Female, 1:Male)')
plt.ylabel('Count')
# Annotate each bar with the percentage of occurrence
#https://stackoverflow.com/questions/55104819/display-count-on-top-of-seaborn-barplot
#https://stackoverflow.com/questions/63603222/how-to-annotate-countplot-with-percentages-by-category
for m in ax.patches:
    ax.annotate(f'\n{(m.get_height()/size)*100:.2f}%', (m.get_x()+0.2, m.get_height()), ha='center', va='top', color='white', size=18)
plt.show()

# Set the parameters for the binomial distribution with trieal 10 to have a better rappresentatation of the random distribution 
n = 100 # Number of trials
p = prob_male # Probability of success
size = 1000

# Generate a random binomial distribution with the specified parameters
male_distribution = np.random.binomial(n=n, p=prob_male, size=size)
# Create a histogram plot with transparency (alpha), a kernel density estimate (kde), and specified bin count
ax = sns.histplot(male_distribution, color='Orange', kde=True, bins=20, alpha=0.7)
ax.lines[0].set_color('crimson')
# Customize the plot by setting the title and labels
#https://stackoverflow.com/questions/69524514/how-to-modify-the-kernel-density-estimate-line-in-a-sns-histplot
ax.set_title(f'Binomial distribution, Trail ({n}) Prob. ({p*100}%), Size ({size})')
ax.set_xlabel("Number of Male Penguins")
ax.set_ylabel("Frequency")

plt.show()

# Species variables
# The distributions that apply are the choice and the Multinomial
#https://stackoverflow.com/questions/37818063/how-to-calculate-conditional-probability-of-values-in-dataframe-pandas-python

# Create a choice Distribution 
#https://medium.com/@brandon93.w/converting-categorical-data-into-numerical-form-a-practical-guide-for-data-science-99fdf42d0e10
# Import the LabelEncoder from scikit-learn for encoding categorical data
# into numerical values.
from sklearn.preprocessing import LabelEncoder
# Initialize the LabelEncoder.
le = LabelEncoder()
# Use the LabelEncoder to transform the 'Species' column from categorical to numerical.
df['Species'] = le.fit_transform(df['Species'])
# Calculate the probabilities based on the group sizes
#https://stackoverflow.com/questions/37818063/how-to-calculate-conditional-probability-of-values-in-dataframe-pandas-python
g = df.groupby('Species').size().div(len(df))
# Specify the sample size
sample_size = 100
# Generate random choices based on the available categories (species)
species_choice_distribution = np.random.choice(g.index, p=g.values, size=sample_size)
# Create a countplot based on the random choices
plt.figure(figsize=(7,6))
ax = sns.countplot(x=species_choice_distribution, palette="Set2", dodge=False)
# Set plot title and labels
ax.set_title(f'Choice Distribution, Size ({sample_size})')
ax.set_xticklabels(['Adelie', 'Chinstrap', 'Gentoo'])
ax.set_xlabel('Species')
ax.set_ylabel('Count')
# Annotate the countplot with percentages
for m in ax.patches:
    ax.annotate(f'\n{(m.get_height()/sample_size)*100:.1f}%', (m.get_x()+0.4, m.get_height()), ha='center', va='top', color='white', size=18)
plt.show()

# Create a Multinomial distribution 

multinomial_distribution = np.random.multinomial(n=6, pvals=g.values, size=1000 )


# Perform the Kolmogorov-Smirnov Test to verify the variables can be rapresented by a normal disstribution 

print(df["Bill_Length(mm)"])

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

# Create a Q-Q plot
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
stats.probplot(df["Body_Mass(g)"], dist="norm", plot=plt)
plt.title("Q-Q Plot")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")

# Create a histogram
plt.subplot(1, 2, 2)
plt.hist(df["Body_Mass(g)"], bins=200, edgecolor='k', density=True, alpha=0.7)
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

statistic, p_value = stats.shapiro(df["Body_Mass(g)"])
alpha = 0.05 
if p_value > alpha:
    print("The data appears to be normally distributed (fail to reject H0)")
else:
    print("The data does not appear to be normally distributed")