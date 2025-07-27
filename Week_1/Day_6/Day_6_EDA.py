import pandas as pd 
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt


def pretty(data):
    print(tabulate(data, headers='keys', tablefmt='fancy_grid'))

df = pd.read_csv('Week_1/Day_5/employee_cleaned.csv')


print('NEW EMPLOYEE DATA ANALYSIS')
print('-'*150)

pretty(df.iloc[:,2:].describe())
pretty(df.info())

df['joining_date'] = pd.to_datetime(df['joining_date'], errors='coerce')
df_numeric = df.select_dtypes(include=['number'])
df_numeric=df_numeric.drop(columns='id')
df_categorical = df.select_dtypes(include=['object'])
df_datetime = df.select_dtypes(include=['datetime'])

pretty(df_numeric)
pretty(df_categorical)
pretty(df_datetime)

for col in df_numeric.columns:
    plt.figure()
    sns.histplot(bins=30,data=df_numeric, x=col, kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# for col in df_numeric.columns:
#     plt.figure()
#     sns.boxplot(data=df_numeric, x=col)
#     plt.title(f'Boxplot of {col}')
#     plt.show()

# sns.pairplot(df_numeric)
# plt.show()

corr_matrix = df_numeric.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(data=corr_matrix, annot=True, fmt=".2f", cmap='magma', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix')
plt.show()


# for col in df_categorical.columns:
#     plt.figure()
#     sns.countplot(data=df_categorical, x=col)
#     plt.title(f'Countplot of {col}')
#     plt.xticks(rotation=45)
#     plt.show()

# for col in df_categorical.columns:
#     plt.figure()
#     sns.boxplot(data=df, x=col, y='salary')
#     plt.title(f'Salary by {col}')
#     plt.xticks(rotation=45)
#     plt.show()

# for col in df_categorical.columns:
#     plt.figure(figsize=(10, 6))
#     sns.barplot(data=df, x=col, y='salary', ci=None)
#     plt.title(f'Average Salary by {col}')
#     plt.xticks(rotation=45)
#     plt.show()

df['joining year'] = df['joining_date'].dt.year
df['joining month'] = df['joining_date'].dt.month
df['joining day'] = df['joining_date'].dt.day

# sns.countplot(data=df, x='joining year')
# plt.title('Count of Employees by Joining Year')
# plt.xticks(rotation=45)
# plt.show()

# sns.countplot(data=df, x='joining month')
# plt.title('Count of Employees by Joining Month')
# plt.xticks(rotation=45)
# plt.show()

df.to_csv('Week_1/Day_6/employee_analysis.csv', index=False)
pretty(df)







import pandas as pd 
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt

def pretty(data):
    print(tabulate(data, headers='keys', tablefmt='fancy_grid'))



df = pd.read_csv("hr_data.csv")
df.head()
df.describe()
df.info()               

df.numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
df.categorical_columns = df.select_dtypes(include=['object']).columns

pretty(df[df.numerical_columns].head())
pretty(df[df.categorical_columns].head())



