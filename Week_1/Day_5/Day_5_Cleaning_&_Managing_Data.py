import pandas as pd 
from tabulate import tabulate

def pretty(data):
    print(tabulate(data, headers='keys', tablefmt='fancy_grid'))

df = pd.read_csv('Week_1/Day_5/employee_messy.csv')

pretty(df)

df['name']=df['name'].str.strip().str.lower()
df['department']=df['department'].str.strip().str.lower()
df['performance_score']=df['performance_score'].str.strip().str.lower()

df['joining_date'] = df['joining_date'].astype(str).str.replace(r"[./]", "-", regex=True)
df['joining_date'] = pd.to_datetime(df['joining_date'], errors='coerce')
df['age'].fillna(df['age'].mean(), inplace=True)
df['salary'].fillna(df['salary'].mean(), inplace=True)

# df=df[df['joining_date'].notnull()]
df.drop_duplicates(inplace=True)
df['Tenure']= 2025 - df['joining_date'].dt.year

df.to_csv('Week_1/Day_5/employee_cleaned.csv', index=False)
pretty(df)

