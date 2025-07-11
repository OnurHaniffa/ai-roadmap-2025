
import pandas as pd 
from tabulate import tabulate

def pretty(data):
    print(tabulate(data, headers='keys', tablefmt='fancy_grid'))


df = pd.read_csv('Week_1/Day_4/employee_data.csv')



pretty(df)


pretty(df.head())
pretty(df.tail())
print(df.shape)
print(df.columns)
print(df.dtypes)

pretty(df[df['department'] == 'Engineering'])

pretty(df[(df['age'] > 30) & (df['gender'] == 'Female')])

pretty(df[df['salary'] > 70000])

pretty(df.sort_values('salary', ascending=False))
pretty(df.sort_values('age'))
pretty(df.sort_values(['department','salary'], ascending=[True, False]))
print(df.groupby("department")[["salary",'age']].mean())
print(df['job_title'].value_counts())
print(df.sort_values(['age']).groupby('department')['age'].first())
df['senior']= df.apply(lambda x: 'Senior' if x['age'] > 40 else 'Junior', axis=1)

pretty(df)
