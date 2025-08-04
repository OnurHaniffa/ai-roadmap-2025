import pandas as pd 

df_raw=pd.read_csv('Week_2/Day_8/hr_data.csv')

df_raw['Attrition_num']=df_raw['Attrition'].map({'Yes':1,'No':0})
df_clean=df_raw.drop(columns=['Attrition','DailyRate',"EmployeeNumber","EmployeeCount","MonthlyRate","HourlyRate","StandardHours","OverTime","Over18","Gender"])

print(df_clean.head())

df_clean.to_csv('hr_data_cleaned_for_pipeline',index=False)