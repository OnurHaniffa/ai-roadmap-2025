import pandas as pd

df = pd.read_excel("default of credit card clients.xls", skiprows=1)

df.to_csv("UCI_Credit_Card.csv", index=False)
