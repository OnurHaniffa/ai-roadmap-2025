from tabulate import tabulate

def pretty(df):
    print(tabulate(df, headers="keys", tablefmt="psql"))
