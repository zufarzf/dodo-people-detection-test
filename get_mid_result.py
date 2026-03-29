import pandas as pd

df = pd.read_csv("report.csv")
print(df["delay_sec"].mean())