import pandas as pd

df = pd.read_csv("optimizer_accuracy_results.csv", index_col=0)

print(df)