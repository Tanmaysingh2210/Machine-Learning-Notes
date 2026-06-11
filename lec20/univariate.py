import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "train.csv")
df = pd.read_csv(csv_path)

print(df.head())
print(df['Survived'])

# survived and pclass are categorical clumns
# analysing each column seperatly is called unvariate analysis

sns.countplot(df['Survived'])
plt.show()