import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Titanic dataset from Kaggle
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
df.head()

df.describe()

df.info()
df.isnull().sum()

df.hist(bins=20, figsize=(14, 10), color='skyblue')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y='Age', data=df)
plt.subplot(1, 2, 2)
sns.boxplot(y='Fare', data=df)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
numerical_df = df.select_dtypes(include=np.number)  
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm')
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()



sns.pairplot(df[['Age', 'Fare', 'Survived', 'Pclass']].dropna(), hue='Survived')
plt.show()
