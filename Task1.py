import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv("titanic.csv")

#Understand the data
print(df.info())
# View first 5 rows
print(df.head())
# View last 5 rows
print(df.tail())
#Shape of the dataset
print("Shape of the dataset:", df.shape)
print("Columns in the dataset:", df.columns)
print("Data types of each column:\n", df.dtypes)
print("Statistical summary of numerical columns:\n", df.describe())
print("Statistical summary of categorical columns:\n", df.describe(include=['O']))
print("Number of missing values in each column:\n", df.isnull().sum())
print("Number of duplicate rows:", df.duplicated().sum())
print("Unique values in each column:\n", df.nunique())
print("Unique Embarked:\n", df["Embarked"].unique())
#Cleaning the data
# Handling missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)
# Removing duplicates
df.drop_duplicates(inplace=True)
# Verify cleaning steps
print("Missing values after cleaning:\n", df.isnull().sum())
print("Shape after cleaning:", df.shape)
# Exploratory Data Analysis (EDA)

# UNIVARIATE ANALYSIS
plt.figure(figsize=(12,5))
plt.subplot(121)
sns.histplot(df['Age'], kde=True)
plt.title("Age Distribution")

plt.subplot(122)
sns.histplot(df['Fare'], kde=True)
plt.title("Fare Distribution")
plt.show()

# OUTLIER DETECTION USING BOXPLOTS 
plt.figure(figsize=(12,5))
plt.subplot(121)
sns.boxplot(y=df['Age'])
plt.title("Age Outliers")

plt.subplot(122)
sns.boxplot(y=df['Fare'])
plt.title("Fare Outliers")
plt.show()

# CATEGORICAL ANALYSIS 
plt.figure(figsize=(6,5))
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title("Survival by Sex")
plt.show()

plt.figure(figsize=(6,5))
sns.countplot(data=df, x='Embarked', hue='Survived')
plt.title("Survival by Embarked")
plt.show()

plt.figure(figsize=(6,5))
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title("Survival by Class")
plt.show()

# BIVARIATE ANALYSIS 
sns.histplot(data=df, x='Age', hue='Survived', bins=30, multiple='stack')
plt.title("Age vs Survival")
plt.show()

sns.histplot(data=df, x='Fare', hue='Survived', bins=30, multiple='stack')
plt.title("Fare vs Survival")
plt.show()

# FEATURE ENGINEERING
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1   # FIXED
df['IsAlone'] = np.where(df['FamilySize'] > 1, 0, 1)
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.')

print(df[['SibSp','Parch','FamilySize','IsAlone','Title']].head())

# SURVIVAL RATES
print("Survival rate:\n", df['Survived'].value_counts(normalize=True)*100)
print("Survival by Sex:\n", df.groupby('Sex')['Survived'].mean())
print("Survival by Pclass:\n", df.groupby('Pclass')['Survived'].mean())
print("Survival by FamilySize:\n", df.groupby('FamilySize')['Survived'].mean())

# CORRELATION (NUMERICAL ONLY)
corr = df.corr(numeric_only=True)

plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# PAIRPLOT
sns.pairplot(df, vars=['Age','Fare','FamilySize'], hue='Survived')
plt.show()

# SAVE CLEANED DATA
df.to_csv("titanic_cleaned.csv", index=False)