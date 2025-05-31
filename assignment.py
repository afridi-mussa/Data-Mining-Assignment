import pandas as pd
import numpy as np

# Sample Titanic-like data
sample_data = {
    'PassengerId': [1, 2, 3, 4, 5],
    'Survived': [0, 1, 1, 1, 0],
    'Pclass': [3, 1, 3, 1, 3],
    'Name': [
        "Braund, Mr. Owen Harris",
        "Cumings, Mrs. John Bradley",
        "Heikkinen, Miss. Laina",
        "Futrelle, Mrs. Jacques Heath",
        "Allen, Mr. William Henry"
    ],
    'Sex': ['male', 'female', 'female', 'female', 'male'],
    'Age': [22, 38, 26, 35, np.nan],
    'SibSp': [1, 1, 0, 1, 0],
    'Parch': [0, 0, 0, 0, 0],
    'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450'],
    'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05],
    'Cabin': [np.nan, "C85", np.nan, "C123", np.nan],
    'Embarked': ['S', 'C', 'S', 'S', np.nan]
}

df = pd.DataFrame(sample_data)

# Missing values before
print("Missing values before cleaning:\n", df.isnull().sum(), "\n")

# Data Cleaning
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Cabin'].fillna("Unknown", inplace=True)
df.dropna(subset=['Survived'], inplace=True)

# Age binning
def age_bin(age):
    if age <= 12:
        return "Child"
    elif age <= 19:
        return "Teen"
    elif age <= 35:
        return "Adult"
    elif age <= 60:
        return "Middle-Aged"
    else:
        return "Senior"

df['AgeGroup'] = df['Age'].apply(age_bin)

# Data Integration
df['FamilyName'] = df['Name'].apply(lambda x: x.split(',')[0])

# Final output
print("Missing values after cleaning:\n", df.isnull().sum(), "\n")
print("Cleaned and Integrated Sample:\n", df[['PassengerId', 'Survived', 'Age', 'AgeGroup', 'Embarked', 'Cabin', 'FamilyName']])
