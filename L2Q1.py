import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)

#create target variable as a Pandas Series
target = pd.Series(housing.target, name="MedHouseVal")

# Add target to the DataFrame
df = pd.concat([df, target], axis=1)

print(df.head())

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("Column names:")
print(df.columns.tolist())

print("MedInc       - Median income")
print("HouseAge     - Median house age")
print("AveRooms     - Average number of rooms")
print("AveBedrms    - Average number of bedrooms")
print("Population   - Population of the area")
print("AveOccup     - Average occupancy")
print("Latitude     - Latitude")
print("Longitude    - Longitude")
print("MedHouseVal  - Median house value (Target)")

df.info()

print(df.dtypes)

print(df.describe())

print("Missing values in each column:")
print(df.isnull().sum())

print("\nTotal missing values:",
      df.isnull().sum().sum())

print("Number of duplicate records:",
      df.duplicated().sum())

df.hist(figsize=(14, 10), bins=30)

plt.suptitle("Distribution of California Housing Attributes")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))

sns.boxplot(data=df)

plt.title("Box Plot of California Housing Dataset")
plt.xticks(rotation=45)

plt.show()

#Q1 and Q3
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)

#IQR
IQR = Q3 - Q1

#outliers
outliers = (
    (df < (Q1 - 1.5 * IQR)) |
    (df > (Q3 + 1.5 * IQR))
)

print("Number of outliers in each column:")
print(outliers.sum())

plt.figure(figsize=(8, 5))

plt.scatter(
    df["MedInc"],
    df["MedHouseVal"],
    alpha=0.3
)

plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Median Income vs Median House Value")

plt.show()

plt.figure(figsize=(8, 5))

plt.scatter(
    df["HouseAge"],
    df["MedHouseVal"],
    alpha=0.3
)

plt.xlabel("House Age")
plt.ylabel("Median House Value")
plt.title("House Age vs Median House Value")

plt.show()

plt.figure(figsize=(8, 5))

plt.scatter(
    df["AveRooms"],
    df["MedHouseVal"],
    alpha=0.3
)

plt.xlabel("Average Number of Rooms")
plt.ylabel("Median House Value")
plt.title("Average Rooms vs Median House Value")

plt.show()

correlation = df.corr()

correlation

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()

#filling missing values with median
for column in df.columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].median())

print("Total missing values after treatment:",
      df.isnull().sum().sum())

#columns where outliers will be treated
outlier_columns = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "MedHouseVal"
]

for column in outlier_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    df[column] = df[column].clip(
        lower_limit,
        upper_limit
    )

print("Outliers have been treated.")

plt.figure(figsize=(14, 8))

sns.boxplot(data=df)

plt.title("Box Plot After Outlier Treatment")
plt.xticks(rotation=45)

plt.show()

#input features
X = df.drop("MedHouseVal", axis=1)

#target variable
y = df["MedHouseVal"]

print("Features:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

#standard scaler
scaler = StandardScaler()

#scale training data
X_train_scaled = scaler.fit_transform(X_train)

#scale testing data
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed.")

#convert scaled training data into a dataframe
X_train_scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)

X_train_scaled_df.head()

print("FINAL CHECK: ")

print("\nDataset shape:")
print(df.shape)

print("\nTraining features:")
print(X_train.shape)

print("\nTesting features:")
print(X_test.shape)

print("\nTraining target:")
print(y_train.shape)

print("\nTesting target:")
print(y_test.shape)

print("\nTotal missing values:")
print(df.isnull().sum().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())


