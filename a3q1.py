import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.head())

print("Shape of dataset:", df.shape)

print("\nColumn names:")
print(df.columns)

print("\nDataset information:")
df.info()

print(df.isnull().sum())

print("Number of duplicate records:", df.duplicated().sum())

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

df = df.drop("customerID", axis=1)

print(df.head())

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print(df["Churn"].value_counts())

print(df.describe())

plt.figure(figsize=(6, 4))

sns.countplot(x="Churn", data=df)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.show()

churn_percentage = df["Churn"].value_counts(normalize=True) * 100

print(churn_percentage)

plt.figure(figsize=(7, 5))

sns.boxplot(x="Churn", y="MonthlyCharges", data=df)

plt.title("Monthly Charges vs Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.show()

plt.figure(figsize=(7, 5))

sns.boxplot(x="Churn", y="tenure", data=df)

plt.title("Tenure vs Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")

plt.show()

plt.figure(figsize=(8, 5))

sns.countplot(x="Contract", hue="Churn", data=df)

plt.title("Contract Type vs Customer Churn")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.xticks(rotation=15)

plt.show()

plt.figure(figsize=(8, 5))

sns.countplot(x="InternetService", hue="Churn", data=df)

plt.title("Internet Service vs Customer Churn")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.show()

#separate input features and target variable
X = df.drop("Churn", axis=1)
y = df["Churn"]

print("X shape:", X.shape)
print("y shape:", y.shape)

#converting categorical columns into numerical columns

X = pd.get_dummies(X, drop_first=True)

print("Shape after encoding:", X.shape)

X.head()

#checking the missing values

print("Total missing values before fixing:")
print(X.isnull().sum().sum())

#filling missing values with the median
X = X.fillna(X.median())

print("\nTotal missing values after fixing:")
print(X.isnull().sum().sum())

#split the dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

#standardize the features

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#check if there are still NaN values

print("NaN values in training data:",
      np.isnan(X_train_scaled).sum())

print("NaN values in testing data:",
      np.isnan(X_test_scaled).sum())

# Create classification models

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    
    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),
    
    "SVM": SVC(
        probability=True,
        random_state=42
    )
}

print("Five models created successfully.")

#store the results of all models

results = []

for name, model in models.items():
    
    #train the model
    model.fit(X_train_scaled, y_train)
    
    #make predictions
    y_pred = model.predict(X_test_scaled)
    
    #get probability of churn
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    #calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    #store results
    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ])

print("All models trained successfully.")

#creating a table of results

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ]
)

#rounding up the values
print(results_df.round(3))

#plot confusion matrices for all models

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

axes = axes.ravel()

for i, (name, model) in enumerate(models.items()):
    
    #predictions
    y_pred = model.predict(X_test_scaled)
    
    #confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    #plot
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[i]
    )
    
    axes[i].set_title(name)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

#hide unused subplot
axes[5].axis("off")

plt.tight_layout()
plt.show()

#plot ROC curves for all models
plt.figure(figsize=(8, 6))

for name, model in models.items():
    
    # probability of churn
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    #calculate ROC values
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    #calculate AUC
    auc = roc_auc_score(y_test, y_prob)
    
    # Plot
    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.3f})"
    )

#random classifier line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()
plt.show()

#plot Precision-Recall curves
plt.figure(figsize=(8, 6))

for name, model in models.items():
    
    #probability of churn
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    #calculate precision and recall
    precision, recall, thresholds = precision_recall_curve(
        y_test,
        y_prob
    )
    
    #plot
    plt.plot(
        recall,
        precision,
        label=name
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.legend()
plt.show()

#compare all evaluation metrics
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]

results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Comparison of Classification Models")
plt.xlabel("Model")
plt.ylabel("Score")

plt.ylim(0, 1)

plt.xticks(rotation=20)

plt.legend()

plt.show()

#find the best model based on ROC-AUC
best_model = results_df.loc[
    results_df["ROC-AUC"].idxmax()
]

print("Best Model:", best_model["Model"])
print("Accuracy:", round(best_model["Accuracy"], 3))
print("Precision:", round(best_model["Precision"], 3))
print("Recall:", round(best_model["Recall"], 3))
print("F1-Score:", round(best_model["F1-Score"], 3))
print("ROC-AUC:", round(best_model["ROC-AUC"], 3))


