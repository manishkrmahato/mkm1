import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)

target = pd.Series(housing.target, name="MedHouseVal")

df = pd.concat([df, target], axis=1)

print(df.head())

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

model = LinearRegression()

rfe = RFE(
    estimator=model,
    n_features_to_select=5
)

rfe.fit(X_train, y_train)

selected_features = X.columns[rfe.support_]

print("Selected features:")
print(selected_features.tolist())

print("\nFeature ranking:")
print(pd.DataFrame({
    "Feature": X.columns,
    "Ranking": rfe.ranking_
}))

model_all = LinearRegression()

model_all.fit(X_train, y_train)

pred_all = model_all.predict(X_test)

mae_all = mean_absolute_error(y_test, pred_all)
mse_all = mean_squared_error(y_test, pred_all)
rmse_all = np.sqrt(mse_all)
r2_all = r2_score(y_test, pred_all)

model_selected = LinearRegression()

model_selected.fit(
    X_train[selected_features],
    y_train
)

pred_selected = model_selected.predict(
    X_test[selected_features]
)

mae_selected = mean_absolute_error(y_test, pred_selected)
mse_selected = mean_squared_error(y_test, pred_selected)
rmse_selected = np.sqrt(mse_selected)
r2_selected = r2_score(y_test, pred_selected)

results = pd.DataFrame({
    "Model": [
        "All Features",
        "RFE Selected Features"
    ],
    "Number of Features": [
        8,
        len(selected_features)
    ],
    "MAE": [
        mae_all,
        mae_selected
    ],
    "MSE": [
        mse_all,
        mse_selected
    ],
    "RMSE": [
        rmse_all,
        rmse_selected
    ],
    "R2 Score": [
        r2_all,
        r2_selected
    ]
})

print(results.round(4))

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model_all.coef_
})

importance["Absolute Coefficient"] = importance["Coefficient"].abs()

importance = importance.sort_values(
    "Absolute Coefficient",
    ascending=False
)

importance

print("Selected features using RFE:")
print(selected_features.tolist())

print("\nModel comparison:")
print(results.round(4))

print("\nConclusion:")

if r2_selected > r2_all:
    print("The RFE model has a higher R2 Score than the model using all features.")
    print("Therefore, reducing the number of features improved prediction accuracy.")
elif r2_selected < r2_all:
    print("The model using all features has a higher R2 Score than the RFE model.")
    print("Therefore, reducing the number of features decreased prediction accuracy.")
else:
    print("Both models have the same R2 Score.")
    print("Therefore, reducing the number of features maintained prediction accuracy.")

print("\nSelected features are:")
print(selected_features.tolist())


