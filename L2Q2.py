import pandas as pd
import numpy as np
import time

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

correlation = df.corr()["MedHouseVal"].drop("MedHouseVal")

correlation = correlation.abs().sort_values(ascending=False)

print("Features ranked by correlation:")
print(correlation)

# Model A: top 3 features
features_A = correlation.head(3).index.tolist()

# Model B: top 5 features
features_B = correlation.head(5).index.tolist()

# Model C: all features
features_C = X.columns.tolist()

print("Model A:", features_A)
print("Model B:", features_B)
print("Model C:", features_C)

model_A = LinearRegression()
model_B = LinearRegression()
model_C = LinearRegression()

start = time.time()
model_A.fit(X_train[features_A], y_train)
time_A = time.time() - start

start = time.time()
model_B.fit(X_train[features_B], y_train)
time_B = time.time() - start

start = time.time()
model_C.fit(X_train[features_C], y_train)
time_C = time.time() - start

print("Model A training time:", time_A)
print("Model B training time:", time_B)
print("Model C training time:", time_C)

pred_A = model_A.predict(X_test[features_A])
pred_B = model_B.predict(X_test[features_B])
pred_C = model_C.predict(X_test[features_C])

mae_A = mean_absolute_error(y_test, pred_A)
mse_A = mean_squared_error(y_test, pred_A)
rmse_A = np.sqrt(mse_A)
r2_A = r2_score(y_test, pred_A)

mae_B = mean_absolute_error(y_test, pred_B)
mse_B = mean_squared_error(y_test, pred_B)
rmse_B = np.sqrt(mse_B)
r2_B = r2_score(y_test, pred_B)

mae_C = mean_absolute_error(y_test, pred_C)
mse_C = mean_squared_error(y_test, pred_C)
rmse_C = np.sqrt(mse_C)
r2_C = r2_score(y_test, pred_C)

results = pd.DataFrame({
    "Model": ["Model A", "Model B", "Model C"],
    "Features": [3, 5, 8],
    "MAE": [mae_A, mae_B, mae_C],
    "MSE": [mse_A, mse_B, mse_C],
    "RMSE": [rmse_A, rmse_B, rmse_C],
    "R2 Score": [r2_A, r2_B, r2_C],
    "Training Time": [time_A, time_B, time_C]
})

print(results.round(4))

best_model = results.loc[results["R2 Score"].idxmax(), "Model"]

print("Best performing model:", best_model)

print("\nFeature combinations:")
print("Model A:", features_A)
print("Model B:", features_B)
print("Model C:", features_C)

print("\nTraining times:")
print("Model A:", time_A, "seconds")
print("Model B:", time_B, "seconds")
print("Model C:", time_C, "seconds")


