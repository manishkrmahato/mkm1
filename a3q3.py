import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("kc_house_data.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("\nShape of dataset:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("Missing values in each column:")
print(df.isnull().sum())

#removing unnecessary columns
df = df.drop(columns=['id', 'date'])

#checking categorical columns
print("\nData types:")
print(df.dtypes)

#convert categorical column 'zipcode' into dummy variables
df = pd.get_dummies(df, columns=['zipcode'], drop_first=True)

#checking the dataset after preprocessing
print("\nShape after preprocessing:", df.shape)

print(df.head())

#separating input features and target variable
X = df.drop(columns=['price'])
y = df['price']

#train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)

#scaling the numerical features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")

#calculating all evaluation metrics
def evaluate_model(model, X_train, X_test, y_train, y_test):
    
    model.fit(X_train, y_train)
    
    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, train_prediction)
    test_mse = mean_squared_error(y_test, test_prediction)
    
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    
    train_mae = mean_absolute_error(y_train, train_prediction)
    test_mae = mean_absolute_error(y_test, test_prediction)
    
    train_r2 = r2_score(y_train, train_prediction)
    test_r2 = r2_score(y_test, test_prediction)
    
    return {
        'Train MSE': train_mse,
        'Test MSE': test_mse,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train MAE': train_mae,
        'Test MAE': test_mae,
        'Train R2': train_r2,
        'Test R2': test_r2
    }


#multiple linear regression
linear_model = LinearRegression()

linear_results = evaluate_model(
    linear_model,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)

print("Multiple Linear Regression Results")
print("----------------------------------")

for metric, value in linear_results.items():
    print(f"{metric}: {value:.4f}")

lambdas = [0.01, 0.1, 1, 10, 100]

ridge_results = []
lasso_results = []
elastic_results = []

for alpha in lambdas:

    # Ridge Regression
    ridge = Ridge(alpha=alpha)

    ridge_result = evaluate_model(
        ridge,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )

    ridge_result['Lambda'] = alpha
    ridge_results.append(ridge_result)


    # Lasso Regression
    lasso = Lasso(
        alpha=alpha,
        max_iter=100000,
        tol=0.001
    )

    lasso_result = evaluate_model(
        lasso,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )

    lasso_result['Lambda'] = alpha
    lasso_results.append(lasso_result)


    # Elastic Net Regression
    elastic = ElasticNet(
        alpha=alpha,
        l1_ratio=0.5,
        max_iter=100000,
        tol=0.001
    )

    elastic_result = evaluate_model(
        elastic,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )

    elastic_result['Lambda'] = alpha
    elastic_results.append(elastic_result)


# Convert results into DataFrames
ridge_df = pd.DataFrame(ridge_results)
lasso_df = pd.DataFrame(lasso_results)
elastic_df = pd.DataFrame(elastic_results)


print("Ridge Regression Results")
print(ridge_df)

print("Lasso Regression Results")
print(lasso_df)

print("Elastic Net Regression Results")
print(elastic_df)

#ploting RMSE against lambda for all regularized models
plt.figure(figsize=(10, 6))

plt.plot(
    lambdas,
    ridge_df['Train RMSE'],
    marker='o',
    label='Ridge Train'
)

plt.plot(
    lambdas,
    ridge_df['Test RMSE'],
    marker='o',
    label='Ridge Test'
)

plt.plot(
    lambdas,
    lasso_df['Train RMSE'],
    marker='s',
    label='Lasso Train'
)

plt.plot(
    lambdas,
    lasso_df['Test RMSE'],
    marker='s',
    label='Lasso Test'
)

plt.plot(
    lambdas,
    elastic_df['Train RMSE'],
    marker='^',
    label='Elastic Net Train'
)

plt.plot(
    lambdas,
    elastic_df['Test RMSE'],
    marker='^',
    label='Elastic Net Test'
)

plt.xscale('log')

plt.xlabel('Lambda (α)')
plt.ylabel('RMSE')
plt.title('Training and Testing RMSE vs Lambda')

plt.legend()
plt.grid(True)
plt.show()

#training the models using a selected lambda
#I used lambda = 1 for comparing coefficients
ridge_model = Ridge(alpha=1)
lasso_model = Lasso(alpha=1, max_iter=10000)
elastic_model = ElasticNet(alpha=1, l1_ratio=0.5, max_iter=10000)

ridge_model.fit(X_train_scaled, y_train)
lasso_model.fit(X_train_scaled, y_train)
elastic_model.fit(X_train_scaled, y_train)

#creating coefficient comparison table
coefficient_df = pd.DataFrame({
    'Feature': X.columns,
    'Linear': linear_model.coef_,
    'Ridge': ridge_model.coef_,
    'Lasso': lasso_model.coef_,
    'Elastic Net': elastic_model.coef_
})

print(coefficient_df.head(20))

#plot coefficients of the main features
top_features = coefficient_df.copy()

#select features based on absolute Linear Regression coefficient
top_features['absolute_value'] = top_features['Linear'].abs()
top_features = top_features.sort_values(
    'absolute_value',
    ascending=False
).head(15)

top_features = top_features.drop(columns=['absolute_value'])

top_features.set_index('Feature').plot(
    kind='bar',
    figsize=(14, 7)
)

plt.title('Comparison of Regression Coefficients')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.legend()
plt.tight_layout()
plt.show()

#create final comparison table
final_results = []

#linear Regression
final_results.append({
    'Model': 'Linear Regression',
    'Lambda': '-',
    'Test MSE': linear_results['Test MSE'],
    'Test RMSE': linear_results['Test RMSE'],
    'Test MAE': linear_results['Test MAE'],
    'Test R2': linear_results['Test R2']
})

#best Ridge model based on Test RMSE
best_ridge = ridge_df.loc[ridge_df['Test RMSE'].idxmin()]

final_results.append({
    'Model': 'Ridge Regression',
    'Lambda': best_ridge['Lambda'],
    'Test MSE': best_ridge['Test MSE'],
    'Test RMSE': best_ridge['Test RMSE'],
    'Test MAE': best_ridge['Test MAE'],
    'Test R2': best_ridge['Test R2']
})

#best Lasso model
best_lasso = lasso_df.loc[lasso_df['Test RMSE'].idxmin()]

final_results.append({
    'Model': 'Lasso Regression',
    'Lambda': best_lasso['Lambda'],
    'Test MSE': best_lasso['Test MSE'],
    'Test RMSE': best_lasso['Test RMSE'],
    'Test MAE': best_lasso['Test MAE'],
    'Test R2': best_lasso['Test R2']
})

#best Elastic Net model
best_elastic = elastic_df.loc[
    elastic_df['Test RMSE'].idxmin()
]

final_results.append({
    'Model': 'Elastic Net',
    'Lambda': best_elastic['Lambda'],
    'Test MSE': best_elastic['Test MSE'],
    'Test RMSE': best_elastic['Test RMSE'],
    'Test MAE': best_elastic['Test MAE'],
    'Test R2': best_elastic['Test R2']
})

final_comparison = pd.DataFrame(final_results)

print("Final Model Comparison")
print("======================")

print(final_comparison)

#find the best model based on RMSE
best_model = final_comparison.loc[
    final_comparison['Test RMSE'].idxmin()
]

print("\nBest model based on Test RMSE:")
print(best_model['Model'])

print("\nBest Lambda:")
print(best_model['Lambda'])
