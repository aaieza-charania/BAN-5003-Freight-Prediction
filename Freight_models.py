
import pandas as pd

df = pd.read_csv("olist_freight.csv")

print("Total rows:", len(df))
print(df.head())

# Information used to predict freight cost
X = df[[
"product_weight_g",
"product_length_cm",
"product_height_cm",
"product_width_cm",
"customer_state"
]]

# Actual freight cost we want to predict
y = df["freight_value"]

print("Input data:", X.shape)
print("Target data:", y.shape)

from sklearn.model_selection import train_test_split

# First, separate 70% for training
X_train, X_temp, y_train, y_temp = train_test_split(
X, y, test_size=0.30, random_state=42
)

# Divide the remaining 30% equally
X_val, X_test, y_val, y_test = train_test_split(
X_temp, y_temp, test_size=0.50, random_state=42
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))

from sklearn.linear_model import LinearRegression

X_train = pd.get_dummies(X_train, columns=["customer_state"], dtype=int)
X_val = pd.get_dummies(X_val, columns=["customer_state"], dtype=int)
X_test = pd.get_dummies(X_test, columns=["customer_state"], dtype=int)

X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

model_a = LinearRegression()
model_a.fit(X_train, y_train)

print("Model A trained successfully")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pred_a = model_a.predict(X_val)

mae_a = mean_absolute_error(y_val, pred_a)
rmse_a = mean_squared_error(y_val, pred_a) ** 0.5
r2_a = r2_score(y_val, pred_a)

print("Model A Validation Results")
print("MAE:", round(mae_a, 2))
print("RMSE:", round(rmse_a, 2))
print("R2:", round(r2_a, 3))

from sklearn.ensemble import RandomForestRegressor

model_b = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

model_b.fit(X_train, y_train)

print("Model B trained successfully")

pred_b = model_b.predict(X_val)

mae_b = mean_absolute_error(y_val, pred_b)
rmse_b = mean_squared_error(y_val, pred_b) ** 0.5
r2_b = r2_score(y_val, pred_b)

print("Model B Validation Results")
print("MAE:", round(mae_b, 2))
print("RMSE:", round(rmse_b, 2))
print("R2:", round(r2_b, 3))

# Final test results for Model A
test_pred_a = model_a.predict(X_test)

print("Model A Test Results")
print("MAE:", round(mean_absolute_error(y_test, test_pred_a), 2))
print("RMSE:", round(mean_squared_error(y_test, test_pred_a) ** 0.5, 2))
print("R2:", round(r2_score(y_test, test_pred_a), 3))

# Final test results for Model B
test_pred_b = model_b.predict(X_test)

print("Model B Test Results")
print("MAE:", round(mean_absolute_error(y_test, test_pred_b), 2))
print("RMSE:", round(mean_squared_error(y_test, test_pred_b) ** 0.5, 2))
print("R2:", round(r2_score(y_test, test_pred_b), 3))

# Compare over-quoting and under-quoting
for name, predictions in [("Linear Regression", test_pred_a),
                          ("Random Forest", test_pred_b)]:
    errors = predictions - y_test

    print(name)
    print("Over-quoted orders:", (errors > 0).sum())
    print("Under-quoted orders:", (errors < 0).sum())
    print("Average over-quote:", round(errors[errors > 0].mean(), 2))
    print("Average under-quote:", round(-errors[errors < 0].mean(), 2))