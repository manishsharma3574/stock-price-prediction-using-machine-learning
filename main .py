import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------------------
#  Load Data

df = pd.read_csv("/content/cleaned_stock_data.csv")

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

# -------------------------------
# Feature Engineering

df['Lag_2'] = df['Close'].shift(2)
df['Lag_3'] = df['Close'].shift(3)

df['Daily_Return'] = df['Close'].pct_change()
df['Volatility_7d'] = df['Daily_Return'].rolling(7).std()

# -------------------------------
#  Target
df['Target'] = df['Close'].shift(-1)

df = df.dropna()

# -------------------------------
# Features

features = [
    'Lag_2',
    'Lag_3',
    'RSI_14',
    'Daily_Return',
    'Volatility_7d',
    'Volume'
]

X = df[features]
y = df['Target']

# -------------------------------
#  Train-Test Split

split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# -------------------------------
#  Models

lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)

# Train
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Predict
pred_lr = lr.predict(X_test)
pred_rf = rf.predict(X_test)
pred_xgb = xgb.predict(X_test)

# -------------------------------
#  Evaluation

def evaluate(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\n{name}")
    print("R2:", r2_score(y_true, y_pred))
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("MSE:", mse)
    print("RMSE:", rmse)

evaluate("Linear Regression", y_test, pred_lr)
evaluate("Random Forest", y_test, pred_rf)
evaluate("XGBoost", y_test, pred_xgb)

# -------------------------------
# Graph (ALL MODELS)

plt.figure(figsize=(10,5))

plt.plot(y_test.values, label='Actual', linewidth=2)
plt.plot(pred_lr, label='Linear')
plt.plot(pred_rf, label='Random Forest')
plt.plot(pred_xgb, label='XGBoost')

plt.legend()
plt.title("Model Comparison")
plt.xlabel("Time")
plt.ylabel("Price")

plt.show()
