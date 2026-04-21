import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("/content/cleaned_stock_data.csv")

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

df = df[['Close']]

# 🔥 RETURNS BASED MODEL
df['Return'] = df['Close'].pct_change()

df['Lag_1'] = df['Return'].shift(1)
df['Lag_2'] = df['Return'].shift(2)
df['Lag_3'] = df['Return'].shift(3)

# Remove strongest leakage-like feature
df.drop(['Lag_1'], axis=1, inplace=True)

# Target
df['Target'] = df['Return'].shift(-1)

df = df.dropna()

X = df.drop(['Close','Target'], axis=1)
y = df['Target']

split = int(len(df)*0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))


