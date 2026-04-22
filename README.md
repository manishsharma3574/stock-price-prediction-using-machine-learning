# 📈 Stock Price Prediction using Machine Learning

This project focuses on predicting stock closing prices using machine learning regression models.
It leverages historical stock data and engineered features to forecast the next day's price.

---

##  Models Used

The following regression models are implemented and compared:

*  **Linear Regression**
*  **Random Forest Regressor**
*  **XGBoost Regressor**

---

##  Key Concepts

* Time Series Feature Engineering
* Lag-based features (Lag_2, Lag_3)
* Technical Indicators:

  * Daily Returns
  * Rolling Volatility (7-day)
  * RSI (Relative Strength Index)
* Model comparison using evaluation metrics

---

## 📊 Features Used

The models are trained using the following features:

* `Lag_2` – Closing price from 2 days ago
* `Lag_3` – Closing price from 3 days ago
* `RSI_14` – Momentum indicator
* `Daily_Return` – Percentage change in price
* `Volatility_7d` – Rolling standard deviation
* `Volume` – Trading volume

---

## 🎯 Target

* `Target` = Next day closing price (`Close.shift(-1)`)

---

##  Evaluation Metrics

Model performance is evaluated using:

* 📌 **R² Score** (Goodness of fit)
* 📌 **MAE** (Mean Absolute Error)
* 📌 **MSE** (Mean Squared Error)
* 📌 **RMSE** (Root Mean Squared Error)

---

## 📈 Visualization

* Actual vs Predicted price comparison
* Multi-model comparison graph

---

##  Project Structure

```
stock-price-prediction/
│
├── data/
│   └── cleaned_stock_data.csv
│
├── notebooks/
│   └── model_training.ipynb
│
├── predictions.xlsx
│
└── README.md
```

---

##  How to Run

1. Clone the repository
2. Install required libraries:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib
```

3. Run the Python script or notebook
4. View model performance and graphs

---

## 📌 Key Observations

* Linear Regression performs well due to strong price continuity
* Random Forest captures non-linear patterns
* XGBoost provides balanced performance with error correction
* Lag features are highly important in stock prediction

---

##  Limitations

* Does not include external factors (news, macroeconomics)
* Market randomness limits prediction accuracy
* Based only on historical price data

---

##  Conclusion

* Machine learning models can effectively capture patterns in stock prices
* Simpler models like Linear Regression can perform surprisingly well
* Tree-based models provide robustness and flexibility
* Feature engineering plays a crucial role in performance

---
