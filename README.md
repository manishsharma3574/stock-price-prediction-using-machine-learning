## Stock Price Prediction using Machine Learning
** Project Overview **

This project aims to predict stock prices using regression-based machine learning models. The primary objective is to analyze how different algorithms perform on financial time series data and to understand the impact of feature engineering on prediction accuracy.

Stock markets are inherently noisy and influenced by multiple external factors, making prediction a challenging task. This project focuses on building reliable baseline models and comparing their performance using standard evaluation metrics.

 Objectives
**To implement regression models for stock price prediction*
**To perform feature engineering on financial data*
**To compare model performance using statistical metrics*
**To analyze the strengths and limitations of each model*
  # Models Used

**The following regression models are implemented:*

- Linear Regression -
A simple and interpretable model used as a baseline for comparison
- Decision Tree Regressor -
Captures non-linear relationships but may overfit on training data
  - Random Forest Regressor -
 An ensemble model that improves performance and reduces overfitting
 - Feature Engineering -

To improve model performance, several technical indicators and features are created:

*Simple Moving Average (SMA)*
*Exponential Moving Average (EMA)*
*Relative Strength Index (RSI)*
*Lag Features (previous returns)*
*Daily Returns using percentage change*

These features help the models capture trends and patterns in stock price movements.

 - Evaluation Metrics

The models are evaluated using the following metrics:

*Mean Squared Error (MSE)*
*Measures average squared difference between predicted and actual values*
*Mean Absolute Error (MAE)*
*Measures average absolute difference*
*R² Score (Coefficient of Determination)*
*Indicates how well the model explains variance in the data*
Results and Analysis
--Model	Observation--
Linear Regression	Provides a good baseline but struggles with non-linearity
 Key Insights
*Stock price prediction is highly uncertain and noisy*
*Feature engineering significantly impacts model performance*
*Simple models can perform competitively with good features*
*Ensemble methods like Random Forest improve stability*
*Overfitting is a common issue in tree-based models*
 Conclusion

This project demonstrates that:

*Regression models are appropriate for stock price prediction tasks*
*Linear Regression serves as a strong baseline*
*Decision Trees can model complex relationships but require control*
*Random Forest provides better generalization and robustness*
*Model performance depends more on feature quality than model complexity*
 #Future Improvements
*Hyperparameter tuning for better performance*
*Incorporating more technical indicators*
*Using deep learning models (LSTM, RNN)*
*Including external factors such as news sentiment*
Decision Tree	Captures patterns well but prone to overfitting
Random Forest	Offers better generalization and more stable predictions
