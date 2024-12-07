# Project 7: Stock Market Analysis and Prediction Using Statistical and Machine Learning Methods

This project aims to analyze stock market data from the Tehran Stock Exchange (TSE) and predict stock returns using statistical and machine learning models. The project uses stock prices and indices over a two-year period, applying advanced techniques to understand trends, assess correlations, and develop predictive models for future performance.

---
# Content
- Table of Contents
  * [Dataset Overview](#Dataset-Overview) 
  * [Implementation Details](#Implementation-Details)  
  * [Evaluation Metrics](#Evaluation-Metrics)
  * [METHOD](#METHOD)
---
### **Dataset Overview**  
The data includes stock market details for several companies, including key metrics such as:  
- **Date**  
- **Opening Price**  
- **Closing Price**  
- **Highest Price**  
- **Lowest Price**  
- **Trade Volume**  
- **Trade Value**  
- **Last Trade Price**  

Additionally, data for the TSE market index, gold prices, and exchange rates are included for broader analysis.

### **Implementation Details**

#### **1. Probabilistic Analysis of Stock Returns**  
1. **Return Calculation**:  
   Stock return for each day is calculated using the formula:  
   \[
   r_i = \frac{P_{i} - P_{i-1}}{P_{i-1}}
   \]  
   where \( P_{i} \) is the closing price on day \( i \).
   The stock return is calculated as `r_i = (P_i - P_(i-1)) / P_(i-1)`.

3. **Probability Distribution**:  
   - The return distribution is analyzed using two approaches:  
     - **Parametric**: Fitting the data to known distributions like normal or log-normal distributions.  
     - **Non-Parametric**: Using histograms and kernel density estimations to understand the return distribution.  

4. **Symmetry Analysis**:  
   - Statistical tests are performed to evaluate whether the return distribution is symmetric or skewed.

5. **Mean and Variance Estimation**:  
   - The mean and variance of stock returns are computed and compared across time windows.  
   - Observations are linked to stock performance trends and market fluctuations.

6. **Moving Window Analysis**:  
   - A rolling analysis is conducted using windows of 10, 20, and 50 days to evaluate the temporal changes in mean and variance.

---

#### **2. Correlation Analysis**  
1. **Autocorrelation**:  
   - Autocorrelation of stock returns over lag windows (1 to 5 days) is calculated to assess temporal dependencies.  

2. **Cross-Correlation**:  
   - Correlation between stock returns and the TSE market index is analyzed.  
   - Correlations with external factors like gold prices and currency exchange rates are optionally computed to understand broader influences.

---

#### **3. Stock Return Prediction**  
1. **Linear Regression**:  
   - A regression model is trained to predict stock returns based on historical data.  
   - Input features include stock returns from previous days. The number of days is chosen based on prior analysis.

2. **Binary Classification of Trends**:  
   - A classification model predicts whether the stock return for a given day will be positive or negative.  
   - Features include recent stock and index return values. The performance is evaluated using accuracy.

3. **Hybrid Model**:  
   - A combined model predicts both the return value and its trend, leveraging stock and market indices.  

4. **Lasso Regression**:  
   - Feature selection is performed using Lasso regression, focusing on selecting the most relevant predictors for return prediction.

---

### **Evaluation Metrics**  
1. **Return Prediction**:  
   - **Mean Squared Error (MSE)** is used to assess the accuracy of predicted returns.  

2. **Trend Prediction**:  
   - **Accuracy** is used to evaluate the model's ability to predict positive or negative trends.


---

# METHOD
