# Stock Market Analysis and Prediction Using Statistical and Machine Learning Methods

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

$$
r_i = \frac{P_i - P_{i-1}}{P_{i-1}}
$$
   
     where $$( P_{i} )$$ is the closing price on day $$( i )$$.

2. **Probability Distribution**:  
   - The return distribution is analyzed using two approaches:  
     - **Parametric**: Fitting the data to known distributions like normal or log-normal distributions.  
     - **Non-Parametric**: Using histograms and kernel density estimations to understand the return distribution.  

3. **Symmetry Analysis**:  
   - Statistical tests are performed to evaluate whether the return distribution is symmetric or skewed.

4. **Mean and Variance Estimation**:  
   - The mean and variance of stock returns are computed and compared across time windows.  
   - Observations are linked to stock performance trends and market fluctuations.

5. **Moving Window Analysis**:  
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

### **Report: Stock Market Analysis and Prediction**

---

### **Introduction**  
In this project, I analyzed stock market data from the Tehran Stock Exchange (TSE) and developed predictive models using statistical and machine learning techniques. The primary goals were to explore trends, evaluate correlations, and predict stock returns. Using data on prices, indices, gold rates, and exchange rates, I employed various models to understand the market's behavior and forecast its future performance.

---

### **Dataset Overview**  
The dataset included critical features such as:  
- **Date**: Persian calendar dates converted to Gregorian for compatibility.  
- **Stock Prices**: Opening, closing, highest, and lowest prices.  
- **Trade Metrics**: Trade volume and value.  
- **Market Indices**: TSE index and other related metrics.  
- **External Factors**: Gold prices and USD exchange rates.  

These features were meticulously cleaned and preprocessed to ensure consistency and usability.

---

### **Data Preprocessing**  

1. **Date Conversion**:  
   - Persian dates were converted to the Gregorian calendar using the **PersianTools** library for integration with modern analytical tools.  

2. **Missing Values**:  
   - Missing data was either filled using forward-fill methods or replaced with zeros for trade metrics.  
   - This ensured no critical gaps in the dataset during analysis.  

3. **Normalization**:  
   - Features like stock returns and prices were scaled to standard ranges to improve model convergence and interpretability.  

* Before Preprocessing:
* 
| *Every Dataset shape* |
|:--:|
| <img style="width:500px" src="https://github.com/user-attachments/assets/bf6668a8-8e55-4999-b172-c88f0167c311"> |

* After Preprocessing:

| *Nan Amount* |
|:--:|
| <img style="width:500px" src="https://github.com/user-attachments/assets/6f35805b-a9a0-46bd-be46-c70341267ba6"> |

| *Every Dataset shape* |
|:--:|
| <img style="width:500px" src="https://github.com/user-attachments/assets/c85dc1c3-a5cd-40c2-9aaf-462acef5cea2"> |

---

### **Exploratory Analysis**  

#### **1. Stock Returns Distribution**  
To understand stock performance, I calculated daily returns using the formula:  
$$
r_i = \frac{P_i - P_{i-1}}{P_{i-1}}
$$
   
     where $$( P_{i} )$$ is the closing price on day $$( i )$$. 

<img style="width:500px" src="https://github.com/user-attachments/assets/7229a317-8fca-4d76-aeac-48438e7db0e7">

- **Probability Distribution**:  
   - Histograms and kernel density estimations revealed that stock returns were not perfectly symmetric, suggesting irregularities and skewness in market behavior.

 <img style="width:500px" src="https://github.com/user-attachments/assets/50f33dba-98c2-407f-acb8-c7aacfd796d2">
 
- **Volatility Analysis**:  
   - Using rolling windows of 10, 20, and 50 days, I observed changes in the mean and variance of returns, identifying periods of high market volatility.
   - 
#### Mobarake Steel Market:
* Before Normalization:
  
| *10-day Window* |
|:--:|
| <img style="width:500px" src="https://github.com/user-attachments/assets/c0b23c42-9c02-4fd6-ac44-3e83e0aab9f3"> |

* After Normalization:

| *10-day Window* |
|:--:|
| <img style="width:500px" src="https://github.com/user-attachments/assets/50f33dba-98c2-407f-acb8-c7aacfd796d2"> |

| *20-day Window* |
|:--:|
| <img style="width:500px" src="https://github.com/user-attachments/assets/50f33dba-98c2-407f-acb8-c7aacfd796d2"> |

| *50-day Window* |
|:--:|
| <img style="width:500px" src="https://github.com/user-attachments/assets/50f33dba-98c2-407f-acb8-c7aacfd796d2"> |


#### Another Market (Tose Atlas Mofid):

| *10-day Window* |
|:--:|
| <img style="width:500px" src="https://github.com/user-attachments/assets/c15e4b92-9b0f-46bb-afc5-02628fc1bb87"> |

| *20-day Window* |
|:--:|
| <img style="width:500px" src="https://github.com/user-attachments/assets/48984fbf-cf11-446d-8b86-ce295763ab77"> |

| *50-day Window* |
|:--:|
| <img style="width:500px" src="https://github.com/user-attachments/assets/10a1c1ea-26bc-4522-95f5-1a15ee542ee3"> |

---

#### **2. Correlation Analysis**  

- **Autocorrelation**:  
   - Autocorrelation analysis over lag windows (1–5 days) indicated short-term dependencies in stock returns, with correlations tapering off as lag increased.  

<img style="width:500px" src="https://github.com/user-attachments/assets/e668e27c-af17-4664-aa5b-9a5f03739105">

- **Cross-Correlation**:  
   - By comparing stock returns with external factors like gold prices and exchange rates, I found significant relationships, particularly during periods of economic uncertainty.  

<img style="width:500px" src="https://github.com/user-attachments/assets/98186bf6-e107-42a7-8971-5ec43e3d5265">
<img style="width:500px" src="https://github.com/user-attachments/assets/9489eaa5-94a7-4a37-be74-1ada6eccc62f">
---

### **Predictive Modeling**  

#### **1. Regression-Based Return Prediction**  

**Model Development**:  
- Using past stock returns as input, I trained a linear regression model to predict future returns.  
- Data was split into 70% training and 30% testing subsets.  

**Feature Engineering**:  
- Past returns over a window of several days were used as predictors.  
- Window sizes were optimized through experimentation, with 10-day windows yielding the best results.  

**Performance**:  
- **Evaluation Metrics**:  
   - The model was assessed using **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)**.  
   - While the model captured short-term trends effectively, it struggled with abrupt shifts caused by unforeseen market events.  

---

#### **2. Binary Trend Prediction**  

**Model Development**:  
- I converted stock returns into binary labels to classify trends (positive or negative).  
- Both **Logistic Regression** and **Gradient Boosting Trees** were implemented.  

**Optimization**:  
- Using grid search, I optimized hyperparameters like the number of trees and learning rate for gradient boosting.  

**Results**:  
- Gradient boosting outperformed logistic regression, achieving an accuracy of ~77%.  
- The confusion matrix revealed that the model was better at predicting positive trends than negative ones, which aligned with the dataset's class imbalance.

<img style="width:500px" src="https://github.com/user-attachments/assets/e49e6755-7f8d-4ba5-bc0a-ac46cae5d300">

#### Logestic Regression:

| *Best Results for 1-day Prediction* |
|:--:|
| <img style="width:500px" src="https://github.com/user-attachments/assets/cb6e490f-4aec-457b-91b0-5e4047f54ad6"> |

| *Best Results for 2-day Prediction* |
|:--:|
| <img style="width:500px" src="https://github.com/user-attachments/assets/257f99ec-ae36-457c-9a05-14598290c5aa"> |

| *Best Results for 3-day Prediction* |
|:--:|
| <img style="width:500px" src="https://github.com/user-attachments/assets/0de447df-f244-428a-be9b-1ccee779cc7f"> |

| *Best Results for 4-day Prediction* |
|:--:|
| <img style="width:500px" src="https://github.com/user-attachments/assets/85eecf2b-c4da-4b24-a425-0f9bf4d2e043"> |

#### Geadiant Boosting Tree:

| *Best Results for 1-day Prediction* |
|:--:|
| <img style="width:500px" src="https://github.com/user-attachments/assets/9ca64ddb-8de4-47cd-be38-4b637035b5ad"> |

| *Best Results for 2-day Prediction* |
|:--:|
| <img style="width:500px" src="https://github.com/user-attachments/assets/bff080b3-3e84-4acc-b56e-b4f222c36c9f"> |

| *Best Results for 3-day Prediction* |
|:--:|
| <img style="width:500px" src="https://github.com/user-attachments/assets/dec24ec3-22ec-4bb7-8b11-b837c7e7b50b"> |

| *Best Results for 4-day Prediction* |
|:--:|
| <img style="width:500px" src="https://github.com/user-attachments/assets/1ca43cf8-8f91-4f69-868b-fbe0b0fe44ed"> |

---

#### **3. Feature Selection with Lasso Regression**  

**Objective**:  
- To identify the most influential predictors for stock returns.  

**Implementation**:  
- Using **Lasso Regression**, I assigned weights to each feature, with larger weights indicating higher importance.  
- The model highlighted specific past days (e.g., the 3rd, 8th, and 10th) as the most predictive, providing valuable insights into market dependencies.

<img style="width:500px" src="https://github.com/user-attachments/assets/a7f1c55a-a963-443b-bf2b-8ed5c0b229e2">

| *Best Results for 10-day Prediction* |
|:--:|
| <img style="width:500px" src="https://github.com/user-attachments/assets/48171312-6c89-47d6-96b4-c572bb8e9eb5"> |

---

### **Analysis of External Factors**  
- **Gold Prices and Exchange Rates**:  
   - These features added context to stock market trends, with strong correlations observed during periods of inflation or political instability.  
   - Integrating these factors improved model predictions, especially for indices sensitive to global markets.  

---

### **Observations and Challenges**  

1. **Performance of Regression Models**:  
   - Linear regression captured short-term dependencies well but was overly sensitive to noise in volatile markets.  
   - Incorporating external factors partially mitigated this limitation.  

2. **Trend Classification Accuracy**:  
   - Binary models excelled in capturing general trends but struggled with minor fluctuations.  
   - Gradient boosting provided more robust results compared to simpler models like logistic regression.

3. **Feature Engineering Challenges**:  
   - Choosing optimal window sizes and feature sets required extensive experimentation.  
   - Lasso regression effectively reduced feature redundancy, simplifying the model without sacrificing accuracy.


### For step-by-step results, check the [Stock MarKet Analysis and Prediction](https://github.com/peymanHashemi/Image-Alignment-With-SIFT-and-FREAK/blob/3275f2b70cc22a0a535b2ac5ba8f90ad379e15dd/SIFT.ipynb).

---

### **Conclusion**  
This project provided a comprehensive framework for analyzing stock market data and predicting returns. By combining statistical insights with machine learning models, I gained valuable experience in financial forecasting. While the models performed well for short-term predictions, they highlighted the challenges of long-term forecasting in volatile markets. Future improvements could include neural network-based models or integrating more granular external data for enhanced accuracy.
