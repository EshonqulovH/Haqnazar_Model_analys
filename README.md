# Logistic Regression with Correlation Analysis

This project performs logistic regression analysis on a dataset to predict customer churn (`Exited`). Several feature engineering techniques are applied, followed by a variety of correlation analyses to understand relationships between the target variable (`Exited`) and other features. This README provides a breakdown of the steps taken in the project.

## Table of Contents
1. [Project Description](#project-description)
2. [Dataset Overview](#dataset-overview)
3. [Feature Engineering](#feature-engineering)
4. [Encoding](#encoding)
5. [Correlation Analysis](#correlation-analysis)
   - Point-Biserial Correlation
   - Phi Correlation
   - Cramér's V Correlation
   - Kendall's Tau Correlation
6. [Visualization](#visualization)
7. [Installation](#installation)
8. [How to Run](#how-to-run)
9. [Conclusion](#conclusion)
10. [License](#license)

## Project Description

This project uses logistic regression to predict customer churn (whether a customer exits or not). Various correlation methods, including **Point-Biserial**, **Phi**, **Cramér's V**, and **Kendall's Tau**, are used to analyze relationships between features and the target variable. The insights from these analyses can help improve feature selection and model performance.

## Dataset Overview

The dataset includes the following files:

- `train.csv`: The training dataset with customer features and target labels (Exited).
- `test.csv`: The test dataset with customer features only.
- `sample_submission.csv`: A sample submission file for predictions.

The dataset contains customer data such as age, credit score, balance, number of products, and other features that might influence whether a customer leaves the company.

## Feature Engineering

Several new features are created to enhance the predictive power of the model:

- `Age_to_NumOfProducts`: Ratio of customer's age to the number of products they use.
- `Balance_Tenure`: Product of balance and tenure (the number of years the customer has been with the company).
- `Balance_to_CreditScore`: Ratio of balance to credit score.
- `Balance_to_NumOfProducts`: Ratio of balance to the number of products used.
- `Balance_to_Salary`: Ratio of balance to estimated salary.
- `Tenure_to_Age`: Ratio of tenure to age.
- `CreditScore_to_Age`: Ratio of credit score to age.
- `Balance_NumOfProducts`: Product of balance and the number of products used.
- `CreditScore_IsActive`: Product of credit score and whether the customer is an active member.

## Encoding

Categorical variables such as `Geography` and `Gender` are encoded using `LabelEncoder`. In cases with more categories, `One-Hot Encoding` could be used for better model performance.

```python
le = LabelEncoder()
for column in data_encod.select_dtypes(include=['object']).columns:
    data_encod[column] = le.fit_transform(data_encod[column])
```

## Correlation Analysis

To understand the relationship between features and the target variable (`Exited`), several correlation methods are applied:

### 1. **Point-Biserial Correlation**
This measures the correlation between a binary variable (`Exited`) and continuous variables.

```python
from scipy.stats import pointbiserialr
korrelyatsiya, p_value = pointbiserialr(binar_ustunlar, uzluksiz_ustunlar[uzluksiz_ustun])
```

### 2. **Phi Correlation**
This is used to measure the correlation between two binary variables.

```python
from scipy.stats import chi2_contingency
contingency_table = pd.crosstab(target_ustun, binar_ustunlar[binar_ustun])
chi2, p_value, _, _ = chi2_contingency(contingency_table)
phi_korrelyatsiya = np.sqrt(chi2 / n)
```

### 3. **Cramér's V Correlation**
Used to measure association between two categorical variables.

```python
def cramers_v(chi2, n, k1, k2):
    return np.sqrt(chi2 / (n * (min(k1 - 1, k2 - 1))))
```

### 4. **Kendall's Tau Correlation**
This measures the ordinal correlation between two continuous variables.

```python
from scipy.stats import kendalltau
tau, p_value = kendalltau(target_ustun, uzluksiz_ustunlar[uzluksiz_ustun])
```

## Visualization

Correlation results are visualized using **Seaborn** heatmaps for better interpretation of the relationships between features and the target variable.

```python
sns.heatmap(korrelyatsiya_df, annot=True, cmap='coolwarm', center=0)
plt.title('Exited and Continuous Features: Point-Biserial Correlation')
plt.show()
```

## Installation

To run this project, you need to have Python installed along with the following libraries:

- pandas
- numpy
- scipy
- seaborn
- scikit-learn
- matplotlib

You can install them using the following command:

```bash
pip install pandas numpy scipy seaborn scikit-learn matplotlib
```

## How to Run

1. Clone the repository.
2. Place the `train.csv`, `test.csv`, and `sample_submission.csv` files in the working directory.
3. Run the script to execute the feature engineering, correlation analysis, and visualization steps.

```bash
python logistic_regression_correlation.py
```

## Conclusion

This project applies multiple correlation techniques to evaluate relationships between customer features and their likelihood to churn. These insights can guide feature selection and improve the performance of a logistic regression model.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**MIT License**

```
MIT License

Copyright (c) [2024] [Eshonqulov Haqnazar]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

This version includes the MIT License section. You can modify the year and other details if needed. Let me know if you'd like any further changes!
