# credit-risk-classification - Logistic Regression Prediction 
## Overview
This project aims to predict credit risk using logistic regression. It involves analyzing loan data to classify loans as either healthy (0) or high-risk (1) based on various features.


## Instructions
The project instructions are divided into several key steps:

### 1. Split the Data into Training and Testing Sets
- Read the `lending_data.csv` dataset from the `Resources` folder into a Pandas DataFrame.
- Create feature (`X`) and label (`y`) datasets where `loan_status` is the target variable.
- Split the data into training and testing sets using `train_test_split`.

``` Python
import pandas as pd
lending_data_df = pd.read_csv("Resources/lending_data.csv")

loan_size	interest_rate	borrower_income	debt_to_income	num_of_accounts	derogatory_marks	total_debt	loan_status
0	10700.0	  7.672	        52800	          0.431818	        5	              1	              22800	        0
1	8400.0	  6.692	        43600	          0.311927	        3	              0	              13600	        0
2	9000.0	  6.963	        46100	          0.349241	        3	              0	              16100	        0
3	10700.0	  7.664	        52700	          0.430740	        5	              1	              22700	        0
4	10800.0	  7.698	        53000	          0.433962	        5	              1	              23000	        0

# Separate the data into labels and features
y = lending_data_df['loan_status']
X = lending_data_df.drop(columns='loan_status')

# Split the data using train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```
### 2. Create a Logistic Regression Model with the Original Data
- Fit a logistic regression model using the training data (`X_train` and `y_train`).
- Predict labels for the testing data (`X_test`) using the trained model.
- Evaluate the model's performance:
- Generate a confusion matrix.
- Print the classification report.
```Python
model = LogisticRegression(solver='lbfgs',random_state=1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
c_matrix = confusion_matrix(y_test, predictions)
print(classification_report(y_test, predictions))

              precision    recall  f1-score   support
           0       1.00      0.99      1.00     18765
           1       0.85      0.91      0.88       619

    accuracy                           0.99     19384
   macro avg       0.92      0.95      0.94     19384
weighted avg       0.99      0.99      0.99     19384
```
  

### 3. Write a Credit Risk Analysis Report
As per the Classification report, it is evident that both models represents higher and reliable predictions.  '0' healthy loan precison to 100% , recall of 99 and f1 score of 100 
whereas '1' - higher risk loan  too has a precision of 85, recall 91, f1-score of 88 and with Overall accuracy of 99% is a great prediction model.


#### Conclusion
In conclusion, this project successfully demonstrates the application of logistic regression for credit risk prediction. The model performs well in identifying healthy loans, with room for improvement in detecting high-risk loans. Future work could explore additional features or alternative modeling techniques to enhance predictive accuracy.


## Tools and Libraries Used
- Jupyter Notebook
- Python
- Libraries:
- Pandas
- Scikit-learn
 

