### **Title: Bank Customer Churn Prediction**

---

## **Overview**

This project focuses on predicting customer churn in a bank using Support Vector Machine (SVM) as the classification algorithm. We work with a dataset that contains customer details, and the objective is to determine which customers are likely to leave the bank based on various factors like credit score, balance, geography, etc.

## **Project Structure**

- **Data Loading and Preprocessing**
- **Handling Imbalanced Data**
- **Feature Scaling**
- **Model Training with SVM**
- **Hyperparameter Tuning**
- **Evaluation and Comparison**

## **Dataset**

The dataset used in this project is publicly available on GitHub, provided by the YBI Foundation. The dataset consists of 10,000 rows and 13 columns, including customer details such as Credit Score, Geography, Age, Balance, and whether or not they churned.

### **Data Loading**

```python
import pandas as pd

df = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Bank%20Churn%20Modelling.csv')
print(df.head())
print(df.info())
print(df.describe())
```

### **Define Labels and Features**

```python
X = df.drop(columns=['Churn'])
y = df['Churn']
```

### **Handling Imbalanced Data**

This dataset is imbalanced with the majority of the customers not churning. To handle this, we will use Random Under Sampling and Random Over Sampling techniques.

```python
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

rus = RandomUnderSampler(random_state=42)
ros = RandomOverSampler(random_state=42)

X_rus, y_rus = rus.fit_resample(X, y)
X_ros, y_ros = ros.fit_resample(X, y)
```

### **Encoding Categorical Features**

```python
X_encoded = pd.get_dummies(X, drop_first=True)
X_rus_encoded = pd.get_dummies(X_rus, drop_first=True)
X_ros_encoded = pd.get_dummies(X_ros, drop_first=True)

X_encoded, X_rus_encoded, X_ros_encoded = X_encoded.align(X_rus_encoded, join='left', axis=1, fill_value=0)
X_encoded, X_ros_encoded = X_encoded.align(X_ros_encoded, join='left', axis=1, fill_value=0)
```

### **Train-Test Split**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=25)
X_train_rus, X_test_rus, y_train_rus, y_test_rus = train_test_split(X_rus_encoded, y_rus, test_size=0.3, random_state=25)
X_train_ros, X_test_ros, y_train_ros, y_test_ros = train_test_split(X_ros_encoded, y_ros, test_size=0.3, random_state=25)
```

### **Feature Scaling**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_rus_scaled = scaler.fit_transform(X_train_rus)
X_test_rus_scaled = scaler.transform(X_test_rus)

X_train_ros_scaled = scaler.fit_transform(X_train_ros)
X_test_ros_scaled = scaler.transform(X_test_ros)
```

### **Support Vector Machine Classifier with Raw Data**

```python
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

svm_classifier = SVC()
svm_classifier.fit(X_train_scaled, y_train)
y_pred = svm_classifier.predict(X_test_scaled)

print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))
```

## **Conclusion**

This project demonstrates how to handle an imbalanced dataset, implement and tune an SVM model, and compare its performance using different sampling techniques. The tuned model can now be used to predict whether a customer is likely to churn, enabling the bank to take proactive measures.

--- 

## **Requirements**

- Python 3.7+
- Pandas
- Scikit-learn
- Imbalanced-learn

## **How to Run**

1. Clone the repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the Jupyter Notebook or Python script to execute the code.

## **References**

- [Scikit-learn Documentation](https://scikit-learn.org/stable/user_guide.html)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
