# Random-forest-documentation



# 🫀 Heart Failure Prediction – Data Analysis & Machine Learning

This project explores the **Heart Failure Prediction** dataset, performing data cleaning, transformation, visualization, and training machine learning models (SVM, Decision Tree, Random Forest) to predict heart disease. It uses pandas, scikit-learn, seaborn, and matplotlib.

---

## 📦 Dataset

- **File**: `heart.csv`
- **Source**: `/kaggle/input/heart-failure-prediction/heart.csv`
- **Target Variable**: `HeartDisease` (1 = has disease, 0 = healthy)

---

## 📊 Data Exploration

```python
dataframe.describe()
dataframe.info()
```

### Simulate Missing Data:
```python
dataframe.loc[5, 'Cholesterol'] = np.nan
dataframe.isnull().sum()
dataframe.dropna(inplace=True)
```

---

## 📉 Data Cleaning

Remove rows where `Cholesterol` is zero:

```python
for x in dataframe.index:
    if dataframe.loc[x, "Cholesterol"] == 0:
        dataframe.drop(x, inplace=True)
```

---

## 📈 Data Visualization

```python
import matplotlib.pyplot as plt
dataframe.plot(kind='scatter', x='Cholesterol', y='RestingBP', color='red')
```

### Correlation Heatmap:

```python
import seaborn as sns
corr_matrix = dataframe.corr()
sns.heatmap(corr_matrix, cmap='Purples', annot=True)
plt.show()
```

---

## 🔁 Categorical Data Encoding

### One-Hot Encoding:

```python
temp_df = pd.get_dummies(dataframe["Sex"], dtype=int)
temp_df.drop(columns=["F"], inplace=True)
dataframe.drop(columns=["Sex"], inplace=True)
dataframe = pd.concat([dataframe, temp_df], axis=1)
dataframe.rename(columns={"M": "Sex"}, inplace=True)
```

### Label Encoding:

```python
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()
dataframe['ChestPainTypeLE'] = LE.fit_transform(dataframe['ChestPainType'])
dataframe['RestingECGLE'] = LE.fit_transform(dataframe['RestingECG'])
dataframe['ExerciseAnginaLE'] = LE.fit_transform(dataframe['ExerciseAngina'])
dataframe['ST_SlopeLE'] = LE.fit_transform(dataframe['ST_Slope'])

dataframe.drop(columns=["ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"], inplace=True)
```

---

## 🧪 Train-Test Split

```python
from sklearn.model_selection import train_test_split

X = dataframe.drop(['HeartDisease'], axis=1)
y = dataframe['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

---

## 🤖 Model Training

### ✅ Support Vector Machine (SVM)

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svm = SVC(random_state=42)
svm.fit(X_train, y_train)
y_predicted_svm = svm.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, y_predicted_svm))
```

### 🌳 Decision Tree

```python
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_predicted_DT = clf.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_predicted_DT))
```

### 🌲 Random Forest (Method 1)

```python
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
y_predicted_RF = classifier.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_predicted_RF))
```

### 🌲 Random Forest (Method 2 – Alternate Instance)

```python
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=100, random_state=42)
RF.fit(X_train, y_train)
y_predicted_RF = RF.predict(X_test)

print("Model Accuracy :{0:0.4f}".format(accuracy_score(y_test, y_predicted_RF)))
```

> ✅ Both RandomForestClassifier instances perform the same operation but can be used for testing different hyperparameters or experiments.

---

## 📊 Model Evaluation (SVM Example)

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_predicted_svm)
precision = precision_score(y_test, y_predicted_svm, average='macro')
recall = recall_score(y_test, y_predicted_svm, average='macro')
f1 = f1_score(y_test, y_predicted_svm, average='macro')

print("------ Evaluation: SVM ------")
print(f"Accuracy : {accuracy}")
print(f"Precision: {precision}")
print(f"Recall   : {recall}")
print(f"F1 Score : {f1}")
print("-----------------------------")
```

---

## 📌 Conclusion

- Proper data cleaning (handling nulls, invalid values) improves accuracy.
- Label encoding and one-hot encoding are vital for categorical variables.
- SVM, Decision Tree, and Random Forest can all provide good baseline classification performance.

---

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```


---

## ⚖️ Handling Class Imbalance with SMOTE

### 🔍 Step 1: Check for Class Imbalance

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Count values in target variable
class_counts = dataframe['HeartDisease'].value_counts()
print(class_counts)

# Visualize the class imbalance
sns.countplot(x='HeartDisease', data=dataframe)
plt.title('Class Distribution')
plt.show()
```

### 🔁 Step 2: Apply SMOTE

```python
from imblearn.over_sampling import SMOTE
from collections import Counter

# Prepare features and labels
X = dataframe.drop(['HeartDisease'], axis=1)
y = dataframe['HeartDisease']

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check the new class distribution
print("After SMOTE:", Counter(y_resampled))
```

### 📊 Optional: Visualize Balanced Classes

```python
sns.countplot(x=y_resampled)
plt.title('Class Distribution After SMOTE')
plt.show()
```

> ✅ After SMOTE, proceed with train-test split using `X_resampled` and `y_resampled`.
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)
```


---

## 🏷️ Label Encoding (All At Once)

Use `LabelEncoder` to convert multiple categorical columns into numerical values:

```python
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

# Apply label encoding to all categorical columns
categorical_cols = ['ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
for col in categorical_cols:
    dataframe[col + '_LE'] = LE.fit_transform(dataframe[col])

# Drop original categorical columns
dataframe.drop(columns=categorical_cols, inplace=True)
```

---

## 🔍 Hyperparameter Tuning using Grid Search

Use `GridSearchCV` to find the best parameters for models like SVM or Random Forest:

### 🧪 Grid Search on SVM

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

grid_svm = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_svm.fit(X_train, y_train)

# Best parameters and accuracy
print("Best Parameters (SVM):", grid_svm.best_params_)
print("Best Accuracy (SVM):", grid_svm.best_score_)

# Use best estimator to predict
best_svm = grid_svm.best_estimator_
y_pred_grid_svm = best_svm.predict(X_test)
```

### 🌲 Grid Search on Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train, y_train)

# Best parameters and accuracy
print("Best Parameters (RF):", grid_rf.best_params_)
print("Best Accuracy (RF):", grid_rf.best_score_)

# Use best estimator to predict
best_rf = grid_rf.best_estimator_
y_pred_grid_rf = best_rf.predict(X_test)
```

> 🔧 Grid search may take time but helps find optimal hyperparameters for better performance.

