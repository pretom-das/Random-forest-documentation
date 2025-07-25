# Random-forest-documentation


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
