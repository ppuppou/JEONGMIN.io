import pandas as pd 
import numpy as np 
from sklearn import set_config
set_config(display="diagram")
#pd.set_option('display.max_columns', None) # 모든 칼럼이 출력되게 조절

from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data
y = iris.target
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = LogisticRegression()
model.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average="macro")
recall_macro = recall_score(y_test, y_pred, average="micro")
f1_macro = f1_score(y_test, y_pred, average="weighted")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision_macro:.2f}")
print(f"Recall: {recall_macro:.2f}")
print(f"F1 Score: {f1_macro:.2f}")

from sklearn.metrics import roc_auc_score
y_prob = model.predict_proba(X_test)
auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
print(f"AUC Score (One-vs-Rest, Macro Average): {auc:.4f}")