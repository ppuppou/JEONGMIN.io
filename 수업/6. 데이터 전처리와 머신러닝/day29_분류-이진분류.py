import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y
df.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='target'), 
                                                    df['target'], 
                                                    test_size=0.3, 
                                                    random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=10000, random_state = 0)
model.fit(X_train, y_train)

y_prob_org = model.predict_proba(X_test)
print(pd.DataFrame(y_prob_org[:4].round(3)))
y_pred = model.predict(X_test)



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred);
isp = ConfusionMatrixDisplay(confusion_matrix=cm);
isp.plot(cmap=plt.cm.Blues);
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

precision2 = precision_score(y_test, y_pred, pos_label = 1)
recall2 = recall_score(y_test, y_pred, pos_label = 1)
f12 = f1_score(y_test, y_pred, pos_label = 1)
print(f"Precision: {precision2:.2f}")