import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
# 1. 불균형 데이터 생성 (정상: 0, 이상: 1)
X, y = make_classification(n_samples=1000, n_features=5, 
                           n_informative=2, n_redundant=0, 
                           weights=[0.95, 0.05], # 95% : 5%
                           random_state=42)
print("클래스 분포:", pd.Series(y).value_counts())
# 2. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# 3. 로지스틱 회귀 모형 적합
model = LogisticRegression()
model.fit(X_train, y_train)
# 4. 예측
y_pred = model.predict(X_test)
# 5. 평가
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("혼동 행렬:\n", cm)
print(f"정확도(Accuracy): {acc:.3f}")
print(f"민감도(Recall): {recall:.3f}")




from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
ros = RandomOverSampler(random_state=42)
X_res_over, y_res_over = ros.fit_resample(X_train, y_train)
model_over = LogisticRegression()
model_over.fit(X_res_over, y_res_over)
y_pred_over = model_over.predict(X_test)
print("\n[RandomOverSampler 결과]")
print("혼동 행렬:\n", confusion_matrix(y_test, y_pred_over))
print("정확도:", accuracy_score(y_test, y_pred_over))
print("민감도:", recall_score(y_test, y_pred_over))
# ----------------------------
# [언더샘플링] RandomUnderSampler 적용
# ----------------------------
under = RandomUnderSampler(random_state=42)
X_res_under, y_res_under = under.fit_resample(X_train, y_train)
model_under = LogisticRegression()
model_under.fit(X_res_under, y_res_under)
y_pred_under = model_under.predict(X_test)
print("\n[RandomUnderSampler 결과]")
print("혼동 행렬:\n", confusion_matrix(y_test, y_pred_under))
print("정확도:", accuracy_score(y_test, y_pred_under))
print("민감도:", recall_score(y_test, y_pred_under))