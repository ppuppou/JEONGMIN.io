import numpy as np
import pandas as pd

# 데이터 불러오기
df = pd.read_csv('../data/haetae.csv')
df.info()
df['label'].value_counts(normalize=True)
# time 컬럼을 datetime으로 변환
df['time'] = pd.to_datetime(df['time'])
# 시간대(hour) 추출
df['hour'] = df['time'].dt.hour
# 요일 추출 (0=월요일, 6=일요일)
df['weekday'] = df['time'].dt.weekday

df.loc[df.Oil_temperature != df.Fryer_temperature, :]

del df['Fryer_temperature']
# 요일을 한글/영문 이름으로 뽑고 싶다면:
#df['weekday_name'] = df['time'].dt.day_name()  # 영어
# df['weekday_name'] = df['time'].dt.day_name(locale='ko_KR')  # 한글 (locale 설정 필요)
print(df[['time', 'hour', 'weekday']].head())

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, 
                  test_size = 0.2,
                  random_state = 0,
                  shuffle = True,
                  stratify = df['label']
                  )


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# 시간 컬럼을 datetime으로 변환
df['time'] = pd.to_datetime(df['time'])

plt.figure(figsize=(15,6))

# 온도 시계열
ax1 = plt.gca()
ax1.plot(df['time'], df['Max_Bake'], label='Max_Bake', color='red')
ax1.plot(df['time'], df['Min_Bake'], label='Min_Bake', color='orange')
ax1.plot(df['time'], df['Oil_temperature'], label='Oil_temperature', color='blue')
ax1.scatter(df['time'][df['label']==1], df['Oil_temperature'][df['label']==1], 
            color='black', s=20, label='label=1', marker='x')

ax1.set_xlabel('Day')
ax1.set_ylabel('Temperature (°C)')
ax1.grid(True)

# X축 날짜만 표시
ax1.xaxis.set_major_locator(mdates.DayLocator())  # 모든 날짜 표시
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d'))  # 일(day)만 표시

# Paddle_Speed 시계열 (보조 Y축)
ax1.plot(df['time'], df['paddle_speed'], label='paddle_speed', color='green')
ax1.plot(df['time'], df['Conv_speed'], label='conv_speed', color='pink')

plt.legend()
plt.title('Time Series of Max_Bake, Min_Bake, Oil_temperature, and Paddle_Speed with Labels')
plt.tight_layout()
plt.show()

df2425 = df.loc[(df['time'].dt.day == 24)|(df['time'].dt.day == 25),:]

df.Conv_speed.min()




import pandas as pd

raw = pd.read_csv("../data/haetae.csv")
df = raw.copy()

### 날짜 추출
df['label'].value_counts(normalize=True)
# time 컬럼을 datetime으로 변환
df['time'] = pd.to_datetime(df['time'])
# 시간대(hour) 추출
df['tt'] = df['time'].dt.time
df['hour'] = df['time'].dt.hour
# 요일 추출 (0=월요일, 6=일요일)
df['weekday'] = df['time'].dt.weekday

# 요일을 한글/영문 이름으로 뽑고 싶다면:
#df['weekday_name'] = df['time'].dt.day_name()  # 영어
# train_df['weekday_name'] = train_df['time'].dt.day_name(locale='ko_KR')  # 한글 (locale 설정 필요)

print(df[['time', 'hour', 'weekday']].head())
del df['Fryer_temperature']
df['Mean_Bake'] = (df['Max_Bake'] + df['Min_Bake'])/2
df["Bake_diff"] = df["Max_Bake"] - df['Min_Bake']
df.info()
df1 = df.drop(columns=["Max_Bake", "Min_Bake"])

df1.describe()
### 테스트 및 트레인 

train, test = train_test_split(df1, 
                  test_size = 0.2,
                   random_state = 0,
                   shuffle = True,
                   stratify = df1['label']
                   )

train.info()
train_sum = train[['Oil_temperature', 'paddle_speed', 'Conv_speed', 'Mean_Bake', "Bake_diff"]]

train_sum.describe()

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# 1) 입력/정답 분리
# =========================
features = ["Oil_temperature", "Conv_speed",
            "Mean_Bake", "Bake_diff", "weekday", "hour"]
target   = "label"

X_train = train[features].copy()
y_train = train[target]

X_test = test[features].copy()
y_test = test[target]


# =========================
# 3) RF + 교차검증
# =========================
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
RandomForestClassifier().get_params()
rf_grid = {
    "n_estimators": [200, 300, 500],               # 트리 개수, 너무 많으면 시간 증가
    "min_samples_leaf": [1, 2, 4, 8],             # 리프 노드 최소 샘플
    "max_features": [0.3, 0.5, 0.7, 'sqrt'],     # 각 분할 시 고려할 feature 비율
    "min_samples_split": [2, 5, 10, 20],          # 내부 노드 분할 최소 샘플
    "max_leaf_nodes": [None, 64, 128, 256, 512],  # 최대 리프 노드 수 제한
    "class_weight" : ['balanced']
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)
grid = GridSearchCV(
    estimator=rf,
    param_grid=rf_grid,
    scoring="accuracy", # 불량(1) 검출 중요하면 f1/recall 권장, 정확도면 'accuracy'
    cv=cv,
    n_jobs=-1,
    verbose=0
)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("Best params:", grid.best_params_)
print(grid.best_score_)

# =========================
# 4) 테스트 성능 확인
# =========================
y_pred = best_model.predict(X_test)
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

# =========================
# 5) 트리 시각화
# =========================
plt.figure(figsize=(20, 12))
plot_tree(
    best_model,
    feature_names=features,
    class_names=["Good(0)", "Defect(1)"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Best Decision Tree (GridSearchCV)")
plt.show()

# 변수 중요도 도출

import pandas as pd
import matplotlib.pyplot as plt

# best_model 은 GridSearchCV에서 얻은 최적 트리
importances = best_model.feature_importances_

# DataFrame으로 정리
feat_imp = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print(feat_imp)

# 막대그래프로 시각화
plt.figure(figsize=(8,5))
plt.barh(feat_imp["Feature"], feat_imp["Importance"], color="skyblue")
plt.xlabel("Importance")
plt.title("Feature Importances (Decision Tree)")
plt.gca().invert_yaxis()  # 중요도 높은 것이 위로 오게
plt.show()