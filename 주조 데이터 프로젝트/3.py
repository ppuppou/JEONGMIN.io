import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score

# 1. 모델이 포함된 딕셔너리 불러오기
try:
    loaded_object = joblib.load('final_model.pkl')
    
    # 👇👇👇 이 부분이 핵심! 딕셔너리에서 실제 모델을 추출합니다. 👇👇👇
    if isinstance(loaded_object, dict):
        model = loaded_object.get("model")
        print("✅ 딕셔너리에서 모델을 성공적으로 추출했습니다.")
    else:
        model = loaded_object # 만약 딕셔너리가 아니면, 그 자체가 모델
        print("✅ 모델을 성공적으로 불러왔습니다.")

    if not hasattr(model, 'predict'):
        print("🚨 오류: 추출된 객체에 predict 기능이 없습니다. .pkl 파일의 구조를 다시 확인해주세요.")
        exit()

except FileNotFoundError:
    print("🚨 'final_model.pkl' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    exit()
except Exception as e:
    print(f"🚨 모델 로드 중 오류 발생: {e}")
    exit()

# 2. 테스트 데이터 및 실제 정답 불러오기
try:
    X_test = pd.read_csv('./test.csv')
    y_true = pd.read_csv('./.csv').squeeze()
    print("✅ 테스트 데이터와 정답 데이터를 성공적으로 불러왔습니다.")
except FileNotFoundError as e:
    print(f"🚨 파일을 찾을 수 없습니다: {e.filename}. 파일 경로를 확인해주세요.")
    exit()

# 3. 모델을 사용하여 예측 수행
predictions = model.predict(X_test)
print("\n✅ 예측을 완료했습니다.")

# 결과 확인
results_df = pd.DataFrame({'Actual': y_true, 'Predicted': predictions})
print("\n[실제 값과 예측 값 비교 (상위 5개)]")
print(results_df.head())

# 4. 성능 평가
print("\n--- 모델 성능 평가 ---")

if y_true.dtype == 'object' or y_true.nunique() < 20:
    print("📈 **분류 모델 성능**")
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, average='macro')
    recall = recall_score(y_true, predictions, average='macro')
    f1 = f1_score(y_true, predictions, average='macro')

    print(f"  - 정확도 (Accuracy): {accuracy:.4f}")
    print(f"  - 정밀도 (Precision): {precision:.4f}")
    print(f"  - 재현율 (Recall): {recall:.4f}")
    print(f"  - F1 스코어 (F1 Score): {f1:.4f}")
else:
    print("📈 **회귀 모델 성능**")
    mse = mean_squared_error(y_true, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, predictions)

    print(f"  - 평균 제곱 오차 (MSE): {mse:.4f}")
    print(f"  - 평균 제곱근 오차 (RMSE): {rmse:.4f}")
    print(f"  - R-제곱 (R²): {r2:.4f}")