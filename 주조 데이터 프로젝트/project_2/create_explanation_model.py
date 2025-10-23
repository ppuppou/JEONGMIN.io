import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings("ignore")

print("--- 이상치 설명 모델 생성 시작 ---")
print("shared.py에서 필요 모듈 로드 중...")

try:
    # 1. shared.py에서 '진짜' 공유 자원만 로드합니다.
    from shared import (
        train_df, anomaly_transform, 
        ANOM_FEATURE_ORDER, feature_cols
    )
    print("✅ shared.py 모듈 로드 성공 (train_df 등)")
except ImportError as e:
    print(f"❌ shared.py 로드 실패: {e}")
    exit()
except Exception as e:
    print(f"❌ shared.py 실행 중 오류 (데이터 로드 등): {e}")
    exit()

print("app.py의 설명 모델(NN) 로직을 복제하여 실행합니다...")

try:
    # 2. app.py의 헬퍼 변수 및 함수 정의를 '복사' (app.py 원본 176, 182행)
    CAL_K = 80
    INPUT_FEATURES = ANOM_FEATURE_ORDER if len(ANOM_FEATURE_ORDER) else feature_cols

    if train_df.empty:
        raise ValueError("train_df가 비어있습니다.")
    if not INPUT_FEATURES:
         raise ValueError("INPUT_FEATURES가 비어있습니다.")

    # 3. 🔴 OOM을 유발하는 무거운 계산 실행
    print("1/3: _Z_ref 계산 중 (anomaly_transform)... (시간이 걸릴 수 있습니다)")
    _Z_ref = anomaly_transform(train_df[INPUT_FEATURES])
    print(f"✅ _Z_ref 생성 완료 (Shape: {_Z_ref.shape})")

    print("2/3: NearestNeighbors 모델 피팅 중...")
    _nn_model = NearestNeighbors(n_neighbors=min(CAL_K, len(_Z_ref)), metric="euclidean")
    _nn_model.fit(_Z_ref)
    print("✅ NearestNeighbors 모델 피팅 완료")

    # 4. 결과물 저장
    app_dir = Path(__file__).parent
    output_path = app_dir / "data/explanation_model.pkl"
    joblib.dump({
        "Z_ref": _Z_ref,
        "nn_model": _nn_model
    }, output_path)
    
    print(f"\n🎉 성공: 이상치 설명 모델이 {output_path} 에 저장되었습니다.")
    print("--- 이제 app.py를 수정하고 앱을 배포하세요. ---")

except Exception as e:
    print(f"\n❌ 스크립트 실행 중 치명적 오류 발생: {e}")