import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from shared import (
    train_df, hdb_prep, hdb_global, hdb_segments, 
    _get_feature_order, _N_NUM_COLS
)
from hdbscan.prediction import approximate_predict

print("참조 분포 계산 시작...")
app_dir = Path(__file__).parent

try:
    _ANOM_SCORE_REF, _RD2_REF = None, None
    feat_order_ref = _get_feature_order(hdb_prep, hdb_global)
    
    if feat_order_ref and not train_df.empty:
        X_ref_raw = train_df.copy()
        for c in feat_order_ref:
            if c not in X_ref_raw.columns:
                X_ref_raw[c] = 0
        Xp_ref = hdb_prep.transform(X_ref_raw[feat_order_ref])

        if _N_NUM_COLS > 0:
            rd2_ref = np.sum(np.square(Xp_ref[:, :_N_NUM_COLS]), axis=1)
            _RD2_REF = np.sort(rd2_ref[np.isfinite(rd2_ref)])
            print(f"✅ RD2 참조(n={len(_RD2_REF)}) 생성 완료")

        hdb_model_ref = hdb_global.get("hdb")
        if hdb_model_ref is not None:
            pca_ref = hdb_global.get("pca")
            Z_ref = pca_ref.transform(Xp_ref) if pca_ref is not None else Xp_ref
            labels_ref, strengths_ref = approximate_predict(hdb_model_ref, Z_ref)
            scores_ref = np.where(np.asarray(labels_ref) == -1, 1.0, 1.0 - np.asarray(strengths_ref))
            _ANOM_SCORE_REF = np.sort(scores_ref[np.isfinite(scores_ref)])
            print(f"✅ Score 참조(n={len(_ANOM_SCORE_REF)}) 생성 완료")
            
        # 🔴 계산된 결과를 파일로 저장
        output_path = app_dir / "data/hdbscan_reference_stats.pkl"
        joblib.dump({
            "ANOM_SCORE_REF": _ANOM_SCORE_REF,
            "RD2_REF": _RD2_REF,
            "N_NUM_COLS": _N_NUM_COLS
        }, output_path)
        
        print(f"✅ 참조 분포가 {output_path} 에 저장되었습니다.")
        
    else:
        print("⚠️ train_df가 비어있거나 피처 순서를 찾을 수 없습니다.")

except Exception as e:
    print(f"❌ 참조 분포 생성 중 오류 발생: {e}")