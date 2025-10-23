import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

print("--- Isotonic 보정 모델 생성 시작 ---")
print("shared.py에서 필요 모듈 로드 중...")

try:
    # 1. shared.py에서 '진짜' 공유 자원만 로드합니다.
    from shared import (
        train_df, test_df, predict_anomaly,
        anomaly_transform, ANOM_FEATURE_ORDER, feature_cols,
        ANOM_PREPROCESSOR, get_preproc_feature_names_out
    )
    print("✅ shared.py 모듈 로드 성공")
except ImportError as e:
    print(f"❌ shared.py 로드 실패: {e}")
    exit()
except Exception as e:
    print(f"❌ shared.py 실행 중 오류 (데이터 로드 등): {e}")
    exit()

print("app.py의 보정 로직을 복제하여 실행합니다...")

try:
    # 2. app.py의 헬퍼 변수 및 함수 정의를 '복사' (app.py 원본 182-227행)
    
    # 🔴 [복사 1] INPUT_FEATURES (app.py 182행)
    INPUT_FEATURES = ANOM_FEATURE_ORDER if len(ANOM_FEATURE_ORDER) else feature_cols

    # 🔴 [복사 2] PREPROC_FEATURES (app.py 185-188행)
    try:
        PREPROC_FEATURES = list(get_preproc_feature_names_out())
    except Exception:
        PREPROC_FEATURES = [f"feat_{i}" for i in range(anomaly_transform(train_df[INPUT_FEATURES]).shape[1])]

    # 🔴 [복사 3] _base_from_preproc_name (app.py 190-194행)
    def _base_from_preproc_name(name: str) -> str:
        n = str(name).split("__")[-1]
        if "=" in n:
            return n.split("=")[0]
        cand = n.rsplit("_", 1)[0]
        return cand if cand in INPUT_FEATURES else n

    # 🔴 [복사 4] _GROUP_IDX, _PREPROC_BASES (app.py 195-201행)
    _GROUP_IDX = defaultdict(list)
    _PREPROC_BASES = []
    for j, pname in enumerate(PREPROC_FEATURES):
        base = _base_from_preproc_name(pname)
        if base == pname and pname.startswith("feat_"):
            if j < len(INPUT_FEATURES):
                base = INPUT_FEATURES[j]
        _PREPROC_BASES.append(base)
        _GROUP_IDX[base].append(j)

    # 🔴 [복사 5] _NUM_IMPUTE_STATS (app.py 200-215행)
    _NUM_IMPUTE_STATS = {}
    try:
        if hasattr(ANOM_PREPROCESSOR, "transformers_"):
            for _, trans, cols in ANOM_PREPROCESSOR.transformers_:
                if hasattr(trans, "named_steps") and "imputer" in trans.named_steps:
                    imp = trans.named_steps["imputer"]
                    if isinstance(imp, SimpleImputer) and hasattr(imp, "statistics_"):
                        for c, v in zip(cols, imp.statistics_):
                            _NUM_IMPUTE_STATS[c] = v
                elif isinstance(trans, SimpleImputer) and hasattr(trans, "statistics_"):
                    for c, v in zip(cols, trans.statistics_):
                        _NUM_IMPUTE_STATS[c] = v
    except Exception:
        pass

    # 🔴 [복사 6] _imputed_raw (app.py 217-221행)
    def _imputed_raw(row: pd.Series, base: str):
        v = row.get(base, np.nan)
        if pd.isna(v) and base in _NUM_IMPUTE_STATS:
            return _NUM_IMPUTE_STATS[base]
        return v

    # 🔴 [복사 7] 참조 행렬 및 NN 모델 (app.py 224-227행)
    _Z_ref = anomaly_transform(train_df[INPUT_FEATURES])
    CAL_K = 80    # app.py의 176행
    _nn_model = NearestNeighbors(n_neighbors=min(CAL_K, len(_Z_ref)), metric="euclidean")
    _nn_model.fit(_Z_ref)
    
    _RIDGE = 1e-6 # app.py의 178행

    # 🔴 [복사 8] _xi_from_row (app.py 229-232행)
    def _xi_from_row(row: pd.Series) -> np.ndarray:
        X1 = pd.DataFrame([row]).reindex(columns=INPUT_FEATURES)
        Zi = anomaly_transform(X1)[0]
        return Zi

    # 🔴 [복사 9] _local_mahalanobis_parts (app.py 234-243행)
    def _local_mahalanobis_parts(xi: np.ndarray):
        _, idx = _nn_model.kneighbors([xi], n_neighbors=min(CAL_K, len(_Z_ref)))
        neigh = _Z_ref[idx[0]]
        mu = neigh.mean(axis=0)
        Sigma = np.cov(neigh, rowvar=False)
        Sigma = Sigma + np.eye(Sigma.shape[0]) * _RIDGE
        invS = np.linalg.pinv(Sigma)
        diff = xi - mu
        parts = diff * (invS @ diff)
        D2 = float(diff @ invS @ diff)
        return parts, D2, mu

    # 🔴 [복사 10] _effective_anom_proba (app.py 245-253행, 이전 단계에서 추가함)
    def _effective_anom_proba(ana_res: pd.DataFrame):
        aprob_raw = float(ana_res.get("anom_proba", [np.nan])[0]) if "anom_proba" in ana_res.columns else np.nan
        strength = float(ana_res.get("prob", [np.nan])[0]) if "prob" in ana_res.columns else np.nan
        if (not np.isfinite(aprob_raw)) or (aprob_raw <= 0):
            aprob_eff = 1.0 - (strength if np.isfinite(strength) else 0.0)
        else:
            aprob_eff = aprob_raw
        return aprob_eff, strength

    # 3. app.py의 _collect_calibration_pairs 함수 '복사' (app.py 원본 230-252행)
    _MAX_CAL_SAMPLES = 500 # app.py의 177행
    
    def _collect_calibration_pairs(max_samples=_MAX_CAL_SAMPLES):
        print(f"\n보정용 샘플 수집 시작 (max_samples={max_samples})...")
        if len(test_df) == 0:
            return None, None
        idxs_eq = np.linspace(0, len(test_df)-1, num=min(max_samples, len(test_df)), dtype=int)
        idxs_rd = np.random.RandomState(42).choice(len(test_df), size=min(max_samples//2, len(test_df)), replace=False)
        idxs = np.unique(np.concatenate([idxs_eq, idxs_rd]))
        
        d2_list, p_list = [], []
        
        processed_count = 0
        for i in idxs:
            r = test_df.iloc[i]
            try:
                xi = _xi_from_row(r)
                parts, D2, _ = _local_mahalanobis_parts(xi)
                # 🔴 이 부분이 바로 OOM을 유발하는 무거운 계산입니다.
                ana_res = predict_anomaly(pd.DataFrame([r])) 
                aprob_eff, _ = _effective_anom_proba(ana_res)
                
                if np.isfinite(D2) and np.isfinite(aprob_eff):
                    d2_list.append(D2)
                    p_list.append(aprob_eff)
                
                processed_count += 1
                if processed_count % 50 == 0:
                    print(f"  ... {processed_count}/{len(idxs)} 샘플 처리 중 ...")

            except Exception as e:
                # ℹ️ Fallback 로그 등이 여기서 출력될 수 있습니다 (정상)
                print(f"  ⚠️ 샘플 {i} 처리 중 경고 (무시): {e}")
                continue
        
        if len(d2_list) < 20:
            print("❌ 유효한 (D2, P) 쌍이 20개 미만이라 보정 모델 생성 실패.")
            return None, None
        
        print(f"✅ 총 {len(d2_list)}개의 유효한 보정 샘플 수집 완료.")
        d2 = np.asarray(d2_list)
        p = np.asarray(p_list)
        order = np.argsort(d2)
        return d2[order], p[order]

    # 4. 계산 실행 및 저장
    _cal_d2, _cal_p = _collect_calibration_pairs()
    
    if _cal_d2 is not None:
        print("\nIsotonicRegression 모델 피팅 중...")
        _iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        _iso.fit(_cal_d2, _cal_p)
        _cal_bounds = (float(np.min(_cal_d2)), float(np.max(_cal_d2)))
        print("✅ 모델 피팅 완료.")

        # 5. 결과물 저장
        app_dir = Path(__file__).parent
        output_path = app_dir / "data/calibration_model.pkl"
        joblib.dump({
            "iso_model": _iso,
            "cal_bounds": _cal_bounds
        }, output_path)
        
        print(f"\n🎉 성공: Isotonic 보정 모델이 {output_path} 에 저장되었습니다.")
        print("--- 이제 app.py를 수정하고 앱을 배포하세요. ---")
    else:
        print("\n❌ 실패: 보정 모델을 생성하지 못했습니다.")

except Exception as e:
    print(f"\n❌ 스크립트 실행 중 치명적 오류 발생: {e}")