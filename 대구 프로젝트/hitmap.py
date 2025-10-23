# 상관관계
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import platform

# 1. 한글 폰트 설정 (OS에 맞게 자동 설정)
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
else: # Linux (Colab 등)
    # 나눔고딕 폰트 설치가 필요할 수 있습니다.
    # !sudo apt-get install -y -qq fonts-nanum
    plt.rc('font', family='NanumGothic')

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False


# 2. 데이터 불러오기 및 전처리
# 사용자가 업로드한 파일을 불러옵니다.
df = pd.read_csv('df1_main_data.csv')
print("✓ 'df1_main_data.csv' 파일을 성공적으로 불러왔습니다.")


if df is not None:
    # '소계' 행 제거
    df = df[df['행정동'] != '소계'].copy()

    # 분석에 사용할 수치형 데이터만 선택
    df_numeric = df.select_dtypes(include=np.number)

    # 3. 상관관계 분석
    # 상관계수 행렬 계산
    corr_matrix = df_numeric.corr()
    # 타겟 변수 설정
    target_var = '범죄발생수(면적기준)'

    if target_var in corr_matrix:
        # 4. 필터링 및 정렬
        # 타겟 변수와의 상관계수 절대값 계산
        target_corr = corr_matrix[target_var].abs()

        # 상관계수가 높은 순으로 내림차순 정렬
        sorted_vars = target_corr.sort_values(ascending=False)
        
        # 정렬된 변수 목록 (타겟 변수가 맨 앞에 오도록)
        final_vars_list = sorted_vars.index.tolist()
        
        print(f"\n[분석 결과] '{target_var}'와 상관관계가 0.15 이상인 변수 (상관도 높은 순):")
        for i, var in enumerate(final_vars_list):
            # --- ✨ 수정된 부분: np.mean()을 사용하여 Series가 반환되어도 오류가 나지 않도록 처리 ---
            print(f"{i+1:2d}. {var} (상관계수: {np.mean(corr_matrix.loc[target_var, var]):.3f})")

        # 5. 시각화
        # 최종 변수들로 상관계수 행렬 재구성
        final_corr_matrix = corr_matrix.loc[final_vars_list, final_vars_list]

        # 히트맵 생성
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            final_corr_matrix,
            annot=True,          # 각 셀에 값 표시
            fmt='.2f',           # 값의 소수점 자리수
            cmap='coolwarm',     # 색상 맵
            linewidths=.5,
            cbar_kws={'label': '상관계수'}
        )
        plt.title(f"'{target_var}' 중심의 주요 변수 상관관계 히트맵", fontsize=18, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    else:
        print(f"✗ 오류: '{target_var}' 컬럼을 데이터에서 찾을 수 없습니다.")



# p
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import platform
from scipy.stats import pearsonr

# 1. 한글 폰트 설정 (OS에 맞게 자동 설정)
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
else: # Linux (Colab 등)
    plt.rc('font', family='NanumGothic')

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False


# 2. 데이터 불러오기 및 전처리

# 사용자가 제공한 'df1_main_data_nontotal.csv' 파일로 수정
df = pd.read_csv('df1_main_data.csv')

if df is not None:
    # '소계' 행 제거
    df = df[df['행정동'] != '소계'].copy()
    
    # 수치형 데이터만 선택
    df_numeric = df.select_dtypes(include=np.number)

    # 3. 분석 대상 변수 선정 (이전과 동일한 기준 적용)
    target_var = '범죄발생수(면적기준)'
    if target_var in df_numeric.columns:
        corr_matrix = df_numeric.corr()
        target_corr = corr_matrix[target_var].abs()
        significant_vars = target_corr[target_corr > 0.15]
        sorted_vars = significant_vars.sort_values(ascending=False)
        final_vars_list = sorted_vars.index.tolist()
        
        # --- ✨ 수정된 부분: 결측치를 제거하는 대신 0으로 채웁니다 ---
        # p-value 계산을 위한 최종 데이터프레임에서 결측치를 0으로 대체
        df_final = df_numeric[final_vars_list].fillna(0)

        # 4. P-value 행렬 계산
        print("\nP-value 행렬을 계산하는 중입니다...")
        # 빈 데이터프레임을 생성하여 p-value 값을 저장
        p_value_matrix = pd.DataFrame(index=df_final.columns, columns=df_final.columns, dtype=float)

        # 모든 변수 쌍에 대해 p-value 계산
        for col1 in df_final.columns:
            for col2 in df_final.columns:
                # 결측치를 0으로 채웠으므로 모든 컬럼의 길이가 동일하여 바로 계산 가능
                _, p_value = pearsonr(df_final[col1], df_final[col2])
                p_value_matrix.loc[col1, col2] = p_value
        
        print("✓ P-value 계산 완료!")

        # 5. P-value 히트맵 시각화
        plt.figure(figsize=(16, 14))
        
        # p-value가 0.05 이하인 유의미한 관계만 강조하기 위한 마스크 생성
        mask = p_value_matrix > 0.05
        
        sns.heatmap(
            p_value_matrix,
            annot=True,          # 각 셀에 p-value 값 표시
            fmt='.3f',           # 값의 소수점 3자리까지 표시
            cmap='viridis_r',    # 색상 맵 (값이 낮을수록 진한 색)
            linewidths=.5,
            mask=mask,           # p-value가 0.05보다 큰 값들은 히트맵에 표시하지 않음
            cbar_kws={'label': 'P-value (유의확률)'}
        )
        # 유의미하지 않은 부분도 연하게 표시하고 싶을 경우
        sns.heatmap(
            p_value_matrix,
            cmap='Greys',
            cbar=False,
            mask=~mask,
            alpha=0.3 # 연한 회색으로 표시
        )

        plt.title('주요 변수 간 관계의 통계적 유의성 (P-value Heatmap)', fontsize=18, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    else:
        print(f"✗ 오류: '{target_var}' 컬럼을 데이터에서 찾을 수 없습니다.")



# pca 분석
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import platform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. 한글 폰트 설정 (OS에 맞게 자동 설정)
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
else: # Linux (Colab 등)
    plt.rc('font', family='NanumGothic')

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 2. 데이터 불러오기 및 전처리
try:
    df = pd.read_csv('df1_main_data.csv')
    print("✓ 'df1_main_data.csv' 파일을 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print("✗ 오류: 'df1_main_data.csv' 파일을 찾을 수 없습니다.")
    df = None

if df is not None:
    # '소계' 행 제거 및 인덱스 리셋
    df = df[df['행정동'] != '소계'].reset_index(drop=True)
    
    # 지역명(구, 행정동) 정보 따로 저장
    location_info = df[['col_1', '행정동']].rename(columns={'col_1': '구'})
    
    # 수치형 데이터만 선택
    df_numeric = df.select_dtypes(include=np.number)
    
    # 결측치를 0으로 대체
    df_numeric = df_numeric.fillna(0)

    # 3. 데이터 표준화 (Standard Scaling)
    # PCA는 변수의 스케일에 민감하므로 표준화가 필수적입니다.
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)
    df_scaled = pd.DataFrame(df_scaled, columns=df_numeric.columns)
    print("\n✓ 데이터 표준화 완료!")

    # 4. PCA(주성분 분석) 수행
    pca = PCA()
    pca_result = pca.fit_transform(df_scaled)
    print("✓ PCA 분석 완료!")

    # 5. 결과 분석 및 시각화
    # 5-1. 스크리 플롯 (Scree Plot) - 각 주성분의 설명력 시각화
    plt.figure(figsize=(12, 6))
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.8, align='center', label='개별 주성분 설명 분산')
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='누적 설명 분산')
    plt.ylabel('설명 분산 비율')
    plt.xlabel('주성분 번호')
    plt.title('주성분 분석 스크리 플롯 (Scree Plot)', fontsize=16)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    print(f"\n첫 2개의 주성분이 전체 분산의 {cumulative_variance[1]*100:.2f}%를 설명합니다.")
    print(f"첫 5개의 주성분이 전체 분산의 {cumulative_variance[4]*100:.2f}%를 설명합니다.")


    # 5-2. 주성분 로딩 히트맵 (Component Loadings Heatmap)
    # 각 주성분이 어떤 변수와 관련이 깊은지 확인
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(len(df_numeric.columns))], index=df_numeric.columns)
    
    # 상위 2개 주성분의 로딩 값 확인
    plt.figure(figsize=(10, 12))
    sns.heatmap(loadings[['PC1', 'PC2']], annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('주요 주성분 로딩 (PC1 & PC2)', fontsize=16)
    plt.tight_layout()
    plt.show()

    # 5-3. 2D 주성분 평면도 (PCA Biplot)
    # 각 행정동을 PC1과 PC2 평면에 시각화
    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(len(df_numeric.columns))])
    pca_df = pd.concat([location_info, pca_df], axis=1)

    plt.figure(figsize=(14, 10))
    sns.scatterplot(x='PC1', y='PC2', data=pca_df, hue='구', s=100, alpha=0.8)
    plt.title('행정동별 주성분 분석 결과 (PC1 vs PC2)', fontsize=16)
    plt.xlabel('PC1 (제1 주성분)')
    plt.ylabel('PC2 (제2 주성분)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='grey', lw=1)
    plt.axvline(0, color='grey', lw=1)
    plt.legend(title='구분')
    
    # 주요 지역 텍스트 라벨링 (일부만)
    for i, txt in enumerate(pca_df['행정동']):
        if abs(pca_df['PC1'][i]) > 5 or abs(pca_df['PC2'][i]) > 5: # 값이 큰 일부 지역만 표시
            plt.text(pca_df['PC1'][i], pca_df['PC2'][i], txt, fontsize=9)

    plt.tight_layout()
    plt.show()