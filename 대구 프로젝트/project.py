import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
import json
import os
from tqdm import tqdm

# 위도 경도 추출
df = pd.read_csv('./경찰서,지구대,파출소,치안센터,초.중.고.대학교,유흥주점.csv')

# 업로드된 파일 경로
file_path = '경찰서,지구대,파출소,치안센터,초.중.고.대학교,유흥주점.csv'

# 카카오 API 키 (가장 중요! 새로 발급받은 키를 정확히 입력해주세요)
KAKAO_API_KEY = 'c6c8a4613ef4138eea36dbf1178a0d29'
API_URL = 'https://dapi.kakao.com/v2/local/search/address.json'

# --- 2. 개선된 기능 ---

# API 키가 기본값 그대로인지 확인
if 'YOUR_NEW_SECURE_API_KEY' in KAKAO_API_KEY:
    print("🛑 [중요] 카카오 API 키가 설정되지 않았습니다. 코드의 'YOUR_NEW_SECURE_API_KEY' 부분을 실제 키로 변경해주세요.")
    # 키가 없으면 여기서 실행 중단
else:
    try:
        # 파일 불러오기
        df = pd.read_csv('./경찰서,지구대,파출소,치안센터,초.중.고.대학교,유흥주점.csv')
        print(f"✅ 파일을 성공적으로 불러왔습니다. (경로: {file_path})")

        # 주소를 좌표로 변환하는 함수
        def get_coords_from_address(address, api_key):
            if not isinstance(address, str) or not address.strip():
                return None, None, "주소 없음"
            
            cleaned_address = re.sub(r'\s*\([^)]*\)', '', address).strip()
            headers = {'Authorization': f'KakaoAK {api_key}'}
            params = {'query': cleaned_address}
            
            try:
                response = requests.get(API_URL, headers=headers, params=params)
                if response.status_code == 401:
                    return None, None, "API 키 인증 실패 (401)"
                response.raise_for_status()
                data = response.json()
                
                if data['documents']:
                    lon = data['documents'][0]['x']
                    lat = data['documents'][0]['y']
                    return lat, lon, "성공"
                else:
                    return None, None, "결과 없음"
            except requests.exceptions.RequestException as e:
                return None, None, f"네트워크 오류: {e}"
            except Exception as e:
                return None, None, f"알 수 없는 오류: {e}"

        # --- 3. 지오코딩 실행 및 상세 로그 출력 ---
        
        print("\n--- 지오코딩 작업을 시작합니다 ---")
        
        # 성공, 실패 카운터 초기화
        success_count = 0
        fail_count = 0

        # 위도가 비어있는 행만 대상으로 작업 수행
        rows_to_process = df[df['위도'].isnull()].index
        if not rows_to_process.empty:
            for index in tqdm(rows_to_process, desc="진행률"):
                row = df.loc[index]
                lat, lon, status = None, None, "시작"

                # 1차: 도로명 주소 시도
                if pd.notna(row['도로명']):
                    lat, lon, status = get_coords_from_address(row['도로명'], KAKAO_API_KEY)

                # 2차: 도로명 실패 시 지번 주소 시도
                if not lat and pd.notna(row['지번']):
                    jibun_address = f"대구광역시 {row['지번']}"
                    lat, lon, status = get_coords_from_address(jibun_address, KAKAO_API_KEY)
                
                # 결과에 따라 처리 및 로그 출력
                if lat and lon:
                    df.loc[index, '위도'] = lat
                    df.loc[index, '경도'] = lon
                    success_count += 1
                    # print(f"  [성공] index {index}: {row['이름']} -> {status}") # 너무 많은 로그 대신 최종 요약으로 대체
                else:
                    fail_count += 1
                    print(f"  [실패] index {index}: '{row['이름']}' / 주소: '{row['도로명']}' -> (원인: {status})")

        # --- 4. 최종 결과 요약 및 저장 ---
        
        print("\n--- 작업이 모두 완료되었습니다 ---")
        print(f"🎉 총 {success_count + fail_count}건 중")
        print(f"  - 성공: {success_count}건")
        print(f"  - 실패: {fail_count}건")

        # API 키 인증 실패가 한 번이라도 있었다면 안내 메시지 출력
        if "API 키 인증 실패" in locals().get('status', ''):
             print("\n⚠️ 'API 키 인증 실패' 오류가 발생했습니다. 카카오 개발자 사이트에서 API 키가 정상적인지, 사용 권한이 있는지 확인해주세요.")

        output_filename = 'geocoded_final_data_log.csv'
        df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n✨ 최종 결과가 '{output_filename}' 파일로 저장되었습니다.")

    except FileNotFoundError:
        print(f"🛑 [오류] 파일을 찾을 수 없습니다. 스크립트가 있는 폴더에 '{file_path}' 파일이 있는지 확인해주세요.")
    except Exception as e:
        print(f"🛑 [오류] 예상치 못한 문제가 발생했습니다: {e}")





# 행정동 추가하는 코드

# --- 1. 설정 단계 ---
FILE_PATH = 'geocoded_final_data_log.csv' 
KAKAO_API_KEY = 'c6c8a4613ef4138eea36dbf1178a0d29' # 본인의 카카오 REST API 키를 입력하세요.
API_URL = 'https://dapi.kakao.com/v2/local/geo/coord2address.json'

# --- 2. 수정된 리버스 지오코딩 함수 ---
def get_dong_from_coords(latitude, longitude, api_key):
    """
    좌표를 행정동으로 변환하되, 행정동이 없으면 법정동을 대신 반환하는 함수
    """
    if pd.isna(latitude) or pd.isna(longitude):
        return None
        
    headers = {'Authorization': f'KakaoAK {api_key}'}
    params = {'x': longitude, 'y': latitude}
    
    try:
        response = requests.get(API_URL, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['documents']:
            # 주소 정보를 변수에 저장
            address_info = data['documents'][0]['address']
            
            # 1. 행정동(region_3depth_h_name)을 우선적으로 가져오기
            dong_name = address_info.get('region_3depth_h_name')
            
            # 2. 만약 행정동 정보가 없다면, 대신 법정동(region_3depth_name)을 가져오기
            if not dong_name:
                dong_name = address_info.get('region_3depth_name')
                
            return dong_name
        else:
            return None
    except Exception:
        return None

# --- 3. 데이터 처리 및 실행 ---
try:
    df = pd.read_csv('./geocoded_final_data_log.csv')
    print(f"✅ 파일을 성공적으로 불러왔습니다: {FILE_PATH}")

    if 'YOUR_NEW_SECURE_API_KEY' in KAKAO_API_KEY:
        print("🛑 [중요] 카카오 API 키를 코드에 입력해주세요.")
    else:
        tqdm.pandas(desc="행정동 변환 최종 진행률")

        # 수정된 함수를 적용하여 '행정동' 열 채우기
        df['행정동'] = df.progress_apply(
            lambda row: get_dong_from_coords(row['위도'], row['경도'], KAKAO_API_KEY),
            axis=1
        )

        # --- 4. 결과 저장 ---
        output_filename = 'final_data_with_dong_completed.csv'
        df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        
        print("\n--- ✨ 최종 작업 완료 ✨ ---")
        print("결과 미리보기 (상위 5개):")
        print(df[['이름', '위도', '경도', '행정동']].head())
        print(f"\n모든 작업이 완료되었습니다! 결과가 '{output_filename}' 파일로 저장되었습니다.")

except FileNotFoundError:
    print(f"🛑 [오류] 파일을 찾을 수 없습니다: {FILE_PATH}")
except Exception as e:
    print(f"🛑 [오류] 예상치 못한 문제가 발생했습니다: {e}")


final = pd.read_csv('./final_data_with_dong_completed.csv')
final['행정동'].unique()




# 세부행정동
# --- 1. 설정 단계 ---

# 좌표 데이터가 있는 파일 경로를 지정하세요.
FILE_PATH = 'geocoded_final_data_log.csv' 
# VWorld에서 발급받은 개발키(API 키)를 입력하세요.
VWORLD_API_KEY = '1E64090E-B7B1-39A7-A068-2525ED16CABC'
API_URL = 'http://api.vworld.kr/req/address'

# --- 2. VWorld 리버스 지오코딩 함수 ---

def get_dong_from_vworld(latitude, longitude, api_key):
    """
    VWorld API를 사용하여 좌표를 세부 행정동으로 변환하는 함수
    """
    if pd.isna(latitude) or pd.isna(longitude):
        return None
        
    params = {
        'service': 'address',
        'request': 'getaddress',
        'version': '2.0',
        'crs': 'epsg:4326',
        'point': f'{longitude},{latitude}', # VWorld는 경도, 위도 순서
        'format': 'json',
        'type': 'both', # 도로명, 지번 주소 모두 조회
        'zipcode': 'true',
        'simple': 'false',
        'key': api_key
    }
    
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        # API 응답 상태가 정상('OK')인지 확인
        if data['response']['status'] == 'OK':
            # 행정동 이름은 'level4A' 필드에 있습니다.
            dong_name = data['response']['result'][0]['structure']['level4A']
            return dong_name
        else:
            return None
    except Exception:
        # API 호출 중 오류가 발생하면 None을 반환
        return None

# --- 3. 데이터 처리 및 실행 ---
try:
    df = pd.read_csv('./geocoded_final_data_log.csv')
    print(f"✅ 파일을 성공적으로 불러왔습니다: {FILE_PATH}")

    if 'YOUR_VWORLD_API_KEY' in VWORLD_API_KEY:
        print("🛑 [중요] VWorld API 키를 코드에 입력해주세요.")
    else:
        tqdm.pandas(desc="VWorld 행정동 변환 진행률")

        # VWorld 함수를 적용하여 '행정동' 열 채우기
        df['행정동'] = df.progress_apply(
            lambda row: get_dong_from_vworld(row['위도'], row['경도'], VWORLD_API_KEY),
            axis=1
        )

        # --- 4. 최종 결과 저장 ---
        output_filename = 'final_data_completed.csv'
        df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        
        print("\n--- ✨ 모든 작업 완료 ✨ ---")
        print("최종 결과 미리보기 (상위 5개):")
        # 이전에 있던 '행정동' 열을 삭제하고 새로 채운 결과를 보여줍니다.
        if '행정동_x' in df.columns:
             df = df.drop(columns=['행정동_x', '행정동_y'])
        print(df[['이름', '도로명', '행정동']].head())
        print(f"\n모든 작업이 완료되었습니다! 최종 결과가 '{output_filename}' 파일로 저장되었습니다.")

except FileNotFoundError:
    print(f"🛑 [오류] 파일을 찾을 수 없습니다: {FILE_PATH}")
except Exception as e:
    print(f"🛑 [오류] 예상치 못한 문제가 발생했습니다: {e}")

final = pd.read_csv('./final_data_completed.csv')
final.isna().sum()


# 가로등 세부행정동 
# --- 1. 설정 단계 ---

# 좌표 데이터가 있는 파일 경로를 지정하세요.
FILE_PATH = '가로등, 보안등, 안전벨, CCTV.csv' 
# VWorld에서 발급받은 개발키(API 키)를 입력하세요.
VWORLD_API_KEY = '1E64090E-B7B1-39A7-A068-2525ED16CABC'
API_URL = 'http://api.vworld.kr/req/address'

# --- 2. VWorld 리버스 지오코딩 함수 ---

def get_dong_from_vworld(latitude, longitude, api_key):
    """
    VWorld API를 사용하여 좌표를 세부 행정동으로 변환하는 함수
    """
    if pd.isna(latitude) or pd.isna(longitude):
        return None
        
    params = {
        'service': 'address',
        'request': 'getaddress',
        'version': '2.0',
        'crs': 'epsg:4326',
        'point': f'{longitude},{latitude}', # VWorld는 경도, 위도 순서
        'format': 'json',
        'type': 'both', # 도로명, 지번 주소 모두 조회
        'zipcode': 'true',
        'simple': 'false',
        'key': api_key
    }
    
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        # API 응답 상태가 정상('OK')인지 확인
        if data['response']['status'] == 'OK':
            # 행정동 이름은 'level4A' 필드에 있습니다.
            dong_name = data['response']['result'][0]['structure']['level4A']
            return dong_name
        else:
            return None
    except Exception:
        # API 호출 중 오류가 발생하면 None을 반환
        return None

# --- 3. 데이터 처리 및 실행 ---
try:
    df1 = pd.read_csv('./가로등, 보안등, 안전벨, CCTV.csv', encoding='cp949')
    print(f"✅ 파일을 성공적으로 불러왔습니다: {FILE_PATH}")

    if 'YOUR_VWORLD_API_KEY' in VWORLD_API_KEY:
        print("🛑 [중요] VWorld API 키를 코드에 입력해주세요.")
    else:
        tqdm.pandas(desc="VWorld 행정동 변환 진행률")

        # VWorld 함수를 적용하여 '행정동' 열 채우기
        df1['행정동'] = df1.progress_apply(
            lambda row: get_dong_from_vworld(row['위도'], row['경도'], VWORLD_API_KEY),
            axis=1
        )

        # --- 4. 최종 결과 저장 ---
        output_filename = 'CCTV_data_completed.csv'
        df1.to_csv(output_filename, index=False, encoding='utf-8-sig')
        
        print("\n--- ✨ 모든 작업 완료 ✨ ---")
        print("최종 결과 미리보기 (상위 5개):")
        # 이전에 있던 '행정동' 열을 삭제하고 새로 채운 결과를 보여줍니다.
        if '행정동_x' in df1.columns:
             df1 = df1.drop(columns=['행정동_x', '행정동_y'])
        print(df1[['이름', '도로명', '행정동']].head())
        print(f"\n모든 작업이 완료되었습니다! 최종 결과가 '{output_filename}' 파일로 저장되었습니다.")

except FileNotFoundError:
    print(f"🛑 [오류] 파일을 찾을 수 없습니다: {FILE_PATH}")
except Exception as e:
    print(f"🛑 [오류] 예상치 못한 문제가 발생했습니다: {e}")

CCTV_final = pd.read_csv('./CCTV_data_completed.csv')

CCTV_final['행정동'].isna().sum()
CCTV_final.loc[(CCTV_final['구분']=='안전비상벨'),:].shape



# CCTV 필터링
# --- 1. 설정 단계 ---
# 원본 데이터 파일 경로를 지정하세요.
FILE_PATH = 'CCTV_data_coml.csv'
# --- 2. 데이터 필터링 및 삭제 ---
try:
    # 삭제할 조건들을 정의합니다.
    # 조건 1: '구분' 칼럼이 'CCTV'인 것
    condition1 = CCTV_final['구분'] == 'CCTV'
    
    # 조건 2: '설치목적' 칼럼이 ['차량방범', '재난재해', '교통단속'] 중 하나에 포함되는 것
    purpose_to_remove = ['차량방범', '재난재해', '교통단속']
    condition2 = CCTV_final['설치목적'].isin(purpose_to_remove)

    # 두 조건을 모두 만족하는 행들을 찾습니다.
    rows_to_remove = CCTV_final[condition1 & condition2]
    print(f"  - 삭제 대상 데이터 개수: {len(rows_to_remove)}개")

    # 위 조건에 해당하지 않는 행들만 남깁니다 (즉, 조건에 맞는 행들을 삭제).
    df_filtered = CCTV_final.drop(rows_to_remove.index)
    
    print(f"  - 필터링 후 남은 데이터 개수: {len(df_filtered)}개")
    
    # --- 3. 결과 저장 ---
    output_filename = 'CCTV_filtered.csv'
    df_filtered.to_csv(output_filename, index=False, encoding='utf-8-sig')

    print(f"\n✨ 작업이 완료되었습니다! 결과가 '{output_filename}' 파일로 저장되었습니다.")

except FileNotFoundError:
    print(f"🛑 [오류] 파일을 찾을 수 없습니다. '{FILE_PATH}' 파일이 있는지, 파일 이름을 정확히 입력했는지 확인해주세요.")
except KeyError as e:
    print(f"🛑 [오류] '{e}' 칼럼을 찾을 수 없습니다. 데이터 파일에 '구분'과 '설치목적' 칼럼이 있는지 확인해주세요.")
except Exception as e:
    print(f"🛑 [오류] 예상치 못한 문제가 발생했습니다: {e}")





# 이웃주소로 주소 할당
from sklearn.neighbors import KNeighborsClassifier

# --- 1. 설정 단계 ---

# 행정동 데이터가 일부 비어있는 파일 경로를 지정하세요.
FILE_PATH = 'CCTV_filtered.csv'
# 예측 결과를 저장할 파일 이름을 지정하세요.
OUTPUT_FILENAME = 'predicted_dong_data.csv'


# --- 2. 데이터 준비 ---

try:
    print(f"✅ 데이터를 불러오는 중입니다: {FILE_PATH}")
    df = pd.read_csv('./CCTV_filtered.csv')
    
    # '행정동' 열이 있는 데이터(학습용)와 없는 데이터(예측용)를 분리합니다.
    df_train = df.dropna(subset=['행정동'])
    df_predict = df[df['행정동'].isnull()]

    print(f"  - 학습용 데이터 (행정동 O): {len(df_train)}개")
    print(f"  - 예측용 데이터 (행정동 X): {len(df_predict)}개")

    # 예측할 데이터가 없으면 종료
    if df_predict.empty:
        print("\n✅ 비어있는 행정동 데이터가 없습니다. 작업을 종료합니다.")
    else:
        # --- 3. 머신러닝 모델 학습 ---
        
        # 특성(X)은 위도, 경도 / 타겟(y)은 행정동으로 지정합니다.
        X_train = df_train[['위도', '경도']]
        y_train = df_train['행정동']

        print("\n⚙️ 머신러닝 모델을 학습시키는 중입니다... (데이터 양에 따라 시간이 걸릴 수 있습니다)")
        # KNN 모델 생성 (가장 가까운 5개의 이웃을 참고)
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1) # n_jobs=-1은 모든 CPU 코어를 사용해 속도 향상

        # 모델 학습
        knn.fit(X_train, y_train)
        print("  - 모델 학습 완료!")


        # --- 4. 비어있는 데이터 예측 ---

        # 예측해야 할 데이터의 위도, 경도 정보를 가져옵니다.
        X_predict = df_predict[['위도', '경도']]
        
        print("\n🔮 비어있는 행정동을 예측하는 중입니다...")
        
        # 학습된 모델로 예측 실행
        predicted_dongs = knn.predict(X_predict)
        print("  - 예측 완료!")

        # --- 5. 결과 합치기 및 저장 ---

        # 예측된 결과를 원본 데이터프레임의 비어있는 부분에 채워넣습니다.
        df.loc[df['행정동'].isnull(), '행정동'] = predicted_dongs
        
        # 결과를 새로운 CSV 파일로 저장
        df.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8-sig')
        
        print(f"\n✨ 모든 작업이 완료되었습니다! 예측 결과가 '{OUTPUT_FILENAME}' 파일로 저장되었습니다.")
        print("\n결과 미리보기 (상위 5개):")
        print(df.head())


except FileNotFoundError:
    print(f"🛑 [오류] 파일을 찾을 수 없습니다: '{FILE_PATH}'")
except KeyError as e:
    print(f"🛑 [오류] 필수 칼럼을 찾을 수 없습니다: {e}. '위도', '경도', '행정동' 칼럼이 있는지 확인해주세요.")
except Exception as e:
    print(f"🛑 [오류] 예상치 못한 문제가 발생했습니다: {e}")


# 메인데이터 통합
main = pd.read_excel('./읍면동인구.xlsx')
main.columns

safe = pd.read_csv('./predicted_dong_data.csv')

kid_cctv = safe.loc[(safe['구분']=='CCTV')&(safe['설치목적']=='어린이보호'),:]
kid_cctv = kid_cctv.groupby('행정동')['위도'].count().reset_index()
kid_cctv
main = pd.merge(main,kid_cctv,on='행정동',how='left')
main['위도'].fillna(0, inplace=True)
main['위도'].sum()
main

other_cctv = safe.loc[(safe['구분']=='CCTV')&(safe['설치목적']!='어린이보호'),:]
other_cctv = other_cctv.groupby('행정동')['경도'].count().reset_index()
other_cctv
main = pd.merge(main,other_cctv,on='행정동',how='left')
main['경도'].fillna(0, inplace=True)
main['경도'].sum()
main

bell = safe.loc[(safe['구분']=='안전비상벨'),:]
bell = bell.groupby('행정동')['설치목적'].count().reset_index()
bell
main = pd.merge(main,bell,on='행정동',how='left')
main['설치목적'].fillna(0, inplace=True)
main['설치목적'].sum()
main

main.to_csv('./정리1.csv')
main = pd.read_excel('./정리1.xlsx')
del main['Unnamed: 0.1']

light = safe.loc[(safe['구분']=='가로등'),:]
light = light.groupby('행정동')['위도'].count().reset_index()
light
main = pd.merge(main,light,on='행정동',how='left')
main['위도'].fillna(0, inplace=True)
main['위도'].sum()
main

light2 = safe.loc[(safe['구분']=='보안등'),:]
light2 = light2.groupby('행정동')['경도'].count().reset_index()
light2
main = pd.merge(main,light2,on='행정동',how='left')
main['경도'].fillna(0, inplace=True)
main['경도'].sum()
main

main.to_csv('./정리2.csv')
main = pd.read_excel('./정리2.xlsx')
del main['Unnamed: 0.1']
main

danger = pd.read_csv('./최종 위험도측정 데이터.csv')
danger
police = danger.loc[(danger['구분']=='경찰서')|
                    (danger['구분']=='지구대')|
                    (danger['구분']=='파출소')|
                    (danger['구분']=='치안센터'),:]
police  = police .groupby('행정동')['이름'].count().reset_index()
police
main = pd.merge(main,police,on='행정동',how='left')
main['이름'].fillna(0, inplace=True)
main['이름'].sum()
main

uheong = danger.loc[(danger['구분']=='유흥주점영업'),:]
uheong  = uheong .groupby('행정동')['도로명'].count().reset_index()
uheong
main = pd.merge(main,uheong,on='행정동',how='left')
main['도로명'].fillna(0, inplace=True)
main['도로명'].sum()
main

main.to_csv('./정리3.csv')
main = pd.read_excel('./정리3.xlsx')
del main['Unnamed: 0.1']
main

ele = danger.loc[(danger['구분']=='초등학교'),:]
ele  = ele .groupby('행정동')['이름'].count().reset_index()
ele
main = pd.merge(main,ele,on='행정동',how='left')
main['이름'].fillna(0, inplace=True)
main['이름'].sum()
main

m_h = danger.loc[(danger['구분']=='중학교')|(danger['구분']=='고등학교'),:]
m_h  = m_h .groupby('행정동')['도로명'].count().reset_index()
m_h
main = pd.merge(main,m_h,on='행정동',how='left')
main['도로명'].fillna(0, inplace=True)
main['도로명'].sum()
main

uni = danger.loc[(danger['구분']=='대학교'),:]
uni  = uni .groupby('행정동')['지번'].count().reset_index()
uni.shape
main = pd.merge(main,uni,on='행정동',how='left')
main['지번'].fillna(0, inplace=True)
main['지번'].sum()
main

main.to_csv('./정리4.csv')

main = pd.read_excel('./정리4.xlsx')
main

com = pd.read_excel('./상권정보_동추가.xlsx')
com = com.rename(columns={'행정동명':'행정동'})
com.shape
com = com.groupby('행정동')['경도'].count().reset_index()
main = pd.merge(main,com,on='행정동',how='left')
main['경도'].fillna(0, inplace=True)
main['경도'].sum()
main.to_csv('./메인.csv')

main = pd.read_excel('./main data.xlsx')
main
com = pd.read_excel('./상권정보_동추가.xlsx')
com = com.rename(columns={'행정동명':'행정동'})
com = com.loc[(com['상권업종소분류명']=='입시·교과학원')|
              (com['상권업종소분류명']=='요리 주점')|
              (com['상권업종소분류명']=='일반 유흥 주점'),:]
com = com.groupby(['행정동','상권업종소분류명'])['상호명'].count().reset_index()
com = com.pivot_table(index='행정동',
                      columns='상권업종소분류명',
                      values='상호명').fillna(0)
main = pd.merge(main,com,on='행정동',how='left')
main
main.to_excel('./Main data.xlsx')











main = pd.read_csv('./df1_main_data_detailed.csv')
main
del main['col_0']
main

safe = pd.read_csv('./최종 안전점수 데이터.csv')
cctv = safe.loc[(safe['구분']=='CCTV'),:]
cctv['설치목적'].unique()
sang_cctv = cctv.loc[(cctv['설치목적']=='생활방범'),:]
sang = sang_cctv.groupby('행정동')['개수'].sum().reset_index()
main = pd.merge(main,sang,on='행정동',how='left')
main['개수'].fillna(0,inplace=True) # 개수_x

si = cctv.loc[(cctv['설치목적']=='시설물관리'),:]
si = si.groupby('행정동')['개수'].sum().reset_index()
main = pd.merge(main,si,on='행정동',how='left')
main['개수'].fillna(0,inplace=True) # 개수

gi = cctv.loc[(cctv['설치목적']=='기타'),:]
gi = gi.groupby('행정동')['개수'].sum().reset_index()
main = pd.merge(main,gi,on='행정동',how='left', suffixes=('_total', '_gi'))
main['개수'].fillna(0,inplace=True) 

ss = cctv.loc[(cctv['설치목적']=='쓰레기단속'),:]
ss = ss.groupby('행정동')['개수'].sum().reset_index()
main = pd.merge(main,ss,on='행정동',how='left', suffixes=('_si', '_ss'))
main['개수_ss'].fillna(0,inplace=True) 


other = cctv.loc[(cctv['설치목적']!='어린이보호'),:]
other = other.groupby('행정동')['개수'].sum().reset_index()
main = pd.merge(main,other,on='행정동',how='left')
main['개수'].fillna(0,inplace=True) # 개수_total


main.to_csv('./df1_main_data.csv')

main


# 분기별 범죄수/범죄율 증감 시각화
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import platform

# 1. 데이터 준비
# './범죄율 증감.xlsx' 파일에서 데이터를 불러옵니다.
# 파일 경로가 코드 실행 위치와 맞는지 확인해주세요.
try:
    # 사용자가 요청한 대로 Excel 파일에서 데이터를 읽어옵니다.
    df = pd.read_excel('./범죄율 증감.xlsx')
except FileNotFoundError:
    print("오류: './범죄율 증감.xlsx' 파일을 찾을 수 없습니다.")
    print("코드와 같은 폴더에 파일이 있는지 확인해주세요.")
    # 파일이 없을 경우, 빈 데이터프레임을 생성하여 오류를 방지합니다.
    df = pd.DataFrame()

# 데이터가 성공적으로 로드되었을 경우에만 후속 코드를 실행합니다.
if not df.empty:
    # 2. 데이터 전처리
    # 'Unnamed: 0' 컬럼을 인덱스로 설정하고, 불필요한 첫 번째 인덱스 컬럼은 제거합니다.
    # Excel 파일의 첫 번째 열 이름이 다를 경우, 'Unnamed: 0' 부분을 실제 열 이름으로 수정해야 합니다.
    df = df.set_index('Unnamed: 0').iloc[:, 1:]

    # 행과 열을 바꿔서(Transpose) 분기별 데이터를 다루기 쉽게 만듭니다.
    df_t = df.transpose()

    # 3. 시각화 (메시지 강조 버전)
    # 한글 폰트 설정 (Windows, Mac, Linux 환경에 맞게 자동 설정)
    if platform.system() == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif platform.system() == 'Darwin': # Mac
        plt.rc('font', family='AppleGothic')
    else: # Linux
        plt.rc('font', family='NanumGothic')

    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

    # 2개의 그래프를 담을 Figure와 Axes 객체를 생성합니다.
    fig, axes = plt.subplots(2, 1, figsize=(12, 14))
    fig.suptitle('분기별 범죄 동향 분석', fontsize=22, y=1.03)

    # --- 그래프 1: 전년 동분기 대비 범죄 발생 증감률 ---
    colors = ['crimson' if x > 0 else 'royalblue' for x in df_t['전년동분기 대비 발생건수 증감률(%)']]
    sns.barplot(x=df_t.index, y=df_t['전년동분기 대비 발생건수 증감률(%)'], ax=axes[0], palette=colors)
    axes[0].set_title('전년 동분기 대비 범죄 발생률은 계속 증가 추세', fontsize=16, pad=20)
    axes[0].set_ylabel('증감률 (%)')
    axes[0].axhline(0, color='grey', linewidth=0.8) # 0% 기준선 추가
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)

    # 막대 위에 값 표시
    for p in axes[0].patches:
        axes[0].annotate(f'{p.get_height()}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=11)

    axes[0].annotate('24년 3분기를 제외하고\n매 분기 전년 대비 범죄 증가', 
                     xy=(0.95, 0.95), xycoords='axes fraction',
                     ha='right', va='top',
                     bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5),
                     fontsize=13)

    # --- 그래프 2: 검거율 하락 추세 강조 ---
    sns.lineplot(data=df_t['발생건수대비 검거건수(%)'], ax=axes[1], marker='o', color='darkorange', linewidth=2.5)
    axes[1].set_title('증가하는 범죄율과 달리 검거율은 하락세', fontsize=16, pad=20)
    axes[1].set_ylabel('검거율 (%)')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # 최저점(가장 낮은 검거율)에 텍스트 강조
    min_period = df_t['발생건수대비 검거건수(%)'].idxmin()
    min_value = df_t['발생건수대비 검거건수(%)'].min()
    axes[1].text(min_period, min_value, f'  {min_value}% (최저)', 
                 color='blue', verticalalignment='top', fontsize=12, fontweight='bold')
    
    # 하락 추세 강조를 위한 화살표와 텍스트 추가
    axes[1].annotate('대응 효율성 저하', 
                     xy=(len(df_t)-2, df_t['발생건수대비 검거건수(%)'].iloc[-3]), 
                     xytext=(len(df_t)-4, df_t['발생건수대비 검거건수(%)'].iloc[-3] - 10),
                     arrowprops=dict(facecolor='royalblue', shrink=0.05, alpha=0.7, connectionstyle="arc3,rad=0.2"),
                     fontsize=14, color='royalblue', fontweight='bold')


    # 전체 레이아웃을 깔끔하게 조정합니다.
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # 그래프를 화면에 보여줍니다.
    plt.show()







# 스케일링
df = pd.read_csv('./df1_main_data - 위험도 계산용.csv')
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

# 수치형 컬럼 선택
numeric_cols = ['치안기관','유흥업소 수','초등학교 수',
                '중,고등학교 수','어린이용 CCTV 수','안전비상벨 수',
                '요리 주점','입시·교과학원','기타 CCTV 수',
                '시설물 CCTV 수', '쓰레기단속 CCTV 수']

# RobustScaler + MinMaxScaler 파이프라인
scaler = Pipeline([
    ('robust', RobustScaler()), 
    ('minmax', MinMaxScaler(feature_range=(0, 10)))
])
scaled_values = scaler.fit_transform(df[numeric_cols])
# 새로운 데이터프레임 생성
df_scaled = pd.DataFrame(scaled_values, columns=numeric_cols, index=df.index)
# 범주형 열 붙이기
df_scaled = pd.concat([df.drop(columns=numeric_cols), df_scaled], axis=1)
print(df_scaled.head())
df_scaled.to_csv('./1.csv')



total = pd.read_csv('./빨간색,노란색.csv')
total.drop(0,axis=0,inplace=True)
total

