import pandas as pd
import numpy as np

data = {
    'date': ['2024-01-01 12:34:56', '2024-02-01 23:45:01', '2024-03-01 06:07:08', '2021-04-01 14:15:16'],
    'value': [100, 201, 302, 404]
}
df = pd.DataFrame(data)
df.info()
# to_datetime() : 날짜 형식으로 변환
df['date'] = pd.to_datetime(df['date'])
df.info()

pd.to_datetime('02-01-2024')

# 날짜형식 변환시 주의사항
# 비표준화 형식의 날짜 문자열의 경우 날짜 형식으로 자동변환되지 않음
pd.to_datetime('02-2024-01',format='%m-%Y-%d')
# 소문자 m은 월, 대문자 M은 분으로 해석됨
# 소문자 y는 2자리, 대문자 Y는 4자리 년도에 적용 가능

# 원하는 정보만 뽑아내기
df['date'].dt.year  
df['date'].dt.month
df['date'].dt.day
df['date'].dt.minute

# 요일 추출
df['date'].dt.day_name()
df['date'].dt.weekday # 0이 월요일
df['date'].dt.weekday>=5 # 주말만 필터링
current_date = pd.to_datetime('2025-07-21')
current_date - df['date'] # 오늘과의 날짜 차이
(current_date - df['date']).dt.days # days 정보만 뽑기

# 날짜 범위 생성
date_range = pd.date_range(start='2021-01-01',
                            end='2021-03-10',
                              freq='D') # 날짜 단위?

# 넘파이 벡터를 하나로 합치기
df['date'].dt.year  
df['date'].dt.month
df['date'].dt.day
df['date2'] = pd.to_datetime(
    dict(
        year=df['date'].dt.year,
        month=df['date'].dt.month,
        day=df['date'].dt.day
    )
)


# 문자열 다루기
import pandas as pd
data = {
    '가전제품': ['냉장고', '세탁기', '전자레인지', '에어컨', '청소기'],
    '브랜드': ['LG', 'Samsung', 'Panasonic', 'Daikin', 'Dyson']
}
df = pd.DataFrame(data)
df.info()
df['가전제품'].str.len() # 가전제품의 values의 문자열 길이
df['제품명_길이'] = df['가전제품'].str.len()
df['브랜드_길이'] = df['브랜드'].str.len()
df['브랜드'] = df['브랜드'].str.lower() # 소문자로 변경

# 특정 문자 포함 여부 확인
df['브랜드'].str.contains('i') # 불리안값으로 나옴
df['브랜드'].str.startswith('s')
df['브랜드'].str.endswith('g')

# 원하는 문자열 변경 
df['가전제품'].str.replace('에어컨','선풍기')
df['가전제품'].str.replace('기','') # '기' 라는 문자를 없애는 방법
df['브랜드'].str.split('a') # 'a'를 기준으로 문자열을 나눔
df['브랜드'].str.split('a',expand=True) # 분할 결과를 여러 행으로 확장

# 문자열 결합
df['제품_브랜드'] = (
    df['가전제품'].str.cat(df['브랜드'], # 가전제품과 브랜드를 결합
                          sep=',') # 결합 사이에 , 추가
                    )
df

# 좌 우 공백 제거
df['가전제품'] = df['가전제품'].str.replace('전자레인지',' 전자 레인지  ')
df['가전제품'].str.strip() # 좌우 공백만 제거됨
df['가전제품'].str.replace(' ',"") # 모든 공백을 제거하는 방법


# 정규표현식 : 특정한 규칙을 가진 문자열의 패턴을 정의
# .str.extract(r'') 형태로 사용
df = pd.DataFrame({
    'text': [
        'apple',        # [aeiou], (a..e), ^a
        'banana',       # [aeiou], (ana), ^b
        'Hello world',  # ^Hello, world$
        'abc',          # (abc), a.c
        'a1c',          # a.c
        'xyz!',         # [^aeiou], [^0-9]
        '123',          # [^a-z], [0-9]
        'the end',      # d$, e.
        'space bar',    # [aeiou], . (space)
        'hi!',          # [^0-9], [aeiou]
        'blue',
        'lue'
    ]
})
df
df['text'].str.extract(r'([aeiou])') # []와 매칭되는 가장 첫 단어
df['text'].str.extractall(r'([aeiou])') # []와 매칭되는 모든 단어
df['text'].str.extract(r'([^0-9])') # []를 제외한 첫 단어 매칭
df['text'].str.extractall(r'([^0-9])') # []를 제외한 모든 단어 매칭
                                       # 공백도 포함
df['text'].str.extract(r'((ba))') # ba 자체와 매칭되는 첫 단어
df['text'].str.extract(r'(a.c)') # 임의의 한 문자와 매칭 (줄바꿈은 제외)
                                 # . 위치에는 아무거나 있어도 된다는 뜻
df['text'].str.extract(r'^(Hello)') # 문자열 시작에 매칭
df['text'].str.extract(r'(bar)$') # 문자열 끝에에 매칭
df['text'].str.extract(r'(b?lue)') # b는 있어도되고 없어도돼