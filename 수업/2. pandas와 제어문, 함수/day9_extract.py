import numpy as np
import pandas as pd

data = {
    '주소': ['서울특별시 강남구 테헤란로 123',
            '부산광역시 해운대구 센텀중앙로 45',
            '대구광역시 수성구 동대구로 77-9@@##',
            '인천광역시 남동구 예술로 501&amp;&amp;, 아트센터',
            '광주광역시 북구 용봉로 123']
}
df = pd.DataFrame(data)

df['도시'] = df['주소'].str.extract(r'([가-힣]+광역시|[가-힣]+특별시)',
                                expand=False)
# 특수문자 추출
special_chars = df['주소'].str.extractall(r'([^a-zA-Z0-9가-힣\s])')
# 특수문자 제거
df['주소_특수문자제거'] = df['주소'].str.replace(r'([^a-zA-Z0-9가-힣\s])',
                                       '',
                                       regex=True) # replace는 넣어야함

df['amp 제거'] = df['주소_특수문자제거'].str.replace('amp','')
df

# 연습
df = pd.read_csv("./data/regex_practice_data.csv")
df
# 1. 이메일
df['email'] = df['전체_문자열'].str.extract(r'([a-z]+.+[a-z]+@+[a-z]+.+[a-z])')
# 정답
df['email'] = df['전체_문자열'].str.extract(r'([\w\.]+@+[\w\.])')

# 휴대폰번호 찾기
df['휴대폰번호'] = df['전체_문자열'].str.extract(r'(010+-[0-9]+-[0-9]+\s)')
# 정답
df['휴대폰번호'] = df['전체_문자열'].str.extract(r'(010-[0-9\-]+)')

# 일반전화번호 찾기 실패
df['일반전화번호'] = df['전체_문자열'].str.extract[r'((^010)+-[0-9]+-[0-9]+\s)']
# 정답
phone_num = df['전체_문자열'].str.extract(r'(\d+-[0-9\-]+)')
~phone_num.iloc[:,0].str.startswith('01')
phone_num.loc[~phone_num.iloc[:,0].str.startswith('01'),:]

# 주소중 구 단위만 추출
df['지역구'] = df['전체_문자열'].str.extract(r'([가-힣]+구\s)')
# or
df['지역구'] = df['전체_문자열'].str.extract(r'(\b\w+구\b)')


# YYYY-MM-DD 날짜형식 찾기 실패
df['날짜'] = df['전체_문자열'].str.extract(r'([0-9])+[-]+[0-9]+[-]+[0-9]')
# 정답
df['-날짜'] = df['전체_문자열'].str.extract(r'(\d{4}-\d{2}-\d{2})')

# 모든 날짜형식 찾기 실패
# 정답
df['날짜'] = df['전체_문자열'].str.extract(r'(\d{4}\W\d{2}\W\d{2})')
df['날짜'] = df['전체_문자열'].str.extract(r'(\d{4}[-/.]\d{2}[-/.]\d{2})')

# 가격정보 찾기 실패
# 정답
df['전체_문자열'].str.extract(r'(₩[\d,]+)')
#  가격에서 숫자만 추출 실패
# 정답
df['전체_문자열'].str.extract(r'₩([\d,]+)')

# 이메일의 도메인 추출 실패
# 정답
df['전체_문자열'].str.extract(r'@([\w.]+)')

# 이름만 추출
df['한글이름'] = df['전체_문자열'].str.extract(r'([가-힣]{3})')
