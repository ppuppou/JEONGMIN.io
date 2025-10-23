import numpy as np # numpy를 약어로 np로 쓰겠다는 뜻
import pandas as pd

#numpy : Numerical Python. 강력한 수치 계산을 수행하기 위해 개발된 라이브러리

# 벡터 생성하기 예제
a = np.array([1, 2, 3, 4, 5])  # 숫자형 벡터 생성
b = np.array(["apple", "banana", "orange"])  # 문자형 벡터 생성
c = np.array([True, False, True, True])  # 논리형 벡터 생성
print("Numeric Vector:", a)
type(a)
a[1]

# numpy 벡터는 type을 항상 일치시킴
d = np.array([(1,),[1]]) # 튜플이 리스트로 바뀜
e = np.array(['p',2]) # 숫자형이 문자형으로 바뀜
f = np.array(['p',2,[1,2]]) # type을 맞출 수가 없어서 에러
a + 3 # 각 원소에 모두 적용됨
a ** 2 

# numpy는 벡터연산을 지원함
b = np.array([6,7,8,9,10])
a + b

a.cumsum() # 누적합

#numpy 함수
vec_a = np.arange(7,1001,7) # 일정 간격을 가지는 숫자배열 형성
vec_a
len(vec_a)
sum(vec_a) # or vec_a.sum() : 총합
np.cumsum(vec_a) # or vec_a.cumsum() : 누적합