import numpy as np
np.arange(1,10,0.5)
np.linspace(1,10,20,True,True)
# first True : 맨 마지막 수를 포함할것인지
# second True : 간격을 반환할 것인지

np.repeat([[1,4,2]],10)
# axis=0은 수직반복, axis=1은 수평반복
np.tile([1,4,2],10) # 리스트 자체를 반복
np.repeat([1,2,4], repeats=[1,2,3]) # 각각 반복할 횟수도 control 가능

arr = np.array([[1,2], 
                 [3,4]])
vec_a = np.array([2,1,4])
len(vec_a)
vec_a.shape
vec_a.size
len(arr) 
arr.shape
arr.size  # 벡터와 행렬에서의 length, size는 다른 의미를 가짐

# 브로드캐스팅 : 길이가 다른 배열 간 연산을 가능하게 해주는 매커니즘
a = np.array([1, 2, 3, 4])
b = np.array([1, 2])
a + np.tile(b,2)
c = 2
a * c # 1차원 브로드캐스팅

# 2차원 브로드캐스팅
matrix = np.array([[ 0.0,  0.0,  0.0],
                   [10.0, 10.0, 10.0],
                   [20.0, 20.0, 20.0],
                   [30.0, 30.0, 30.0]])
vector = np.array([1.0, 2.0, 3.0])
print(vector.shape, matrix.shape)
matrix + vector
vector_v = np.array([[1,2,3,4]]).reshape(4,1) 
# reshape : shape를 바꾸는 함수
matrix + vector_v
matrix_v = matrix.reshape(3,4)
matrix_v = matrix.reshape(3,-1)
# -의 의미 : 나머지는 알아서 계산좀
matrix_v
np.linspace(0,100,20).reshape(10,-1)

vec_a = np.arange(20)
vec_a [1:4] # indexing 가능

# 조건을 만족하는 위치 탐색 np.where()
a = np.array([1, 5, 7, 8, 10])
result = np.where(a < 7)
result

# 벡터의 내적
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.dot(a,b)

# 빈 벡터
y = np.zeros((2,2))

z = np.arange(1,5).reshape(2,-1, order='f')
z
# order='c' : 행 먼저, order='f' : 열 먼저 채우기

# 행렬 원소에 접근하기
x = np.arange(1,11).reshape(5,2)*2
x[2][1]
x[2,0] # 행렬은 indexing이 행 >>> 열 순서

x[[2,4],0]
# 행렬의 indexing은 형태가 유지가 안되고 벡터형태로 출력됨
x[2:4,[0]]
# 이렇게 하면 행렬의 형태를 유지하며 출력 가능
x[1:4,:]
x[[1,3,4],0]
x[[1,2,3],[0,1,1]]

x[x > 10]
# 중간고사 점수가 10점 초과인 학생들의 중간/기말 데이터 필터링
x[:,0] > 10 # 0열의 숫자중 10을 초과하는 요소 
x[x[:,0]>10,:]
# 기말고사 점수가 10점 이하인 학생들의 중간/기말 데이터 필터링
x[x[:,1]<=10,:]

# 연습문제
# random이어도 seed를 맞추면 똑같음
np.random.seed(2025)
vec_a = np.random.choice(np.arange(1,101), size=3000, replace=True)
mat_a = vec_a.reshape(-1,2)

# 0. 중간, 기말 평균
mid_avg = mat_a[:,0].mean()
fin_avg = mat_a[:,1].mean()
# 혹은
result = mat_a.mean(axis=0)
mid_avg = result[0]
fin_avg = result[1]
# 1. 중간고사 점수가 50점 이상인 학생의 데이터와, 명수 구하기
mat_a[mat_a[:,0] >= 50,:].shape[0]

mid_score = mat_a[:,0]
fin_score = mat_a[:,1]

# 2. 그 학생들의 기말 성적 평균 구하기
over_50 = mat_a[mat_a[:,0] >= 50,1]
over_50.mean()

# 3. 중간고사 최고점을 맞은 학생의 기말 성적
mat_a[mat_a[:,0] == mat_a.max(),1]

# 4. 중간 성적이 평균보다 높은 학생들의 기말 성적 평균 실패
over_half = mat_a[mid_score > mid_avg,1]
over_half.mean()

# 5. 중간고사 대비 기말 성적이 향상된 학생의 수
mat_a[mat_a[:,0]<mat_a[:,1],:].shape # 740
sum(mat_a[:,0]<mat_a[:,1])
# 6. 반대로 성적이 떨어진 학생들의 위치(index) 실패
np.where(mid_score>fin_score)

# np.where(조건문, 조건이 맞으면 a, 아니면 b)