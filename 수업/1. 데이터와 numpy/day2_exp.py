## 데이터타입
# 연습문제 2
nums = [5, 10, 10, 15]
tup = tuple(nums)
set1 = set(tup)
len(set1)

# 연습문제 3
profile = {
    'name' : 'jane',
    'age' : 27,
    'city' : 'Busan'
}
del profile['city']
profile['age'] = 28 #다시 보기
print(profile)

# 연습문제 4
set_x = {1,2,3,4}
set_y = {3,4,5,6}
print(set_x & set_y)
print(set_x | set_y)
print(set_x - set_y)
print(set_y - set_x)

## numpy 연습문제
import numpy as np
# 1
a = np.array([1, 2, 3, 4, 5])
a + 5
print(a + 5)
# print(a+5) 로 하면 리스트가 나오는데 왜그럴까요?

# 2 # slicing에서 맨 마지막에 숫자를 넣으면, 그건 주기가 된다고 합니다
a = np.array([12, 21, 35, 48, 5])
a[::2]

# 3
a = np.array([1,22,93,64,54])
a.max()

# 4 #다시 보기
a = np.array([1,2,3,2,4,5,4,6])
np.unique(a)

# 5 # 다시 보기. slicing에도 값을 차례대로 채워넣을 수 있음 
    # + empty 배열을 만들 땐 배열의 길이를 정해주어야함. 안넣으면 0
a = np.array([21, 31, 58])
b = np.array([24, 44, 67])
c = np.empty(len(a)+len(b))
c[::2] = a
c[1::2] = b
c

#질문, 그렇다면 이 문제에서 꼭 길이를 6으로 정해야하나요?

# 6
a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9])
c = a[:-1] + b  #다시 보기
c

# 7
a = np.array([1, 3, 3, 2, 1, 3, 4, 2, 2, 2, 5, 6, 6, 6, 6])
c, d = np.unique(a, return_counts=True) 
# a의 중복값을 지우고, 개수를 세어달라는 뜻. 그러면 두개의 데이터가 나오니까 
# c,d에 두 데이터를 할당
d == np.max(d)  
# d는 개수, 그래서 개수가 가장 많은 값을 T/F로 표현. ==를 써서 불값이 나옴
c[d == np.max(d)]  # c에서 True인 값만 indexing


# 8 # 다시 보기
a = np.array([12, 5, 18, 21, 7, 9, 30, 25, 3, 6])
multiples_of_three = a[a % 3 == 0]
multiples_of_three

# 9 # 다시 보기
a = np.array([10, 20, 5, 7, 15, 30, 25, 8])
median_value = np.median(a)
first = a[a < median_value]
second = a[a > median_value]
first
second

# 10
a = np.array([12, 45, 8, 20, 33, 50, 19])
median_value = np.median(a)
differences = np.abs(a - median_value) 
#abs는 절댓값을 말함. a 값에서 중앙값을 빼고, 여기에 절대값을 한다는 뜻. 
# 즉, 각 요소의 중앙값과의 차이를 나타내는 코드
closest_value = a[np.where(differences == np.min(differences))][0] 
#나는 where로 어떤 위치를 찾을거야. 어떤 위치냐면 diff. 배열에서 값이 
#diff. 배열의 최소값과 같은 데이터의 위치를 찾을거야. 근데 [0]은 뭐냐? 
#여러개 필요 없고 1개만 (0번째만) 가져올거야. 라는 뜻의 코드
closest_value