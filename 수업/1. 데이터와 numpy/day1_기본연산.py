number = 10
greeting = "안녕하세요!"
number

user_age = 27
user_age

x, y, z = 1,2,3
y

print("x + z =", x+z)
10 % 3

# % : 나머지, // : 몫, ** : 거듭제곱
# == : 같다, != : 부등, <=, >= : 부등호

True # 1
False # 0
True + True # = 2
True + False # = 1

True and True 
True and False 
True or False
not True

a = 33
(a % 2 == 0) and (a >= 10) 
(a % 2 == 0) or (a >= 10) 
not (a % 2 == 0)

# b 변수의 값이 100 이하이면서 7의 배수인지 체크하는 코드
b = 59
(a % 2 == 0) and (a >= 10) 

"안녕" * 3
len('안녕')

str_a = "안녕" + " python"
len(str_a) # 길이
str_a[1] # 인덱싱 indexing
str_a[-2]
str_a[1:3] # 슬라이싱 slicing
str_a[3:] # 맨 처음부터/ 혹은 마지막까지 할거면 생략 가능
str_a[:5]
str_a[:]
str_a[1:-3]

# 포맷팅
name = '정민' ; age = 27
f'나의 이름은 {name}, 나이는 {27}입니다.'

type(str_a)

f'{(str_a):=^15}'
num_a = 10
type(num_a)

# 박스 뒤에 .을 찍으면, 각각 정보의 종류에 따른 가용가능 함수(method)가 나타남
# 아래의 함수들은 문자열 에만 적용 가능한 함수 예시
str_a.count('a')  # 문자열 중 'a'의 개수를 리턴
str_a.find('녕') # 문자열에 '녕' 이 처음 나온 위치
str_a.index('y') # 문자열에 'y'의 위치
','.join(str_a) # 각각 문자 사이에 ','을 삽입
str_a.upper() ; str_a.lower() # 대, 소문자 변경
str_a.lstrip() ; str_a.rstrip() ; str_a.strip() # 좌, 우, 양옆 공백 지우기
str_a.replace('안녕', '반가워') # 문자열 바꾸기
str_a = str_a.replace('안녕', '반가워') # 원 변수를 변경
str_a 
str_a.split() # 공백을 기준으로 나눔. ()안에 문자를 넣으면 문자를 기준으로
type(str_a.split())


name = '정민'
age = 27
f'나의 이름은 {name}이며 나이는 {age}살 입니다'
