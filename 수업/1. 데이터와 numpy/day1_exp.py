# 1. 국 영 수 의 평균 구하기
korean = 80
english = 75
math = 55
print("평균점수 = ", (korean + english + math)/3)

# 2. n의 짝/홀수 판별하기
n = 14
n % 2 == 1

# print("n은 홀수인가요?", n % 2 == 1)

# 3. pin에서 생년월일, 일련번호, 성별 정보 추출하기
pin = "881120-1068234"
print('yyyymmdd = ', '19' + pin[:6])
print('num = ', pin[7:])

# yyyydddd = '19' + pin[:pin.find('-')]
# num = pin[pin.find('-')+1:]

print(pin[7])

# 4. : 대신 #로 바꾸기
a = "a:b:c:d"
a.replace(':','#')

# 5. b를 [5,4,3,2,1] 로 변경하여 출력하기
b = [1, 3, 5, 4, 2]
b.sort()
b.reverse()
b

# 6. list를 문자열로 바꾸기
c = ['life', 'is', 'too', 'short']
result = ' '.join(c)
result

# 7. tuple d에 4 원소 추가하기
d = (1, 2, 3)
d = d + (4,)
d

#연습문제 1
city_name = 'seong_nam'
age = 27
is_student = 'no'
f'저는 {city_name}에 살고있는 {age}살 오정민입니다. 학생은 {is_student}입니다.'

#연습문제 2
# a, c, e

#연습문제 3
x = 12
y = 4
(x - y) * y / x
x ** y // x
(x % y) + 2

#연습문제 4
age >= 18
is_raining = False
is_warm = True
is_raining and is_warm
not is_raining or is_warm
not is_raining

#연습문제 5
price = 12000
quantity = 3
total_price = price * 3
10000 <= total_price <= 50000

#data type 연습문제
#연습문제 1
num = 100
pi = 3.14
name = "111"
fruits = ["apple", "banana", "cherry"]
data = (10, 20, 30)
person = {"name": "Tom", "age": 25}
flags = {True, False}
results = [
    type(num) == int,
    type(pi) == float,
    type(name) == int,
    type(fruits) == list,
    type(data) == tuple,
    type(person) == dict,
    type(flags) == set
]
print('검증결과 : ', results)
print("True 개수 : ", sum(results))
 
#연습문제 5 - 1
sentence = "Python Is FUN"
sentence = sentence.upper()
sentence
sentence[:6] + sentence[6:].lower()

# 5 - 2
sentence = "Python Is FUN"
sentence.lower().replace('python','PYTHON')

#Q12
a = b = [1, 2, 3]
a[1] = 4
a
b
# a = b 라서 a가 변경함에 따라 수식을 맞추기 위해 b도 바뀌지 않았을까영