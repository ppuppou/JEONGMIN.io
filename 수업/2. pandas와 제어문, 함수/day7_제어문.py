import numpy as np
import pandas as pd

# 제어문 
# if문 
money = 1000
card = True
if((money > 4500) | (card == True)): # 조건이 여러개여도 가능
    print('택시를 타세여.')
else :
    print('걸어가세여')

# 시험점수가 60점 이상이면 합격, 그렇지 않으면 불합격을 출력
score = 50
if(score >= 60) :
    print('합격')
else :
    print('불합격')
# x가 홀수이면서 7의 배수이면 '7의배수이면서 홀수', 
# 그렇지 않으면 '조건불만족' 출력
x = 14
if((x%2==1)and(x%7==0)):
    print('7의배수이면서 홀수')
else :
    print('조건불만족')

# 결과가 여러개인 경우
gender = 'M'
if(gender == 'M') :
    print('남성입니다.')
elif(gender == 'W') :
    print('여성입니다.')
else :
    print('중성입니다.')

# 유아(7세이하)는 무료, 어린이는 3000원,
# 성인은 7000원, 노인(60세이상)은 5000원을 출력하는 함수를 만드세요
def cal_price(age) :
    if(age <= 7):
        price = '무료'
    elif(7 < age < 20): # 이전 if문에서 이미 <=7이 False라서,
        price = '2000원' #  7<age는 넣어도 되고 안넣어도 됨
    elif(20<= age <60):
        price = '7000원'
    else:
        price = '5000원'
    return price
cal_price(15)

# whlie문
# 조건을 만족하는(True)동안 코드를 반복실행
treehit = 0
while treehit < 10:
    treehit += 1
    print(f'나무를 {treehit}번 찍었습니다.')
    if treehit == 10:
        print('나무 넘어갑니다~ 슈루루루룩푸쾅팡퓨수슈슈ㅜㅅㅅ슈슈ㅜㅅ규슈ㅜ규수으억당했다')
# while문 강제로 빠져나가기
coffee = 10
money = 300
while money:
    print('돈을 받았으니 커피를 주겠다.')
    coffee -= 1
    print(f'커피가 {coffee}개 남았도다')
    if coffee == 0:
        print('이제 커피 없다.')
        break
# break와 continue
a = 0
while a < 10:
    a += 1
    if (a%2 == 0): # a가 짝수인 경우
        continue   # while 루프 처음으로 넘어가라(print를 건너뛴다는 뜻)
    print(a)

# for문 : for 변수 in 순서가 있는 객체:
#             반복할 내용 1
#             반복할 내용 2
#             ...
test_list = ['one','two','three']
for i in test_list: # one, two, three를 순서대로 i에 대입한다
    print(i)

a = [(1,2),(3,4),(5,6)]
for (first,last) in a:
    print(first+last)

a = [(1,2),(3,4),(5,6),7] # 이 경우, 7은 형식이 맞지 않아서 오류
for (first,last) in a:
    print(first+last)

a = np.arange(1,101,1) # nparray는 순서가 있기 때문에 for문 사용 가능
for i in a:
    if(i%7!=0):
        continue
    print(i)   


# list comprehension : list 안에 for문을 포함 >> 결과가 list
numbers = [x for x in range(1, 6)] # range는 np.arange와 비슷
numbers   # 맨 앞의 x는 출력될 값. print(x)라고 생각하면 됨
a = [1,2,3,4,5]
result = []
for i in a: 
    result.append(i**2+3)
print(result)
[x**2 + 3 for x in range(1,6)] # 위 5줄을 1줄로 줄이는 효과

# 1~10 정수중 각 수의 제곱값을 요소로 가지는 리스트 만들기
[i**2 for i in range(1,11)]
# 1~20까지의 정수중 짝수만 담은 리스트 만들기
for i in range(1,21):
    if(i%2==1):
        continue
    print(i)

[i for i in range(1,21) if(i%2==0)] # if만 있으면 for 뒤에
# 음수는 0으로 바꾸고 양수는 그대로 유지하는 리스트 만들기
nums = [-3, 5, -1, 0, 8]
for x in nums:
    if(x<0):
        print(0)
    else: print(x)

[x if x>=0 else 0 for x in nums ] # else도 있으면 for 앞에
# 리스트에서 'a'로 시작하는 문자열만 추출하기
words = ['apple','banana','cherry','avocado']
for i in words:
    if i[0]!='a': # or, if i.startswith('a')도 가능
        continue
    print(i)

[i for i in words if i[0]=='a']