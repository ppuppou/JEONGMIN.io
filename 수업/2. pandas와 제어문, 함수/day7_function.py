import numpy as np
import pandas as pd

# function 만들기
def add(a,b) : 
    result = a + b
    return result
add(3,4)

# 매개변수 : parameters
# 인수 : arguments

# 입력값이 없는 함수
def say() :
    return 'hi'
say()

# 매개변수 지정해서 호출하기
def sub(a,b) :
    return a-b
sub(b=7,a=10) # 순서가 상관없어짐

def add_many(*nums): # 입력값의 개수를 모를 땐 *를 붙임
    result=0
    for i in nums:
        result += i
    return result
add_many(3,4,2)

def cal_how(method, *nums) :
    if (method == 'add'):
        result=0
        for i in nums:
            result += i
    elif (method == 'mul'):
        result = 1
        for i in nums:
            result *= i
    else :
        print('해당 연산은 수행할 수 없습니다.')
        result = None
    return(result)
cal_how('add',4,5)

def add_and_mul(a,b):
    return a+b,a*b
add_and_mul(3,4) # 함수의 리턴값은 하나로만 나온다 (이경우 tuple 1개)
result1, result2 = add_and_mul(3,4) # 이런식으로 따로 받을수는 있음
# 함수의 parameter를 미리 설정할수도 있음
def add_and_mul(a=5,b=4):
    return (a+b,a*b)
add_and_mul()
add_and_mul(3)
add_and_mul(b=3)