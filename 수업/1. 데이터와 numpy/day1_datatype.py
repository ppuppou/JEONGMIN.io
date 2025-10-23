# list type : 대괄호 [] 안에 쉼표로 구분된 여러 type의 원소를 가짐
fruits = ['apple', 'banana', 'cherry']
numbers = [1, 2, 3, 4, 5]
mixed_list = [1, "hello", [1, 2, 3]] #원소에 리스트가 있어도 괜찮음

fruits[0] # list type도 마찬가지로 대괄호로 indexing 가능
fruits[1:]
numbers[1:4]

numbers + numbers # lsit type도 합, 곱 연산 가능, 하나의 list로 합쳐짐
numbers * 3
numbers + fruits # list끼리는 문자, 숫자 상관 없음

mixed_list[2][1]

len(mixed_list[2])

fruits[2] + ' hi'
numbers[2] + 'hi' #숫자 + 문자라 오류
str(numbers[2]) + 'hi' #숫자를 문자로 바꿔서 가능

numbers[2] = 10 # list는 원소를 수정 가능 == mutable
numbers
del numbers[1] # 1번자리 삭제
numbers.append(10) # 맨 뒤에 10(원소) 추가 
numbers.sort() # 작은 숫자부터 정렬(문자도 정렬 가능)
# 숫자와 문자가 섞이면 sorting이 불가능

# tuple type : 소괄호, 또는 괄호 없이 쉼표로 요소 구분
a = (10, 20, 30) # a = 10, 20, 30 과 동일
b = (2, 4)
c = (2,) # 원소가 하나여도 쉼표가 필요함, 안그러면 숫자로 인식
print("좌표:", a)
# 튜플도 indexing, slicing, 합/곱 연산 가능]
c[0]
a[2] = 3 # tuple은 수정이 불가능함 == immutable

tup_e = (1, 3, 'a', [1,2], (3,1)) # 역시나 원소 종류는 상관 X

a = a + tup_e
a