# dictionary type : 중괄호를 사용해 형성. 각 항목은 키:값 형태로 표현됨
person = {
    'name' : 'john',
    'age' : 30,
    'city' : 'New York'
}
person[0] # index 지원 안됨 = 순서가 없다
person.get('name') # 키에 해당하는 값을 추출
person.keys() # 모든 key 추출
person.values() # 모든 value 추출
person['family_name'] = 'Oh' # 쌍 추가 가능
del person['family_name'] # 쌍 삭제 가능
person['name'] # .get과 같은 역할

list(person.keys()) # 결과값을 list로 바꿔서 indexing이 가넝하게 만듬
tuple(person.keys())

# set type : 키가 없이 값만 존재
s1 = set([1,2,3])
s1 = {1,2,3}
s2 = set('hello')
s3 = {4,3,6,2,6,1}
s3 # 집합은 중복을 허락하지 않으며, 순서가 없어서 항상 정렬됨?
s4 = {'a','c','b','b'}
s1 & s3 # 교집합
s1 | s3 # 합집합
s1 - s3 # 차집합

# Q9 
a = {}
a['name']='python'
a[('a',)]='python'
a[[1]]='python'
a[250]='python'
# key 값에 list는 들어갈 수 없음. key는 변경이 불가능한 data여야 함

# Q10 딕셔너리에서 B값을 반환하고 B를 제거
b = {'A':90, 'B':80, 'C':70}
result = b.pop('B')
print(b)
print(result)

# Q11 리스트에서 중복된 숫자를 제거
a = [1,1,1,2,2,3,3,3,4,4,5]
aSet = set(a)
b = list(aSet)
print(b)