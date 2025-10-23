import numpy as np
import pandas as pd

# 제어문, 함수 연습문제
# 1
def add_numbers(a=1, b=2):
    return a+b
add_numbers()
add_numbers(5,7)

# 2
def check_sign(x):
    if(x>0):
        print('양수')
    elif(x<0):
        print('음수')
    else :
        print('0')
check_sign(10)
check_sign(-5)
check_sign(0)

# 3
def print_numbers():
    for i in range(1,11):
        print(i)
print_numbers()

# 4 
def outer_function(x):
    def inner_function(y):
        return y + 2
    return inner_function(x)
outer_function(5)

# 5
def find_even(start):
    while start%2 == 1:
        start += 1
        if (start%2==0):
            break
    print(start)
find_even(3)

# 추가 연습문제 1
[x**2 for x in range(1,11) if(x%2==0)]

# 정답
even_squares = []
for x in range(1,11):
    if x % 2 == 0:
        even_squares.append(x**2)
print(even_squares)

# 추가 연습문제 2
customers = [
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 34},
    {"name": "Charlie", "age": 29},
    {"name": "David", "age": 32},
    {"name": "Eve", "age": 22}
]
max_ind = len(customers)
# 2-1
for i in range(0,max_ind):
    customers[i]['age'] += 1

[customers]
# 정답 
updates_age = []
for i in customers:
    up_customer = {'name':i['name'], 'age':i['age']+1}
    updates_age.append(up_customer)

# 2-2
over_30 = [i for i in customers if i['age']>=30]
# 정답
age_30 = []
for i in customers:
    if i['age']>=30:
        age_30.append(i)
age_30

# 2-3
sum([customers[i]['age'] for i in range(0,5)])

# 2-4
under_30 = [i for i in customers if i['age']<30]
for i in range(0,2):
    if (under_30[i]['name'][0] == 'A'):
        print (customers[i])

# 추가 연습문제 3
sales_data = {
    "January": 90,"February": 110,
    "March": 95,"April": 120,
    "May": 80,"June": 105,
    "July": 135,"August": 70,
    "September": 85,"October": 150,
    "November": 125,"December": 95
}
# 3-1
over100 = [i for i in sales_data.keys() if(sales_data[i]>=100)]
#정답
over_100 = {}
for month,sales in sales_data.items():
    if sales >= 100:
        over_100[month] = sales

# 3-2
total = sum(sales_data.values())
mean_month = total/len(sales_data.keys())

# 3-3 # 실패
top = list(sales_data.values())
top.sort()
top.reverse()
result = (
    [list(sales_data.items())[i] for i in range(12) 
    if(list(sales_data.values())[i]==top[0])]
)
result

# 정답
high = list(sales_data.keys())[0]
highest_sales = sales_data[high]
for month, sales in sales_data.items():
    if sales > highest_sales:
        high = month
        highest_sales = sales
print(high,highest_sales)