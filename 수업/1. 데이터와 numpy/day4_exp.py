import numpy as np
# 1
A = np.array([1,2,3,4]).reshape(2,2)
B = np.array([5,6,7,8]).reshape(2,2)
np.matmul(A,B)
B @ A
# A @ B != B @ A 

# 2
a = np.arange(1,7).reshape(2,3)
b = np.arange(7,13).reshape(3,2)
np.matmul(a,b)
a @ b

# 3
a = np.arange(2,6).reshape(2,2)
i = np.eye(2)
a @ i
i @ a
a @ i == i @ a
# ia = ai = a

# 4
a = np.arange(1,5).reshape(2,2)
z = np.array([0,0,0,0]).reshape(2,2)
a @ z

# 5
d = np.array([[2,0],[0,3]])
a = np.arange(4,8).reshape(2,2)
d @ a
a @ d

# 6
a = np.arange(1,7).reshape(3,2)
v = np.array([0.4,0.6]).reshape(2,-1)
a @ v # 가중치

a = np.array([1,2,5,3,4,2,5,6,1]).reshape(3,3)
v = np.array([0.3,0.3,0.4]).reshape(3,1)
a @ v

# 7
A = np.array([1,2,3,4]).reshape(2,2)
B = np.array([5,6,7,8]).reshape(2,2)
C = np.array([9,10,11,12]).reshape(2,2)
T = np.array([[A],[B]])
T @ C # == array([[A @ C] , [B @ C]])

# 8 # 멱등행렬 == 자기 자신을 곱해도 자기 자신이 나오는 행렬
s = np.array([[3,-6],[1,-2]])
s @ s
np.linalg.inv(s) 

# 9 
A = np.array([1,2,3,4]).reshape(2,2)
B = np.array([5,6,7,8]).reshape(2,2)
C = np.array([9,10,11,12]).reshape(2,2)
(A @ B) @ C
A @ B @ C
B @ C @ A
C @ A @ B

# 10
a = np.array([[3,2,-1],[2,-2,4],[-1,0.5,-1]])
b = np.array([1,-2,0]).reshape(3,-1)
inv_a = np.linalg.inv(a)
x = inv_a @ b
a @ x
c = np.array([1,-2,-2]).reshape(3,1)
a @ c