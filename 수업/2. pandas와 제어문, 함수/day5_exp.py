import numpy as np
import pandas as pd

# 연습문제 
# 11
a = np.array([[3,5,7,2,3,6]]).reshape(2,3)

# 12
np.random.seed(2023)
B = np.random.choice(range(1, 11), 20, replace=True).reshape(5, 4)
New_B = B[[1,3,4],:]
# replace= True 는 중복추출을 허용한다는 뜻

# 13
over_3 = B[B[:,2]>3,:]

# 14
over_20 = B[np.sum(B,axis=1)>=20,:]

# 15
np.where(np.mean(B,axis=0)>=5)[0]

# 16
B > 7
B[np.any(B>7, axis=1),:]

# 17
x = np.array([1, 2, 3, 4, 5])  
y = np.array([2, 4, 5, 4, 5]) 
beta1 = np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2)

# 18
X = np.array([[2, 4, 6],
              [1, 7, 2],
              [7, 8, 12]])
y = np.array([[10],
              [5],
              [15]])
beta = (np.linalg.inv(X.transpose()@X)@X.transpose())@y
# transpose는 X.T 해도 가능함

# 판다스 연습문제
df = pd.read_csv('../data/grade.csv')
print(df.head())

# 1 
df.info()

# 2 
df.loc[df['midterm']>=85,:]

# 3
df.sort_values(by='final', ascending=False).head()

# 4
df.groupby('gender')[['midterm','final']].mean()

# 6
Max_row = df.loc[df['assignment'] == df['assignment'].max(),:]
Max_row.info()
min_row = df.loc[df['assignment'].idxmax(),:]
# 다른 해답
Max_row2 = df.loc[df['assignment'].idxmax(),:] 
Max_row2.info()
min_row = df.loc[df['assignment'].idxmin(),:]
Max_row = df.loc[df['assignment'].idxmax(),:]
# 왜 series로 나오나염?

# 10
df.iloc[:,3:6].mean()
df['average'] = ((df['midterm']+df['final']+df['assignment'])/3)
df1 = df.groupby('gender')[['assignment','average','final','midterm']].mean()

# 10 답안
df.iloc[:,3:6].mean()
df['average'] = ((df['midterm']+df['final']+df['assignment'])/3)
df1 = df.groupby('gender',as_index=False)[['assignment','average','final','midterm']].mean()
df1.melt(id_vars='gender',
         var_name='variable',
         value_name='score').sort_values('gender')

# 10 답안 2
df.iloc[:,3:6].mean()
df['average'] = ((df['midterm']+df['final']+df['assignment'])/3)
df1 = df.melt(id_vars=['gender'],value_vars=['midterm','final','assignment','average'],var_name='variable',value_name='score')
df2 = df1.groupby(['gender','variable'],as_index=False).mean()

# 11
df.loc[df['average']==df['average'].max(),['name','average']]
df