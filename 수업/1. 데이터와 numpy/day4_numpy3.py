import numpy as np
import matplotlib.pyplot as plt
# 난수 생성하여 3x3 크기의 행렬 생성
np.random.seed(2024)
img1 = np.random.rand(3, 3) # randum.rand : 무작위로 0 ~ 1까지 
img1

# 행렬 이미지
plt.figure(figsize=(10, 5))  # (가로, 세로) 크기 설정
plt.imshow(img1, cmap='gray', interpolation='nearest');
plt.colorbar();
plt.show();

img_mat = np.loadtxt('./data/img_mat.csv', delimiter=',', skiprows=1)
img_mat.shape
np.max(img_mat)
img_mat.min()
# 행렬 값을 0과 1 사이로 변환
img_mat = img_mat / 255.0
# 밝기를 올리려면 전체에 0.2를 더하고 1이상 값을 가지는 애들을 1로 변환해야함
img_mat = img_mat + 0.2
img_mat[img_mat > 1.0] = 1
# 행렬을 이미지로 변환하여 출력
plt.imshow(img_mat, cmap='gray', interpolation='nearest');
plt.colorbar();
plt.show();

# 행렬 연산
# 행렬 뒤집기 : transpose()
x = np.arange(2, 11).reshape((5, 2)) * 2
print("원래 행렬 x:\n", x)
transposed_x = x.transpose()
print("전치된 행렬 x:\n", transposed_x)

# 행렬 곱셈 
# 곱셉은 앞 행렬의 열과 뒷 행렬의 행의 크기가 맞아야 가능
y = np.arange(1, 7).reshape((2, 3))
print("행렬 y:\n", y)
# 행렬 x, y의 크기: (5, 2) (2, 3)
dot_product = x.dot(y)
# or, np.matmul(x, y) or, x @ y
# x * y 는 원소별 곱셈. 행렬곱과는 다름
dot_product

mat_a = np.array([[1,2],
                  [4,3]])
mat_b = np.array([[2,1],
                  [3,1]])
mat_a.dot(mat_b)

# 행렬의 역행렬 : np.linalg.inv() 
inv_a = np.linalg.inv(mat_a)
mat_a @ inv_a
mat_a @ np.eye(2)
np.eye(3) # 3 by 3 짜리 단위행렬
# 행렬은 역행렬이 존재하는 행렬과, 존재하지 않는 행렬로 나뉨
# non-singular vs. singular
# 역행렬이 존재하지 않는 행렬은, 각각의 열에 사칙연산을 적용해서 
# 다른 열을 만들 수 있음. 즉, 역행렬이 존재하려면 각각의 열이 선형독립이어야함

# 고차원행렬
mat1 = np.arange(1, 7).reshape(2, 3)
mat2 = np.arange(7, 13).reshape(2, 3)
# 3차원 배열로 합치기
my_array = np.array([mat1, mat2])
my_array.shape
my_array[:,1,:]
my_array.reshape(3,2,2)

# 이미지 가져오기
import imageio # 이미지 읽기
jelly = imageio.imread("./data/cat.png")
type(jelly)
jelly.shape

import matplotlib.pyplot as plt
plt.imshow(jelly);
plt.axis('off');
plt.show();
# 흑백으로 변환
bw_jelly = np.mean(jelly[:, :, :3], axis=2) # axis=2는 
plt.imshow(bw_jelly, cmap='gray');          # 채널 축(rgb축)을 따라
plt.axis('off');                            # 평균을 계산한다는 뜻
plt.show();                                 # >> 색을 2가지로 좁히겠다
