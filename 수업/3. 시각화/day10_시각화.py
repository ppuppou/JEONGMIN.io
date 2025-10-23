import matplotlib.pyplot as plt
plt.plot([4,1,3,2])
plt.ylabel('Some Numbers')
plt.show()

plt.plot([4,1,3,2], marker='x', # 점을 표현하고(o는 점, x는 x, ^는 세모...)
         linestyle='None')      # 선은 지워라
plt.ylabel('Some Numbers')
plt.show()

# x-y plot | 산점도 
plt.plot([1,2,3,4],[1,4,9,16], # x와 y를 둘다 지정
         marker='o', 
         linestyle='None'
         )
plt.show()

import pandas as pd
import numpy as np
df = pd.read_csv('./data/penguins.csv')
df.info()
plt.plot(np.arange(10),
         np.arange(10),
         marker='o',
         linestyle='None')
plt.plot(df['bill_length_mm'],df['bill_depth_mm'],
         marker='o', linestyle='None', color = 'green')
plt.ylabel('depth')
plt.xlabel('length')
plt.show()

# 산점도 간소화
plt.scatter(df['bill_length_mm'],df['bill_depth_mm'], #c = 'red',
            c = np.repeat('red',344)) # 각 점에 대한 정보가 다 있어야함
plt.ylabel('depth')
plt.xlabel('length')
plt.show()

np.repeat('green',100)
np.repeat('red',244)
mycolor = np.concatenate([np.repeat('green',100),np.repeat('red',244)])
plt.scatter(df['bill_length_mm'],df['bill_depth_mm'],
            c = mycolor) # 각 점에 대한 색 벡터가 존재한다
plt.show()

df['color'] = 'blue'
df.loc[df['species']=='Adelie','color'] = 'red'
df.loc[df['species']=='Gentoo','color'] = 'green'
df.head()
plt.scatter(df['bill_length_mm'],df['bill_depth_mm'],
            c = df['color'])
plt.show()

plt.scatter(df['bill_length_mm'],df['bill_depth_mm'],
            c = np.arange(0,344))
plt.show()

df['species'] = df['species'].astype('category')
df.info()
df['species'].cat.codes
plt.scatter(df['bill_length_mm'],df['bill_depth_mm'],
            c = df['species'].cat.codes)
plt.show()

# 범주형 데이터 시각화
names = ['A', 'B', 'C'] # 범주 정보
values = [1, 10, 100]
plt.figure(figsize=(9, 3)) # 가로9, 세로3 도화지 만들어
plt.subplot(131) # 1:세로 1등분 3: 가로 3등분 1: 1번째
plt.bar(names, values)  # 막대 그래프
plt.subplot(132)
plt.scatter(names, values)  # 산점도
plt.subplot(133)
plt.plot(names, values)  # 선 그래프
plt.suptitle('Categorical Plotting')
plt.show()

# 선 속성 제어
plt.plot([1, 2, 3, 4], 
         [1, 4, 9, 16], 
         linewidth=2, # 선 두께 
         color='r', # 'r'은 빨강
         linestyle='--') # '-', '--', ':'
plt.show()

# 텍스트 추가 및 주석 처리
plt.plot([1, 2, 3, 4], [10, 20, 30, 40])
plt.text(2, 25, # 특정 위치에 텍스트 추가
        'Important Point', 
        fontsize=12, 
        color='red')
plt.show()

# 범례 지정
plt.plot([1, 2, 3, 4], 
         [1, 4, 9, 16], 
        label="y = x^2") 
plt.legend(loc="upper left")
plt.show()

# 실습
# 종별 부리길이 평균 막대그래프
df = pd.read_csv('./data/penguins.csv')
df.info()
mean_bill_length = df.groupby('species')['bill_length_mm'].mean().reset_index()
plt.bar(mean_bill_length['species'],mean_bill_length['bill_length_mm'])
plt.xlabel('species')
plt.ylabel('mean bill length')
plt.title('species vs mean bill length')
plt.show

# 섬별 몸무게 평균 막대그래프

# 한글 폰트 설정: 맑은 고딕
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 음수 부호 깨짐 방지

# 몸무게 평균 계산
mean_mass = df.groupby('island')['body_mass_g'].mean().sort_values(ascending=True)

# 색상 지정: 가장 큰 값은 빨간색, 나머지는 회색
min_island = mean_mass.idxmin()
colors = ['blue' if island == min_island else 'gray' for island in mean_mass.index]

# 막대그래프 그리기
plt.figure(figsize=(8, 6))
bars = plt.bar(mean_mass.index, mean_mass.values, color=colors)

# y축 숫자 표시
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 10,
             f'{height:.1f}g', ha='center', va='bottom', fontsize=11)

# 축 이름 한글로
plt.xlabel('섬 이름', fontsize=13)
plt.ylabel('몸무게 평균 (g)', fontsize=13)
plt.title('섬별 펭귄 몸무게 평균', fontsize=14)
plt.tight_layout()
plt.show()




# 펭귄 종별 부리길이 VS 부리깊이 산점도
# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 종별 색상 매핑
color_map = {
    'Adelie': 'red',
    'Chinstrap': 'gray',
    'Gentoo': 'gray'
}

# 유효 데이터만 필터링
scatter_df = df.dropna(subset=['species', 'bill_length_mm', 'bill_depth_mm'])

# 산점도 그리기
plt.figure(figsize=(9, 6))
adelie_df = scatter_df[scatter_df['species'] == 'Adelie']
plt.scatter(adelie_df['bill_length_mm'],
            adelie_df['bill_depth_mm'],
            label='Adelie',
            color='red',
            alpha=0.6)

# 나머지 종 묶어서 기타 처리
other_species = scatter_df[scatter_df['species'] != 'Adelie']
plt.scatter(other_species['bill_length_mm'],
            other_species['bill_depth_mm'],
            label='기타',
            color='gray',
            alpha=0.6)

# 평균 중심점 계산 및 표시
mean_df = scatter_df.groupby('species')[['bill_length_mm', 'bill_depth_mm']].mean()

for species, row in mean_df.iterrows():
    x, y = row['bill_length_mm'], row['bill_depth_mm']
    
    # 중심점 X 마커
    plt.scatter(x, y, color=color_map[species], edgecolor='black', s=100, marker='X', zorder=5)
    
    # 화살표 + 텍스트
    plt.annotate(
        f'평균 부리길이: {x:.2f} mm\n평균 부리깊이: {y:.2f} mm',
        xy=(x, y),                  # 중심점
        xytext=(x + 2, y + 1),      # 텍스트 위치 (조정 가능)
        textcoords='data',
        fontsize=10,
        ha='left',
        va='center',
        arrowprops=dict(arrowstyle='->', color='black')
    )

# 라벨 설정
plt.xlabel('부리 길이 (mm)', fontsize=12)
plt.ylabel('부리 깊이 (mm)', fontsize=12)
plt.title('펭귄 종별 부리길이 vs 부리깊이 산점도', fontsize=14)
plt.legend(title='펭귄 종', loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()



# 연속적인 값을 가지는 데이터를 쪼개서 범주형 변수로 변환시키는 테크닉
min_weight = 2700.0
max_weight = 6300.0

# 구간 경계 계산
bins = [min_weight,                         # 2700.0
        min_weight + (max_weight - min_weight) / 3,  # 3900.0
        min_weight + 2 * (max_weight - min_weight) / 3,  # 5100.0
        max_weight]                         # 6300.0

# 라벨
labels = ['low', 'middle', 'high']

# 범주형 변수 생성 (결측치 제외 자동)
df['body_mass_cat'] = pd.cut(df['body_mass_g'], bins=bins,
                              labels=labels, include_lowest=True)
# 각 범주별 마리 수 계산
mass_counts = df['body_mass_cat'].value_counts().sort_index()

print(mass_counts)

mass_counts.plot(kind='bar', color='skyblue',
                 edgecolor='black', figsize=(6,4),
                 alpha=0.5)
plt.xlabel('몸무게 구간', fontsize=12)
plt.ylabel('펭귄 마리 수', fontsize=12)
plt.title('몸무게 구간별 펭귄 마리 수', fontsize=14)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()











import numpy as np
import pandas as pd
import nycflights13 as flights

# 항공편 데이터 (main dataset)
df_flights = flights.flights
df_airlines = flights.airlines
df_airports = flights.airports
df_planes = flights.planes
df_weather = flights.weather

# 필터링할 항공사 리스트
airlines_filter = ['AA', 'AS', 'B6', 'DL', 'HA', 'OO', 'UA', 'US', 'WN']

# loc와 isin을 사용한 필터링
df_filtered = df_flights.loc[df_flights['carrier'].isin(airlines_filter)]

# tailnum를 기준으로 병합
merged1 = pd.merge( df_filtered,
                    df_planes,
                    on='tailnum',
                    how="left"
                    )

# 제조년도에 따른 구분을 위한 함수 설정
def classify(year):
    if pd.isna(year):
        return 'unknown'
    elif year > 1993:
        return 'after_1993'
    else:
        return 'before_1993'

# 제조년도 분류 함수를 적용한 후 원 데이터에 추가
merged1['era'] = merged1['year_y'].apply(classify)

# 데이터 그룹화
result = merged1.groupby(['carrier','tailnum','era']
                        ).size().reset_index(name='flight_count')
result.head()

# 데이터 longterm으로 변환
pivot_result1 = result.pivot_table(index='carrier',
                            columns='era',
                            values='flight_count',
                            aggfunc='sum'  # 중복시 합계
                            ).fillna(0).astype(int)
total = pivot_result1["after_1993"] + pivot_result1["before_1993"] + pivot_result1["unknown"]

# 백분율 변환
pivot_result1['after%'] = pivot_result1["after_1993"]*100/total
pivot_result1['before%'] = pivot_result1["before_1993"]*100/total
pivot_result1['unknown%'] = pivot_result1["unknown"]*100/total
pivot_result1 = pivot_result1.round(2)
pivot_result1



# 1 
plot_data_count = pivot_result1[['before_1993', 'after_1993', 'unknown']]

ax = plot_data_count.plot(kind='bar',
                         stacked=False,
                         color=['indianred', 'skyblue', 'darkgray'],
                         figsize=(10, 6),
                         edgecolor='black',
                         width=0.9)  # 막대폭 0.9 (기본 0.8, 1.0이 최대)

plt.ylabel('운행 횟수 (flight count)', fontsize=12)
plt.xlabel('항공사 (carrier)', fontsize=12)
plt.title('항공사별 1993년 기준 기체 운항 횟수', fontsize=14)
plt.legend(['1993년 이전', '1993년 이후', '제조년도 미상'], loc='upper right')

plt.xticks(rotation=0)

for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.text(p.get_x() + p.get_width()/2, height + 50, f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# 2
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 항공사 순서 고정
plot_data = pivot_result1[['before%', 'after%', 'unknown%']]

# 가로 100% 누적 막대그래프
ax = plot_data.plot(kind='barh',
                    stacked=True,
                    color=['indianred', 'skyblue', 'darkgray'],
                     figsize=(10, 6),
                    edgecolor='black')

plt.xlabel('비율 (%)', fontsize=12)
plt.ylabel('항공사 (carrier)', fontsize=12)
plt.title('항공사별 1993년 기준 기체 운항 비율', fontsize=14)
plt.legend(['1993년 이전', '1993년 이후', '제조년도 미상'], loc='lower right')
plt.grid(axis='x', linestyle='--', alpha=1)

# 각 막대에 비율 표시 + 임계값 이하 화살표 표시
threshold = 5  # 임계값 (%)
for i, (before, after, unknown) in enumerate(zip(plot_data['before%'], plot_data['after%'], plot_data['unknown%'])):
    y_pos = i
    cum = 0

    for value, color, label in zip([before, after, unknown],
                                   ['indianred', 'skyblue', 'darkgray'],
                                   ['before%', 'after%', 'unknown%']):
        x_pos = cum + value / 2
        if value >= threshold:
            ax.text(x_pos, y_pos, f'{value:.1f}%', ha='center', va='center',color='black', fontsize=9)
        else:
            # 바깥쪽에 화살표로 값 표시
            ax.annotate(f'{value:.1f}%',
                        xy=(cum + value, y_pos),               # 화살표 시작점 (막대 끝)
                        xytext=(cum + value + 3, y_pos),       # 텍스트 위치
                        textcoords='data',
                        arrowprops=dict(arrowstyle="->", color='black', lw=1),
                        ha='left', va='center',
                        fontsize=9)
        cum += value
plt.tight_layout()
plt.show()



