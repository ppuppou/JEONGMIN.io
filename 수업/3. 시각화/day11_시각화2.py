import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('./data/penguins.csv')
df.info()

# 파이차트
islands = ['Torgersen', 'Biscoe', 'Dream']
island_names_kor = {
    'Torgersen': '토르거센(Torgersen)',
    'Biscoe': '비스코(Biscoe)',
    'Dream': '드림(Dream)'
}

# 종별 색상 정의 (파스텔톤)
species_colors = {
    'Adelie': '#ff9999',    # 연빨강
    'Chinstrap': '#99cc99', # 연녹색
    'Gentoo': '#99ccff'     # 연파랑
}

# 서브플롯 생성
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# 각 섬에 대해 파이차트 생성
for i, island in enumerate(islands):
    island_df = df[df['island'] == island]
    species_counts = island_df['species'].value_counts()
    
    # 순서를 종 이름 기준으로 맞춰서 색상 고정
    labels = species_counts.index.tolist()
    sizes = species_counts.values
    colors = [species_colors[label] for label in labels]
    
    wedges, texts, autotexts = axes[i].pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 20}
    )
    axes[i].set_title(island_names_kor[island], fontsize=16)
    axes[i].axis('equal')

# 전체 제목
fig.suptitle('각 섬별 펭귄 종 서식 현황', fontsize=22)

# 공통 범례 추가 (오른쪽 하단)
handles = [plt.Line2D([0], [0], marker='o', color='w', label=species,
                      markerfacecolor=color, markersize=15)
           for species, color in species_colors.items()]
fig.legend(handles=handles,
           loc='lower right',
           title='펭귄 종',
           fontsize=14,
           title_fontsize=15)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 전체 제목과 범례 공간 확보
plt.show()




# 펭귄 종별로 성비를 나타내는 파이차트 작성
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('penguins.csv')

df = df.dropna(subset=['sex'])

# 종 목록 및 한글 이름 매핑
species_list = ['Adelie', 'Chinstrap', 'Gentoo']
species_names_kor = {
    'Adelie': '아델리(Adelie)',
    'Chinstrap': '친스트랩(Chinstrap)',
    'Gentoo': '젠투(Gentoo)'
}

# ✅ 한글 라벨을 위한 매핑
sex_label_map = {
    'Male': '수컷',
    'Female': '암컷'
}

# ✅ 한글 라벨 기준 색상 정의
sex_colors = {
    '수컷': '#99ccff',    # 연하늘색
    '암컷': '#ffcccc'     # 연분홍
}
# 성별 색상 (젠투-암컷 강조 색 추가)
sex_colors_default = {
    '수컷': '#99ccff',
    '암컷': '#ffcccc'
}
gentoo_female_highlight = '#ff9999'  # 더 진한 핑크
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for i, species in enumerate(species_list):
    species_df = df[df['species'] == species]
    sex_counts_raw = species_df['sex'].value_counts()
    sex_counts = {
        '수컷': sex_counts_raw.get('Male', 0),
        '암컷': sex_counts_raw.get('Female', 0)
    }

    labels = list(sex_counts.keys())
    sizes = list(sex_counts.values())

    if species == 'Gentoo':
        colors = [
            gentoo_female_highlight if label == '암컷' else sex_colors_default[label]
            for label in labels
        ]
        explode = [0.1 if label == '암컷' else 0 for label in labels]
        shadow = True
    else:
        colors = [sex_colors_default[label] for label in labels]
        explode = [0, 0]
        shadow = False

    wedges, texts, autotexts = axes[i].pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        explode=explode,
        shadow=shadow,
        textprops={'fontsize': 20}
    )
    axes[i].set_title(species_names_kor[species], fontsize=16)
    axes[i].axis('equal')

    # 젠투 종에서 암컷 라벨만 스타일 강조
    if species == 'Gentoo':
        for label, text, autotext in zip(labels, texts, autotexts):
            if label == '암컷':
                text.set_fontweight('bold')
                text.set_color('darkred')
                autotext.set_fontweight('bold')
                autotext.set_color('darkred')
            else:
                # 수컷은 기본 스타일 유지 (원하면 조절 가능)
                text.set_fontweight('normal')
                text.set_color('black')
                autotext.set_fontweight('normal')
                autotext.set_color('black')

fig.suptitle('펭귄 종별 성비 현황', fontsize=22)

handles = [plt.Line2D([0], [0], marker='o', color='w', label=sex,
                      markerfacecolor=color, markersize=15)
           for sex, color in sex_colors_default.items()]
fig.legend(handles=handles,
           loc='lower right',
           title='성별 구분',
           fontsize=14,
           title_fontsize=15)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()



# 레이더 차트
df = df.dropna(subset=['bill_depth_mm', 'bill_length_mm', 'body_mass_g', 'flipper_length_mm'])

features = ['bill_depth_mm', 'bill_length_mm', 'body_mass_g', 'flipper_length_mm']
feature_labels = ['부리 깊이', '부리 길이', '몸무게', '날개 길이']

species_list = ['Adelie', 'Chinstrap', 'Gentoo']
colors = {
    'Adelie': '#ff9999',
    'Chinstrap': '#99ff99',
    'Gentoo': '#9999ff'
}

# 종별 평균 계산
grouped = df.groupby('species')[features].mean()

# 각 변수별 3종 평균값 등수 계산 - 큰 값이 1위가 되도록
ranks = grouped.rank(axis=0, method='min', ascending=False)

# 최대 등수 (3)
max_rank = ranks.max().max()

# 점수는 max_rank - rank + 1 형태가 아니라, 그냥 max_rank - rank + 1이므로
# 1등(rank=1) -> 점수 = 3
# 2등(rank=2) -> 점수 = 2
# 3등(rank=3) -> 점수 = 1
scores = max_rank - ranks + 1

num_vars = len(features)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

lines = []
for species in species_list:
    values = scores.loc[species, :].tolist()
    values += values[:1]
    line, = ax.plot(angles, values, color=colors[species], linewidth=2, label=species)
    ax.fill(angles, values, color=colors[species], alpha=0.25)
    lines.append(line)

ax.legend(handles=lines, labels=['아델리', '친스트랩', '젠투'], loc='upper right', fontsize=12)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(feature_labels, fontsize=12)

ax.set_yticks([1, 2, 3])
ax.set_yticklabels(['3등', '2등', '1등'], fontsize=10)

plt.title('펭귄 종별 부리 깊이, 부리 길이, 몸무게, 날개 길이 (등수 기반)', size=16, y=1.1)
plt.show()





df = df.dropna(subset=['bill_length_mm', 'bill_depth_mm', 'body_mass_g', 'flipper_length_mm', 'sex'])

features = ['bill_length_mm', 'bill_depth_mm', 'body_mass_g', 'flipper_length_mm']
feature_labels = ['부리 길이', '부리 깊이', '몸무게', '날개 길이']

species_list = ['Adelie', 'Chinstrap', 'Gentoo']
species_labels = {'Adelie':'아델리', 'Chinstrap':'친스트랩', 'Gentoo':'젠투'}
colors = {'Adelie':'#d9534f', 'Chinstrap':'#5cb85c', 'Gentoo':'#337ab7'}

sex_list = ['Male', 'Female']
sex_labels = {'Male':'수컷', 'Female':'암컷'}

grouped = df.groupby(['sex', 'species'])[features].mean()

ranks = pd.DataFrame(index=grouped.index, columns=features, dtype=float)
for sex in sex_list:
    for feature in features:
        temp = grouped.loc[sex, feature].sort_values(ascending=False)
        rank_scores = temp.rank(method='min', ascending=False)
        max_score = len(rank_scores)
        rank_scores = max_score - rank_scores + 1
        for species in species_list:
            ranks.loc[(sex, species), feature] = rank_scores.get(species, np.nan)

N = len(features)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(1, 2, subplot_kw=dict(polar=True), figsize=(16, 8))
fig.suptitle('펭귄 종별 신체 특성 순위', fontsize=18)

for i, sex in enumerate(sex_list):
    ax = axes[i]
    ax.set_ylim(0, 3.5)  # 0~3.5로 지정해 0부터 그리되 3점까지 커버

    # 내/외부 원 가이드라인 직접 그리기 (0, 1, 2, 3)
    for r in [1, 2, 3]:
        ax.text(np.pi / 4, r, str(r),
                color='gray', fontsize=16, fontweight='bold',
                ha='center', va='bottom')

    ax.set_yticklabels([])  # 축 눈금 숨김
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels, fontsize=14)
    for r in [1, 2, 3]:
        ax.text(np.pi / 2, r, str(r), color='gray', fontsize=12, ha='center', va='bottom')

    # '부리 길이' 라벨 위치 약간 내리기 (필요시 미세 조정 가능)
    labels = ax.get_xticklabels()
    for label in labels:
        if label.get_text() == '부리 길이':
            label.set_y(label.get_position()[1] - 0.1)

    # 수컷/암컷 제목 위치 올리기
    ax.set_title(sex_labels[sex], fontsize=16, pad=30)
    for species in species_list:
        values = ranks.loc[(sex, species), features].tolist()
        values += values[:1]
        ax.plot(angles, values, color=colors[species], linewidth=2, label=species_labels[species], zorder=3)
        ax.fill(angles, values, color=colors[species], alpha=0.4, zorder=2)

        for angle, val in zip(angles, values):
            ax.text(angle, val, f'{int(val)}', fontsize=14, fontweight='bold',
                    ha='center', va='center', color=colors[species], zorder=4)

handles = [plt.Line2D([0], [0], color=colors[sp], lw=4) for sp in species_list]
labels = [species_labels[sp] for sp in species_list]
fig.legend(handles, labels, loc='upper right', fontsize=14, title='종류')

plt.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.show()