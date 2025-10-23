import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- 폰트 및 마이너스 부호 설정 ---
# 사용자의 환경에 'Malgun Gothic' 폰트가 설치되어 있어야 합니다.
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
# --------------------------------

# --- 데이터 파일 불러오기 ---
try:
    df = pd.read_csv('./df1_main_data.csv')
except FileNotFoundError:
    print("'df1_main_data.csv' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    exit()
# ---------------------------

# --- 시각화할 컬럼을 여기에서 변경하세요 ---
COLUMN_TO_VISUALIZE = '치안기관'
# 변수 : 세대수, 등록인구, 등록인구(남), 등록인구(여), 한국인, 한국인(남),
#       한국인(여), 외국인, 외국인(남), 외국인(여), 세대당인구, 65세이상고령자,
#       평균연령, 인구밀도 (명/㎢), 면적 (㎢), 치안기관, 유흥업소 수,
#       초등학교 수, 중,고등학교 수, 대학교 수, 가로등 수, 보안등 수,
#       어린이용 CCTV 수, 전체 CCTV수(어린이보호 제외), 안전비상벨 수,
#       상가 수, 요리 주점, 일반 유흥 주점, 입시·교과학원, 생활방범 CCTV 수,
#       기타 CCTV 수, 시설물 CCTV 수, 쓰레기단속 CCTV 수, 범죄발생수(유동인구기준)
# -----------------------------------------

def preprocess_data(dataframe, column_name, keep_zeros=False):
    """데이터 전처리 공통 함수"""
    df_filtered = dataframe[dataframe['행정동'] != '소계'].copy()
    df_filtered[column_name] = pd.to_numeric(df_filtered[column_name], errors='coerce').fillna(0)
    if not keep_zeros:
        # 0 이하의 값은 트리맵 등에서 오류를 일으킬 수 있으므로 제외
        df_filtered = df_filtered[df_filtered[column_name] > 0]
    return df_filtered

def visualize_vertical_bar(dataframe, column_name):
    """
    세로 막대그래프로 행정동별 데이터를 시각화하는 함수 (X축: 행정동)
    """
    df_processed = preprocess_data(dataframe, column_name, keep_zeros=True)
    df_sorted = df_processed.sort_values(by=column_name, ascending=False) # 세로는 보통 큰 값이 앞에 오도록 정렬

    plt.figure(figsize=(max(12, len(df_sorted) * 0.4), 8)) # 너비를 동적으로 조절
    
    bars = plt.bar(df_sorted['행정동'], df_sorted[column_name], color='skyblue')
    
    plt.title(f'대구시 행정동별 {column_name} 현황', fontsize=16, pad=20)
    plt.xlabel('행정동', fontsize=12)
    plt.ylabel(column_name, fontsize=12)
    
    # X축 라벨 90도 회전으로 가독성 확보
    plt.xticks(rotation=90)
    
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout() # 라벨이 잘리지 않도록 레이아웃 조정
    plt.show()
    plt.close()
def visualize_horizontal_bar(dataframe, column_name):
    """
    수평 막대그래프로 행정동별 데이터를 시각화하는 함수 (상위 5개 강조)
    """
    df_processed = preprocess_data(dataframe, column_name, keep_zeros=True)
    df_sorted = df_processed.sort_values(by=column_name, ascending=True)

    top_5_dongs = df_sorted['행정동'][-5:].tolist()
    colors = ['tomato' if dong in top_5_dongs else 'skyblue' for dong in df_sorted['행정동']]

    num_dongs = len(df_sorted['행정동'])
    plt.figure(figsize=(12, max(8, num_dongs * 0.4)))
    
    bars = plt.barh(df_sorted['행정동'], df_sorted[column_name], color=colors)
    
    plt.title(f'대구시 행정동별 {column_name} 현황', fontsize=16, pad=20)
    plt.xlabel(column_name, fontsize=12)
    plt.ylabel('행정동', fontsize=12)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2., f'{int(width)}', 
                 ha='left', va='center')

    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    plt.close()

def visualize_lollipop(dataframe, column_name):
    """
    롤리팝 차트(Lollipop Chart)로 행정동별 데이터를 시각화하는 함수
    """
    df_processed = preprocess_data(dataframe, column_name)
    df_sorted = df_processed.sort_values(by=column_name, ascending=True)

    plt.figure(figsize=(12, max(8, len(df_sorted) * 0.4)))
    
    plt.hlines(y=df_sorted['행정동'], xmin=0, xmax=df_sorted[column_name], color='skyblue', alpha=0.7, linewidth=2)
    plt.scatter(df_sorted[column_name], df_sorted['행정동'], color='dodgerblue', s=100, alpha=0.8, zorder=3)

    plt.title(f'대구시 행정동별 {column_name} 롤리팝 차트', fontsize=16, pad=20)
    plt.xlabel(column_name, fontsize=12)
    plt.ylabel('행정동', fontsize=12)
    
    for index, value in enumerate(df_sorted[column_name]):
        plt.text(value + 0.1, index, str(int(value)), ha='left', va='center', fontsize=9)
        
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    plt.close()

def visualize_vertical_bar_split(dataframe, column_name, n=10, highlight_n=3):
    """
    상위 n개와 하위 n개만 보여주는 세로 막대그래프 (그 안에서 상/하위 highlight_n개 강조)
    """
    df_processed = preprocess_data(dataframe, column_name, keep_zeros=True)
    df_sorted = df_processed.sort_values(by=column_name, ascending=False)
    
    if len(df_sorted) <= n * 2:
        print(f"데이터가 {n*2}개 이하이므로, 일반 강조 그래프를 표시합니다.")
        visualize_vertical_bar(dataframe, column_name, highlight='both', n=highlight_n)
        return

    # --- [수정된 부분] ---
    # 표시할 데이터만 합쳐서 새로운 DataFrame 생성
    df_plot = pd.concat([df_sorted.head(n), df_sorted.tail(n)])

    top_names = df_sorted.head(n)['행정동'].tolist()
    bottom_names = df_sorted.tail(n)['행정동'].tolist()

    top_highlight_names = df_sorted.head(highlight_n)['행정동'].tolist()
    bottom_highlight_names = df_sorted.tail(highlight_n)['행정동'].tolist()

    colors = []
    for dong in df_plot['행정동']:
        if dong in top_highlight_names:
            colors.append('tomato')
        elif dong in top_names:
            colors.append('lightcoral')
        elif dong in bottom_highlight_names:
            colors.append('dodgerblue')
        else:
            colors.append('skyblue')

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(max(10, n * 0.8), 10))
    fig.subplots_adjust(hspace=0.1)

    # 두 subplot에 동일한 데이터를 그리되, y축 범위만 다르게 설정
    ax1.bar(df_plot['행정동'], df_plot[column_name], color=colors)
    ax2.bar(df_plot['행정동'], df_plot[column_name], color=colors)
    
    ax1.set_ylim(bottom=df_sorted.head(n).iloc[-1][column_name] * 0.95)
    ax2.set_ylim(0, df_sorted.tail(n).iloc[0][column_name] * 1.05)
    
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(axis='x', length=0)
    ax2.xaxis.tick_bottom()
    
    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, lw=1)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    
    fig.suptitle(f'대구시 행정동별 {column_name} 상·하위 {n}개 현황', fontsize=16)
    plt.xlabel('행정동', fontsize=12)
    fig.supylabel(column_name, fontsize=12)
    plt.xticks(rotation=90)
    plt.show()
    plt.close()

def visualize_horizontal_bar_split(dataframe, column_name, n=10, highlight_n=3):
    """
    상위 n개와 하위 n개만 보여주는 수평 막대그래프 (그 안에서 상/하위 highlight_n개 강조)
    """
    df_processed = preprocess_data(dataframe, column_name, keep_zeros=True)
    df_sorted = df_processed.sort_values(by=column_name, ascending=True)
    
    if len(df_sorted) <= n * 2:
        print(f"데이터가 {n*2}개 이하이므로, 일반 강조 그래프를 표시합니다.")
        visualize_horizontal_bar(dataframe, column_name, highlight='both', n=highlight_n)
        return
    
    # --- [수정된 부분] ---
    # 표시할 데이터만 합쳐서 새로운 DataFrame 생성
    df_plot = pd.concat([df_sorted.head(n), df_sorted.tail(n)])

    top_names = df_sorted.tail(n)['행정동'].tolist()
    bottom_names = df_sorted.head(n)['행정동'].tolist()

    top_highlight_names = df_sorted.tail(highlight_n)['행정동'].tolist()
    bottom_highlight_names = df_sorted.head(highlight_n)['행정동'].tolist()

    colors = []
    for dong in df_plot['행정동']:
        if dong in top_highlight_names:
            colors.append('tomato')
        elif dong in top_names:
            colors.append('lightcoral')
        elif dong in bottom_highlight_names:
            colors.append('dodgerblue')
        else:
            colors.append('skyblue')

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 10))
    fig.subplots_adjust(wspace=0.05)
    
    # 두 subplot에 동일한 데이터를 그리되, x축 범위만 다르게 설정
    ax1.barh(df_plot['행정동'], df_plot[column_name], color=colors)
    ax2.barh(df_plot['행정동'], df_plot[column_name], color=colors)
    
    ax1.set_xlim(0, df_sorted.head(n).iloc[-1][column_name] + 0.99)
    ax2.set_xlim(df_sorted.tail(n).iloc[0][column_name] * 0.99, df_sorted.tail(n).iloc[-1][column_name] * 1.01)
    ax1.margins(x=0)

    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(axis='y', length=0)
    
    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    
    fig.suptitle(f'대구시 행정동별 {column_name} 상·하위 {n}개 현황', fontsize=16)
    fig.supxlabel(column_name, fontsize=12)
    plt.show()
    plt.close()




# --- 실행하고 싶은 시각화 함수의 주석을 해제하세요 (하나만 선택) ---
# visualize_vertical_bar(df, COLUMN_TO_VISUALIZE)
visualize_horizontal_bar(df, COLUMN_TO_VISUALIZE)
# visualize_lollipop(df, COLUMN_TO_VISUALIZE)
# visualize_vertical_bar_split(df, COLUMN_TO_VISUALIZE, n=10, highlight_n=3)
# visualize_horizontal_bar_split(df, COLUMN_TO_VISUALIZE, n=10, highlight_n=3)
# -----------------------------------------------------------





# def visualize_horizontal_bar_single_ax(dataframe, column_name, n=10, highlight_n=3):
#     """
#     (하나의 그래프 사용) 상위 n개와 하위 n개만 보여주는 수평 막대그래프
#     """
#     df_processed = preprocess_data(dataframe, column_name, keep_zeros=True)
#     df_sorted = df_processed.sort_values(by=column_name, ascending=True)

#     if len(df_sorted) <= n * 2 + 1: # +1 for the gap
#         print(f"데이터가 {n*2+1}개 이하이므로, 일반 강조 그래프를 표시합니다.")
#         # Fallback to a simpler graph if not enough data
#         # For simplicity, let's call a non-split version here if it exists, or just plot all
#         df_plot = df_sorted
#     else:
#         df_top = df_sorted.tail(n)
#         df_bottom = df_sorted.head(n)
#         # 중간 생략을 위한 빈 데이터프레임 생성
#         df_gap = pd.DataFrame([{'행정동': '...', column_name: 0}])
#         df_plot = pd.concat([df_bottom, df_gap, df_top]).reset_index(drop=True)

#     top_highlight_names = df_sorted.tail(highlight_n)['행정동'].tolist()
#     bottom_highlight_names = df_sorted.head(highlight_n)['행정동'].tolist()
    
#     colors = []
#     for dong in df_plot['행정동']:
#         if dong in top_highlight_names:
#             colors.append('tomato')
#         elif dong in df_top['행정동'].tolist():
#             colors.append('lightcoral')
#         elif dong in bottom_highlight_names:
#             colors.append('dodgerblue')
#         elif dong in df_bottom['행정동'].tolist():
#             colors.append('skyblue')
#         else: # '...'에 해당
#             colors.append('white') # 보이지 않게 처리

#     fig, ax = plt.subplots(figsize=(10, 12))
    
#     bars = ax.barh(df_plot['행정동'], df_plot[column_name], color=colors)
    
#     # '...' 라벨의 눈금선 제거
#     gap_index = df_plot[df_plot['행정동'] == '...'].index[0]
#     ax.get_yticklabels()[gap_index].set_fontsize(15)
#     ax.get_yticklabels()[gap_index].set_color('grey')
#     ax.get_yticklines()[gap_index*2].set_visible(False)
#     ax.get_yticklines()[gap_index*2+1].set_visible(False)

#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
    
#     fig.suptitle(f'대구시 행정동별 {column_name} 상·하위 {n}개 현황', fontsize=16)
#     ax.set_xlabel(column_name, fontsize=12)
    
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()
#     plt.close()

# visualize_horizontal_bar_single_ax(df, COLUMN_TO_VISUALIZE, n=10, highlight_n=3)