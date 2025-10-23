# 파일명: app_multivariate.py

from shiny import reactive
from shiny.express import input, render, ui
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# --- 한글 폰트 설정 ---
import matplotlib as mpl
try:
    mpl.rcParams['font.family'] = 'Malgun Gothic'
except:
    mpl.rcParams['font.family'] = 'AppleGothic'
mpl.rcParams['axes.unicode_minus'] = False

# --- Phase 0: 사전 준비된 다변량 파라미터 로드 ---
try:
    params = joblib.load('multivariate_params.joblib')
    MEAN_VECTOR = params['mean_vector']
    INV_COV_MATRIX = params['inv_cov_matrix']
    UCL = params['ucl']
    NUMERIC_COLS = params['columns']

    test_df = pd.read_csv("test.csv")
    test_df['registration_time'] = pd.to_datetime(test_df['registration_time'])
    test_numeric = test_df[NUMERIC_COLS].ffill().dropna()
    test_time = test_df.loc[test_numeric.index, 'registration_time']

except FileNotFoundError as e:
    ui.h3("🚨 필수 파일이 없습니다!", style="color: red;")
    ui.p(f"에러: '{e.filename}' 파일을 찾을 수 없습니다.")
    ui.p("먼저 'train_multivariate.py' 스크립트를 실행하여 파라미터 파일을 생성해주세요.")
    exit()

# --- 실시간 데이터 스트리밍 관리 클래스 (초고속 리스트 방식) ---
class MultivariateStreamer:
    def __init__(self, numeric_data, time_data):
        self.full_numeric_data = numeric_data
        self.full_time_data = time_data
        self.current_index = 0
        self.streamed_rows = []

    def get_next_data_point(self):
        if self.current_index < len(self.full_numeric_data):
            x_vector = self.full_numeric_data.iloc[self.current_index]
            timestamp = self.full_time_data.iloc[self.current_index]
            diff = x_vector - MEAN_VECTOR
            t_squared = diff.T @ INV_COV_MATRIX @ diff
            is_anomaly = 1 if t_squared > UCL else 0

            self.streamed_rows.append({"time": timestamp, "t_squared": t_squared, "anomaly": is_anomaly})
            self.current_index += 1
            return True
        return None

    def get_current_data_as_df(self):
        if not self.streamed_rows:
            return pd.DataFrame(columns=["time", "t_squared", "anomaly"])
        return pd.DataFrame(self.streamed_rows)

    def reset(self):
        self.current_index = 0
        self.streamed_rows = []

# --- Shiny 앱 UI 설정 ---
ui.page_opts(title="📈 다변량 시계열 관리도", fillable=True)

# --- 반응형 값 초기화 ---
streamer = reactive.Value(MultivariateStreamer(test_numeric, test_time))
current_data = reactive.Value(pd.DataFrame())
is_streaming = reactive.Value(False)

# --- [핵심] UI 레이아웃 (상단 가로 제어판) ---
# 1. 상단에 제어판 카드 배치
with ui.card(height="90px"): # 카드 높이 지정
    with ui.layout_columns(col_widths=[8, 4]):
        # 왼쪽 컬럼에 버튼들 배치
        with ui.div():
            ui.input_action_button("start", "▶ 시작", class_="btn-success")
            ui.input_action_button("pause", "⏸ 일시정지", class_="btn-warning")
            ui.input_action_button("reset", "🔄 리셋", class_="btn-danger")
        # 오른쪽 컬럼에 상태 표시
        with ui.div(style="text-align: right; padding-top: 5px;"):
            @render.ui
            def stream_status():
                if is_streaming():
                    return ui.div("🟢 스트리밍 중", class_="badge bg-success", style="font-size: 1.1rem;")
                return ui.div("🔴 정지됨", class_="badge bg-secondary", style="font-size: 1.1rem;")

# 2. 메인 그래프 카드
with ui.card(full_screen=True):
    ui.card_header("Hotelling T-제곱 실시간 시계열 관리도")
    @render.plot
    def stream_plot():
        df = current_data()
        fig, ax = plt.subplots(figsize=(12, 6))
        if df.empty:
            ax.text(0.5, 0.5, "▶ 시작 버튼을 눌러 스트리밍을 시작하세요", ha="center", va="center", fontsize=15)
            ax.set_xticks([]); ax.set_yticks([])
            return fig

        # 그래프 상한 처리
        display_limit = UCL * 2.0
        df_plot = df.copy()
        clipped_points = df_plot[df_plot["t_squared"] > display_limit]
        df_plot.loc[df_plot["t_squared"] > display_limit, "t_squared"] = display_limit
        
        normal_points = df_plot[df["anomaly"] == 0]
        anomaly_points = df_plot[df["anomaly"] == 1]
        
        # X축을 시간(df['time'])으로 설정
        ax.plot(normal_points['time'], normal_points["t_squared"], marker='o', linestyle='-', color='dodgerblue', label='T² 통계량')
        if not anomaly_points.empty:
            ax.scatter(anomaly_points['time'], anomaly_points["t_squared"], color='red', s=100, zorder=5, label='이상 감지!')
        if not clipped_points.empty:
            ax.scatter(clipped_points['time'], clipped_points["t_squared"], color='magenta', marker='^', s=100, zorder=6, label='차트 범위 초과')

        ax.axhline(UCL, color='red', linestyle='--', label=f'UCL ({UCL:.2f})')
        ax.set_ylim(bottom=-1, top=display_limit * 1.1)
        ax.legend(loc='upper right')
        ax.set_title("실시간 T-제곱 모니터링", fontsize=16)
        ax.set_xlabel("시간 (registration_time)", fontsize=12)
        ax.set_ylabel("T-제곱 값")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig.autofmt_xdate(rotation=25, ha='right')
        plt.tight_layout()
        return fig

# 3. 하단 테이블 카드
with ui.card():
    ui.card_header("최신 T-제곱 값 (10개)")
    @render.ui
    def recent_data_table():
        df = current_data()
        if df.empty: return ui.p("데이터 없음")
        
        display_df = df.copy()
        display_df["anomaly"] = display_df["anomaly"].apply(lambda x: '🔴 이상' if x == 1 else '🟢 정상')
        display_df['time'] = display_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df = display_df.set_index('time')
        
        return ui.HTML(
            display_df[["t_squared", "anomaly"]].tail(10).round(2).style.set_table_styles(
                [dict(selector="th, td", props=[("text-align", "right")])]
            ).to_html(classes="table table-striped")
        )

# --- 서버 로직 (변경 없음) ---
@reactive.effect
@reactive.event(input.start)
def _(): is_streaming.set(True)

@reactive.effect
@reactive.event(input.pause)
def _(): is_streaming.set(False)

@reactive.effect
@reactive.event(input.reset)
def _():
    is_streaming.set(False)
    s = streamer()
    s.reset()
    streamer.set(s)
    current_data.set(pd.DataFrame())

@reactive.effect
def _():
    if not is_streaming(): return
    reactive.invalidate_later(0.5) # 1초 간격
    s = streamer()
    if s.get_next_data_point():
        current_data.set(s.get_current_data_as_df())
    else:
        is_streaming.set(False)
        ui.notification_show("테스트 데이터 스트리밍이 완료되었습니다.", duration=5)