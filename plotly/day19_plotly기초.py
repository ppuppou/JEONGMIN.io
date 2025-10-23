import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 데이터 생성
df = pd.DataFrame({'x': [1, 2, 3, 4],
'y': [1, 4, 9, 16]})
# 산점도 (Scatter Plot) 생성
fig = px.scatter(df,x='x', # X축 데이터 설정
    y='y', # Y축 데이터 설정
    title='Scatter Plot Example', # 그래프 제목 설정
    size_max=10 # 마커 크기 최대값 설정
)
fig.show() # 그래프 출력
# 선그래프
fig = px.line(df, x='x', # X축 데이터 설정
    y='y', # Y축 데이터 설정
    title='Line Plot Example', # 그래프 제목 설정
    markers=True # 마커(점) 추가
)
fig.update_layout(
    title='내마음',
    xaxis_title='X값',
    yaxis_title='Y값 (제곱)',
    width=600, height=400,
    template='plotly_dark',
    paper_bgcolor='lightgray',
    plot_bgcolor='black',
    legend=dict(x=0.5,y=1)
)
fig.show() # 그래프 출력

x = [1, 2, 3, 4]
y = [10, 20, 30, 40]
fig = go.Figure()
fig.add_trace(go.Scatter(x=x,y=y,mode='lines+markers',
    marker=dict(size=10, color="red"),name="Line Plot")
    )
fig.add_annotation(x=2,y=25,text="Important Point",
    showarrow=True,arrowhead=2,font=dict(color="blue", size=14)
    )
fig.show()


t = np.arange(0, 5, 0.5)
fig = go.Figure()
styles = ['solid', 'dash', 'dot', 'dashdot']
colors = ['blue', 'green', 'orange', 'red']
for i, dash_style in enumerate(styles):
    fig.add_trace(
        go.Scatter(
            x=t,y=t + i,mode='lines',
            name=f'dash="{dash_style}"',
            line=dict(dash=dash_style,width=3,color=colors[i])
    )
)
fig.show()

