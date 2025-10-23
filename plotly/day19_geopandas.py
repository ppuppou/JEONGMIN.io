import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

lcd_df = pd.read_csv('./seoul_bike.csv')
lcd_df.columns

fig = px.scatter_mapbox(
    lcd_df,
    lat="lat",lon="long",size="LCD거치대수",
    color="자치구",hover_name="대여소명", # 마우스 오버 시 표시한 텍스트
    hover_data={"lat": False, "long": False, "LCD거치대수": True, "자치구": True},
    text="text",zoom=11,height=650,
)
# carto-positron : 무료, 지도 배경 스타일 지정
fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

import pandas as pd
pd.set_option('display.max_columns', None)
import geopandas as gpd
gdf = gpd.read_file("./서울시군구/TL_SCCO_SIG_W.shp")
# 좌표계 정보
print(gdf.crs)

gdf.to_file("./seoul_districts.geojson", driver="GeoJSON")
import json
with open('./seoul_districts.geojson', encoding='utf-8') as f:
    geojson_data = json.load(f)
print(geojson_data.keys())
geojson_data['features']
print(geojson_data['features'][0]['properties'])

agg_df = (lcd_df.groupby("자치구",
as_index=False)["LCD거치대수"].sum())
agg_df.columns = ["자치구", "LCD합계"]
# 컬럼 이름을 GeoJSON과 맞추기
agg_df = agg_df.rename(columns={"자치구": "SIG_KOR_NM"})
print(agg_df.head(2))

import plotly.express as px
fig = px.choropleth_mapbox(
    agg_df,
    geojson=geojson_data,
    locations="SIG_KOR_NM",
    featureidkey="properties.SIG_KOR_NM",
    color="LCD합계",
    color_continuous_scale="Blues",
    mapbox_style="carto-positron",
    center={"lat": 37.5665, "lon": 126.9780},
    zoom=10,
    opacity=0.7,
    title="서울시 자치구별 LCD 거치대 수"
)
fig.update_layout(
    margin={"r": 0, "t": 30, "l": 0, "b": 0}
)
fig.show()

