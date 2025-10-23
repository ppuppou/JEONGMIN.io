import sqlite3
# DB 파일 연결 (없으면 자동 생성됨)
conn = sqlite3.connect("./data/penguins.db")

import pandas as pd
# SELECT 쿼리 결과를 DataFrame으로 읽기
df = pd.read_sql_query("SELECT * FROM penguins;", conn)
print(df.head())