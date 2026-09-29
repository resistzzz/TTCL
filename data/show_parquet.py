import pandas as pd
import pprint

# df = pd.read_parquet("data/dapo17k/train.parquet")

df = pd.read_parquet("data/amc23/amc23_placeholder_conf.parquet")

# 展示总行数
print(f"Total rows: {len(df)}")

# 展示指定行
row_id = 3
print(f"\nRow {row_id}:")
pprint.pprint(df.iloc[row_id].to_dict(), width=120)