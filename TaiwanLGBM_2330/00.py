from FinMind.data import DataLoader
import pandas as pd

api = DataLoader()

df_div = api.taiwan_stock_dividend_result(
    stock_id="2330",
    start_date="2000-05-01",
    end_date="2025-05-31",
)
df_div["adj_factor"] = df_div["after_price"] / df_div["before_price"]
print(df_div[["date", "before_price", "after_price", "adj_factor"]].head())


print(df_div.head())
print(df_div.columns)
