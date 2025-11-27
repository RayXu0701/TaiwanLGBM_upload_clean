import os, certifi
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import pandas as pd
import numpy as np
from datetime import datetime
import time
from scipy import stats
import warnings
import twstock

warnings.filterwarnings("ignore")

path_pc = 'C:/Users/ray92/Desktop/TaiwanLGBM_upload/'
symbols_all50 = ['0050']

def get_twse_ohlcv_monthly(symbols, start_year, start_month, end_year, end_month):
    all_data = []
    for symbol in symbols:
        print(f"Downloading {symbol} monthly data...")
        data = []

        # 只建一次 Stock 物件
        stock = twstock.Stock(symbol)

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # 超出結束年月就停
                if (year > end_year) or (year == end_year and month > end_month):
                    break

                print(f"  Fetching {year}-{month:02d} ...")
                try:
                    monthly_data = stock.fetch(year, month)  
                    time.sleep(2)  

                    if not monthly_data:
                        continue

                    for rec in monthly_data:
                        data.append({
                            'date': rec.date,
                            'symbol': symbol,
                            'open': rec.open,
                            'high': rec.high,
                            'low': rec.low,
                            'close': rec.close,
                            'volume': rec.capacity,
                            'turnover': rec.turnover,
                            'transaction': rec.transaction
                        })
                except Exception as e:
                    print(f"Error: {symbol} {year}-{month:02d}", e)

        df_symbol = pd.DataFrame(data)
        df_symbol.set_index(['symbol', 'date'], inplace=True)
        all_data.append(df_symbol)

    return pd.concat(all_data)


# 輸入抓取區間
start_year, start_month = 2009, 1
end_year, end_month = 2025, 5

df = get_twse_ohlcv_monthly(symbols_all50, start_year, start_month, end_year, end_month)

# 指標計算
def calc_features(df, symbols):
    ma_50 = lambda x: x.rolling(50, min_periods=1).mean()
    ema_50 = lambda x: x.ewm(span=50, min_periods=1).mean()
    macd = lambda x: x.ewm(span=12, min_periods=1).mean() - x.ewm(span=26, min_periods=1).mean()
    macd_signal = lambda x: x.ewm(span=9, min_periods=1).mean()
    bbands50_upper = lambda x: x.rolling(50, min_periods=1).mean() + 2 * x.rolling(50, min_periods=1).std(ddof=0)
    bbands50_lower = lambda x: x.rolling(50, min_periods=1).mean() - 2 * x.rolling(50, min_periods=1).std(ddof=0)
    
    for symbol in symbols:
        arr_close = df.loc[(symbol, slice(None)), 'close']
        df.loc[(symbol, slice(None)), 'ema50'] = ema_50(arr_close)
        df.loc[(symbol, slice(None)), 'macd'] = macd(arr_close)
        df.loc[(symbol, slice(None)), 'macd_signal'] = macd_signal(df.loc[(symbol, slice(None)), 'macd'])
        df.loc[(symbol, slice(None)), 'macd_signal_pct_diff'] = (
            (df.loc[(symbol, slice(None)), 'macd'] - df.loc[(symbol, slice(None)), 'macd_signal']) /
            df.loc[(symbol, slice(None)), 'macd_signal']
        )
        df.loc[(symbol, slice(None)), 'sma50'] = ma_50(arr_close)
        df.loc[(symbol, slice(None)), 'bbands50_upper'] = bbands50_upper(arr_close)
        df.loc[(symbol, slice(None)), 'bbands50_lower'] = bbands50_lower(arr_close)
        
        # RSI
        df_symbol = df.loc[df.index.get_level_values('symbol') == symbol]
        deltas = df_symbol['close'].diff().replace({np.nan: 0.0})
        dUp = deltas.clip(lower=0)
        dDown = deltas.clip(upper=0)
        RolUp = dUp.rolling(14, min_periods=1).mean()
        RolDown = -dDown.rolling(14, min_periods=1).mean()
        rsi = pd.Series(np.zeros(len(deltas)), index=df_symbol.index)
        for i in range(16, len(deltas)):
            RolUp.iloc[i] = (RolUp.iloc[i - 1] * 13 + dUp.iloc[i]) / 14
            RolDown.iloc[i] = (RolDown.iloc[i - 1] * 13 - dDown.iloc[i]) / 14
            rsi.iloc[i] = 100 - 100 / (1 + RolUp.iloc[i] / RolDown.iloc[i]) if RolDown.iloc[i] != 0 else 100
        df.loc[(symbol, slice(None)), 'rsi14'] = rsi
        
        # 其他技術指標
        df.loc[(symbol, slice(None)), 'volume_pct_change_1_day'] = df.loc[(symbol, slice(None)), 'volume'].pct_change(1)
        df.loc[(symbol, slice(None)), 'close_pct_change_5_day'] = df.loc[(symbol, slice(None)), 'close'].pct_change(5)
        df.loc[(symbol, slice(None)), 'intraday_chg'] = (
            (df.loc[(symbol, slice(None)), 'close'] - df.loc[(symbol, slice(None)), 'open']) /
            df.loc[(symbol, slice(None)), 'open']
        )
        df.loc[(symbol, slice(None)), 'log_volume'] = np.log(df.loc[(symbol, slice(None)), 'volume'].replace(0, np.nan))
        std_50 = lambda x: x.rolling(window=50, min_periods=20).std()
        df.loc[(symbol, slice(None)), 'volatility50'] = std_50(df.loc[(symbol, slice(None)), 'close'])
        df.loc[(symbol, slice(None)), 'volatility50_ratio'] = (
            df.loc[(symbol, slice(None)), 'volatility50'] / df.loc[(symbol, slice(None)), 'close']
        )
        zscore_50 = lambda x: (x - x.rolling(window=50, min_periods=1).mean()) / x.rolling(window=50, min_periods=1).std()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df.loc[(symbol, slice(None)), col + '_scaled50'] = zscore_50(df.loc[(symbol, slice(None)), col])
        for t_col in ['ema50', 'sma50']:
            df.loc[(symbol, slice(None)), t_col + '_scaled50'] = zscore_50(df.loc[(symbol, slice(None)), t_col])
        df.loc[(symbol, slice(None)), '(close/ema)-1'] = (df.loc[(symbol, slice(None)), 'close'] / df.loc[(symbol, slice(None)), 'ema50']) - 1
        df.loc[(symbol, slice(None)), '(close/sma)-1'] = (df.loc[(symbol, slice(None)), 'close'] / df.loc[(symbol, slice(None)), 'sma50']) - 1
        df.loc[(symbol, slice(None)), 'upper/close-1'] = (df.loc[(symbol, slice(None)), 'bbands50_upper'] / df.loc[(symbol, slice(None)), 'close']) - 1
        df.loc[(symbol, slice(None)), '1-lower/close'] = 1 - (df.loc[(symbol, slice(None)), 'bbands50_lower'] / df.loc[(symbol, slice(None)), 'close'])
        
        def slope(ts):
            x = np.arange(len(ts))
            mask = ~np.isnan(x) & ~np.isnan(ts)
            regress = stats.linregress(x[mask], ts[mask])
            return regress[0]
        for col in ['bbands50_lower','bbands50_upper','close','ema50','high','low','open','sma50','volume','rsi14']:
            arr = df.loc[(symbol, slice(None)), col]
            df.loc[(symbol, slice(None)), col + '_slope'] = arr.rolling(14, min_periods=1).apply(slope)
        def momentum_score(ts):
            x = np.arange(len(ts))
            log_ts = np.log(ts)
            mask = ~np.isnan(x) & ~np.isnan(log_ts)
            regress = stats.linregress(x[mask], log_ts[mask])
            annualized_slope = (np.power(np.exp(regress[0]), 252) - 1) * 100
            return annualized_slope * (regress[2] ** 2)
        arr = df.loc[(symbol, slice(None)), 'close']
        df.loc[(symbol, slice(None)), 'momentum'] = arr.rolling(50, min_periods=1).apply(momentum_score)
        for r in [1,2,3,4,5,10]:
            df.loc[(symbol, slice(None)),f'return_{r}'] = (arr.shift(-r)-arr) / arr
            df.loc[(symbol, slice(None)),f'log_return_{r}'] = np.log(arr.shift(-r)/arr)
            df.loc[(symbol, slice(None)),f'past_return_{r}'] = (arr-arr.shift(r)) / arr.shift(r)
            df.loc[(symbol, slice(None)),f'past_log_return_{r}'] = np.log(arr / arr.shift(r))
        if 'log_volume' in df.columns:
            df.loc[(symbol, slice(None)), 'log_volume_std50'] = std_50(df.loc[(symbol, slice(None)), 'log_volume'])
    return df

df = calc_features(df, symbols_all50)

# 輸出 CSV
last_date = pd.to_datetime(sorted(list(set(df.index.get_level_values('date'))))[-1])
out_csv = f'outcomes_twse_{last_date.strftime("%Y-%m-%d")}.csv'
df.to_csv(path_pc + out_csv)
print("Done! 資料輸出檔名:", out_csv)
