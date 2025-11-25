import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import warnings
from yahooquery import Ticker

warnings.filterwarnings("ignore")

# 路徑設定
path_pc = 'C:/Users/ray92/Desktop/TaiwanLGBM_upload/'
symbols_all50 = ['0056.TW']

def get_original_yq(symbols_list, start, end):
    all_result = []
    print("Downloading ORIGINAL price data with yahooquery...")
    for symbol in symbols_list:
        try:
            tq = Ticker(symbol)
            data = tq.history(start=start, end=end)
            if isinstance(data, pd.DataFrame):
                df0 = data.copy()
            else:
                df0 = data.loc[symbol].copy()
            df0 = df0.reset_index()
            df0.columns = [col.lower() for col in df0.columns]
            if 'date' not in df0.columns and 'timestamp' in df0.columns:
                df0['date'] = df0['timestamp']
            df0['symbol'] = symbol
            df0.set_index(['symbol', 'date'], inplace=True)
            # 這裡「不要做任何覆蓋」 open/high/low/close
            all_result.append(df0)
        except Exception as e:
            print(f"Error with {symbol}: {e}")
    return pd.concat(all_result) if all_result else pd.DataFrame()

df = get_original_yq(symbols_all50, start='2003-06-29', end='2025-10-24')
if df.empty:
    raise ValueError("資料下載失敗或Yahoo封鎖！")

print("Downloaded:", df.shape)
print(df.index.names)

# 技術指標 lambda（請以原始close欄為主計算）
ma_50 = lambda x: x.rolling(50, min_periods=1).mean()
ema_50 = lambda x: x.ewm(span=50, min_periods=1).mean()
macd = lambda x: x.ewm(span=12, min_periods=1).mean() - x.ewm(span=26, min_periods=1).mean()
macd_signal = lambda x: x.ewm(span=9, min_periods=1).mean()
bbands50_upper = lambda x: x.rolling(50, min_periods=1).mean() + 2 * x.rolling(50, min_periods=1).std(ddof=0)
bbands50_lower = lambda x: x.rolling(50, min_periods=1).mean() - 2 * x.rolling(50, min_periods=1).std(ddof=0)

for symbol in symbols_all50:
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

def RSI(df_all, symbol, n=14):
    df_symbol = df_all.loc[df_all.index.get_level_values('symbol') == symbol]
    deltas = df_symbol['close'].diff().replace({np.nan: 0.0})
    dUp = deltas.clip(lower=0)
    dDown = deltas.clip(upper=0)
    RolUp = dUp.rolling(n, min_periods=1).mean()
    RolDown = -dDown.rolling(n, min_periods=1).mean()
    rsi = pd.Series(np.zeros(len(deltas)), index=deltas.index)

    for i in range(n + 2, len(deltas)):
        RolUp.iloc[i] = (RolUp.iloc[i - 1] * (n-1) + dUp.iloc[i]) / n
        RolDown.iloc[i] = (RolDown.iloc[i - 1] * (n-1) - dDown.iloc[i]) / n
        rsi.iloc[i] = 100 - 100 / (1 + RolUp.iloc[i] / RolDown.iloc[i]) if RolDown.iloc[i] != 0 else 100
    return rsi

print("Calculating RSI...")
for symbol in symbols_all50:
    df.loc[(symbol, slice(None)), 'rsi14'] = RSI(df, symbol)

for symbol in symbols_all50:
    df.loc[(symbol, slice(None)), 'volume_pct_change_1_day'] = df.loc[(symbol, slice(None)), 'volume'].pct_change(1)
    df.loc[(symbol, slice(None)), 'close_pct_change_5_day'] = df.loc[(symbol, slice(None)), 'close'].pct_change(5)
    df.loc[(symbol, slice(None)), 'intraday_chg'] = (
        (df.loc[(symbol, slice(None)), 'close'] - df.loc[(symbol, slice(None)), 'open']) /
        df.loc[(symbol, slice(None)), 'open']
    )
    # 修正欄位名稱，避免 KeyError
    df.loc[(symbol, slice(None)), 'log_volume'] = np.log(
        df.loc[(symbol, slice(None)), 'volume'].replace(0, np.nan)
    )

    std_50 = lambda x: x.rolling(window=50, min_periods=20).std()
    df.loc[(symbol, slice(None)), 'volatility50'] = std_50(df.loc[(symbol, slice(None)), 'close'])
    df.loc[(symbol, slice(None)), 'volatility50_ratio'] = (
        df.loc[(symbol, slice(None)), 'volatility50'] /
        df.loc[(symbol, slice(None)), 'close']
    )

    zscore_50 = lambda x: (
        (x - x.rolling(window=50, min_periods=1).mean()) /
        x.rolling(window=50, min_periods=1).std()
    )

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df.loc[(symbol, slice(None)), col + '_scaled50'] = zscore_50(df.loc[(symbol, slice(None)), col])

    for t_col in ['ema50', 'sma50']:
        df.loc[(symbol, slice(None)), t_col + '_scaled50'] = zscore_50(df.loc[(symbol, slice(None)), t_col])

    df.loc[(symbol, slice(None)), '(close/ema)-1'] = (
        df.loc[(symbol, slice(None)), 'close'] / df.loc[(symbol, slice(None)), 'ema50']
    ) - 1

    df.loc[(symbol, slice(None)), '(close/sma)-1'] = (
        df.loc[(symbol, slice(None)), 'close'] / df.loc[(symbol, slice(None)), 'sma50']
    ) - 1

    df.loc[(symbol, slice(None)), 'upper/close-1'] = (
        df.loc[(symbol, slice(None)), 'bbands50_upper'] / df.loc[(symbol, slice(None)), 'close']
    ) - 1

    df.loc[(symbol, slice(None)), '1-lower/close'] = 1 - (
        df.loc[(symbol, slice(None)), 'bbands50_lower'] / df.loc[(symbol, slice(None)), 'close']
    )

    def slope(ts):
        x = np.arange(len(ts))
        mask = ~np.isnan(x) & ~np.isnan(ts)
        regress = stats.linregress(x[mask], ts[mask])
        return regress[0]

    for col in [
        'bbands50_lower', 'bbands50_upper', 'close', 'ema50', 'high',
        'low', 'open', 'sma50', 'volume', 'rsi14'
    ]:
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

    # 回報率相關特徵
    for r in [1, 2, 3, 4, 5, 10]:
        df.loc[(symbol, slice(None)), f'return_{r}'] = (arr.shift(-r) - arr) / arr
        df.loc[(symbol, slice(None)), f'log_return_{r}'] = np.log(arr.shift(-r) / arr)
        df.loc[(symbol, slice(None)), f'past_return_{r}'] = (arr - arr.shift(r)) / arr.shift(r)
        df.loc[(symbol, slice(None)), f'past_log_return_{r}'] = np.log(arr / arr.shift(r))

# 修正「log volume」→「log_volume」欄位下游使用
for symbol in symbols_all50:
    if 'log_volume' in df.columns:
        std_50 = lambda x: x.rolling(50, min_periods=20).std()
        df.loc[(symbol, slice(None)), 'log_volume_std50'] = std_50(df.loc[(symbol, slice(None)), 'log_volume'])

# 輸出 CSV
last_date = pd.to_datetime(sorted(list(set(df.index.get_level_values('date'))))[-1])
out_csv = f'outcomes_original_{last_date.strftime("%Y-%m-%d")}.csv'
df.to_csv(path_pc + out_csv)

print("Done! 資料輸出檔名:", out_csv)
