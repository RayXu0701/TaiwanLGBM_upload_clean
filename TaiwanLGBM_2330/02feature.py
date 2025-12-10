import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")

path_pc = 'C:/Users/ray92/Desktop/TaiwanLGBM_upload/'
symbols_all50 = [2330]

step1_file = 'outcomes_twse_2025-05-29.csv'
print(f"載入資料: {step1_file}")
df = pd.read_csv(path_pc + step1_file, index_col=[0, 1])
df.index.names = ['symbol', 'date']

print("前 5 筆 index：")
print(df.index[:5])
print("所有 symbol 值：", sorted(set(df.index.get_level_values('symbol'))))

# 只保留目標標的
df = df[df.index.get_level_values('symbol').isin(symbols_all50)]

# 刪掉每檔一開始 open/close 還是 NaN 前面的列
for symbol in symbols_all50:
    if symbol in df.index.get_level_values('symbol'):
        this_df = df.loc[symbol]
        fvi = this_df[['open', 'close']].first_valid_index()
        if fvi is not None:
            drop_idx = this_df.loc[:fvi].index
            df = df.drop(drop_idx, errors='ignore')

valid_symbols = sorted(set(df.index.get_level_values('symbol')))
print(f"過濾後的台股清單: {valid_symbols}")

# ========= 多 N、多 gate 標籤 =========
N_list = [3, 5, 7, 9]
gate_list = [0.03, 0.05, 0.07]   # 三組門檻

for N in N_list:
    future_max_col = f'future_max_return_{N}'
    future_min_col = f'future_min_return_{N}'
    if future_max_col not in df.columns or future_min_col not in df.columns:
        print(f"補充計算 {future_max_col} / {future_min_col} ...")
        df[future_max_col] = np.nan
        df[future_min_col] = np.nan
        for symbol in valid_symbols:
            symbol_df = df.loc[df.index.get_level_values('symbol') == symbol].copy()
            closes = symbol_df['close'].values
            for i, idx in enumerate(symbol_df.index):
                if i + N < len(closes):
                    future_window = closes[i+1:i+1+N]
                    this_close = closes[i]
                    future_max = (np.max(future_window) - this_close) / this_close
                    future_min = (np.min(future_window) - this_close) / this_close
                    df.loc[idx, future_max_col] = future_max
                    df.loc[idx, future_min_col] = future_min

    for gate in gate_list:
        up_gate, down_gate = gate, -gate
        label_col = f'label_N{N}_G{int(gate*100)}'
        df[label_col] = 0
        df.loc[df[future_max_col] > up_gate, label_col] = 1
        df.loc[df[future_min_col] < down_gate, label_col] = -1
        print(f"\n--- N={N}, gate={gate*100:.1f}% 標籤分布 ---")
        print(df[label_col].value_counts(normalize=True))

# =========== 以下特徵工程區塊（全保留，欄位全用 log_volume） ===========

def wwma(values, n):
    return values.ewm(alpha=1 / n, adjust=False).mean()

def atr(df, symbol, n=14):
    df_symbol = df.loc[df.index.get_level_values('symbol') == symbol].copy()
    high = df_symbol['high']
    low = df_symbol['low']
    close = df_symbol['close']
    tr = pd.concat([
        abs(high - low),
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    return wwma(tr, n)

for symbol in valid_symbols:
    df.loc[(symbol, slice(None)), 'atr14'] = atr(df, symbol, n=14)
    df.loc[(symbol, slice(None)), 'atr10'] = atr(df, symbol, n=10)
    df.loc[(symbol, slice(None)), 'atr20'] = atr(df, symbol, n=20)
    df.loc[(symbol, slice(None)), 'atr100'] = atr(df, symbol, n=100)

df['atr10/atr100'] = df['atr10'] / df['atr100']
df['atr10/atr20'] = df['atr10'] / df['atr20']

for symbol in valid_symbols:
    df.loc[(symbol, slice(None)), 'delta_atr10/atr100_10'] = (
        df.loc[(symbol, slice(None)), 'atr10/atr100']
        - df.loc[(symbol, slice(None)), 'atr10/atr100'].shift(10)
    )
    df.loc[(symbol, slice(None)), 'delta_atr10/atr100_3'] = (
        df.loc[(symbol, slice(None)), 'atr10/atr100']
        - df.loc[(symbol, slice(None)), 'atr10/atr100'].shift(3)
    )

std_func = lambda x, n: x.rolling(n, min_periods=1).std()
for symbol in valid_symbols:
    df.loc[(symbol, slice(None)), 'volatility5'] = std_func(df.loc[(symbol, slice(None)), 'close'], 5)
    df.loc[(symbol, slice(None)), 'volatility10'] = std_func(df.loc[(symbol, slice(None)), 'close'], 10)
    df.loc[(symbol, slice(None)), 'volatility25'] = std_func(df.loc[(symbol, slice(None)), 'close'], 25)
    df.loc[(symbol, slice(None)), 'volatility100'] = std_func(df.loc[(symbol, slice(None)), 'close'], 100)

df['volatility5_ratio'] = df['volatility5'] / df['close']
df['volatility10_ratio'] = df['volatility10'] / df['close']
df['volatility25_ratio'] = df['volatility25'] / df['close']
df['volatility100_ratio'] = df['volatility100'] / df['close']

def momentum_score(ts):
    x = np.arange(len(ts))
    log_ts = np.log(ts)
    mask = ~np.isnan(x) & ~np.isnan(log_ts)
    regress = stats.linregress(x[mask], log_ts[mask])
    annualized_slope = (np.power(np.exp(regress[0]), 252) - 1) * 100
    return annualized_slope * (regress[2] ** 2)

momentum_windows = [(5, 3), (10, 7), (25, 20), (50, 40), (100, 70)]
for window, min_mom in momentum_windows:
    for symbol in valid_symbols:
        arr = df.loc[(symbol, slice(None)), 'close']
        df.loc[(symbol, slice(None)), f'momentum_{window}'] = (
            arr.rolling(window, min_periods=min_mom).apply(momentum_score)
        )

ema_40 = lambda x: x.ewm(span=40, min_periods=1).mean()
ema_80 = lambda x: x.ewm(span=80, min_periods=1).mean()
for symbol in valid_symbols:
    df.loc[(symbol, slice(None)), 'ema40'] = ema_40(df.loc[(symbol, slice(None)), 'close'])
    df.loc[(symbol, slice(None)), 'ema80'] = ema_80(df.loc[(symbol, slice(None)), 'close'])

df['bull'] = df['ema40'] > df['ema80']
df['bull_ratio'] = df['ema40'] / df['ema80'] - 1

max_50 = lambda x: x.rolling(50, min_periods=1).max()
for symbol in valid_symbols:
    df.loc[(symbol, slice(None)), '50d_high'] = max_50(df.loc[(symbol, slice(None)), 'close'])
    df.loc[(symbol, slice(None)), '50d_high_volume'] = max_50(df.loc[(symbol, slice(None)), 'volume'])

df['close>50d_high'] = df['close'] >= df['50d_high']
df['volume_vs_50d_high'] = df['volume'] / df['50d_high_volume']
df['close_vs_50d_high'] = df['close'] / df['50d_high']

zscore_50 = lambda x: (x - x.rolling(50, min_periods=1).mean()) / x.rolling(50, min_periods=1).std()
scale_cols = [
    'log_volume', 'volume_pct_change_1_day', 'volume_pct_change_5_day', 'past_return_1', 'past_return_2',
    'past_return_3', 'past_return_4', 'past_return_5', 'past_return_10', 'volatility50', 'delta_atr10/atr100_3',
    'delta_atr10/atr100_10', 'atr10', 'atr14', 'atr20', 'atr100', 'atr10/atr100', 'atr10/atr20'
]
for col in scale_cols:
    if col in df.columns:
        for symbol in valid_symbols:
            arr = df.loc[(symbol, slice(None)), col]
            df.loc[(symbol, slice(None)), col + '_scaled50'] = zscore_50(arr)

# ========= 修正版 MMI =========
def mmi(window_values):
    closes = pd.Series(window_values)
    if len(closes) < 2:
        return np.nan
    m = closes.median()
    nh = 0
    nl = 0
    for i in range(1, len(closes)):
        if closes.iloc[i] > m and closes.iloc[i] > closes.iloc[i - 1]:
            nl += 1
        elif closes.iloc[i] < m and closes.iloc[i] < closes.iloc[i - 1]:
            nh += 1
    return 100 * (nl + nh) / (len(closes) - 1)

for symbol in valid_symbols:
    arr = df.loc[(symbol, slice(None)), 'close']
    df.loc[(symbol, slice(None)), 'mmi50'] = arr.rolling(50, min_periods=2).apply(mmi, raw=True)

ema_std50 = lambda x: x.ewm(span=50, min_periods=20).std()
for symbol in valid_symbols:
    df.loc[(symbol, slice(None)), 'past_return_1_ema_std50'] = ema_std50(df.loc[(symbol, slice(None)), 'past_return_1'])
    df.loc[(symbol, slice(None)), 'price_chg_1'] = (
        df.loc[(symbol, slice(None)), 'close'] - df.loc[(symbol, slice(None)), 'close'].shift(1)
    )
    df.loc[(symbol, slice(None)), 'price_chg_1_ema_std50'] = ema_std50(df.loc[(symbol, slice(None)), 'price_chg_1'])

df['past_return_1_ema_std50*2.2'] = df['past_return_1_ema_std50'] * 2.2
df['price_chg_1_ema_std50*2.2'] = df['price_chg_1_ema_std50'] * 2.2 / 50

std_50 = lambda x: x.rolling(50, min_periods=20).std()
for symbol in valid_symbols:
    df.loc[(symbol, slice(None)), 'volume_std50'] = std_50(df.loc[(symbol, slice(None)), 'volume'])
    df.loc[(symbol, slice(None)), 'log_volume_std50'] = std_50(df.loc[(symbol, slice(None)), 'log_volume'])
    df.loc[(symbol, slice(None)), 'past_return_5_std50'] = std_50(df.loc[(symbol, slice(None)), 'past_return_5'])
    df.loc[(symbol, slice(None)), 'past_return_4_std50'] = std_50(df.loc[(symbol, slice(None)), 'past_return_4'])
    df.loc[(symbol, slice(None)), 'past_return_3_std50'] = std_50(df.loc[(symbol, slice(None)), 'past_return_3'])
    df.loc[(symbol, slice(None)), 'past_return_2_std50'] = std_50(df.loc[(symbol, slice(None)), 'past_return_2'])
    df.loc[(symbol, slice(None)), 'past_return_1_std50'] = std_50(df.loc[(symbol, slice(None)), 'past_return_1'])
    df.loc[(symbol, slice(None)), 'past_return_10_std50'] = std_50(df.loc[(symbol, slice(None)), 'past_return_10'])

df['target_upper'] = df['close'] * (1 + df['past_return_5_std50'])
df['target_lower'] = df['close'] * (1 - df['past_return_5_std50'])
df['target_upper_v2'] = df['close'] * (1 + df['past_return_1_ema_std50*2.2'])
df['target_lower_v2'] = df['close'] * (1 - df['past_return_1_ema_std50*2.2'])

# ======== 結果儲存 ========
last_date = pd.to_datetime(sorted(list(set(df.index.get_level_values('date'))))[-1])
out_csv = f'outcomes_new_features_{last_date.strftime("%Y-%m-%d")}_multiG.csv'
full_path = os.path.join(path_pc, out_csv)

try:
    df.to_csv(full_path)
    print("Done! STEP2完成（多Nx多G標籤與全特徵）, 輸出檔名:", out_csv)
except PermissionError:
    print("【權限錯誤】請確認該CSV未被Excel等程式佔用後再執行！")

