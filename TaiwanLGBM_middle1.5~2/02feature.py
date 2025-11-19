import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")
path_pc = 'C:/Users/ray92/Desktop/TaiwanLGBM_upload/'


# 九支中波動股票
symbols_middle_9 = ['2886.TW', '2317.TW', '1216.TW', '2395.TW', '1101.TW', '2382.TW', '4958.TW', '2356.TW', '3711.TW']

# 載入step1匯出檔案（請確認用對檔案版本）
step1_file = 'outcomes_middle_features_2025-10-23.csv'
print(f"載入資料: {step1_file}")
df = pd.read_csv(path_pc + step1_file, index_col=[0,1])
df.index.names = ['symbol','date']
df = df[df.index.get_level_values('symbol').isin(symbols_middle_9)]

valid_symbols = sorted(set(df.index.get_level_values('symbol')))
print(f"過濾後的台股清單: {valid_symbols}")

# N日最大漲跌幅
N = 10
percentile_upper = 0.7
percentile_lower = 0.3

df['future_max_return'] = np.nan
df['future_min_return'] = np.nan
for symbol in valid_symbols:
    symbol_df = df.loc[df.index.get_level_values('symbol') == symbol].copy()
    closes = symbol_df['close'].values
    for i, idx in enumerate(symbol_df.index):
        if i + N < len(closes):
            future_window = closes[i+1:i+1+N]
            this_close = closes[i]
            future_max = np.max(future_window)
            future_min = np.min(future_window)
            max_return = (future_max - this_close) / this_close
            min_return = (future_min - this_close) / this_close
            df.loc[idx, 'future_max_return'] = max_return
            df.loc[idx, 'future_min_return'] = min_return

upper_cut = df['future_max_return'].quantile(percentile_upper)
lower_cut = df['future_min_return'].quantile(percentile_lower)
df['label_v2'] = 0
df.loc[df['future_max_return'] > upper_cut, 'label_v2'] = 1
df.loc[df['future_min_return'] < lower_cut, 'label_v2'] = -1
print(f"label分布: {df['label_v2'].value_counts(normalize=True)}")

# ATR 多期
def wwma(values, n): return values.ewm(alpha=1/n, adjust=False).mean()
def atr(df, symbol, n=14):
    df_symbol = df.loc[df.index.get_level_values('symbol') == symbol].copy()
    high = df_symbol['high']
    low = df_symbol['low']
    close = df_symbol['close']
    tr = pd.concat([abs(high-low), abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
    return wwma(tr, n)
for symbol in valid_symbols:
    df.loc[(symbol, slice(None)), 'atr14'] = atr(df, symbol, n=14)
    df.loc[(symbol, slice(None)), 'atr10'] = atr(df, symbol, n=10)
    df.loc[(symbol, slice(None)), 'atr20'] = atr(df, symbol, n=20)
    df.loc[(symbol, slice(None)), 'atr100'] = atr(df, symbol, n=100)

df['atr10/atr100'] = df['atr10'] / df['atr100']
df['atr10/atr20'] = df['atr10'] / df['atr20']
for symbol in valid_symbols:
    df.loc[(symbol, slice(None)), 'delta_atr10/atr100_10'] = df.loc[(symbol, slice(None)), 'atr10/atr100'] - df.loc[(symbol, slice(None)), 'atr10/atr100'].shift(10)
    df.loc[(symbol, slice(None)), 'delta_atr10/atr100_3'] = df.loc[(symbol, slice(None)), 'atr10/atr100'] - df.loc[(symbol, slice(None)), 'atr10/atr100'].shift(3)

# 波動特徵
std_func = lambda x,n: x.rolling(n,min_periods=1).std()
for symbol in valid_symbols:
    df.loc[(symbol, slice(None)), 'volatility5'] = std_func(df.loc[(symbol, slice(None)), 'close'],5)
    df.loc[(symbol, slice(None)), 'volatility10'] = std_func(df.loc[(symbol, slice(None)), 'close'],10)
    df.loc[(symbol, slice(None)), 'volatility25'] = std_func(df.loc[(symbol, slice(None)), 'close'],25)
    df.loc[(symbol, slice(None)), 'volatility100'] = std_func(df.loc[(symbol, slice(None)), 'close'],100)
df['volatility5_ratio'] = df['volatility5'] / df['close']
df['volatility10_ratio'] = df['volatility10'] / df['close']
df['volatility25_ratio'] = df['volatility25'] / df['close']
df['volatility100_ratio'] = df['volatility100'] / df['close']

# 多窗口momentum
def momentum_score(ts):
    x = np.arange(len(ts))
    log_ts = np.log(ts)
    mask = ~np.isnan(x) & ~np.isnan(log_ts)
    regress = stats.linregress(x[mask], log_ts[mask])
    annualized_slope = (np.power(np.exp(regress[0]), 252)-1)*100
    return annualized_slope * (regress[2]**2)
momentum_windows = [(5,3),(10,7),(25,20),(50,40),(100,70)]
for window, min_mom in momentum_windows:
    for symbol in valid_symbols:
        arr = df.loc[(symbol, slice(None)), 'close']
        df.loc[(symbol, slice(None)), f'momentum_{window}'] = arr.rolling(window, min_periods=min_mom).apply(momentum_score)

# Bull/Bull Ratio/High特徵
ema_40 = lambda x: x.ewm(span=40,min_periods=1).mean()
ema_80 = lambda x: x.ewm(span=80,min_periods=1).mean()
for symbol in valid_symbols:
    df.loc[(symbol, slice(None)), 'ema40'] = ema_40(df.loc[(symbol, slice(None)), 'close'])
    df.loc[(symbol, slice(None)), 'ema80'] = ema_80(df.loc[(symbol, slice(None)), 'close'])
df['bull'] = df['ema40'] > df['ema80']
df['bull_ratio'] = df['ema40'] / df['ema80'] - 1
max_50 = lambda x: x.rolling(50,min_periods=1).max()
for symbol in valid_symbols:
    df.loc[(symbol, slice(None)), '50d_high'] = max_50(df.loc[(symbol, slice(None)), 'close'])
    df.loc[(symbol, slice(None)), '50d_high_volume'] = max_50(df.loc[(symbol, slice(None)), 'volume'])
df['close>50d_high'] = df['close'] >= df['50d_high']
df['volume_vs_50d_high'] = df['volume'] / df['50d_high_volume']
df['close_vs_50d_high'] = df['close'] / df['50d_high']

# Scaling
zscore_50 = lambda x: (x-x.rolling(50,min_periods=1).mean())/x.rolling(50,min_periods=1).std()
scale_cols = ['log volume','volume_pct_change_1_day','volume_pct_change_5_day','past_return_1','past_return_2','past_return_3',
              'past_return_4','past_return_5','past_return_10','volatility50','delta_atr10/atr100_3','delta_atr10/atr100_10',
              'atr10','atr14','atr20','atr100','atr10/atr100','atr10/atr20']
for col in scale_cols:
    if col in df.columns:
        for symbol in valid_symbols:
            arr = df.loc[(symbol, slice(None)), col]
            df.loc[(symbol, slice(None)), col+'_scaled50'] = zscore_50(arr)

# Market Meanness Index
def mmi(closes):
    m = closes.median()
    nh = 0; nl = 0
    for i in range(1,len(closes)):
        if closes[i]>m and closes[i]>closes[i-1]: nl+=1
        elif closes[i]<m and closes[i]<closes[i-1]: nh+=1
    return 100*(nl+nh)/(len(closes)-1)
for symbol in valid_symbols:
    arr = df.loc[(symbol,slice(None)),'close']
    df.loc[(symbol,slice(None)),'mmi50'] = arr.rolling(50,min_periods=2).apply(mmi)

# Dynamic targets
ema_std50 = lambda x: x.ewm(span=50,min_periods=20).std()
for symbol in valid_symbols:
    df.loc[(symbol,slice(None)),'past_return_1_ema_std50'] = ema_std50(df.loc[(symbol,slice(None)),'past_return_1'])
    df.loc[(symbol,slice(None)),'price_chg_1'] = df.loc[(symbol,slice(None)),'close'] - df.loc[(symbol,slice(None)),'close'].shift(1)
    df.loc[(symbol,slice(None)),'price_chg_1_ema_std50'] = ema_std50(df.loc[(symbol,slice(None)), 'price_chg_1'])
df['past_return_1_ema_std50*2.2'] = df['past_return_1_ema_std50'] * 2.2
df['price_chg_1_ema_std50*2.2'] = df['price_chg_1_ema_std50'] * 2.2 / 50

# 多期STD
std_50 = lambda x: x.rolling(50, min_periods=20).std()
for symbol in valid_symbols:
    df.loc[(symbol,slice(None)), 'volume_std50'] = std_50(df.loc[(symbol, slice(None)), 'volume'])
    df.loc[(symbol,slice(None)), 'log volume_std50'] = std_50(df.loc[(symbol,slice(None)), 'log volume'])
    df.loc[(symbol,slice(None)), 'past_return_5_std50'] = std_50(df.loc[(symbol,slice(None)), 'past_return_5'])
    df.loc[(symbol,slice(None)), 'past_return_4_std50'] = std_50(df.loc[(symbol,slice(None)), 'past_return_4'])
    df.loc[(symbol,slice(None)), 'past_return_3_std50'] = std_50(df.loc[(symbol,slice(None)), 'past_return_3'])
    df.loc[(symbol,slice(None)), 'past_return_2_std50'] = std_50(df.loc[(symbol,slice(None)), 'past_return_2'])
    df.loc[(symbol,slice(None)), 'past_return_1_std50'] = std_50(df.loc[(symbol,slice(None)), 'past_return_1'])
    df.loc[(symbol,slice(None)), 'past_return_10_std50'] = std_50(df.loc[(symbol,slice(None)), 'past_return_10'])

df['target_upper'] = df['close'] * (1+df['past_return_5_std50'])
df['target_lower'] = df['close'] * (1-df['past_return_5_std50'])
df['target_upper_v2'] = df['close'] * (1+df['past_return_1_ema_std50*2.2'])
df['target_lower_v2'] = df['close'] * (1-df['past_return_1_ema_std50*2.2'])

# Triple Barrier標籤
df['label_v2'] = np.nan
for symbol in valid_symbols:
    symbol_df = df.loc[df.index.get_level_values('symbol') == symbol].copy()
    for i, idx in enumerate(symbol_df.index):
        j = 1
        while i+j < len(symbol_df) and j <= 5:
            future_row = symbol_df.iloc[i+j]
            if pd.isna(future_row['high']) or pd.isna(future_row['low']) or pd.isna(symbol_df.iloc[i]['target_upper_v2']) or pd.isna(symbol_df.iloc[i]['target_lower_v2']):
                j += 1
                continue
            if future_row['high'] > symbol_df.iloc[i]['target_upper_v2']:
                df.loc[idx, 'label_v2'] = 1
                break
            elif future_row['low'] < symbol_df.iloc[i]['target_lower_v2']:
                df.loc[idx, 'label_v2'] = -1
                break
            elif j == 5:
                df.loc[idx, 'label_v2'] = 0
            j += 1

last_date = pd.to_datetime(sorted(list(set(df.index.get_level_values('date'))))[-1])
out_csv = f'outcomes_middle_features_step2_{last_date.strftime("%Y-%m-%d")}.csv'
df.to_csv(path_pc + out_csv)
print("Done! STEP2完成，輸出檔名:", out_csv)
