import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ====== 參數設定 ======
path_pc = 'C:/Users/ray92/Desktop/TaiwanLGBM_upload/'
DATA_PATH = path_pc + 'outcomes_new_features_2025-10-23.csv'

INITIAL_CASH = 1_000_000
MAX_STOCKS = 3                # 只持有三檔，單檔金額大
MAX_HOLDING_DAYS = 30         # 持有天數縮短至30
TRADE_COST = 0.0035
SCORE_DIFF_THRESHOLD = 0.02   # 只要新股比現有好0.02就換股
MIN_HOLDING_DAYS = 2          # 最短持有天數縮短至2
TRAILING_STEP_BASE = 3        # 浮盈僅3%就啟動移動停損
TRAILING_STEP_INC = 2         # 浮盈每多1%，停損往上提升2%

# ====== 讀資料並填補異常 ======
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
with open(path_pc + 'features_list.txt', 'r', encoding='utf-8') as f:
    features = [line.strip() for line in f.readlines()]
for col in features:
    if col not in df.columns:
        df[col] = np.nan
df[features] = df[features].fillna(0)

# 處理價格欄位異常 —— 保證無NaN且>0
price_cols = ['open', 'close', 'high']
for col in price_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)
        df = df[df[col] > 0]

# ====== 找漲幅最大股票 ======
if 'future_max_return' in df.columns:
    # 為確保符號和date都有
    require_cols = ['symbol', 'date', 'future_max_return']
    for c in require_cols:
        if c not in df.columns:
            raise Exception(f"{c} 欄位不存在，請檢查原始資料！")
    max_row = df.loc[df['future_max_return'].idxmax()]
    print(f"漲幅最大股票: {max_row['symbol']}, 日期: {max_row['date']}, 漲幅: {max_row['future_max_return']:.2%}")
else:
    print('future_max_return 欄位不存在！')

test_df = df[df['date'] >= '2016-01-01']
symbols = sorted(test_df['symbol'].unique())
dates = sorted(test_df['date'].unique())

# ====== 查價dict，有效值才存 ======
open_map  = {(row['date'], row['symbol']): row['open']  for _, row in test_df.iterrows() if row['open'] > 0}
close_map = {(row['date'], row['symbol']): row['close'] for _, row in test_df.iterrows() if row['close'] > 0}
high_map  = {(row['date'], row['symbol']): row['high']  for _, row in test_df.iterrows() if row['high'] > 0}

# ====== SmartTradingSystem動態策略（回歸版） ======
class PositionReg:
    def __init__(self, buy_price, amount, score, buy_date):
        self.buy_price = buy_price
        self.amount = amount
        self.score = score
        self.buy_date = buy_date
        self.days = 0
        self.max_high = buy_price
        self.stop_price = buy_price * 0.97
        self.max_amount = INITIAL_CASH / MAX_STOCKS
    def update(self, curr_high, curr_close):
        self.days += 1
        if curr_high > self.max_high:
            self.max_high = curr_high
        self.update_stop_price()
        if curr_close < self.stop_price or self.days >= MAX_HOLDING_DAYS:
            return True
        return False
    def update_stop_price(self):
        float_gain = (self.max_high - self.buy_price) / self.buy_price
        if float_gain >= TRAILING_STEP_BASE / 100:
            step = max(2, int((float_gain * 100) - TRAILING_STEP_BASE + TRAILING_STEP_INC))
            self.stop_price = self.buy_price * (1 + step / 100)
        else:
            self.stop_price = self.buy_price * 0.97

class SmartTradingSystemReg:
    def __init__(self):
        self.cash = INITIAL_CASH
        self.max_amount = INITIAL_CASH / MAX_STOCKS
        self.portfolio = {}
        self.trade_log = []
        self.portfolio_value_log = []
    def alloc_ratio(self, score):
        if score > 0.08:   ratio = 1.0
        elif score > 0.06: ratio = 0.8
        elif score > 0.04: ratio = 0.6
        else:              return 0.0
        return ratio
    def buy_or_add(self, symbol, score, price, date):
        if price is None or price <= 0 or np.isnan(price): return False
        ratio = self.alloc_ratio(score)
        if ratio == 0: return False
        max_val = self.max_amount * ratio
        if symbol in self.portfolio:
            position = self.portfolio[symbol]
            add_amt = max_val - position.amount
            if add_amt > 0 and self.cash >= add_amt:
                self.cash -= add_amt
                position.amount += add_amt
                position.score = score
                self.trade_log.append({'date': date, 'symbol': symbol, 'side': 'add', 'price': price, 'amount': add_amt, 'reason': 'add_up'})
        else:
            if self.cash >= max_val and len(self.portfolio) < MAX_STOCKS:
                self.cash -= max_val
                self.portfolio[symbol] = PositionReg(price, max_val, score, date)
                self.trade_log.append({'date': date, 'symbol': symbol, 'side': 'buy', 'price': price, 'amount': max_val, 'reason': 'init_buy'})
        return True
    def sell(self, symbol, price, date, reason):
        if symbol not in self.portfolio: return False
        if price is None or price <= 0 or np.isnan(price): return False
        position = self.portfolio[symbol]
        if position.buy_price == 0 or np.isnan(position.buy_price): return False
        profit = (price - position.buy_price) / position.buy_price * position.amount
        self.cash += position.amount + profit
        self.trade_log.append({'date': date, 'symbol': symbol, 'side': 'sell', 'price': price, 'amount': position.amount, 'profit': profit, 'reason': reason, 'hold_days': position.days})
        del self.portfolio[symbol]
        return True
    def daily_update(self, date):
        to_sell = []
        for symbol, position in list(self.portfolio.items()):
            curr_high  = high_map.get((date, symbol), position.buy_price)
            curr_close = close_map.get((date, symbol), position.buy_price)
            if curr_high is None or curr_high <= 0 or np.isnan(curr_high): curr_high = position.buy_price
            if curr_close is None or curr_close <= 0 or np.isnan(curr_close): curr_close = position.buy_price
            if position.update(curr_high, curr_close):
                reason = 'trailing_stop' if curr_close < position.stop_price else 'max_days'
                to_sell.append((symbol, curr_close, reason))
        for symbol, price, reason in to_sell:
            self.sell(symbol, price, date, reason)
        port_val = self.cash
        for symbol, position in self.portfolio.items():
            last_close = close_map.get((date, symbol), position.buy_price)
            if last_close is None or last_close <= 0 or np.isnan(last_close):
                last_close = position.buy_price
            if position.buy_price == 0 or np.isnan(position.buy_price):
                continue
            port_val += position.amount * (last_close / position.buy_price)
        self.portfolio_value_log.append({'date': date, 'portfolio_value': port_val, 'cash': self.cash, 'holdings': len(self.portfolio)})
        if np.isnan(port_val):
            print(f"【偵測到nan】on {date}, 詳細持股:", self.portfolio)
            raise Exception(f"偵測到nan, 請檢查該日資產、成交價格與持有清單")
    def batch_replace(self, drop_symbols, add_symbols, next_date):
        for symbol in drop_symbols:
            open_price = open_map.get((next_date, symbol), None)
            if open_price is not None and open_price > 0 and not np.isnan(open_price):
                self.sell(symbol, open_price, next_date, 'multi_replace')
        for symbol, score in add_symbols:
            open_price = open_map.get((next_date, symbol), None)
            if open_price is not None and open_price > 0 and not np.isnan(open_price):
                self.buy_or_add(symbol, score, open_price, next_date)

# ====== 載入LightGBM回歸模型 ======
model = lgb.Booster(model_file=path_pc + 'lightgbm_recall_model.txt')

trading_system = SmartTradingSystemReg()
test_dates = dates

for i in range(len(test_dates) - 1):
    date0, date1 = test_dates[i], test_dates[i+1]
    signal_scores = {}
    for s in symbols:
        X_today = test_df[(test_df['date'] == date0) & (test_df['symbol'] == s)][features].values
        if len(X_today) == 0: continue
        score = model.predict(X_today.reshape(1, -1))[0]
        signal_scores[s] = score
        open_price = open_map.get((date1, s), None)
        if open_price is not None and open_price > 0 and not np.isnan(open_price):
            trading_system.buy_or_add(s, score, open_price, date1)
    trading_system.daily_update(date0)
    curr_symbols = set(trading_system.portfolio.keys())
    drop_symbols, add_symbols = [], []
    scores_sorted = sorted([(s, signal_scores.get(s, 0)) for s in curr_symbols], key=lambda x: x[1])
    for s, old_score in scores_sorted:
        if trading_system.portfolio[s].days >= MIN_HOLDING_DAYS:
            for cand, cand_score in signal_scores.items():
                if cand not in curr_symbols and cand_score > old_score + SCORE_DIFF_THRESHOLD:
                    drop_symbols.append(s); add_symbols.append((cand, cand_score)); break
    if drop_symbols and add_symbols:
        trading_system.batch_replace(drop_symbols, add_symbols, date1)
    if i % 100 == 0 and len(trading_system.portfolio_value_log) > 0:
        print(f'[Regress] {i}/{len(test_dates)} 資產：{trading_system.portfolio_value_log[-1]["portfolio_value"]:.0f}')

# === 主動策略績效報表與勝率 === 
trade_df = pd.DataFrame(trading_system.trade_log)
portfolio_df = pd.DataFrame(trading_system.portfolio_value_log)
final_value = portfolio_df['portfolio_value'].iloc[-1]
total_return = (final_value - INITIAL_CASH) / INITIAL_CASH
sell_trades = trade_df[trade_df['side'] == 'sell']
win_rate_no_stop = (sell_trades[~sell_trades['reason'].isin(['trailing_stop','stop_loss'])].shape[0] / sell_trades.shape[0]) if sell_trades.shape[0] > 0 else 0
max_loss = sell_trades['profit'].min() if not sell_trades.empty else 0
print("\n=== 回歸主動策略回測結果 ===")
print(f"期初資金: {INITIAL_CASH:,.0f}")
print(f"期末資產: {final_value:,.0f}")
print(f"總報酬率: {total_return:.2%}")
print(f"贏率(沒觸及止損離場): {win_rate_no_stop:.2%}")
print(f"最大單筆虧損: {max_loss:,.0f}")

# ====== Buy&Hold 等權 & DCA動態都已做價格防守 ======
portfolio_units_eq = {s: 0 for s in symbols}
portfolio_cost_eq = {s: 0 for s in symbols}
cash_eq = INITIAL_CASH
buy_log_eq = []
value_log_eq = []
X_PER_STOCK = INITIAL_CASH / MAX_STOCKS
first_date = dates[0]
for i, s in enumerate(symbols):
    open_price = open_map.get((first_date, s), None)
    if open_price is None or open_price <= 0 or np.isnan(open_price): continue
    buy_amt = X_PER_STOCK
    units = buy_amt / open_price
    portfolio_units_eq[s] += units
    portfolio_cost_eq[s] += buy_amt
    cash_eq -= buy_amt
    buy_log_eq.append({'date': first_date, 'symbol': s, 'side': 'buy', 'price': open_price, 'units': units, 'amount': buy_amt, 'reason': 'init_equal_in'})
for date in dates:
    pv_eq = cash_eq
    for s in symbols:
        last_close = close_map.get((date, s), 0)
        if last_close is None or last_close <= 0 or np.isnan(last_close): continue
        pv_eq += portfolio_units_eq[s] * last_close
    value_log_eq.append({'date': date, 'portfolio_value': pv_eq, 'cash': cash_eq})
value_df_eq = pd.DataFrame(value_log_eq)
final_value_eq = value_df_eq['portfolio_value'].iloc[-1]
total_return_eq = (final_value_eq - INITIAL_CASH) / INITIAL_CASH
print("\n=== Buy&Hold回測結果 ===")
print(f"期初資金: {INITIAL_CASH:,.0f}")
print(f"期末資產: {final_value_eq:,.0f}")
print(f"總報酬率: {total_return_eq:.2%}")

# ====== DCA 定期定額 (120月、均分9檔) ======
TOTAL_DCA = 1_000_000
N_MONTH = 120
X_PER_STOCK = TOTAL_DCA / MAX_STOCKS
X_PER_MONTH_PER_STOCK = X_PER_STOCK / N_MONTH
portfolio_units_dca = {s: 0 for s in symbols}
portfolio_cost_dca = {s: 0 for s in symbols}
cash_dca = TOTAL_DCA
buy_log_dca = []
value_log_dca = []
month_series = pd.Series(dates).dt.month
year_series = pd.Series(dates).dt.year
for idx, date in enumerate(dates):
    if idx == 0 or (month_series[idx] != month_series[idx - 1] or year_series[idx] != year_series[idx - 1]):
        n_stocks_this_month = 0
        for s in symbols:
            open_price = open_map.get((date, s), None)
            if open_price is None or open_price <= 0 or np.isnan(open_price): continue
            already = portfolio_cost_dca[s]
            remain = X_PER_STOCK - already
            denom = MAX_STOCKS - n_stocks_this_month
            if denom <= 0:
                continue
            buy_amt = min(X_PER_MONTH_PER_STOCK, remain, cash_dca / (MAX_STOCKS - n_stocks_this_month))
            if buy_amt > 0:
                units = buy_amt / open_price
                portfolio_units_dca[s] += units
                portfolio_cost_dca[s] += buy_amt
                cash_dca -= buy_amt
                n_stocks_this_month += 1
                buy_log_dca.append({'date': date, 'symbol': s, 'side': 'buy', 'price': open_price, 'units': units, 'amount': buy_amt, 'reason': 'monthly_in'})
    pv = cash_dca
    for s in symbols:
        close_price = close_map.get((date, s), 0)
        if close_price is None or close_price <= 0 or np.isnan(close_price): continue
        pv += portfolio_units_dca[s] * close_price
    value_log_dca.append({'date': date, 'portfolio_value': pv, 'cash': cash_dca})
value_df_dca = pd.DataFrame(value_log_dca)
final_value_dca = value_df_dca['portfolio_value'].iloc[-1]
total_invest = TOTAL_DCA
total_return_dca = (final_value_dca - total_invest) / total_invest
print("\n=== DCA: 合計一百萬均分9檔 ===")
print(f"期初資金: {TOTAL_DCA:,.0f}")
print(f"期末資產: {final_value_dca:,.0f}")
print(f"總報酬率: {total_return_dca:.2%}")

# ====== 畫圖比較 ======
plt.figure(figsize=(12, 6))
plt.plot(portfolio_df['date'], portfolio_df['portfolio_value'], label='Regress/Active')
plt.plot(value_df_dca['date'], value_df_dca['portfolio_value'], label='DCA (1M total)')
plt.plot(value_df_eq['date'], value_df_eq['portfolio_value'], label='Equal Weight Buy & Hold')
plt.title('Portfolio Value: Regression (Active) vs DCA vs Buy & Hold')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print('\n全部回測完成！（新版回歸主動+比對策略）')
