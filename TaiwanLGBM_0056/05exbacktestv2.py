import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import matplotlib.pyplot as plt
import os

warnings.filterwarnings("ignore")

# ====== Parameters ======
path_pc = 'C:/Users/ray92/Desktop/TaiwanLGBM_upload/'
DATA_PATH = os.path.join(path_pc, 'outcomes_new_features_2025-10-23_N10.csv')
MODEL_PATH = path_pc + 'lightgbm_classify_model.txt'
with open(path_pc + 'features_list.txt', 'r', encoding='utf-8') as f:
    features = [line.strip() for line in f.readlines()]

TARGET_SYMBOL = '0056.TW'
INITIAL_CASH = 1_000_000
MAX_HOLDING_DAYS = 30
TRADE_COST = 0.0035
MIN_HOLDING_DAYS = 2
TRAILING_STEP_BASE = 3
TRAILING_STEP_INC = 2

# ====== 讀入資料 ======
df = pd.read_csv(DATA_PATH)
if 'date' not in df.columns:
    raise ValueError("原始CSV缺少 date 欄位！")
df['date'] = pd.to_datetime(df['date'])
for col in features:
    if col not in df.columns:
        df[col] = np.nan
df[features] = df[features].fillna(0)

# 強制只用0056.TW資料
if 'symbol' not in df.columns:
    raise ValueError("原始CSV缺少 symbol 欄位！")
df = df[df['symbol'] == TARGET_SYMBOL].copy()
if df.empty:
    raise ValueError(f"原始資料無 {TARGET_SYMBOL} 資料，請確認 {DATA_PATH} 是否含正確ETF！")

test_start_date = pd.to_datetime('2019-01-09')
test_df = df[df['date'] >= test_start_date].copy()
if test_df.empty:
    raise ValueError(f"起始日期 {test_start_date} 後無 {TARGET_SYMBOL} 資料，資料過濾結果為空，檢查CSV時間區間！")

symbol = TARGET_SYMBOL
dates = sorted(test_df['date'].unique())
open_map  = {(row['date'], row['symbol']): row['open']  for _, row in test_df.iterrows()}
close_map = {(row['date'], row['symbol']): row['close'] for _, row in test_df.iterrows()}
high_map  = {(row['date'], row['symbol']): row['high']  for _, row in test_df.iterrows()}

# ====== Position Class (略，同舊版) ======
class Position:
    def __init__(self, buy_price, amount, score, buy_date):
        self.buy_price = buy_price
        self.amount = amount
        self.score = score
        self.buy_date = buy_date
        self.days = 0
        self.max_high = buy_price
        self.stop_price = buy_price * 0.97
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

class SingleStockSystem:
    def __init__(self, buy_rule_type='original'):
        self.cash = INITIAL_CASH
        self.position = None
        self.trade_log = []
        self.portfolio_value_log = []
        self.buy_rule_type = buy_rule_type
    def try_entry(self, score, price, date):
        if self.position is not None: return False
        if self.buy_rule_type == 'original':
            if score > 0.8: ratio = 1.0
            elif score > 0.7: ratio = 0.8
            elif score > 0.6: ratio = 0.6
            else: return False
        elif self.buy_rule_type == 'gate_06':
            if score >= 0.6: ratio = 1.0
            else: return False
        elif self.buy_rule_type == 'gate_08':
            if score >= 0.8: ratio = 1.0
            else: return False
        else:
            return False
        invest_amt = self.cash * ratio
        if self.cash >= invest_amt:
            self.position = Position(price, invest_amt, score, date)
            self.cash -= invest_amt
            self.trade_log.append({'date': date, 'symbol': symbol, 'side': 'buy', 'price': price, 'amount': invest_amt, 'reason': 'init_buy'})
            return True
        return False
    def try_exit(self, close_price, date):
        if self.position is None: return False
        need_sell = self.position.update(high_map.get((date, symbol), self.position.buy_price), close_price)
        if need_sell or self.position.days >= MAX_HOLDING_DAYS:
            profit = (close_price - self.position.buy_price) / self.position.buy_price * self.position.amount
            self.cash += self.position.amount + profit
            self.trade_log.append({'date': date, 'symbol': symbol, 'side': 'sell', 'price': close_price, 'amount': self.position.amount, 'profit': profit, 'reason': 'exit', 'hold_days': self.position.days})
            self.position = None
            return True
        return False
    def daily_update(self, date):
        curr_close = close_map.get((date, symbol), None)
        if curr_close is not None:
            self.try_exit(curr_close, date)
        port_val = self.cash
        if self.position is not None:
            last_close = close_map.get((date, symbol), self.position.buy_price)
            port_val += self.position.amount * (last_close / self.position.buy_price)
        self.portfolio_value_log.append({'date': date, 'portfolio_value': port_val, 'cash': self.cash, 'holdings': int(self.position is not None)})

# ====== Load Model ======
model = lgb.Booster(model_file=MODEL_PATH)

# ====== 回測三策略 ======
strategies = {
    'original':      "Tiered Entry (0.6/0.7/0.8)",
    'gate_06':       "Gate 0.6 Full Position",
    'gate_08':       "Gate 0.8 High Confidence Only"
}
results = {}
for key, desc in strategies.items():
    system = SingleStockSystem(buy_rule_type=key)
    for i in range(len(dates) - 1):
        date0, date1 = dates[i], dates[i+1]
        row_today = test_df[(test_df['date'] == date0) & (test_df['symbol'] == symbol)][features]
        if row_today.shape[0] == 0: continue
        row_today = row_today.fillna(0).values.reshape(1, -1)
        score = model.predict(row_today)[0][1]
        open_price = open_map.get((date1, symbol), None)
        if open_price is not None:
            system.try_entry(score, open_price, date1)
        system.daily_update(date0)
    trade_df = pd.DataFrame(system.trade_log)
    portfolio_df = pd.DataFrame(system.portfolio_value_log)
    if portfolio_df.empty or 'portfolio_value' not in portfolio_df.columns:
        print(f"\n[{desc}] 無任何持倉紀錄或欄位，請再檢查流程/資料！")
        final_value, total_return, win_rate, max_loss = np.nan, np.nan, np.nan, np.nan
    else:
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - INITIAL_CASH) / INITIAL_CASH
        sell_trades = trade_df[trade_df['side'] == 'sell']
        win_rate = (sell_trades[sell_trades['profit'] > 0].shape[0] / sell_trades.shape[0]) if sell_trades.shape[0] > 0 else 0
        max_loss = sell_trades['profit'].min() if not sell_trades.empty else 0
    results[key] = {
        'desc': desc,
        'trade_df': trade_df,
        'portfolio_df': portfolio_df,
        'final_value': final_value,
        'total_return': total_return,
        'win_rate': win_rate,
        'max_loss': max_loss
    }
    print(f"\n=== {desc} ===")
    print(f"期初資金: {INITIAL_CASH:,.0f}")
    print(f"期末資產: {final_value:,.0f}")
    print(f"總報酬率: {total_return:.2%}")
    print(f"贏率(賺錢出場比例): {win_rate:.2%}")
    print(f"最大單筆虧損: {max_loss:,.0f}")

# ====== Buy&Hold 回測 ======
portfolio_units_eq = 0
cash_eq = INITIAL_CASH
buy_log_eq = []
value_log_eq = []
first_date = dates[0]
open_price = open_map.get((first_date, symbol), None)
if open_price is not None:
    buy_amt = INITIAL_CASH
    units = buy_amt / open_price
    portfolio_units_eq += units
    buy_log_eq.append({'date': first_date, 'symbol': symbol, 'side': 'buy', 'price': open_price, 'units': units, 'amount': buy_amt, 'reason': 'init_equal_in'})
for date in dates:
    last_close = close_map.get((date, symbol), 0)
    pv_eq = portfolio_units_eq * last_close
    value_log_eq.append({'date': date, 'portfolio_value': pv_eq, 'cash': 0})
value_df_eq = pd.DataFrame(value_log_eq)
if value_df_eq.empty or 'portfolio_value' not in value_df_eq.columns:
    print("Buy&Hold無持倉紀錄或欄位錯誤")
    final_value_eq, total_return_eq = np.nan, np.nan
else:
    final_value_eq = value_df_eq['portfolio_value'].iloc[-1]
    total_return_eq = (final_value_eq - INITIAL_CASH) / INITIAL_CASH
print("\n=== Buy&Hold回測結果 ===")
print(f"期初資金: {INITIAL_CASH:,.0f}")
print(f"期末資產: {final_value_eq:,.0f}")
print(f"總報酬率: {total_return_eq:.2%}")

# ====== DCA定期定額回測 (單一ETF) ======
TOTAL_DCA = 1_000_000
N_MONTH = (dates[-1].year - dates[0].year) * 12 + (dates[-1].month - dates[0].month + 1)
X_PER_MONTH = TOTAL_DCA / N_MONTH
portfolio_units_dca = 0
portfolio_cost_dca = 0
cash_dca = TOTAL_DCA
buy_log_dca = []
value_log_dca = []
month_series = pd.Series(dates).dt.month
year_series = pd.Series(dates).dt.year
for idx, date in enumerate(dates):
    if idx == 0 or (month_series[idx] != month_series[idx - 1] or year_series[idx] != year_series[idx - 1]):
        open_price = open_map.get((date, symbol), None)
        if open_price is None: continue
        remain = TOTAL_DCA - portfolio_cost_dca
        buy_amt = min(X_PER_MONTH, remain, cash_dca)
        if buy_amt > 0:
            units = buy_amt / open_price
            portfolio_units_dca += units
            portfolio_cost_dca += buy_amt
            cash_dca -= buy_amt
            buy_log_dca.append({'date': date, 'symbol': symbol, 'side': 'buy', 'price': open_price, 'units': units, 'amount': buy_amt, 'reason': 'monthly_in'})
    close_price = close_map.get((date, symbol), 0)
    pv_dca = portfolio_units_dca * close_price
    value_log_dca.append({'date': date, 'portfolio_value': pv_dca, 'cash': cash_dca})
value_df_dca = pd.DataFrame(value_log_dca)
if value_df_dca.empty or 'portfolio_value' not in value_df_dca.columns:
    print("DCA無持倉紀錄或欄位錯誤")
    final_value_dca, total_return_dca = np.nan, np.nan
else:
    final_value_dca = value_df_dca['portfolio_value'].iloc[-1]
    total_return_dca = (final_value_dca - TOTAL_DCA) / TOTAL_DCA
print("\n=== DCA定期定額回測結果 ===")
print(f"期初資金: {TOTAL_DCA:,.0f}")
print(f"期末資產: {final_value_dca:,.0f}")
print(f"總報酬率: {total_return_dca:.2%}")

# ====== Plot Comparison ======
plt.figure(figsize=(12, 6))
for key in results:
    pf = results[key]['portfolio_df']
    if pf.empty or 'portfolio_value' not in pf.columns:
        continue
    plt.plot(pf['date'], pf['portfolio_value'], label=strategies[key], linewidth=2)
if not value_df_dca.empty and 'portfolio_value' in value_df_dca.columns:
    plt.plot(value_df_dca['date'], value_df_dca['portfolio_value'], label='DCA (1M total)', linewidth=2)
if not value_df_eq.empty and 'portfolio_value' in value_df_eq.columns:
    plt.plot(value_df_eq['date'], value_df_eq['portfolio_value'], label='Buy&Hold', linewidth=2)
plt.title('Strategy Comparison')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
leg = plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print('\n全部回測完成!')

