import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ====== 參數設定 ======
path_pc = 'C:/Users/ray92/Desktop/Thesis/'
DATA_PATH = path_pc + 'outcomes_new_features_2025-10-23.csv'
MODEL_PATH = path_pc + 'lightgbm_classify_model.txt'
with open(path_pc + 'features_list.txt', 'r', encoding='utf-8') as f:
    features = [line.strip() for line in f.readlines()]

INITIAL_CASH = 1_000_000
MAX_STOCKS = 9
MAX_HOLDING_DAYS = 90
TRADE_COST = 0.0035
SCORE_DIFF_THRESHOLD = 0.05
MIN_HOLDING_DAYS = 3
TRAILING_STEP_BASE = 5
TRAILING_STEP_INC = 1

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
for col in features:
    if col not in df.columns:
        df[col] = np.nan
test_df = df[df['date'] >= '2016-01-01']
symbols = sorted(test_df['symbol'].unique())
dates = sorted(test_df['date'].unique())
open_map  = {(row['date'], row['symbol']): row['open']  for _, row in test_df.iterrows()}
close_map = {(row['date'], row['symbol']): row['close'] for _, row in test_df.iterrows()}
high_map  = {(row['date'], row['symbol']): row['high']  for _, row in test_df.iterrows()}

# ====== 多策略共用類別 ======
class Position:
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

class SmartTradingSystem:
    def __init__(self, buy_rule_type='original'):
        self.cash = INITIAL_CASH
        self.max_amount = INITIAL_CASH / MAX_STOCKS
        self.portfolio = {}
        self.trade_log = []
        self.portfolio_value_log = []
        self.buy_rule_type = buy_rule_type

    def buy_or_add(self, symbol, score, price, date):
        # --- 三種投資規則方案 ---
        if self.buy_rule_type == 'original':
            if score > 0.8:      ratio = 1.0
            elif score > 0.7:    ratio = 0.8
            elif score > 0.6:    ratio = 0.6
            else:                return False
        elif self.buy_rule_type == 'gate_06':
            # 只要score >=0.6才投入1.0，
            if score >= 0.6:     ratio = 1.0
            else:                return False
        elif self.buy_rule_type == 'gate_08':
            # 只要score >=0.8才投入1.0，其餘不進場
            if score >= 0.8:     ratio = 1.0
            else:                return False
        else:
            # fallback
            return False
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
                self.portfolio[symbol] = Position(price, max_val, score, date)
                self.trade_log.append({'date': date, 'symbol': symbol, 'side': 'buy', 'price': price, 'amount': max_val, 'reason': 'init_buy'})
        return True

    def sell(self, symbol, price, date, reason):
        if symbol not in self.portfolio: return False
        position = self.portfolio[symbol]
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
            if position.update(curr_high, curr_close):
                reason = 'trailing_stop' if curr_close < position.stop_price else 'max_days'
                to_sell.append((symbol, curr_close, reason))
        for symbol, price, reason in to_sell:
            self.sell(symbol, price, date, reason)
        port_val = self.cash
        for symbol, position in self.portfolio.items():
            last_close = close_map.get((date, symbol), position.buy_price)
            port_val += position.amount * (last_close / position.buy_price)
        self.portfolio_value_log.append({'date': date, 'portfolio_value': port_val, 'cash': self.cash, 'holdings': len(self.portfolio)})
    def batch_replace(self, drop_symbols, add_symbols, next_date):
        for symbol in drop_symbols:
            open_price = open_map.get((next_date, symbol), None)
            if open_price is not None:
                self.sell(symbol, open_price, next_date, 'multi_replace')
        for symbol, score in add_symbols:
            open_price = open_map.get((next_date, symbol), None)
            if open_price is not None:
                self.buy_or_add(symbol, score, open_price, next_date)

# ====== 載入模型 ======
model = lgb.Booster(model_file=MODEL_PATH)

# ====== 回測三組策略 ======
strategies = {
    'original':      "原本三段進場分級(0.6/0.7/0.8)",
    'gate_06':       "0.6分數門檻，進場才滿額",
    'gate_08':       "0.8分數門檻，僅高信心滿額"
}
results = {}

for key, desc in strategies.items():
    system = SmartTradingSystem(buy_rule_type=key)
    for i in range(len(dates) - 1):
        date0, date1 = dates[i], dates[i+1]
        signal_scores = {}
        for s in symbols:
            X_today = test_df[(test_df['date'] == date0) & (test_df['symbol'] == s)][features].values
            if len(X_today)==0: continue
            score = model.predict(X_today.reshape(1, -1))[0][1]
            signal_scores[s] = score
            open_price = open_map.get((date1, s), None)
            if open_price is not None:
                system.buy_or_add(s, score, open_price, date1)
        system.daily_update(date0)
        curr_symbols = set(system.portfolio.keys())
        drop_symbols, add_symbols = [], []
        scores_sorted = sorted([(s, signal_scores.get(s, 0)) for s in curr_symbols], key=lambda x: x[1])
        for s, old_score in scores_sorted:
            if system.portfolio[s].days >= MIN_HOLDING_DAYS:
                for cand, cand_score in signal_scores.items():
                    if cand not in curr_symbols and cand_score > old_score + SCORE_DIFF_THRESHOLD:
                        drop_symbols.append(s); add_symbols.append((cand, cand_score)); break
        if drop_symbols and add_symbols:
            system.batch_replace(drop_symbols, add_symbols, date1)
    # 績效統計
    trade_df = pd.DataFrame(system.trade_log)
    portfolio_df = pd.DataFrame(system.portfolio_value_log)
    final_value = portfolio_df['portfolio_value'].iloc[-1]
    total_return = (final_value - INITIAL_CASH) / INITIAL_CASH
    sell_trades = trade_df[trade_df['side'] == 'sell']
    win_rate_no_stop = (sell_trades[~sell_trades['reason'].isin(['trailing_stop','stop_loss'])].shape[0] / sell_trades.shape[0]) if sell_trades.shape[0] > 0 else 0
    max_loss = sell_trades['profit'].min() if not sell_trades.empty else 0
    results[key] = {
        'desc': desc,
        'trade_df': trade_df,
        'portfolio_df': portfolio_df,
        'final_value': final_value,
        'total_return': total_return,
        'win_rate': win_rate_no_stop,
        'max_loss': max_loss
    }
    print(f"\n=== {desc} ===")
    print(f"期初資金: {INITIAL_CASH:,.0f}")
    print(f"期末資產: {final_value:,.0f}")
    print(f"總報酬率: {total_return:.2%}")
    print(f"贏率(沒觸及止損離場): {win_rate_no_stop:.2%}")
    print(f"最大單筆虧損: {max_loss:,.0f}")

# ====== 等權Buy&Hold回測 ======
portfolio_units_eq = {s: 0 for s in symbols}
portfolio_cost_eq = {s: 0 for s in symbols}
cash_eq = INITIAL_CASH
buy_log_eq = []
value_log_eq = []

X_PER_STOCK = INITIAL_CASH / MAX_STOCKS
first_date = dates[0]
for i, s in enumerate(symbols):
    open_price = open_map.get((first_date, s), None)
    if open_price is None: continue
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
        pv_eq += portfolio_units_eq[s] * last_close
    value_log_eq.append({'date': date, 'portfolio_value': pv_eq, 'cash': cash_eq})

value_df_eq = pd.DataFrame(value_log_eq)
final_value_eq = value_df_eq['portfolio_value'].iloc[-1]
total_return_eq = (final_value_eq - INITIAL_CASH) / INITIAL_CASH
print("\n=== 等權Buy&Hold回測結果 ===")
print(f"期初資金: {INITIAL_CASH:,.0f}")
print(f"期末資產: {final_value_eq:,.0f}")
print(f"總報酬率: {total_return_eq:.2%}")

# ====== DCA定期定額回測 ======
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
            if open_price is None: continue
            already = portfolio_cost_dca[s]
            remain = X_PER_STOCK - already
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
        pv += portfolio_units_dca[s] * close_price
    value_log_dca.append({'date': date, 'portfolio_value': pv, 'cash': cash_dca})

value_df_dca = pd.DataFrame(value_log_dca)
final_value_dca = value_df_dca['portfolio_value'].iloc[-1]
total_invest = TOTAL_DCA
total_return_dca = (final_value_dca - total_invest) / total_invest
print("\n=== DCA定期定額回測結果 ===")
print(f"期初資金: {TOTAL_DCA:,.0f}")
print(f"期末資產: {final_value_dca:,.0f}")
print(f"總報酬率: {total_return_dca:.2%}")

# ====== 畫圖比較 ======
plt.figure(figsize=(12, 6))
for key in results:
    plt.plot(results[key]['portfolio_df']['date'], results[key]['portfolio_df']['portfolio_value'], label=strategies[key])
plt.plot(value_df_dca['date'], value_df_dca['portfolio_value'], label='DCA (1M total)')
plt.plot(value_df_eq['date'], value_df_eq['portfolio_value'], label='Equal B&H')
plt.title('Strategy Compare')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print('\n全部回測完成！')
