import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ====== 路徑與檔名設定 ======
path_pc = 'C:/Users/ray92/Desktop/TaiwanLGBM_upload/'
DATA_PATH = os.path.join(path_pc, 'outcomes_new_features_2025-05-29_multiG.csv')
FEATURES_PATH = os.path.join(path_pc, 'features_list.txt')
MODEL_PATH = os.path.join(path_pc, 'model_N5_gate03.txt')

print(f"讀取資料檔: {DATA_PATH}")
print(f"讀取特徵清單: {FEATURES_PATH}")
print(f"讀取模型檔: {MODEL_PATH}")

# ====== 讀取特徵清單 ======
with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
    features = [line.strip() for line in f.readlines()]

TARGET_SYMBOL = 50
INITIAL_CASH = 1_000_000
MAX_HOLDING_DAYS = 90
TRADE_COST = 0.0035
MIN_HOLDING_DAYS = 3
TRAILING_STEP_BASE = 5
TRAILING_STEP_INC = 1

# ====== 讀取資料並整理 ======
df = pd.read_csv(DATA_PATH)

if 'date' not in df.columns:
    raise ValueError("原始CSV缺少 date 欄位！")

df['date'] = pd.to_datetime(df['date'])

for col in features:
    if col not in df.columns:
        df[col] = np.nan
df[features] = df[features].fillna(0)

if 'symbol' not in df.columns:
    raise ValueError("原始CSV缺少 symbol 欄位！")

df = df[df['symbol'] == TARGET_SYMBOL].copy()
if df.empty:
    raise ValueError(f"原始資料無 symbol={TARGET_SYMBOL} 資料！")

test_start_date = pd.to_datetime('2019-01-09')
test_df = df[df['date'] >= test_start_date].copy()
if test_df.empty:
    raise ValueError(f"起始日期 {test_start_date} 後無資料！")

symbol = TARGET_SYMBOL
dates = sorted(test_df['date'].unique())

open_map  = {(row['date'], row['symbol']): row['open']  for _, row in test_df.iterrows()}
close_map = {(row['date'], row['symbol']): row['close'] for _, row in test_df.iterrows()}
high_map  = {(row['date'], row['symbol']): row['high']  for _, row in test_df.iterrows()}
low_map   = {(row['date'], row['symbol']): row['low']   for _, row in test_df.iterrows()}

# ====== 持股物件與交易系統定義 ======
class Position:
    def __init__(self, buy_price, amount, score, buy_date):
        self.buy_price = buy_price
        self.amount = amount
        self.score = score
        self.buy_date = buy_date
        self.days = 0
        self.max_high = buy_price
        self.stop_price = buy_price * 0.97

    def update(self, curr_high, curr_low):
        """
        改用當日最低價判斷是否觸及止損
        回傳 (是否觸發止損, 實際賣出價)
        """
        self.days += 1
        if curr_high > self.max_high:
            self.max_high = curr_high
        self.update_stop_price()
        
        # 如果當日最低價觸及止損價，立即以止損價出場
        if curr_low <= self.stop_price:
            return True, self.stop_price
        
        # 如果持有超過最大天數，以當日收盤價出場
        if self.days >= MAX_HOLDING_DAYS:
            return True, None  # None 代表用收盤價
        
        return False, None

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
        if self.position is not None:
            return False

        if self.buy_rule_type == 'original':
            if score > 0.8:
                ratio = 1.0
            elif score > 0.7:
                ratio = 0.8
            elif score > 0.6:
                ratio = 0.6
            else:
                return False
        elif self.buy_rule_type == 'gate_06':
            if score >= 0.6:
                ratio = 1.0
            else:
                return False
        elif self.buy_rule_type == 'gate_08':
            if score >= 0.8:
                ratio = 1.0
            else:
                return False
        else:
            return False

        invest_amt = self.cash * ratio
        if self.cash >= invest_amt and invest_amt > 0:
            units = invest_amt / price
            self.position = Position(price, units, score, date)
            self.cash -= invest_amt
            self.trade_log.append({
                'date': date, 'symbol': symbol, 'side': 'buy',
                'price': price, 'amount': units, 'cash_used': invest_amt
            })
            return True
        return False

    def try_exit(self, date):
        if self.position is None:
            return False

        curr_high = high_map.get((date, symbol), self.position.buy_price)
        curr_low = low_map.get((date, symbol), self.position.buy_price)
        curr_close = close_map.get((date, symbol), self.position.buy_price)

        need_sell, sell_price = self.position.update(curr_high, curr_low)
        
        if need_sell:
            # 如果是觸及止損，用止損價；如果是到期，用收盤價
            if sell_price is None:
                sell_price = curr_close
            
            sell_value = sell_price * self.position.amount
            buy_cost = self.position.buy_price * self.position.amount
            profit = sell_value - buy_cost
            profit_pct = (sell_price - self.position.buy_price) / self.position.buy_price
            
            self.cash += sell_value
            self.trade_log.append({
                'date': date, 'symbol': symbol, 'side': 'sell',
                'price': sell_price,
                'amount': self.position.amount,
                'profit': profit,
                'profit_pct': profit_pct,
                'hold_days': self.position.days,
                'buy_price': self.position.buy_price
            })
            self.position = None
            return True
        return False

    def daily_update(self, date):
        # 先嘗試出場
        self.try_exit(date)

        # 更新投資組合價值
        port_val = self.cash
        if self.position is not None:
            last_close = close_map.get((date, symbol), self.position.buy_price)
            port_val += self.position.amount * last_close
        self.portfolio_value_log.append({
            'date': date,
            'portfolio_value': port_val,
            'cash': self.cash,
            'holdings': int(self.position is not None)
        })


# ====== 載入模型 ======
model = lgb.Booster(model_file=MODEL_PATH)

# ====== 三種策略回測 ======
strategies = {
    'original': "Tiered Entry (0.6/0.7/0.8)",
    'gate_06': "Gate 0.6 Full Position",
    'gate_08': "Gate 0.8 High Confidence Only"
}


results = {}

for key, desc in strategies.items():
    system = SingleStockSystem(buy_rule_type=key)

    for i in range(len(dates) - 1):
        date0, date1 = dates[i], dates[i + 1]
        row_today = test_df[
            (test_df['date'] == date0) & (test_df['symbol'] == symbol)
        ][features]

        if row_today.shape[0] == 0:
            continue

        row_today = row_today.fillna(0).values.reshape(1, -1)
        score_vec = model.predict(row_today)[0]
        score = score_vec[1]

        open_price = open_map.get((date1, symbol), None)
        if open_price is not None:
            system.try_entry(score, open_price, date1)

        system.daily_update(date0)

    trade_df = pd.DataFrame(system.trade_log)
    portfolio_df = pd.DataFrame(system.portfolio_value_log)

    if portfolio_df.empty or 'portfolio_value' not in portfolio_df.columns:
        final_value, total_return, win_rate, max_loss, max_loss_pct = np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - INITIAL_CASH) / INITIAL_CASH

        sell_trades = trade_df[trade_df['side'] == 'sell']
        if sell_trades.shape[0] > 0:
            win_rate = (sell_trades[sell_trades['profit'] > 0].shape[0]
                        / sell_trades.shape[0])

            losing_trades = sell_trades[sell_trades['profit'] < 0]
            if losing_trades.shape[0] > 0:
                max_loss = losing_trades['profit'].min()
                worst_trade = losing_trades.loc[losing_trades['profit'].idxmin()]
                max_loss_pct = worst_trade['profit_pct']
                
                print(f"\n[{desc}] 最大虧損交易明細：")
                print(f"日期: {worst_trade['date']}")
                print(f"買入價: {worst_trade['buy_price']:.2f}")
                print(f"賣出價: {worst_trade['price']:.2f}")
                print(f"虧損金額: {worst_trade['profit']:,.0f}")
                print(f"虧損比例: {worst_trade['profit_pct']:.2%}")
                print(f"持有天數: {int(worst_trade['hold_days'])}")
            else:
                max_loss = 0
                max_loss_pct = 0
                print(f"\n[{desc}] 無虧損交易。")
        else:
            win_rate = 0
            max_loss = 0
            max_loss_pct = 0
            print(f"\n[{desc}] 無任何賣出交易。")

    results[key] = {
        'desc': desc,
        'trade_df': trade_df,
        'portfolio_df': portfolio_df,
        'final_value': final_value,
        'total_return': total_return,
        'win_rate': win_rate,
        'max_loss': max_loss,
        'max_loss_pct': max_loss_pct
    }

    print(f"\n=== {desc} ===")
    print(f"期初資金: {INITIAL_CASH:,.0f}")
    print(f"期末資產: {final_value:,.0f}")
    print(f"總報酬率: {total_return:.2%}")
    print(f"贏率: {win_rate:.2%}")
    print(f"最大單筆虧損: {max_loss:,.0f} ({max_loss_pct:.2%})")


# ====== Buy & Hold 回測 ======
portfolio_units_eq = 0
value_log_eq = []

first_date = dates[0]
open_price = open_map.get((first_date, symbol), None)
if open_price is not None:
    units = INITIAL_CASH / open_price
    portfolio_units_eq = units

for date in dates:
    last_close = close_map.get((date, symbol), 0)
    pv_eq = portfolio_units_eq * last_close
    value_log_eq.append({'date': date, 'portfolio_value': pv_eq})

value_df_eq = pd.DataFrame(value_log_eq)
if not value_df_eq.empty and 'portfolio_value' in value_df_eq.columns:
    final_value_eq = value_df_eq['portfolio_value'].iloc[-1]
    total_return_eq = (final_value_eq - INITIAL_CASH) / INITIAL_CASH
else:
    final_value_eq, total_return_eq = np.nan, np.nan

print("\n=== Buy&Hold ===")
print(f"期初資金: {INITIAL_CASH:,.0f}")
print(f"期末資產: {final_value_eq:,.0f}")
print(f"總報酬率: {total_return_eq:.2%}")


# ====== DCA 定期定額回測 ======
TOTAL_DCA = 1_000_000
N_MONTH = (dates[-1].year - dates[0].year) * 12 + (dates[-1].month - dates[0].month + 1)
X_PER_MONTH = TOTAL_DCA / N_MONTH

portfolio_units_dca = 0
portfolio_cost_dca = 0
cash_dca = TOTAL_DCA
value_log_dca = []

month_series = pd.Series(dates).dt.month
year_series = pd.Series(dates).dt.year

for idx, date in enumerate(dates):
    if idx == 0 or (month_series[idx] != month_series[idx - 1]
                    or year_series[idx] != year_series[idx - 1]):
        open_price = open_map.get((date, symbol), None)
        if open_price is None:
            continue
        remain = TOTAL_DCA - portfolio_cost_dca
        buy_amt = min(X_PER_MONTH, remain, cash_dca)
        if buy_amt > 0:
            units = buy_amt / open_price
            portfolio_units_dca += units
            portfolio_cost_dca += buy_amt
            cash_dca -= buy_amt

    close_price = close_map.get((date, symbol), 0)
    pv_dca = portfolio_units_dca * close_price
    value_log_dca.append({'date': date, 'portfolio_value': pv_dca})

value_df_dca = pd.DataFrame(value_log_dca)
if not value_df_dca.empty and 'portfolio_value' in value_df_dca.columns:
    final_value_dca = value_df_dca['portfolio_value'].iloc[-1]
    total_return_dca = (final_value_dca - TOTAL_DCA) / TOTAL_DCA
else:
    final_value_dca, total_return_dca = np.nan, np.nan

print("\n=== DCA定期定額 ===")
print(f"期初資金: {TOTAL_DCA:,.0f}")
print(f"期末資產: {final_value_dca:,.0f}")
print(f"總報酬率: {total_return_dca:.2%}")


# ====== 策略績效曲線比較圖 ======
plt.figure(figsize=(12, 6))

for key, res in results.items():
    pf = res['portfolio_df']
    if pf.empty or 'portfolio_value' not in pf.columns:
        continue
    plt.plot(pf['date'], pf['portfolio_value'],
             label=res['desc'], linewidth=2)

if not value_df_eq.empty and 'portfolio_value' in value_df_eq.columns:
    plt.plot(value_df_eq['date'], value_df_eq['portfolio_value'],
             label='Buy&Hold', linewidth=2)

if not value_df_dca.empty and 'portfolio_value' in value_df_dca.columns:
    plt.plot(value_df_dca['date'], value_df_dca['portfolio_value'],
             label='DCA (Monthly)', linewidth=2)

plt.title('Strategy Portfolio Value Over Time', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend(loc='upper left', frameon=False)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()

fig_path = os.path.join(path_pc, 'backtest_strategy_comparison.png')
plt.savefig(fig_path, dpi=150)
print(f"\n策略比較圖已存檔：{fig_path}")
plt.close()

print('\n全部回測完成!')
