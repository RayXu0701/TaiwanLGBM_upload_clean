import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# 股票清單
TAIWAN_SYMBOLS = [
    '2412.TW', '5876.TW', '4904.TW', '2002.TW', '2892.TW', '2880.TW', '2884.TW', '2891.TW', '2883.TW'
]

path_pc = 'C:/Users/ray92/Desktop/TaiwanLGBM_upload/'
DATA_PATH = path_pc + 'outcomes_new_features_2025-10-23.csv'

df = pd.read_csv(DATA_PATH, index_col=[0,1])
df.index.names = ['symbol', 'date']
df = df.sort_index()
df = df[df.index.get_level_values('symbol').isin(TAIWAN_SYMBOLS)]
print('\n過濾後有效台股:', sorted(df.index.get_level_values('symbol').unique()))

TARGET = 'future_max_return'
assert TARGET in df.columns, f'目標欄 {TARGET} 不存在，請確認資料欄位！'
print(f'==== {TARGET} 分布 ====')
print(df[TARGET].describe())

# ====== 找最大漲幅股票&日期 ======
max_idx = df[TARGET].idxmax()
max_symbol, max_date = max_idx
max_gain = df.loc[max_idx, TARGET]
print(f'訓練過程中漲幅最大股票: {max_symbol}，日期: {max_date}，漲幅: {max_gain:.2%}')

df = df[df[TARGET].notnull()]

skip_cols = [
    'label_v2', 'target_upper', 'target_lower', 'target_upper_v2', 'target_lower_v2', TARGET
]
features = [col for col in df.columns if (col not in skip_cols and df[col].dtype in [float, np.float64, int, np.int64])]
for col in features:
    if col not in df.columns:
        df[col] = np.nan
df[features] = df[features].fillna(0)
df.reset_index(inplace=True)
df['date'] = pd.to_datetime(df['date'])

# 資料分割
test_df = df[df['date'] >= '2016-01-01']
trainval_df = df[df['date'] < '2016-01-01'].sort_values('date')
trainval_len = len(trainval_df)
train_len = int(trainval_len * 0.75)
train_df = trainval_df.iloc[:train_len]
valid_df = trainval_df.iloc[train_len:]

X_train, y_train = train_df[features], train_df[TARGET]
X_valid, y_valid = valid_df[features], valid_df[TARGET]
X_test, y_test = test_df[features], test_df[TARGET]

print('\n資料 shape:', X_train.shape, X_valid.shape, X_test.shape)
print('訓練集範圍:', train_df['date'].min(), '~', train_df['date'].max())
print('驗證集範圍:', valid_df['date'].min(), '~', valid_df['date'].max())
print('測試集範圍:', test_df['date'].min(), '~', test_df['date'].max())

params = {
    'objective': 'regression',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 8,
    'metric': 'rmse',
    'seed': SEED,
    'verbose': -1
}
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_valid = lgb.Dataset(X_valid, label=y_valid)

print("\n訓練 LightGBM回歸...")

model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_valid],
    num_boost_round=600,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# ==> 英文圖表 + 嚴格方向率 (<1%誤差)
def eval_reg(X, y, name):
    THRESHOLD = 0.01
    y_pred = model.predict(X, num_iteration=model.best_iteration)
    print(f"\n{name} set:")
    print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))
    print("MAE:", mean_absolute_error(y, y_pred))
    print("R2:", r2_score(y, y_pred))
    y_true_dir = np.sign(np.where(np.abs(y) > THRESHOLD, y, 0))
    y_pred_dir = np.sign(np.where(np.abs(y_pred) > THRESHOLD, y_pred, 0))
    dir_correct = ((y_true_dir == y_pred_dir) &
                   (np.abs(y_true_dir) != 0) &
                   (np.abs(y - y_pred) < THRESHOLD))
    dir_acc_strict = dir_correct.mean()
    has_dir = (y_true_dir != 0)
    if has_dir.sum() > 0:
        print(f"方向準確率(誤差<1%): {dir_acc_strict:.4f}")
    else:
        print("無有方向樣本，無法計算方向準確率。")
    plt.figure(figsize=(5,3))
    sns.histplot(y - y_pred, bins=40, kde=True)
    plt.title(f"{name} Residual Distribution")
    plt.xlabel('Residual: True - Predicted')
    plt.ylabel('Count')
    plt.show()
    plt.figure(figsize=(5,3))
    plt.scatter(y, y_pred, s=5, alpha=0.3)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.title(f"{name} True vs Predicted")
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.show()

eval_reg(X_train, y_train, "Train")
eval_reg(X_valid, y_valid, "Validation")
eval_reg(X_test, y_test, "Test")

print("訓練集：", train_df['date'].min(), "~", train_df['date'].max(), "| 筆數:", len(train_df))
print("驗證集：", valid_df['date'].min(), "~", valid_df['date'].max(), "| 筆數:", len(valid_df))
print("測試集：", test_df['date'].min(), "~", test_df['date'].max(), "| 筆數:", len(test_df))


model.save_model(path_pc + 'lightgbm_recall_model.txt')
with open(path_pc + 'features_list.txt', 'w', encoding='utf-8') as f:
    for col in features:
        f.write(col + '\n')
print("模型 .txt、特徵清單都已存到", path_pc)
print('測試集負值比例:', (y_test < 0).mean())
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
print('預測負值比例:', (y_pred < 0).mean())
