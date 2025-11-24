import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random


warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

TAIWAN_SYMBOLS = ['0056.TW']
path_pc = 'C:/Users/ray92/Desktop/TaiwanLGBM_upload/'
DATA_PATH = path_pc + 'outcomes_new_features_2025-10-23_N10.csv'

# 1. 讀檔+台股過濾
df = pd.read_csv(DATA_PATH, index_col=[0,1])
df.index.names = ['symbol','date']
df = df.sort_index()
df = df[df.index.get_level_values('symbol').isin(TAIWAN_SYMBOLS)]
df = df.reset_index()
df['date'] = pd.to_datetime(df['date'])

# 2. 補標籤
N = 10
up_gate, down_gate = 0.03, -0.03
label_col = f'label_fixed_{N}'
if label_col not in df.columns:
    print(f"補充計算 {label_col} ...")
    df[label_col] = 0
    df.loc[df[f'future_max_return_{N}'] > up_gate, label_col] = 1
    df.loc[df[f'future_min_return_{N}'] < down_gate, label_col] = -1
label_prop = df[label_col].value_counts(normalize=True).sort_index()
print(f"\n==== {label_col} 分布 ====")
print(label_prop.rename('proportion'))

# 3. 標籤 -1 換成 2，防止 LightGBM報錯
df = df[df[label_col].notnull()].copy()
df.loc[df[label_col] == -1, label_col] = 2

# 4. 特徵自動篩選
skip_cols = [
    'label_fixed', 'label_test',
    f'future_max_return_{N}', f'future_min_return_{N}',
    'target_upper', 'target_lower', 'target_upper_v2', 'target_lower_v2'
]
features = [
    col for col in df.columns
    if (col not in skip_cols
        and df[col].dtype in [float, np.float64, int, np.int64]
        and 'future' not in col
        and 'label' not in col
        and 'target' not in col)
]
for col in features:
    if col not in df.columns:
        df[col] = 0
df[features] = df[features].fillna(0)

# 5. 資料切分
df = df.sort_values('date')
total_len = len(df)
train_len = int(total_len * 0.5)
valid_len = int(total_len * 0.25)
test_len = total_len - train_len - valid_len

train_df = df.iloc[:train_len]
valid_df = df.iloc[train_len:train_len+valid_len]
test_df  = df.iloc[train_len+valid_len:]

print(f"訓練集：{train_df['date'].min()} ~ {train_df['date'].max()} | 筆數: {len(train_df)}")
print(f"驗證集：{valid_df['date'].min()} ~ {valid_df['date'].max()} | 筆數: {len(valid_df)}")
print(f"測試集：{test_df['date'].min()} ~ {test_df['date'].max()} | 筆數: {len(test_df)}")

df = pd.read_csv(DATA_PATH, index_col=[0,1])
df.index.names = ['symbol','date']
print("原始資料筆數:", len(df))

# 過濾股票代碼
df = df[df.index.get_level_values('symbol').isin(TAIWAN_SYMBOLS)]
print("過濾代碼後筆數:", len(df))

# 檢查標籤欄位與非空
print(df.columns)
print(df[label_col].value_counts(dropna=False))

# 檢查index的日期欄格式及是否有NaT
print(df.index.get_level_values('date').min(), df.index.get_level_values('date').max())


X_train = train_df[features]
X_valid = valid_df[features]
X_test  = test_df[features]
y_train = train_df[label_col].astype(int)
y_valid = valid_df[label_col].astype(int)
y_test  = test_df[label_col].astype(int)

print(f"\n訓練集： {train_df['date'].min()} ~ {train_df['date'].max()} | 筆數: {len(train_df)}")
print(f"驗證集： {valid_df['date'].min()} ~ {valid_df['date'].max()} | 筆數: {len(valid_df)}")
print(f"測試集： {test_df['date'].min()} ~ {test_df['date'].max()} | 筆數: {len(test_df)}")

with open(path_pc + 'features_list.txt', 'w') as f:
    for col in features:
        f.write(col + '\n')
print("特徵欄位清單已存檔：features_list.txt")

params = {
    'objective': 'multiclass',
    'num_class': 3,
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 10,             
    'max_depth': 4,               
    'feature_fraction': 0.7,       
    'bagging_fraction': 0.7,       
    'bagging_freq': 1,
    'lambda_l1': 1.0,
    'lambda_l2': 3.0,
    'metric': 'multi_logloss',
    'seed': SEED,
    'verbose': -1
}

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_valid = lgb.Dataset(X_valid, label=y_valid)
print("\n訓練 LightGBM...")
model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_valid],
    num_boost_round=600,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

def eval_pred(model, X, y, name):
    y_pred = model.predict(X, num_iteration=model.best_iteration)
    y_pred_cls = np.argmax(y_pred, axis=1)
    print(f"\n{name}:")
    print("Accuracy:", accuracy_score(y, y_pred_cls))
    print("F1:", f1_score(y, y_pred_cls, average='macro'))
    print("分類詳情:\n", classification_report(y, y_pred_cls, digits=4))
    # 方向準確率
    y_dir = np.where(y==1, 1, np.where(y==2, -1, 0))
    y_pred_dir = np.where(y_pred_cls==1, 1, np.where(y_pred_cls==2, -1, 0))
    valid_idx = (y_dir != 0)
    if valid_idx.sum() > 0:
        dir_acc = (y_pred_dir[valid_idx] == y_dir[valid_idx]).mean()
        print(f"方向準確率: {dir_acc:.4f}")
    else:
        print("無方向樣本，無法計算方向準確率。")
    plt.figure(figsize=(4,3))
    cm = confusion_matrix(y, y_pred_cls, labels=[0,1,2])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['0: Neutral','1: Up','2: Down'], yticklabels=['0: Neutral','1: Up','2: Down'])
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.show()

eval_pred(model, X_train, y_train, "Train set")
eval_pred(model, X_valid, y_valid, "Validation set")
eval_pred(model, X_test, y_test, "Test set")

model_file = path_pc + f'lightgbm_classify_model_N{N}_pct{int(up_gate*100):02d}.txt'
model.save_model(model_file)
print(f"模型已存檔：{model_file}")
