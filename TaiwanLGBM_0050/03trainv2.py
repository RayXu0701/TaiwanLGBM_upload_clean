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

# 台股清單（1支）
TAIWAN_SYMBOLS = ['0050.TW']

path_pc = 'C:/Users/ray92/Desktop/TaiwanLGBM_upload/'
DATA_PATH = path_pc + 'outcomes_new_features_2025-10-23.csv'

# ==== 讀檔 + 台股過濾 ====
df = pd.read_csv(DATA_PATH, index_col=[0,1])
df.index.names = ['symbol','date']
df = df.sort_index()
df = df[df.index.get_level_values('symbol').isin(TAIWAN_SYMBOLS)]
print('\n過濾後有效台股:', sorted(df.index.get_level_values('symbol').unique()))

# 可用特徵自動篩選
skip_cols = [
    'label_fixed', 'label_test',
    'future_max_return', 'future_min_return',
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
print("最終可用特徵：", features)

for col in features:
    if col not in df.columns:
        df[col] = 0
df[features] = df[features].fillna(0)
df.reset_index(inplace=True)
df['date'] = pd.to_datetime(df['date'])

# ==== 多標籤迴圈訓練（N=3/5/7/10） ====
def align_features(df, features):
    for col in features:
        if col not in df.columns:
            df[col] = 0
    return df[features]

def eval_pred(model, X, y, name):
    y_pred = model.predict(X, num_iteration=model.best_iteration)
    y_pred_cls = np.argmax(y_pred, axis=1)
    print(f"\n{name} set:")
    print("Accuracy:", accuracy_score(y, y_pred_cls))
    print("F1:", f1_score(y, y_pred_cls, average='macro'))
    print("Classification report:\n", classification_report(y, y_pred_cls, digits=4))
    plt.figure(figsize=(4,3))
    cm = confusion_matrix(y, y_pred_cls, labels=[0,1,2])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=['0: Neutral','1: Up','2: Down'], yticklabels=['0: Neutral','1: Up','2: Down'])
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.show()
    # 方向準確率（只計算 up/down 樣本）:
    y_dir = np.where(y==1, 1, np.where(y==2, -1, 0))
    y_pred_dir = np.where(y_pred_cls==1, 1, np.where(y_pred_cls==2, -1, 0))
    valid_idx = (y_dir != 0)
    if valid_idx.sum() > 0:
        dir_acc = (y_pred_dir[valid_idx] == y_dir[valid_idx]).mean()
        print(f"方向準確率: {dir_acc:.4f}")
    else:
        print("無有方向的樣本，無法計算方向準確率。")

params = {
    'objective': 'multiclass',
    'num_class': 3,
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 8,
    'metric': 'multi_logloss',
    'seed': SEED,
    'verbose': -1
}

for N in [3, 5, 7, 10]:
    label_col = f'label_fixed_{N}'
    print(f"\n========================")
    print(f"==== 訓練/驗證/測試 Label: {label_col} ====")
    print("========================\n")

    df_tmp = df[df[label_col].notnull()].copy()
    # 標籤 -1 改 2
    df_tmp.loc[df_tmp[label_col] == -1, label_col] = 2

    # 資料分割
    trainval_df = df_tmp[df_tmp['date'] < '2016-01-01'].sort_values('date')
    trainval_len = len(trainval_df)
    train_len = int(trainval_len * 0.75)
    train_df = trainval_df.iloc[:train_len]
    valid_df = trainval_df.iloc[train_len:]
    test_df = df_tmp[df_tmp['date'] >= '2016-01-01']

    X_train = align_features(train_df, features)
    X_valid = align_features(valid_df, features)
    X_test  = align_features(test_df, features)
    y_train = train_df[label_col].astype(int)
    y_valid = valid_df[label_col].astype(int)
    y_test  = test_df[label_col].astype(int)

    print(f"訓練集範圍: {train_df['date'].min()} ~ {train_df['date'].max()}")
    print(f"驗證集範圍: {valid_df['date'].min()} ~ {valid_df['date'].max()}")
    print(f"測試集範圍: {test_df['date'].min()} ~ {test_df['date'].max()}")
    print(f"樣本數: 訓練:{len(train_df)} 驗證:{len(valid_df)} 測試:{len(test_df)}")

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid)
    print("\n訓練 LightGBM (N=%d)..." % N)
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        num_boost_round=600,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    eval_pred(model, X_train, y_train, f"Train label_fixed_{N}")
    eval_pred(model, X_valid, y_valid, f"Valid label_fixed_{N}")
    eval_pred(model, X_test, y_test, f"Test label_fixed_{N}")

    model_file = path_pc + f'lightgbm_classify_model_N{N}.txt'
    model.save_model(model_file)
    print(f"模型已存檔於: {model_file}\n")

    # 亦可依需求分別存特徵表/結果表

print("==== 所有天數訓練/測試結束 ====")
