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

# 台股清單（中波動九支）
TAIWAN_SYMBOLS = symbols_middle_9 = ['2886.TW', '2317.TW', '1216.TW', '2395.TW', '1101.TW', '2382.TW', '4958.TW', '2356.TW', '3711.TW']

# 1. 讀檔 + 台股過濾
DATA_PATH = 'C:/Users/ray92/Desktop/Thesis/outcomesmiddle_new_features_2025-10-23.csv'
df = pd.read_csv(DATA_PATH, index_col=[0,1])
df.index.names = ['symbol','date']
df = df.sort_index()
df = df[df.index.get_level_values('symbol').isin(TAIWAN_SYMBOLS)]
print('\n過濾後有效台股:', sorted(df.index.get_level_values('symbol').unique()))
print('==== label_v2 分布 ====')
print(df['label_v2'].value_counts(normalize=True))

# 去除NaN標籤，特徵自動選
df = df[df['label_v2'].notnull()]
skip_cols = [
    'label_v2','target_upper','target_lower',
    'target_upper_v2','target_lower_v2'
]
features = [col for col in df.columns if (col not in skip_cols and df[col].dtype in [float, np.float64, int, np.int64])]
df[features] = df[features].fillna(0)

# 日期欄型態修正、index重置
df.reset_index(inplace=True)
df['date'] = pd.to_datetime(df['date'])

# 測試集：近10年
test_df = df[df['date'] >= '2016-01-01']

# 訓練/驗證集：更早（按時序分三七）
trainval_df = df[df['date'] < '2016-01-01'].sort_values('date')
trainval_len = len(trainval_df)
train_len = int(trainval_len * 0.75)
train_df = trainval_df.iloc[:train_len]
valid_df = trainval_df.iloc[train_len:]

# 標籤 -1 改 2（台股三類：0-平 1-漲 2-跌）
for _df in [train_df, valid_df, test_df]:
    _df.loc[_df['label_v2'] == -1, 'label_v2'] = 2

# 分組
X_train, y_train = train_df[features], train_df['label_v2'].astype(int)
X_valid, y_valid = valid_df[features], valid_df['label_v2'].astype(int)
X_test,  y_test  = test_df[features],  test_df['label_v2'].astype(int)

print('\n資料 shape:', X_train.shape, X_valid.shape, X_test.shape)
print('訓練集範圍:', X_train.index.min(), '~', X_train.index.max())
print('驗證集範圍:', X_valid.index.min(), '~', X_valid.index.max())
print('測試集範圍:', X_test.index.min(), '~', X_test.index.max())

# LightGBM 參數
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

# 評估 function（label三類＋confusion matrix＋方向準確率）
def eval_pred(X, y, name):
    y_pred = model.predict(X, num_iteration=model.best_iteration)
    y_pred_cls = np.argmax(y_pred, axis=1)
    print(f"\n{name} set:")
    print("Accuracy:", accuracy_score(y, y_pred_cls))
    print("F1:", f1_score(y, y_pred_cls, average='macro'))
    print("分類詳情:\n", classification_report(y, y_pred_cls, digits=4))
    # 混淆矩陣
    plt.figure(figsize=(4,3))
    cm = confusion_matrix(y, y_pred_cls, labels=[0,1,2])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0:平','1:漲','2:跌'], yticklabels=['0:平','1:漲','2:跌'])
    plt.title(f"{name} confusion matrix")
    plt.xlabel('預測類別')
    plt.ylabel('真實類別')
    plt.show()

    # 方向準確率（只考慮1/2類）
    y_dir = np.where(y==1, 1, np.where(y==2, -1, 0))
    y_pred_dir = np.where(y_pred_cls==1, 1, np.where(y_pred_cls==2, -1, 0))
    valid_idx = (y_dir != 0)
    if valid_idx.sum() > 0:
        dir_acc = (y_pred_dir[valid_idx] == y_dir[valid_idx]).mean()
        print(f"方向準確率: {dir_acc:.4f}")
    else:
        print("無有方向的樣本，無法計算方向準確率。")

# 三組評分
eval_pred(X_train, y_train, "Train")
eval_pred(X_valid, y_valid, "Validation")
eval_pred(X_test,  y_test,  "Test")

print("訓練集：", train_df['date'].min(), "~", train_df['date'].max(), "| 筆數:", len(train_df))
print("驗證集：", valid_df['date'].min(), "~", valid_df['date'].max(), "| 筆數:", len(valid_df))
print("測試集：", test_df['date'].min(), "~", test_df['date'].max(), "| 筆數:", len(test_df))

# ------ LightGBM 內建儲存方式 ------
model.save_model('lightgbm_classify_model.txt')
with open('middlefeatures_list.txt', 'w', encoding='utf-8') as f:
    for col in features:
        f.write(col + '\n')
print("特徵欄位清單已存檔：middlefeatures_list.txt")
