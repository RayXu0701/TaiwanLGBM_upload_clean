import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import random
import os


warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

TAIWAN_SYMBOLS = [50]
path_pc = 'C:/Users/ray92/Desktop/TaiwanLGBM_upload/'
DATA_PATH = os.path.join(path_pc, 'outcomes_new_features_2025-05-29_multiG.csv')

print(f"載入資料檔: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, index_col=[0, 1])
df.index.names = ['symbol', 'date']
df = df.sort_index()
df = df[df.index.get_level_values('symbol').isin(TAIWAN_SYMBOLS)]
df = df.reset_index()
df['date'] = pd.to_datetime(df['date'])

def eval_pred_single(model, X, y, name):
    y_pred = model.predict(X, num_iteration=model.best_iteration)
    y_pred_cls = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y, y_pred_cls)
    f1 = f1_score(y, y_pred_cls, average='macro')
    print(f"\n{name} - Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    print("分類詳情:\n", classification_report(y, y_pred_cls, digits=4))

    y_dir = np.where(y == 1, 1, np.where(y == 2, -1, 0))
    y_pred_dir = np.where(y_pred_cls == 1, 1, np.where(y_pred_cls == 2, -1, 0))
    valid_idx = (y_dir != 0)
    if valid_idx.sum() > 0:
        dir_acc = (y_pred_dir[valid_idx] == y_dir[valid_idx]).mean()
        print(f"方向準確率: {dir_acc:.4f}")
    else:
        dir_acc = None
        print("無方向樣本，無法計算方向準確率。")

    plt.figure(figsize=(4, 3))
    cm = confusion_matrix(y, y_pred_cls, labels=[0, 1, 2])
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['0: Neutral', '1: Up', '2: Down'],
        yticklabels=['0: Neutral', '1: Up', '2: Down']
    )
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    
    
    fig_name = name.replace(' ', '_').replace('=', '').replace(',', '').replace('.', '_')
    fig_path = os.path.join(path_pc, f"confusion_{fig_name}.png")
    plt.savefig(fig_path, dpi=100)
    plt.close()
    print(f"混淆矩陣已存為：{fig_path}")

    return acc, f1, dir_acc

N_list = [3, 5, 7, 9]
gates = [0.03, 0.05, 0.07]

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

all_models = []

for N in N_list:
    for gate in gates:
        label_col = f'label_N{N}_G{int(gate*100)}'
        if label_col not in df.columns:
            print(f"資料需要含有 {label_col}，請確認 feature 工程階段有包含。")
            continue

        df_filtered = df[df[label_col].notnull()].copy()
        df_filtered.loc[df_filtered[label_col] == -1, label_col] = 2

        skip_cols = [
            label_col,
            'symbol', 'date',
            f'future_max_return_{N}', f'future_min_return_{N}',
            'target_upper', 'target_lower', 'target_upper_v2', 'target_lower_v2'
        ]

        features = [
            col for col in df_filtered.columns
            if (
                col not in skip_cols
                and df_filtered[col].dtype in [float, np.float64, int, np.int64]
                and 'future' not in col
                and 'label' not in col
                and 'target' not in col
            )
        ]

        df_filtered[features] = df_filtered[features].fillna(0)
        df_filtered = df_filtered.sort_values('date')

        total_len = len(df_filtered)
        train_len = int(total_len * 0.7)
        valid_len = int(total_len * 0.15)

        train_df = df_filtered.iloc[:train_len]
        valid_df = df_filtered.iloc[train_len:train_len + valid_len]
        test_df = df_filtered.iloc[train_len + valid_len:]

        X_train = train_df[features]
        X_valid = valid_df[features]
        X_test = test_df[features]

        y_train = train_df[label_col].astype(int)
        y_valid = valid_df[label_col].astype(int)
        y_test = test_df[label_col].astype(int)

        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_valid = lgb.Dataset(X_valid, label=y_valid)

        print(f"\n訓練 LightGBM 模型 N={N}, 門檻={gate:.2f} ...")
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            num_boost_round=600,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )

        acc, f1, dir_acc = eval_pred_single(
            model, X_test, y_test, f'Test set N={N}, Gate={gate:.2f}'
        )

        all_models.append({
            'N': N,
            'gate': gate,
            'accuracy': acc,
            'f1_macro': f1,
            'direction_accuracy': dir_acc,
            'model': model,
            'features':features,
        })

results_df = pd.DataFrame([
    {
        'N': m['N'],
        'gate': m['gate'],
        'accuracy': m['accuracy'],
        'f1_macro': m['f1_macro'],
        'direction_accuracy': m['direction_accuracy']
    }
    for m in all_models
])
print("\n全部模型摘要：")
print(results_df)
print("\n請輸入要存檔的 N（如5） 和 gate（如0.03）")

try:
    N_choice = int(input("N = "))
    gate_choice = float(input("gate = "))
except Exception:
    N_choice = 5
    gate_choice = 0.03
    print(f"自動選 N={N_choice} gate={gate_choice}")

selected = None
for m in all_models:
    if m['N'] == N_choice and abs(m['gate'] - gate_choice) < 1e-6:
        selected = m
        break

if selected is not None:
    save_path = os.path.join(path_pc, f"model_N{N_choice}_gate{int(gate_choice*100):02d}.txt")
    selected['model'].save_model(save_path)
    print(f"\n已儲存 N={N_choice}, gate={gate_choice} 模型在：{save_path}")
    
    features_for_selected = selected['features']
    feat_path = os.path.join(path_pc, "features_list.txt")
    with open(feat_path, "w", encoding="utf-8") as f:
        for col in features_for_selected:
            f.write(col + "\n")
    print("特徵欄位清單已存檔：", feat_path)
else:
    print("查無此組合，請確認選項是否正確。")

print("\n訓練、評估與存檔全部完成！")
