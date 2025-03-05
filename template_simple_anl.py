import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# データの読み込み
print("\n=== データの読み込み ===")
df = pd.read_csv("train.csv")  # 適宜ファイルパスを変更
print("データの概要:")
print(df.info())

# データの先頭を表示
print("\n=== データの先頭を確認 ===")
print(df.head())

# 欠損値の確認
print("\n=== 欠損値の確認 ===")
print(df.isnull().sum())

# 数値データの統計情報
print("\n=== 数値データの統計情報 ===")
print(df.describe())

# カテゴリ変数の確認
print("\n=== カテゴリ変数のユニーク値の確認 ===")
print(df.select_dtypes(include=["object"]).nunique())

# データの可視化（数値データのヒストグラム）
print("\n=== ヒストグラムで特徴量の分布を確認 ===")
df.select_dtypes(include=["number"]).hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.show()

# 相関行列の可視化
print("\n=== 相関関係をヒートマップで確認 ===")
plt.figure(figsize=(12, 6))
sns.heatmap(df.select_dtypes(include=["number"]).corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("相関行列")
plt.show()

# カテゴリ変数のエンコード
print("\n=== カテゴリ変数のエンコード ===")
df_encoded = df.copy()
label_encoders = {}

for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le  # エンコーダーを保存
print("エンコード完了！")

# 特徴量とターゲットの分離
print("\n=== 特徴量とターゲットを分割 ===")
target_column = "target"  # 目的変数のカラム名を適宜変更
X = df_encoded.drop(columns=[target_column])
y = df_encoded[target_column]
print(f"特徴量の形状: {X.shape}, 目的変数の形状: {y.shape}")

# データの分割（訓練データ 80% / テストデータ 20%）
print("\n=== 訓練データとテストデータの分割 ===")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"訓練データ: {X_train.shape}, テストデータ: {X_test.shape}")

# 特徴量の標準化
print("\n=== 特徴量の標準化（StandardScaler） ===")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("標準化完了！")

# モデルの学習（RandomForestClassifier）
print("\n=== モデルの学習（Random Forest） ===")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 予測と評価
print("\n=== モデルの評価 ===")
y_pred = model.predict(X_test_scaled)

# 精度の確認
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy（正解率）: {accuracy:.4f}")

# 詳細な評価レポート
print("\n=== クラス分類レポート ===")
print(classification_report(y_test, y_pred))

# 混同行列の可視化
print("\n=== 混同行列 ===")
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("予測値")
plt.ylabel("実際の値")
plt.title("混同行列")
plt.show()

# ハイパーパラメータチューニング（GridSearchCV）
print("\n=== ハイパーパラメータチューニング（GridSearchCV） ===")
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

print(f"\n最適なパラメータ: {grid_search.best_params_}")
print(f"最適なモデルのスコア: {grid_search.best_score_:.4f}")

# 最適なモデルで再評価
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
best_accuracy = accuracy_score(y_test, y_pred_best)

print(f"\n最適なモデルの Accuracy（正解率）: {best_accuracy:.4f}")
