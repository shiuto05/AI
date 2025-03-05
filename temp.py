import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# データの読み込み
df = pd.read_csv("train.csv")  # 適宜ファイルパスを変更

print("=== データの基本情報を表示 ===")
print("データの概要を確認します。列のデータ型や欠損値の有無をチェックしましょう。\n")
print(df.info())

print("\n=== データの先頭・末尾を表示 ===")
print("データの最初と最後の数行を表示して、内容を確認します。\n")
print("【データの先頭】")
print(df.head())
print("\n【データの末尾】")
print(df.tail())

print("\n=== 欠損値の確認 ===")
print("各カラムの欠損値の数を表示します。欠損が多い項目には注意が必要です。\n")
print(df.isnull().sum())

print("\n=== 基本統計量の表示（数値データのみ） ===")
print("数値データの平均値・標準偏差・最大値などを確認します。\n")
print(df.describe())

print("\n=== カテゴリ変数の確認 ===")
print("カテゴリ変数のユニークな値の数を確認し、分類の特徴を掴みます。\n")
print(df.select_dtypes(include=["object"]).nunique())

print("\n=== 数値データの相関関係を可視化 ===")
print("数値データのみを使い、相関関係をヒートマップで確認します。\n")
plt.figure(figsize=(12, 6))
sns.heatmap(df.select_dtypes(include=["number"]).corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("数値データの相関行列")
plt.show()

print("\n=== カテゴリ変数の数値化（Label Encoding） ===")
print("カテゴリ変数を数値に変換し、分析しやすくします。\n")
df_encoded = df.copy()
label_encoders = {}

for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le  # エンコーダーを保存

print("\n=== エンコード後の相関関係を可視化 ===")
print("カテゴリ変数を数値化した後、相関関係を再度ヒートマップで確認します。\n")
plt.figure(figsize=(12, 6))
sns.heatmap(df_encoded.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("エンコード後の相関行列")
plt.show()

print("\n=== 数値データの分布（ヒストグラム） ===")
print("各特徴量の分布をヒストグラムで確認します。\n")
df.select_dtypes(include=["number"]).hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.show()

print("\n=== カテゴリ変数の分布 ===")
print("各カテゴリ変数のデータの分布を可視化します。\n")
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    print(f"\n【{col} の分布】")
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=df, order=df[col].value_counts().index)
    plt.xticks(rotation=45)
    plt.title(f"{col}の分布")
    plt.show()
