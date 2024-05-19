# -*- coding: utf-8 -*-
"""
根據最終結果excel表 畫出誤差圖
Created on Mon Mar  4 15:40:35 2024

@author: zihong
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# 全局设置支持中文显示的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 假設你已經將文件上傳，並且文件路徑是 '/mnt/data/final_acc_file.xlsx'
file_path = './final_acc_file.xlsx'

# 讀取 Excel 文件
df = pd.read_excel(file_path)

# 顯示 DataFrame 的前幾行，以確認內容
print(df.head())

# 確認 'ACC_O' 欄位存在
if 'ACC_O' in df.columns:
    # 計算 'ABS_O' 的值
    df['ABS_O_acc'] = (df['ACC_O'] - 1).abs()
    # 顯示 'ACC_O' 和 'ABS_O' 欄位以確認結果
    print(df[['ACC_O', 'ABS_O_acc']])
else:
    print("DataFrame 中未找到 'ACC_O' 欄位。")
    
    
    
    
# 獲取所有以 "_acc" 結尾的欄位名稱
acc_columns = [col for col in df.columns if col.endswith('_acc')]

# 初始化一個字典來儲存結果
results = {}

# 對每個符合條件的欄位進行操作
for col in acc_columns:
    # 計算每個欄位的值加總後除以 63
    results[col] = df[col].sum() / 63

# 顯示結果
print(results)


# 定義我們感興趣的前綴
prefixes = [
    "Predict_G_Pic_Cross_11",
    "Predict_G_Pic_Plus",
    "Predict_Pic_Cross_11",
    "Predict_Pic_Plus"
]

# 初始化一個字典來儲存每個前綴的最小值和相應的欄位名
min_values = {}

# 對每個前綴進行操作
for prefix in prefixes:
    # 篩選出以該前綴開頭的欄位
    filtered_results = {k: v for k, v in results.items() if k.startswith(prefix)}
    
    # 找出這些欄位中的最小值
    if filtered_results:
        min_col = min(filtered_results, key=filtered_results.get)
        min_value = filtered_results[min_col]
        min_values[prefix] = (min_col, min_value)
    else:
        min_values[prefix] = ("None", "N/A")  # 如果沒有找到匹配的欄位

# 顯示每個前綴的最小值所在的欄位
for prefix, (col, value) in min_values.items():
    print(f"{prefix}: {col} with min value of {value}")
    
    
# 定義欄位名稱
columns = [
    "KC_val",
    "Pic_Gray",
    "G_Pic_Gray",
    "Pic_Global",
    "G_Pic_Global",
    "Pic_Cross",
    "G_Pic_Cross",
]

# # 從 results 字典中提取特定欄位的值
# values = [results.get(col, 0) for col in columns]  # 如果某個欄位不存在於 results 中，使用 0 作為默認值


# # 繪製長條圖
# plt.figure(figsize=(10, 8))  # 設置圖形大小
# plt.bar(columns, values, color='skyblue')  # 繪製長條圖，顏色為天藍色
# plt.xlabel('Columns')  # X軸標籤
# plt.ylabel('Values')  # Y軸標籤
# plt.title('Bar Chart of Selected Columns')  # 圖形標題
# plt.xticks(rotation=90)  # 將X軸標籤旋轉90度，以避免重疊
# plt.tight_layout()  # 自動調整子圖參數，以使之填充整個圖表區域
# plt.show()  # 顯示圖形


# 從 results 字典中提取特定欄位的值
values = [results.get(col, 0) for col in columns]

# # 生成一個顏色列表，為每條長條指定不同的顏色
# colors = plt.cm.jet(np.linspace(0, 1, len(columns)))

# # 繪製橫向長條圖
# plt.figure(figsize=(10, 8))  # 設置圖形大小
# plt.barh(columns, values, color=colors)  # 使用 barh 函數繪製橫向長條圖，並指定顏色
# plt.ylabel('Columns')  # Y軸標籤（現在表示欄位名）
# plt.xlabel('Values')  # X軸標籤（現在表示值）
# plt.title('Horizontal Bar Chart of Selected Columns')  # 圖形標題
# plt.tight_layout()  # 自動調整子圖參數，以使之填充整個圖表區域
# plt.show()  # 顯示圖形

# # 指定顏色列表
# colors = [
#     '#1f77b4',  # 淡蓝色
#     '#ff7f0e',  # 橙色
#     '#2ca02c',  # 淡绿色
#     '#d62728',  # 砖红色
#     '#9467bd',  # 淡紫色
#     '#8c564b',  # 棕色
#     '#e377c2',  # 粉红色
#     '#7f7f7f',  # 灰色
#     '#bcbd22',  # 橄榄绿
#     '#17becf'   # 天蓝色
# ]

# # 繪製橫向長條圖，為每條長條指定顏色
# plt.figure(figsize=(10, 8))
# plt.barh(columns, values, color=colors)

# # 添加平均值參考線
# print('acccccccccccc')
# print(results.get("ABS_O_acc", 0))
# plt.axvline(x=results.get("ABS_O_acc", 0), color='red', linestyle='--', label='參考值')

# plt.xlabel('Values')
# plt.title('Horizontal Bar Chart of Selected Columns with Specified Colors')
# plt.legend()
# plt.tight_layout()
# plt.show()


# 假设的 results 字典
# 原組
results = {
    "Pic_Gray": 1.394,
    "G_Pic_Gray": 1.416,
    "Pic_Global": 1.429,
    "G_Pic_Global": 1.388,
    "Pic_Cross": 1.429,
    "G_Pic_Cross": 1.429,
    "KC_val": 1.48173  # 假设的 ANS_O_acc 值
}
# # 對照組
# results = {
#     "Pic_Gray": 1.397,
#     "G_Pic_Gray": 1.461,
#     "Pic_Global": 1.427,
#     "G_Pic_Global": 1.389,
#     "Pic_Cross": 1.469,
#     "G_Pic_Cross": 1.427,
#     "KC_val": 1.48173  # 假设的 ANS_O_acc 值
# }

# 定义颜色列表
colors = [
    '#7f7f7f',  # 灰色
    '#7f7f7f',  # 灰色
    '#7f7f7f',  # 灰色
    '#7f7f7f',  # 灰色
    '#7f7f7f',  # 灰色
    '#7f7f7f',  # 灰色
    '#7f7f7f',  # 灰色
]
# 定义欄位名稱
values = [results[col] for col in columns]

# 绘制横向长条图
plt.figure(figsize=(10, 8))
bars = plt.barh(columns, values, color=colors[:len(columns)])

# 在每个条形上增加数值标注
for bar in bars:
    width = bar.get_width()  # 获取条形的宽度
    label_x_pos = bar.get_width() + 0.01  # 设置数值标注的位置为条形的右侧
    plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width}', va='center_baseline')

# 使用 ANS_O_acc 的值添加参考线
ans_o_acc_value = results.get("KC_val", 0)  # 使用 get 方法安全地获取值，如果不存在则返回 0
plt.axvline(x=ans_o_acc_value, color='r', linestyle='--', label='KC_val')

plt.xlabel('r')
plt.title('Horizontal Bar Chart of Selected Columns with Specified Colors')
plt.legend()
plt.tight_layout()
plt.show()

