# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:19:30 2023

@author: zihong
"""

# 批次將KC_EXCEL XLSX檔中讀出來
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def plt_hist(img):
    # 將圖片像素數據平鋪
    print(img.ravel().shape)
    # 畫長方圖
    plt.hist(img.ravel(),256,[0,256])
    plt.show



def readCSV():
    CSVArr= ['KC_EXCEL/KC2.xlsx' ,'KC_EXCEL/KC3.xlsx']
    resltArr = []
    print(resltArr)
    for val in CSVArr:
        print(val)
        df = pd.read_excel(val,header=1,usecols='P')
        # for dfVal in df["Defect Name"]:
        for dfVal in df:
            print(dfVal)
            resltArr.append(dfVal)
        # print(resltArr)
    
    return resltArr

# defectTypArr = []
# # 取出KC2,KC3  excel表格內 Defact Name 爛位值, 輸出array
# defectTypArr = readCSV()
# print('==========defectTypArr')
# print(defectTypArr)

CSVArr = ['KC_EXCEL/KC2.xlsx', 'KC_EXCEL/KC3.xlsx']
df_list = []

for val in CSVArr:
    df = pd.read_excel(val, header=1)  # 使用第二行作为列標題
    df_list.append(df)
    
for df in df_list:
    print('df',df)    
    N_column = df['RealSize']
    Q_column = df['Length']
    print('N_column',N_column)    
    print('Q_column',Q_column)    
# CSVArr = ['KC_EXCEL/KC2.xlsx', 'KC_EXCEL/KC3.xlsx']
# resltArr = []
# df_list = []

# for val in CSVArr:
#     print(val)
#     df = pd.read_excel(val,header=1,usecols='P')
#     for dfVal in df:
#         df_list.append(dfVal)

# for val in CSVArr:
#     df = pd.read_excel(val, header=1)  # 读取Excel文件，忽略标题行
#     print(df)
#     df_list.append(df)  # 将每个DataFrame添加到列表中
#     # df_list.append(df[1])  # 将每个DataFrame添加到列表中

# 合并所有数据帧
combined_df = pd.concat(df_list, axis=0, ignore_index=True)

# 如果需要将合并后的DataFrame保存到一个文件中，可以使用以下代码：
# combined_df.to_excel('combined_data.xlsx', index=False)

print(combined_df.head(n=10))  # 查看前10列


# result_folder = 'D:/KC_AOI_Project/wrk/study_wrk/predict_result/'
# final_folder = 'D:/KC_AOI_Project/wrk/study_wrk/p_final_result/'
# # 使用os.walk()遞迴獲取所有子資料夾
# for root, dirs, files in os.walk(result_folder):
#     for folder in dirs:
#         folder_path = os.path.join(root, folder)
#         print("子資料夾:", folder_path)
#         print("子資料夾:", folder)
#         # 檢查是否存在名稱為"AAA"的資料夾
#         chk_folder = os.path.join(final_folder, folder)
        
#         if not os.path.exists(chk_folder):
#             # 如果不存在，則創建它
#             os.makedirs(chk_folder)
#             print("已創建名稱為'"+folder+"'的資料夾。")
#         else:
#             print("名稱為'"+folder+"'的資料夾已存在。")
        
#         # 使用glob.glob()抓取資料夾中的所有圖片文件
#         image_files = glob.glob(os.path.join(folder_path, '*.bmp'))  # 此處可以更改檔案擴展名
#         for image_file in image_files:
#             print("圖片文件:", image_file)

