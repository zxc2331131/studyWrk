# -*- coding: utf-8 -*-
# readImgByOriginal.py
# 將KC_Pic_Demo 經高斯處理/不經高斯處理 存為灰階圖片
"""
Created on Fri Sep 15 11:24:35 2023

@author: zihong
"""

from PIL import Image, ImageFilter
import os

predict_source = 'D:/KC_AOI_Project/wrk/study_wrk/KC_Pic/KC_Pic_demo/'
target_source = 'D:/KC_AOI_Project/wrk/study_wrk/predict_pic/KC_Pic_Original/'
gaus_source = 'D:/KC_AOI_Project/wrk/study_wrk/predict_pic/G_KC_Pic_Original/'
# 高斯濾波的標準差
sigma = 2.0
# 資料夾文件列表
file_list = os.listdir(predict_source)

for file_name in file_list:
    # 檢查文件擴展名是否為常見的圖像格式，如.jpg、.png等
    if file_name.lower().endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp')):
        # 組成完整的文件路徑
        file_path = os.path.join(predict_source, file_name)

        color_image = Image.open(file_path)
        # 不用進行高斯濾波處理
        # 轉換為灰度圖像
        grayscale_image = color_image.convert('L')

        new_file_name = os.path.splitext(file_name)[0] + '.bmp'
        new_file_path = os.path.join(target_source, new_file_name)

        grayscale_image.save(new_file_path)
        
        # 對彩色圖像進行高斯濾波處理
        blurred_image = color_image.filter(ImageFilter.GaussianBlur(sigma))
        # 轉換為灰度圖像
        grayscale_image = blurred_image.convert('L')

        new_file_name = os.path.splitext(file_name)[0] + '.bmp'
        new_file_path = os.path.join(gaus_source, new_file_name)

        grayscale_image.save(new_file_path)

        color_image.close()
