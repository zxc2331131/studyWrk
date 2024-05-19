# -*- coding: utf-8 -*-
# 指定單張彩色圖片 產生出指定r對比強化值的灰階圖片   測試用小工具
"""
doImgCompare_test
Created on Mon Aug 28 15:41:05 2023

@author: zihong
"""

# 指定單張彩色圖片 產生出指定r對比強化值的灰階圖片
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

    
# 傳入x,y座標 及 圖票陣列
# 回傳灰階值
def get_gray_val(x,y,img):
    (b,g,r) = img[x,y]
    b = int(b)
    g = int(g)
    r = int(r)
    gray = r*0.299+g*0.587+b*0.114
    return gray

# 傳入r次方值,原圖像素list 
# 將匯入的原圖像素list中的值 依序的轉換
# 回傳轉換後的像素值list
def do_r_power_test(r,doArr):
    result = []
    str_point = 0
    # doArr 長度 log
    print('doArr長度 = ' + str(len(doArr)))
    # list 最大值
    maxVal = max(doArr)
    # list 最小值
    minVal = min(doArr)
    # 使用python內建的函數，能一次得到index跟值
    for index, val in enumerate(doArr):
        result.insert(index, (((val - minVal)/(maxVal - minVal)) ** r)*255)
        # 暫時用不到
        # if (((((val - minVal)/(maxVal - minVal)) ** r)*255 == 255) and str_point == 0):
        #     print('起始 index' + str(index))
        #     print('起始 value' + str(val))
        #     str_point = val
    return result

# 傳入r次方值,圖片陣列,原圖像素list 
# 將彩色圖片每個像素轉成灰階並套用轉換公式
# 將套用公式轉換後的圖片存檔
def do_r_power_pic(r,imgArr, pixArr):
    # 取得圖片資訊
    imgInfo = imgArr.shape
    # print(img)
    # (140, 163, 3)  高*寬 , 照片類別 全彩=3 灰階=1
    height = imgInfo[0]
    width = imgInfo[1]
    # list 最大值
    maxVal = max(pixArr)
    # list 最小值
    minVal = min(pixArr)
    # 處理圖像時,資料類別都np.uint8
    result = np.zeros((height,width,3),np.uint8)
    # 根據圖片 高,寬 依序取出每個pixel值
    # 將灰階圖 套入r轉換
    for i in range(0,height):
        for j in range(0,width):
            # 取出圖片像素對應灰階值
            gray = get_gray_val(i, j, imgArr)
            # 將灰階值套入公式做轉換
            gray = (((gray - minVal)/(maxVal - minVal)) ** r)*255
            #  以一樣的高 寬像素位置塞回去
            result[i,j] = np.uint8(gray)
    cv2.imshow('grayArr',result)
    grayArr=Image.fromarray(result)
    grayArr.save('compare_Pic/12651test.bmp','bmp')
    # cv2.waitKey(0)
    cv2.destroyWindow('grayArr')
    return result


# 彩色圖片
img = cv2.imread('KC_Pic/KC_Pic_demo/2.bmp',1)
# 取得圖片資訊
imgInfo = img.shape
# print(img)
# (140, 163, 3)  高*寬 , 照片類別 全彩=3 灰階=1
height = imgInfo[0]
width = imgInfo[1]
# original =>原圖像素list
# compare =>公式轉換後像素list
org_arr = {'original':[],
           'compare':[],}
# 基礎轉灰階 產生>原圖像素list
for i in range(0,height):
    for j in range(0,width):
        # 取出圖片像素對應灰階值
        gray = get_gray_val(i, j, img)
        # 先將所有灰階像素都存起來 後續做畫直方圖 及轉換時最大最小值使用
        org_arr['original'].append(gray)
# set()去除list中重複的值
org_arr['original'] = list(set(org_arr['original']))
# 從小到大排序
org_arr['original'].sort()
# 宣告測試的 r次方 , 套入下面判斷
r_p = 0.4
org_arr['compare'] = do_r_power_test(r_p,org_arr['original'])
print(max(org_arr['original']))
print(min(org_arr['original']))
# 將灰階圖 套入r轉換  帶入統一宣告的r次方 , 圖片彩色array, 圖片灰階像素array
# 並產生圖片 
do_r_power_pic(r_p, img, org_arr['original'])
# 畫原圖/對比圖 像素直方圖
plt.plot(org_arr['original'], org_arr['compare'], color='b')
plt.xlabel('original') # 設定x軸標題
plt.ylabel('compare') # 設定y軸標題
plt.title('r') # 設定圖表標題
plt.show()


