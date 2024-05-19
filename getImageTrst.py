# 批次將圖片從KC_EXCEL XLSX檔中讀出來並轉檔另存為灰階圖片 作為YOLOv4 model訓練用
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
        for dfVal in df["Defect Name"]:
            resltArr.append(dfVal)
        # print(resltArr)
    
    return resltArr

defectTypArr = []
# 取出KC2,KC3  excel表格內 Defact Name 爛位值, 輸出array
defectTypArr = readCSV()
# print('==========defectTypArr')
# print(defectTypArr)


# 下for迴圈 處理資料夾內所有圖片檔案
for file_no in range(1, 64): 
    print('=======進行第'+str(file_no)+'張圖')
    # 讀取圖片
    img = cv2.imread('KC_Pic/KC_Pic_demo/'+str(file_no)+'.bmp',1)
    
    # 取得圖片資訊
    imgInfo = img.shape
    # print(img)
    # (140, 163, 3)  高*寬 , 照片類別 全彩=3 灰階=1
    height = imgInfo[0]
    width = imgInfo[1]

    # RGB R=G=B = gray （r*0.299+g*0.587+b*0.114
    # 處理圖像時,資料類別都np.uint8
    # numpy.zeros(shape，dtype=float，order = 'C')
    grayArr = np.zeros((height,width,3),np.uint8)
    # 每個像素位元 依序處理
    print('=======像素bgr')

    for i in range(0,height):
        for j in range(0,width):
            (b,g,r) = img[i,j]
            b = int(b)
            g = int(g)
            r = int(r)
            gray = r*0.299+g*0.587+b*0.114
            #  以一樣的高 寬像素位置
            grayArr[i,j] = np.uint8(gray)

    # print(grayArr)
    cv2.imshow('grayArr',grayArr)
    grayArr=Image.fromarray(grayArr)
    # 取得 defactName 類別順序
    dn_sort = file_no-1
    print(dn_sort)
    print(str(file_no))
    dn_str = defectTypArr[dn_sort]
    dn_str_arr = dn_str.split(' ')
    dn_str = dn_str_arr[0]
    print(dn_str)
    # gray_file = str(file_no)+'_'+ dn_str
    # 產生YOLO 預處理資料檔名從簡
    gray_file = str(file_no)
    # grayArr.save('G_KC_Pic/'+ gray_file +'.bmp','bmp')
    # 轉入YOLO 預處理的檔案格式為jpg
    grayArr.save('D:/Yolo_v4/darknet/build/darknet/x64/data/obj/koi/'+ gray_file +'.jpg')
    # cv2.waitKey(0)
    cv2.destroyWindow('grayArr')

    print('=======進行第'+str(file_no)+'張圖 結束')


