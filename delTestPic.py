# -*- coding: utf-8 -*-
"""
快速刪除 預設背景值 /ostsu / gaus test picture
Created on Wed Mar  8 10:39:24 2023

@author: zihong
"""

import cv2
import numpy as np
import os
import shutil
from PIL import Image
from decimal import Decimal
import statistics
import operator

#刪除路徑檔案
# 刪除陣列中比對用的檔案  jpg檔
predict_source = 'D:/KC_AOI_Project/wrk/study_wrk/compare_Pic/'
gray_source = 'D:/KC_AOI_Project/wrk/study_wrk/G_KC_Pic/'
fileList = os.listdir(predict_source)
for fname in fileList:
    try:
        pic_no = fname.split('.')
        # print(gray_source + pic_no[0] + '.' + pic_no[1] + '_o' + '.bmp')
        os.remove(gray_source + pic_no[0] + '.' + pic_no[1] + '_o' + '.bmp')
        os.remove(gray_source + pic_no[0] + '.' + pic_no[1] + '_gaus' + '.bmp')
        os.remove(gray_source + pic_no[0] + '.' + pic_no[1] + '_otsu' + '.bmp')
    except OSError as e:
        print(e)
    else:
        print(predict_source + fname + "File is deleted successfully")