# 將每張彩色圖片轉成灰階 並保留最佳r值
# 
import cv2
import numpy as np
import os
import shutil
from PIL import Image
from decimal import Decimal
import statistics
import operator

#刪除路徑檔案
def delDir(fileList):
    # 刪除陣列中比對用的檔案  jpg檔
    predict_source = 'D:/KC_AOI_Project/wrk/study_wrk/compare_Pic/'
    gray_source = 'D:/KC_AOI_Project/wrk/study_wrk/G_KC_Pic/'
    for fname in fileList:
        try:
            os.remove(predict_source + fname + '.bmp')
            os.remove(gray_source + fname + '.bmp')
        except OSError as e:
            print(e)
        else:
            print(predict_source + fname + "File is deleted successfully")
    return
#讀取模型與訓練權重
def initNet():
    # CONFIG = './train_finished_1/yolov4-tiny-myobj.cfg'
    CONFIG = 'D:/Yolo_v4/darknet/build/darknet/x64/cfg/yolov4-koi.cfg'
    # WEIGHT = './yolov4-tiny-myobj_last.weights'
    WEIGHT = 'D:/Yolo_v4/darknet/build/darknet/x64/backup/yolov4-koi_last.weights'
    # WEIGHT = './train_finished/yolov4-tiny-myobj_last.weights'
    net = cv2.dnn.readNet(CONFIG, WEIGHT)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255.0)
    model.setInputSwapRB(True)
    return model

#物件偵測
def nnProcess(image, model):
    classes, confs, boxes = model.detect(image, 0.4, 0.1)
    return classes, confs, boxes

#框選偵測到的物件，並裁減
def drawBox(image, classes, confs, boxes):
    new_image = image.copy()
    for (classid, conf, box) in zip(classes, confs, boxes):
        x, y, w, h = box
        if x - 18 < 0:
            x = 18
        if y - 18 < 0:
            y = 18
        cv2.rectangle(new_image, (x - 18, y - 18), (x + w + 20, y + h + 24), (0, 255, 0), 3)
    return new_image

# 裁減圖片
def cut_img(image, classes, confs, boxes):
    cut_img_list = []
    for (classid, conf, box) in zip(classes, confs, boxes):
        x, y, w, h = box
        if x - 18 < 0:
            x = 18
        if y - 18 < 0:
            y = 18
        cut_img = image[y - 18:y + h + 20, x - 18:x + w + 25]
        cut_img_list.append(cut_img)
    return cut_img_list[0]

# 傳入x,y座標 及 圖票陣列
# 回傳灰階值
def get_gray_val(x,y,img):
    (b,g,r) = img[x,y]
    b = int(b)
    g = int(g)
    r = int(r)
    gray = r*0.299+g*0.587+b*0.114
    return gray


# 傳入r次方值,圖片陣列,原圖像素list 
# 將彩色圖片每個像素轉成灰階並套用轉換公式
# 將套用公式轉換後的圖片存檔
# 
def do_best_r_pic(imgArr, pixArr, str_r, do_unit, pic_no):
    # 取得圖片資訊
    imgInfo = imgArr.shape
    # print(img)
    # (140, 163, 3)  高*寬 , 照片類別 全彩=3 灰階=1
    height = imgInfo[0]
    width = imgInfo[1]
    # list 最大值
    # maxVal = max(pixArr)
    # 因精度關係用 Decimal(str()) 避免浮點數小數位過多 
    maxVal = Decimal(str(max(pixArr)))
    # list 最小值
    # minVal = min(pixArr)
    minVal = Decimal(str(min(pixArr)))
    # 處理圖像時,資料類別都np.uint8
    result = np.zeros((height,width,3),np.uint8)
    # 產生檔案Log 陣列
    dir_log = {}
    # for 1-4 迴圈 根據str_r減掉 do_unit*for迴圈當下數字 = 得到r值
    if do_unit == 0.01 :
        str_num = 1
    else:
        str_num = 0
        
    for do_num in range(str_num,10):
        now_r = Decimal(str(str_r)) + Decimal(str(do_unit)) * Decimal(str(do_num))
        print('+項'+str(now_r))
        print(now_r)
        # if do_unit == 0.01 :
        #     print(maxVal)
        #     print('123454654894984894')
        #     print(minVal)
        # 根據圖片 高,寬 依序取出每個pixel值
        # 將灰階圖 套入r轉換
        for i in range(0,height):
            for j in range(0,width):
                # 取出圖片像素對應灰階值
                gray = Decimal(str(get_gray_val(i, j, imgArr)))
                if do_unit == 0.01 :
                    print('Decimal')
                    print(str(get_gray_val(i, j, imgArr)))
                    print('gray')
                    print(gray)
                    print('minVal')
                    print(minVal)
                    print('maxVal')
                    print(maxVal)
                    print('now_r')
                    print(now_r)
                # 將灰階值套入公式做轉換
                # gray = Decimal(((Decimal(gray - minVal)/Decimal(maxVal - minVal)) ** now_r)*Decimal(255))
                gray = (((gray - minVal)/(maxVal - minVal)) ** now_r)*Decimal('255')
                if do_unit == 0.01 :
                    print('score============>')
                    print((((gray - minVal)/(maxVal - minVal)) ** now_r))
                    print('gray============>')
                    print(gray)
                # gray = Decimal((((gray - minVal)/(maxVal - minVal)) ** now_r)*Decimal('255'))
                # gray = (((gray - minVal)/(maxVal - minVal)) ** now_r)*255
                #  以一樣的高 寬像素位置塞回去
                result[i,j] = np.uint8(gray)
        # cv2.imshow('grayArr',result)
        grayArr=Image.fromarray(result)
        # grayArr.save('compare_Pic/' + str(pic_no) + '_' + str(now_r) + '.bmp','bmp')
        grayArr.save('G_KC_Pic/' + str(pic_no) + '_' + str(now_r) + '.bmp','bmp')
        dir_log[str(pic_no) + '_' + str(now_r)] = 0
        grayArr.save('compare_Pic/' + str(pic_no) + '_' + str(now_r) + '.bmp','bmp')
        # grayArr.save('D:/Yolo_v4/darknet/build/darknet/x64/data/test/koi/' + pic_no + '_' + now_r + '.jpg','jpg')
        # cv2.destroyWindow('grayArr')
        if do_num > 0:
            now_r = Decimal(str(str_r)) - Decimal(str(do_unit)) * Decimal(str(do_num))
            # now_r = Decimal(str_r) - (do_unit * Decimal(do_num))
            print('-項'+str(now_r))
            # print(Decimal(str(do_unit * do_num)))
            # 根據圖片 高,寬 依序取出每個pixel值
            # 將灰階圖 套入r轉換
            for i in range(0,height):
                for j in range(0,width):
                    # 取出圖片像素對應灰階值
                    gray = Decimal(str(get_gray_val(i, j, imgArr)))
                    # gray = get_gray_val(i, j, imgArr)
                    # 將灰階值套入公式做轉換
                    gray = (((gray - minVal)/(maxVal - minVal)) ** now_r)*Decimal('255')
                    # gray = (((gray - minVal)/(maxVal - minVal)) ** now_r)*255
                    #  以一樣的高 寬像素位置塞回去
                    result[i,j] = np.uint8(gray)
            # cv2.imshow('grayArr',result)
            grayArr=Image.fromarray(result)
            # grayArr.save('compare_Pic/' + str(pic_no) + '_' + str(now_r) + '.bmp','bmp')
            # 在G_KC_Pic也產生最佳k值灰階圖
            grayArr.save('G_KC_Pic/' + str(pic_no) + '_' + str(now_r) + '.bmp','bmp')
            dir_log[str(pic_no) + '_' + str(now_r)] = 0
            # 產生yolo v4要預測用的圖檔 需要為jpg
            grayArr.save('compare_Pic/' + str(pic_no) + '_' + str(now_r) + '.bmp','bmp')
            # grayArr.save('D:/Yolo_v4/darknet/build/darknet/x64/data/test/koi/' + str(pic_no) + '_' + str(now_r) + '.jpg')
            # cv2.destroyWindow('grayArr')
    print(dir_log)
            
    return dir_log

# 傳入r次方值,圖片陣列,原圖像素list 
# 將彩色圖片每個像素轉成灰階並套用轉換公式
# 將套用公式轉換後的圖片存檔
# 
# def do_r_power_pic(r, imgArr, pixArr):
#     # 取得圖片資訊
#     imgInfo = imgArr.shape
#     # print(img)
#     # (140, 163, 3)  高*寬 , 照片類別 全彩=3 灰階=1
#     height = imgInfo[0]
#     width = imgInfo[1]
#     # list 最大值
#     maxVal = max(pixArr)
#     # list 最小值
#     minVal = min(pixArr)
#     # 處理圖像時,資料類別都np.uint8
#     result = np.zeros((height,width,3),np.uint8)
#     # 根據圖片 高,寬 依序取出每個pixel值
#     # 將灰階圖 套入r轉換
#     for i in range(0,height):
#         for j in range(0,width):
#             # 取出圖片像素對應灰階值
#             gray = get_gray_val(i, j, imgArr)
#             # 將灰階值套入公式做轉換
#             gray = (((gray - minVal)/(maxVal - minVal)) ** r)*255
#             #  以一樣的高 寬像素位置塞回去
#             result[i,j] = np.uint8(gray)
#     cv2.imshow('grayArr',result)
#     grayArr=Image.fromarray(result)
#     grayArr.save('compare_Pic/12651test.bmp','bmp')
#     # cv2.waitKey(0)
#     cv2.destroyWindow('grayArr')
#     return result
        
def do_pic_info():
    # 指定YOLO 4 test圖片路徑
    # source = 'D:/Yolo_v4/darknet/build/darknet/x64/data/test/koi/'
    source = 'D:/KC_AOI_Project/wrk/study_wrk/KC_Pic/KC_Pic_demo/'
    # predict_source = 'D:/KC_AOI_Project/wrk/study_wrk/compare_Pic/'
    predict_source = 'D:/KC_AOI_Project/wrk/study_wrk/G_KC_Pic/'
    # 取出路徑下所有圖片 並存成list
    files = os.listdir(source)
    print('※ 資料夾共有 {} 張圖檔'.format(len(files)))
    print('※ 開始執行YOLOV4 test圖片最佳r次方 模糊化處理...')
    # 宣告 訓練好的YOLOv4 模型
    model = initNet()
    # success = fail = uptwo = 0
    number = 1
    for file in files:
        print(' ▼ 第{}張'.format(number))
        print(' ▼ 檔名: {}'.format(file))
        # img = cv2.imdecode(np.fromfile(source+file, dtype=np.uint8), -1)
        # classes, confs, boxes = nnProcess(img, model)
        # 分割檔名編號
        pic_no = file.split('.')
        # 產生單一張圖片 補資料用
        # pic_no[0] = str(32)
        # 彩色圖片
        img = cv2.imread('KC_Pic/KC_Pic_demo/'+ pic_no[0] +'.bmp',1)
        # 取得圖片資訊
        imgInfo = img.shape
        # print(img)
        # (140, 163, 3)  高*寬 , 照片類別 全彩=3 灰階=1
        height = imgInfo[0]
        width = imgInfo[1]
        # pixel_arr =>原圖像素list
        pixel_arr = []
        # 基礎轉灰階 產生>原圖像素list
        for i in range(0,height):
            for j in range(0,width):
                # 取出圖片像素對應灰階值
                gray = get_gray_val(i, j, img)
                # 先將所有灰階像素都存起來 後續做畫直方圖 及轉換時最大最小值使用
                pixel_arr.append(gray)
        # 取出數量最多的值
        # 保留處理好原圖灰階陣列
        # set()去除list中重複的值
        pixel_arr = list(set(pixel_arr))
        # 從小到大排序
        pixel_arr.sort()
        # print(pixel_arr)
        # for 產生r次方 入暫時dir 輸入單位值 找出最K值
        # 傳入資料都是原圖彩色資料
        pic_log_arr = do_best_r_pic(img, pixel_arr, 1, 0.1,pic_no[0])
        # 存放套入model 預測的值
        for pic_dir  in pic_log_arr:
            # 123
            img = cv2.imdecode(np.fromfile(predict_source + pic_dir+'.bmp', dtype=np.uint8), -1)
            classes, confs, boxes = nnProcess(img, model)
            # 將classes不重複類別取出
            class_uniq = np.unique(classes)
            # 取得target YOLO判斷準確率的最大值
            target_area = 0
            # 如果有判斷出兩個clasees
            if class_uniq.size == 2:
                # 判斷整張圖片 yolo有判斷尺標 跟 目標物 兩種類別
                # 有判斷出兩種類別的, 拿掉尺標得資料陣列,以目標物資料做最大平均值篩選
                print('==========有兩種類別喔===========')
                for (classid, conf, box) in zip(classes, confs, boxes):
                    x, y, w, h = box
                    if (classid == 0):
                        if (target_area > 0):
                            if (w*h > target_area) :
                                target_area = w*h
                        else:
                            target_area = w*h
                # 取目標物面積最大的
                # pic_log_arr[pic_dir] = target_area   
                # classes陣列中只有 0 1兩種 ,argmax抓陣列中最大值一定能取到1在陣列中的位置, 假設1:尺標在一張圖只有一個
                ruler_sort = np.argmax(classes)
                classes = np.delete(classes, ruler_sort, 0)
                confs = np.delete(confs, ruler_sort, 0)
                boxes = np.delete(boxes, ruler_sort, 0)
                # 取加權
                # pic_log_arr[pic_dir] = target_area*statistics.mean(confs)*100
                # 取正確率平均最高ㄉ
                pic_log_arr[pic_dir] = statistics.mean(confs)*100
            # 正常要判斷出兩個classes ,沒有的話考慮有可能被判斷成類別的尺標所以做例外處理
            if class_uniq.size == 1:
                # 判斷整張圖片 yolo有判斷尺標 跟 目標物 兩種類別
                # 有判斷出兩種類別的, 拿掉尺標得資料陣列,以目標物資料做最大平均值篩選
                print('==========有兩種類別喔===========')
                for (classid, conf, box) in zip(classes, confs, boxes):
                    x, y, w, h = box
                    if (target_area > 0):
                        if (w*h > target_area) :
                            target_area = w*h
                    else:
                        target_area = w*h
                # 取目標物面積最大的
                pic_log_arr[pic_dir] = target_area   
        
        print(pic_log_arr)
        print('最大值')
        print(max(pic_log_arr.items(), key=operator.itemgetter(1))[0])
        # 將平均得分最高的圖檔名稱 從log_arr裡面unset掉
        del pic_log_arr[max(pic_log_arr.items(), key=operator.itemgetter(1))[0]]
        print(pic_log_arr)
        
        # 將剩餘的圖檔log arr 丟進delDir()去刪除檔案
        # delDir(pic_log_arr)
        # 產生單一張圖片 補資料用
        # break
    
        # 小數點後兩位k值尚不進行
        # 分割出下一層K 起始值
        # next_k_num = max(pic_log_arr).split('_')
        # print(next_k_num)
        # # for 產生r次方 入暫時dir 輸入單位值 找出最K值
        # pic_log_arr = do_best_r_pic(img, pixel_arr, next_k_num[1], 0.01 ,pic_no[0])
        # for pic_dir  in pic_log_arr:
        #     # 123
        #     img = cv2.imdecode(np.fromfile(predict_source + pic_dir+'.jpg', dtype=np.uint8), -1)
        #     classes, confs, boxes = nnProcess(img, model)
        #     pic_log_arr[pic_dir] = statistics.mean(confs)
        #     print(confs)
        
        # print(pic_log_arr)
        # print(max(pic_log_arr))
        # 將平均得分最高的圖檔名稱 從log_arr裡面unset掉
        # del pic_log_arr[max(pic_log_arr)]
    return imgInfo

if __name__ == '__main__':
    # 先將原有test檔案路經 經取出最R次方依序處理過
    do_pic_info()