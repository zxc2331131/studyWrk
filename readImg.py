# readImg.py
# do_pic_info ==>將每張彩色圖片轉成灰階 並保留最佳r值 => 現已拉出去獨立在doImgCompare.py 但依舊保留原版程式 避免出錯
# doOtsu ==>global對比強化
# 切出灰階 target圖片 每個target 經高斯處理/未經高斯處理 為各自一張, 並產生一張完整otsu bmp檔方便比對
import cv2
import numpy as np
import os
import shutil
from PIL import Image
from decimal import Decimal
import statistics
import operator
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
import copy
from tensorflow.keras.applications import MobileNetV2
from tensorflow import keras
# callback
from IPython.display import clear_output

NUM_CLASSES = 4 # class number + 1 (background)
INPUT_SHAPE = [480, 640, 3] # (H, W, C)
BATCH_SIZE = 2
EPOCHS = 100
VAL_SUBSPLITS = 1


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def normalize(image):
    image = image / 127.5 - 1
    return image


def resize_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


def resize_label(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.NEAREST)
    new_image = Image.new('L', size, (0))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

# 處理 資料夾所有圖片 predict後印出輪廓並存檔
def detect_dir_image(pr_result,do_count,file,do_typ=''):
    if (do_detect_flag == 1):
        if (do_typ == 'gaus'):
            
            model = keras.models.load_model('D:/KC_AOI_Project/wrk/study_wrk/modelLogs/G_Pluslogs2_14/the-last-model.h5')
            # 預測的資料夾
            dir_path = 'D:/KC_AOI_Project/wrk/study_wrk/predict_pic/G_KC_Pic_Plus_14/'
        else:
            model = keras.models.load_model('D:/KC_AOI_Project/wrk/study_wrk/modelLogs/Pluslogs2_14/the-last-model.h5')
            # 預測的資料夾
            dir_path = 'D:/KC_AOI_Project/wrk/study_wrk/predict_pic/KC_Pic_Plus_14/'
            
        voc_no = file.split('_')
        image = Image.open(dir_path + str(voc_no[0]) + '_' + str(do_count) + '.jpg')
        # 將圖片轉為RGB
        image = cvtColor(image)
        ori_h = np.array(image).shape[0]
        ori_w = np.array(image).shape[1]
    
        image_data, nw, nh = resize_image(image, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
    
        image_data = normalize(np.array(image_data, np.float32))
    
        image_data = np.expand_dims(image_data, 0)
        # model預測
        pr = model.predict(image_data)[0]
        
    
        pr = pr[int((INPUT_SHAPE[0] - nh) // 2) : int((INPUT_SHAPE[0] - nh) // 2 + nh), \
                int((INPUT_SHAPE[1] - nw) // 2) : int((INPUT_SHAPE[1] - nw) // 2 + nw)]
    
        pr = cv2.resize(pr, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
    
        pr = pr.argmax(axis=-1)
    
        # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        # for c in range(NUM_CLASSES):
        #     seg_img[:, :, 0] += ((pr[:, :] == c ) * colors[c][0]).astype('uint8')
        #     seg_img[:, :, 1] += ((pr[:, :] == c ) * colors[c][1]).astype('uint8')
        #     seg_img[:, :, 2] += ((pr[:, :] == c ) * colors[c][2]).astype('uint8')
        # 預測面積
        seg_img = np.reshape(
            np.array(colors, np.uint8)[np.reshape(pr, [-1])], [ori_h, ori_w, -1])
                        
        # 骰選掉太小 過度擬和的點
        # 創建一個掩碼，用於標記紅色區域
        red_mask = (seg_img[:, :, 0] == 128) & (seg_img[:, :, 1] == 0) & (seg_img[:, :, 2] == 0)
        
        # 定義周圍的像素偏移
        # 3*3
        # neighborhood_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        # 5*5
        neighborhood_offsets = [
            (2, 0), (-2, 0), (1, 0), (-1, 0), (0, 2), (0, -2), (0, 1), (0, -1),  # 中心像素的上下左右邻居
            (1, 1), (-1, 1), (1, -1), (-1, -1)  # 对角线上的像素
        ]
        # 7*7
        # neighborhood_offsets = [
        #     (3, 0), (-3, 0), (2, 0), (-2, 0), (1, 0), (-1, 0), (0, 3), (0, -3),  # 中心像素的上下左右邻居
        #     (0, 2), (0, -2), (1, 1), (-1, 1), (1, -1), (-1, -1),  # 对角线上的像素
        #     (2, 1), (-2, 1), (2, -1), (-2, -1),  # 更远的像素
        # ]
        
        # 定義周圍紅色點的最小數量閾值
        min_red_neighbors = 9  # 根據需要調整閾值
        
        # 獲取圖像高度和寬度
        height, width = red_mask.shape
        
        # 創建一個標記掩碼，用於記錄已檢查的點
        checked_mask = np.zeros_like(red_mask, dtype=bool)
        
        # 遍歷標記的紅色區域
        for y in range(height):
            for x in range(width):
                if red_mask[y, x] and not checked_mask[y, x]:
                    red_neighbors = 0
                    queue = [(x, y)]
                    checked_mask[y, x] = True
                    # 使用隊列來叠代周圍的紅色點
                    while queue:
                        cx, cy = queue.pop()
                        red_neighbors += 1
                        # 檢查周圍的點是否是紅色點，並且未檢查過
                        for dx, dy in neighborhood_offsets:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < width and 0 <= ny < height and red_mask[ny, nx] and not checked_mask[ny, nx]:
                                queue.append((nx, ny))
                                checked_mask[ny, nx] = True
                    # 如果紅色點數量小於閾值，將當前點標記為背景
                    if red_neighbors < min_red_neighbors:
                        red_mask[y, x] = False
        
        # 將標記的結果應用於原始分割圖像
        seg_img[red_mask == False] = [0, 0, 0]  # 設置背景顏色為黑色

        # 提取seg_img的邊緣  取出輪廓
        edges = cv2.Canny(seg_img, 100, 200)
        
        # 将edges中的白色pixel（canny輪廓）對應的位置在old_img中改成红色
        edges_bool = edges.astype(bool)
        pr_result[edges_bool] = [255, 0, 0]  # BGR 红色為255，绿色和蓝色通道為0
        return pr_result

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

def do_pic_otsu():
    # 宣告 訓練好的YOLOv4 模型
    model = initNet()
    # 指定取出yolo-v4 jpg圖片路徑
    predict_source = 'D:/KC_AOI_Project/wrk/study_wrk/compare_Pic/'
    # 取出路徑下所有圖片 並存成list
    files = os.listdir(predict_source)
    print('※ 資料夾共有 {} 張圖檔'.format(len(files)))
    print('※ 開始執行otsu處理...')
    
    number = 1
    for file in files:
        print(' ▼ 第{}張'.format(number))
        print(' ▼ 檔名: {}'.format(file))
        number = number + 1 
        img = cv2.imdecode(np.fromfile(predict_source + file, dtype=np.uint8), -1)
        classes, confs, boxes = nnProcess(img, model)
        # 取得灰階圖檔
        img = cv2.imread(predict_source+file,0)
        # 針對target做增強對比處理
        get_target(img, classes, confs, boxes, file)
        
        
        
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
                pic_log_arr[pic_dir] = target_area*statistics.mean(confs)*100
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
        delDir(pic_log_arr)
    return imgInfo
# 裁減圖片
def get_target(image, classes, confs, boxes, file):
    # 宣告要區塊camma校正的值
    sc_r = 1.4
    root_path = 'D:/KC_AOI_Project/wrk/study_wrk/KC_Pic/KC_Pic_demo/'
    # 沒有高斯處理
    source = 'D:/KC_AOI_Project/wrk/study_wrk/predict_pic/KC_Pic_Plus_14/'
    if not os.path.exists(source):
        # 如果不存在，則創建它
        os.makedirs(source)
        print("已創建名稱為'"+source+"'的資料夾。")
    else:
        print("名稱為'"+source+"'的資料夾已存在。")
    # 有高斯處理
    gaus_source = 'D:/KC_AOI_Project/wrk/study_wrk/predict_pic/G_KC_Pic_Plus_14/'
    if not os.path.exists(gaus_source):
        # 如果不存在，則創建它
        os.makedirs(gaus_source)
        print("已創建名稱為'"+gaus_source+"'的資料夾。")
    else:
        print("名稱為'"+gaus_source+"'的資料夾已存在。")
    # 取得圖片資訊
    imgInfo = image.shape
    print(imgInfo)
    print('4444')
    # 處理圖像時,資料類別都np.uint8
    # 
    voc_no = file.split('_')
    # 不經高斯處理
    pr_img = Image.open(root_path+str(voc_no[0])+ '.bmp')
    pr_img = np.array(pr_img)
    result_path = 'D:/KC_AOI_Project/wrk/study_wrk/predict_result/Predict_Pic_Plus2_14/'
    if not os.path.exists(result_path):
        # 如果不存在，則創建它
        os.makedirs(result_path)
        print("已創建名稱為'"+result_path+"'的資料夾。")
    else:
        print("名稱為'"+result_path+"'的資料夾已存在。")
    # 高斯處理
    pr_img_g = Image.open(root_path+str(voc_no[0])+ '.bmp')
    pr_img_g = np.array(pr_img_g)
    result_path_g = 'D:/KC_AOI_Project/wrk/study_wrk/predict_result/Predict_G_Pic_Plus2_14/'
    if not os.path.exists(result_path_g):
        # 如果不存在，則創建它
        os.makedirs(result_path_g)
        print("已創建名稱為'"+result_path_g+"'的資料夾。")
    else:
        print("名稱為'"+result_path_g+"'的資料夾已存在。")
    # 計算classid == 0 處理次數,如果>0則不要重製處理後的影像陣列
    do_count = 0
    cut_img_list = []
    # 先判斷classes裡面有沒有0的target 沒有的話考慮有可能被判斷成類別的尺標所以做例外處理
    has_zero = np.any(classes == 0)
    
    print('has_zero',has_zero)
    for (classid, conf, box) in zip(classes, confs, boxes):
        print('classes',classes)
        print('classid',classid)
        print('conf',conf)
        print('box',box)
        print('imgInfo',imgInfo)
        x, y, w, h = box
        print('------------------------')
        if (classid == 0):
            # 計算classid == 0 處理次數
            do_count += 1
            cut_img = image[y:y + h, x :x + w]
            cut_img_list = cut_img.flatten()
            maxVal = max(cut_img_list)
            minVal = min(cut_img_list)
            print('最大值',cut_img_list)
            print('最大值',max(cut_img_list))
            # 各區塊處理
            result1 = np.zeros((h,w,1),np.uint8)
            pr_result = np.zeros((h,w,3),np.uint8)
            pr_result_g = np.zeros((h,w,3),np.uint8)
            for i in range(y,y+h):
                # print('i==>',i)
                for j in range(x,x+w):
                    result1[i-y,j-x] = (((image[i,j] - minVal)/(maxVal - minVal)) ** sc_r)*255
                    pr_result[i-y,j-x] = pr_img[i,j]
                    pr_result_g[i-y,j-x] = pr_img_g[i,j]
                    # result1[i-y,j-x] = (((image[i,j] - minVal)/(maxVal - minVal)) ** 4)*255
            # 高斯濾波器是一個平滑化濾波器，平滑化程度是由標準差σ來控制，σ值越大，平滑程度越高，相對的，影像越模糊
            # 將target做 高斯處理
            gaus_new_image = cv2.GaussianBlur(result1, (5, 5), 0)
            gaus_img = gaus_new_image.astype(np.uint8) 
            gaus_grayArr=Image.fromarray(gaus_img.squeeze(), mode='L')
            # 沒有高斯模糊
            img = result1.astype(np.uint8) 
            grayArr=Image.fromarray(img.squeeze(), mode='L')
            # grayArr.save(source + str(pic_no[0]) + '.' + str(pic_no[1]) + '_t' + '.bmp')
            # 給轉VOC檔用的 產編號.jpg擋
            voc_no = file.split('_')
            grayArr.save(source + str(voc_no[0]) + '_'+ str(do_count) + '.jpg')
            gaus_grayArr.save(gaus_source + str(voc_no[0]) + '_'+ str(do_count) + '.jpg')
            # 如果有訓練model後  利用重新圖片 一起預測並產生出預測結果檔
            # 有gaus沒有gaus分開處理
            pr_result = detect_dir_image(pr_result,do_count,file)
            pr_result_g = detect_dir_image(pr_result,do_count,file,'gaus')
            
            boxInfo = result1.shape     
            box_h = boxInfo[0]
            box_W = boxInfo[1]
            for i in range(0,box_h):
                for j in range(0,box_W):
                    if (do_detect_flag == 1):
                        pr_img[i+y,j+x] = pr_result[i,j]
                        pr_img_g[i+y,j+x] = pr_result_g[i,j]
        # classes裡面有沒有0的target 沒有的話考慮有可能被判斷成類別的尺標所以做例外處理
        if (has_zero == False):
            if (classid == 1):
                # 計算classid == 0 處理次數
                do_count += 1
                cut_img = image[y:y + h, x :x + w]
                cut_img_list = cut_img.flatten()
                maxVal = max(cut_img_list)
                minVal = min(cut_img_list)
                print('最大值',cut_img_list)
                print('最大值',max(cut_img_list))
                # 各區塊處理
                result1 = np.zeros((h,w,1),np.uint8)
                pr_result = np.zeros((h,w,3),np.uint8)
                pr_result_g = np.zeros((h,w,3),np.uint8)
                for i in range(y,y+h):
                    for j in range(x,x+w):
                        result1[i-y,j-x] = (((image[i,j] - minVal)/(maxVal - minVal)) ** sc_r)*255
                        pr_result[i-y,j-x] = pr_img[i,j]
                        pr_result_g[i-y,j-x] = pr_img_g[i,j]
                # 高斯濾波器是一個平滑化濾波器，平滑化程度是由標準差σ來控制，σ值越大，平滑程度越高，相對的，影像越模糊
                # 20230625
                # 將target做 高斯處理
                gaus_new_image = cv2.GaussianBlur(result1, (5, 5), 0)
                gaus_img = gaus_new_image.astype(np.uint8) 
                gaus_grayArr=Image.fromarray(gaus_img.squeeze(), mode='L')
                # 沒有高斯模糊
                img = result1.astype(np.uint8) 
                grayArr=Image.fromarray(img.squeeze(), mode='L')
                # grayArr.save(source + str(pic_no[0]) + '.' + str(pic_no[1]) + '_t' + '.bmp')
                # 給轉VOC檔用的 產編號.jpg擋
                voc_no = file.split('_')
                grayArr.save(source + str(voc_no[0]) + '_'+ str(do_count) + '.jpg')
                gaus_grayArr.save(gaus_source + str(voc_no[0]) + '_'+ str(do_count) + '.jpg')
                # 如果有訓練model後  利用重新圖片 一起預測並產生出預測結果檔
                # 有gaus沒有gaus分開處理
                pr_result = detect_dir_image(pr_result,do_count,file)
                pr_result_g = detect_dir_image(pr_result,do_count,file,'gaus')
                boxInfo = result1.shape     
                box_h = boxInfo[0]
                box_W = boxInfo[1]
                for i in range(0,box_h):
                    for j in range(0,box_W):
                        if (do_detect_flag == 1):
                            pr_img[i+y,j+x] = pr_result[i,j]
                            pr_img_g[i+y,j+x] = pr_result_g[i,j]
    if (do_detect_flag == 1):
        new_file_name = str(voc_no[0])  + '.bmp'
        # 正常 高斯非高斯會一起執行  但是因為時間不構只有先做高斯 所以先註解 之後資料齊全可以打開一起執行
        new_file_path = os.path.join(result_path, new_file_name)
        pr_img = Image.fromarray(pr_img)
        pr_img.save(new_file_path)
        new_file_path = os.path.join(result_path_g, new_file_name)
        pr_img_g = Image.fromarray(pr_img_g)
        pr_img_g.save(new_file_path)
    return 

def majority_color_in_edges(image_array):
    # 取得影像的高度和寬度
    height, width = image_array.shape

    # 提取四個邊的像素值
    top_edge = image_array[0, :]
    bottom_edge = image_array[height - 1, :]
    left_edge = image_array[:, 0]
    right_edge = image_array[:, width - 1]

    # 計算各邊黑白像素的個數
    black_count = 0
    white_count = 0

    for edge in [top_edge, bottom_edge, left_edge, right_edge]:
        black_count += np.count_nonzero(edge == 0)  # 假設黑色像素值為0
        white_count += np.count_nonzero(edge == 255)  # 假設白色像素值為255

    # 判斷佔多數的顏色
    if black_count > white_count:
        return 0
    elif white_count > black_count:
        return 255
    else:
        # 預設底色為白色 255
        return 255
if __name__ == '__main__':
    # 決定要不要做預測
    do_detect_flag = 1
    # 顏色映射表 用於圖像分割任務中 用不同的顏色來表示不同的類別或分割標籤, (0, 0, 0)代表黑色，(128, 0, 0)代表红色，(0, 128, 0)代表绿色 以此類推
    colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
              (128, 0, 128), (0, 128, 128), (128, 128, 128),
              (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
              (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0),
              (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
              (128, 64, 12)]

    # 先將原有test檔案路經 經取出最R次方依序處理過
    # do_pic_info()
    do_pic_otsu()
    # print('※ 程式執行完畢')
    # print('※ 總計：成功 {} 張、失敗 {} 張'.format(success, fail))
    # print('※ 偵測超過兩個字元組 {} 張'.format(uptwo))