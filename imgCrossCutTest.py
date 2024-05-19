# 十字強化對比
# imgCrossCutTest.py 十字強化對  測試小工具
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
    for fname in fileList:
        try:
            os.remove(predict_source + fname + '.jpg')
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
    # [in]	confThreshold	A threshold used to filter boxes by confidences.
    # [in]	nmsThreshold	A threshold used in non maximum suppression.

    # classes, confs, boxes = model.detect(image, 0.4, 0.1)
    classes, confs, boxes = model.detect(image, 0.8, 0.8)
    return classes, confs, boxes

#框選偵測到的尺標，並裁減
def cutRuler(image, classes, confs, boxes):
    print('----------gooooooooooooo---')
    new_image = image.copy()
    # otsu 需要一維值 , 灰階值為一維值
    # image傳入圍rgb三維值
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # 取出現次數最多的數字  推估為背景色
    # print(image)
    # bg_val = np.argmax(np.bincount(gray.tolist()))
    for (classid, conf, box) in zip(classes, confs, boxes):
        if classid == 1:
            print(classid)
            print(conf)
            print(box)
            x, y, w, h = box 
            if x - 18 < 0:
                x = 18
            if y - 18 < 0:
                y = 18
            # cv2.rectangle(new_image, (x - 18, y - 18), (x + w + 20, y + h + 24), (0, 255, 0), 3)
            # cv2.rectangle(new_image, (x , y), (x + w , y + h), (0, 255, 0), 3)
            new_image = image[y :y + h, x :x + w ]
            # 讀取輪廓
            # ret, thresh = cv2.threshold(gray,208,255, cv2.THRESH_BINARY)
            ret, thresh = cv2.threshold(gray,0,255, cv2.THRESH_OTSU)
            print('ret') 
            print(ret)
            print('thresh')
            print(thresh)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            print('contours')
            print(len(contours))
            print(contours)
            c_len = 2
            # print(contours[c_len])
            # print('bg_val')
            # print(bg_val)
            # cv2.drawContours(image, contours[c_len], -1, (0,0,255),3)
            cv2.drawContours(image, contours, -1, (0,0,255),1)
    return image
# 傳入x,y座標 及 圖票陣列
# 回傳灰階值
def get_gray_val(x,y,img):
    (b,g,r) = img[x,y]
    b = int(b)
    g = int(g)
    r = int(r)
    gray = r*0.299+g*0.587+b*0.114
    return gray
# 裁減圖片
def get_target(image, classes, confs, boxes):
    new_image = image.copy()
    # 取得圖片資訊
    imgInfo = image.shape
    print(img)
    print(imgInfo)
    print('4444')
    # (140, 163, 3)  高*寬 , 照片類別 全彩=3 灰階=1
    height = imgInfo[0]
    width = imgInfo[1]
    # 處理圖像時,資料類別都np.uint8
    print(imgInfo)
    # 大圖
    result = np.zeros((height,width,1),np.uint8)
    cut_img_list = []
    for (classid, conf, box) in zip(classes, confs, boxes):
        print('classid',classid)
        print('conf',conf)
        print('box',box)
        print('imgInfo',imgInfo)
        x, y, w, h = box
        print('x')
        print(x)
        print('y')
        print(y)
        print('h')
        print(h)
        print('------------------------')
        if (classid == 0):
            cut_img = image[y:y + h, x :x + w]
            cut_img_list = cut_img.flatten()
            maxVal = max(cut_img_list)
            minVal = min(cut_img_list)
            print('最大值',cut_img_list)
            print('最大值',max(cut_img_list))
            # 各區塊處理
            result1 = np.zeros((h,w,1),np.uint8)
            for i in range(y,y+h):
                # print('i==>',i)
                for j in range(x,x+w):
                    # gray = get_gray_val(i, j, image)
                    # result[i-y,j-x] = np.uint8(gray)
                    result1[i-y,j-x] = image[i,j]
                    # result1[i-y,j-x] = (((image[i,j] - minVal)/(maxVal - minVal)) ** 2.5)*255
            # 高斯濾波器是一個平滑化濾波器，平滑化程度是由標準差σ來控制，σ值越大，平滑程度越高，相對的，影像越模糊
            # result1 = cv2.GaussianBlur(result1, (3, 3), 0)
            # result1 = cv2.Laplacian(result1, -1, 1, 5) 
            # result1 = cv2.GaussianBlur(result1, (5, 5), 0)
            # retval, result1 = cv2.threshold(result1, 0, 255, cv2.THRESH_OTSU)
            boxInfo = result1.shape
            box_h = boxInfo[0]
            box_W = boxInfo[1]
            for i in range(0,box_h):
                print('i==>',i)
                for j in range(0,box_W):
                    print('j==>',j)
                    result[i+y,j+x] = result1[i,j]
        
    result1 = doLocalOtsu(result1)
    # 應用自適應直方圖均衡化（CLAHE）
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # result1 = clahe.apply(result1)
    # # result1 = doLocalOtsu(result1)
    # retval, result1 = cv2.threshold(result1, 0, 255, cv2.THRESH_OTSU)
    return result1
    # return result
# 根據目標區域各點 往上下左右延伸十字線 將線上的點集中計算成目標點的ostsu值 
# 缺點: 有可能不會那麼精準  不過經測試後得到結果相差不大  
# 優點: 速度快很多  (用這個)
def doLocalOtsu(image):
    new_image = image.copy()
    height = image.shape[0]
    print('doLocalOtsu::height',height)
    width  = image.shape[1]
    print('doLocalOtsu::width',width)
    # 先算出y軸往上下沿展的長度 約1/4 高度
    y_ridge = height/4//2
    # 先算出X軸往左右沿展的長度 約1/4 寬度
    x_ridge = width/3//2
    # dfor迴圈跑傳入圖檔每個點
    # 長66 寬128
    for i in range(0,height):
        # print('height==>',height)
        # print('y_ridge==>',y_ridge)
        # print('x_ridge==>',x_ridge)
        for j in range(0,width):
            # 計算XY向延展起迄點
            # X起點
            if(j-x_ridge)<0:
                x_start = 0
            else:
                x_start = int(j-x_ridge)
            # X迄點 
            if(j+x_ridge)>(width-1):
                x_end = width
            else:
                x_end = int(j+x_ridge+1)
            # Y起點
            if(i-y_ridge)<0:
                y_start = 0
            else:
                y_start = int(i-y_ridge)
            # Y迄點 
            if(i+y_ridge)>(height-1):
                y_end = height
            else:
                y_end = int(i+y_ridge+1)
            print('x_start==>',x_start)
            print('x_end==>',x_end)
            print('y_start==>',y_start)
            print('y_end==>',y_end)
            print('i==>',i)
            print('x_ridge==>',x_ridge)
            # print('image==>',image[111,128])

            # 取得目標點所屬y軸點X向所有點
            x_row = image[i,x_start:x_end]
            # 取得目標點所屬x軸點上Y向所有點
            y_row = image[y_start:y_end,j]
            # 因目標點再鈐個row有取出 所以去除點重複的值
            y_row = np.delete(y_row, int(y_ridge))
            result1 = np.append(x_row, y_row)
            result1 = cv2.GaussianBlur(result1, (3, 3), 0)
            retval, result1 = cv2.threshold(result1, 0, 255, cv2.THRESH_OTSU)
            # print('x_row==>',x_row)
            # print('do_row==>',result1)
            # print('result1==>',len(result1))
            new_image[i,j] = result1[int(x_ridge)]
    
    # print('doLocalOtsu::new_image',new_image)
    return new_image
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
    return cut_img_list[1]

# 裁減圖片
def cut_img2(image, classes, confs, boxes):
    cut_img_list = []
    for (classid, conf, box) in zip(classes, confs, boxes):
        x, y, w, h = box
        if x - 18 < 0:
            x = 18
        if y - 18 < 0:
            y = 18
        cut_img = image[y - 18:y + h + 20, x - 18:x + w + 25]
        cut_img_list.append(cut_img)
    return cut_img_list[2]



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
        if do_unit == 0.01 :
            print(maxVal)
            print('123454654894984894')
            print(minVal)
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
        cv2.imshow('grayArr',result)
        grayArr=Image.fromarray(result)
        # grayArr.save('compare_Pic/' + str(pic_no) + '_' + str(now_r) + '.bmp','bmp')
        dir_log[str(pic_no) + '_' + str(now_r)] = 0
        grayArr.save('compare_Pic/' + str(pic_no) + '_' + str(now_r) + '.jpg')
        # grayArr.save('D:/Yolo_v4/darknet/build/darknet/x64/data/test/koi/' + pic_no + '_' + now_r + '.jpg','jpg')
        cv2.destroyWindow('grayArr')
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
            cv2.imshow('grayArr',result)
            grayArr=Image.fromarray(result)
            # grayArr.save('compare_Pic/' + str(pic_no) + '_' + str(now_r) + '.bmp','bmp')
            dir_log[str(pic_no) + '_' + str(now_r)] = 0
            # 產生yolo v4要預測用的圖檔 需要為jpg
            grayArr.save('compare_Pic/' + str(pic_no) + '_' + str(now_r) + '.jpg')
            # grayArr.save('D:/Yolo_v4/darknet/build/darknet/x64/data/test/koi/' + str(pic_no) + '_' + str(now_r) + '.jpg')
            cv2.destroyWindow('grayArr')
            # 釋放資源
            cv2.destroyAllWindows()
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
    # 指定取出最佳k值的灰階圖片路徑
    source = 'D:/KC_AOI_Project/wrk/study_wrk/compare_Pic/'
    # 取出路徑下所有圖片 並存成list
    files = os.listdir(source)
    print('※ 資料夾共有 {} 張圖檔'.format(len(files)))
    print('※ 開始執行otsu處理...')
    number = 1
    for file in files:
        print(' ▼ 第{}張'.format(number))
        print(' ▼ 檔名: {}'.format(file))
        number = number + 1 
        img = cv2.imread(source+file,0)
        # img = cv2.imdecode(np.fromfile(source+file, dtype=np.uint8), -1)
        print(img)
        # 分割檔名編號
        pic_no = file.split('.')
        print(pic_no)
        retval, a_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        grayArr=Image.fromarray(a_img)
        grayArr.save(source + str(pic_no[0]) + '.' + str(pic_no[1]) + '_otsu' + '.bmp')
        # 高斯模糊
        img = cv2.GaussianBlur(img, (5, 5), 0)
        retval, a_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        print('使用opencv函数的方法:'+str(retval))
        # cv2.imshow('a_img',a_img)
        # cv2.waitKey(0)
        grayArr=Image.fromarray(a_img)
        grayArr.save(source + str(pic_no[0]) + '.' + str(pic_no[1]) + '_gaus' + '.bmp')
        # cv2.destroyWindow('a_img')
        
def do_pic_info():
    # 指定YOLO 4 test圖片路徑
    # source = 'D:/Yolo_v4/darknet/build/darknet/x64/data/test/koi/'
    source = 'D:/KC_AOI_Project/wrk/study_wrk/KC_Pic/KC_Pic_demo/'
    predict_source = 'D:/KC_AOI_Project/wrk/study_wrk/compare_Pic/'
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
        # set()去除list中重複的值
        pixel_arr = list(set(pixel_arr))
        # 從小到大排序
        pixel_arr.sort()
        # print(pixel_arr)
        # for 產生r次方 入暫時dir 輸入單位值 找出最K值
        # 傳入資料都是原圖彩色資料
        pic_log_arr = do_best_r_pic(img, pixel_arr, 1, 0.1,pic_no[0])
        # 存放套入model 預測的值
        print(pic_log_arr)
        for pic_dir  in pic_log_arr:
            # 123
            img = cv2.imdecode(np.fromfile(predict_source + pic_dir+'.jpg', dtype=np.uint8), -1)
            classes, confs, boxes = nnProcess(img, model)
            pic_log_arr[pic_dir] = statistics.mean(confs)
            print(confs)
        
        print(pic_log_arr)
        print('最大值')
        print(max(pic_log_arr.items(), key=operator.itemgetter(1))[0])
        # 將平均得分最高的圖檔名稱 從log_arr裡面unset掉
        del pic_log_arr[max(pic_log_arr.items(), key=operator.itemgetter(1))[0]]
        print(pic_log_arr)
        
        # 將剩餘的圖檔log arr 丟進delDir()去刪除檔案
        delDir(pic_log_arr)
    return imgInfo

# 儲存已完成前處理之圖檔(中文路徑)
def saveClassify(image, output):
    cv2.imencode(ext='.jpg', img=image)[1].tofile(output)
    
    
    
if __name__ == '__main__':
    # 先將原有test檔案路經 經取出最R次方依序處理過
    print(123)
    model = initNet()
    file_root = 'D:/KC_AOI_Project/wrk/study_wrk/G_KC_PIC/21_1.6.bmp'
    # file_root = 'D:/KC_AOI_Project/wrk/study_wrk/G_KC_PIC/20_1.2.bmp'
    # file_root = 'D:/KC_AOI_Project/wrk/study_wrk/G_KC_PIC/32_0.4.bmp'
    file_root = 'D:/KC_AOI_Project/wrk/study_wrk/G_KC_PIC/34_1.8.bmp'
    # file_root = 'D:/KC_AOI_Project/wrk/study_wrk/G_KC_PIC/32_0.2.bmp'
    # file_root = 'D:/KC_AOI_Project/wrk/study_wrk/G_KC_PIC/31_1.9.bmp'
    img = cv2.imdecode(np.fromfile(file_root, dtype=np.uint8), -1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # print(gray)
    classes, confs, boxes = nnProcess(img, model)
    print(classes)
    
    img = cv2.imread(file_root,0)
    
    
    # 找出尺標
    # frame = cutRuler(img, classes, confs, boxes)
    # test append list of target
    frame = get_target(img, classes, confs, boxes)
    # # 框選後圖檔
    # frame = drawBox(img, classes, confs, boxes)
    # print(frame)
    # 裁剪後圖檔 
    cut = cut_img(img, classes, confs, boxes)
    # cut2 = cut_img2(img, classes, confs, boxes)
    # gooo = np.concatenate((cut2,cut))
    # print(cut)
    # 儲存裁剪後圖檔
    # saveClassify(cut, 'D:/KC_AOI_Project/wrk/study_wrk/compare_Pic/47_0.6_cut.jpg')
    # saveClassify(cut, './public_training_data/YOLOV4_pre/success/' + file)
    # saveClassify(img, './test123/success/' + file)
    cv2.imshow('img', frame)
    # cv2.imshow('cut', cut) 
    # cv2.imshow('cut2', cut2) 
    # cv2.imshow('gooo', gooo) 
    print('=' * 60)
    print(boxes)
    cv2.waitKey(0)
    # 釋放資源
    cv2.destroyAllWindows()
    # do_pic_info()
    # do_pic_otsu()
    
    
    
    
    