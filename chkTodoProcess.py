# -*- coding: utf-8 -*-
# chkTodoProcess.py
# 
# doOtsu ==>getTarget 取圖片yolo判斷汙點區塊 做對比度判斷 適合global還是 LCGC
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.filters import laplace

# Function to apply Gamma correction
def adjust_gamma(image, gamma=1.0):
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# Function to apply Local Contrast Gain Control (LCGC)
def local_contrast_gain_control(image, alpha=1.0, beta=1.0):
    # Apply Gaussian blur to the image
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=alpha, sigmaY=alpha)
    
    # Compute the difference (local contrast)
    local_contrast = cv2.subtract(image, blurred)
    
    # Scale the local contrast
    scaled_contrast = cv2.convertScaleAbs(local_contrast, alpha=beta)
    
    # Enhance the image by adding the scaled local contrast
    enhanced_image = cv2.add(image, scaled_contrast)
    
    return enhanced_image

def calculate_dynamic_threshold(image, percentile=90):
    """
    Calculate a dynamic threshold based on the global standard deviation
    and a specified percentile of the brightness histogram.
    """
    print('讓我看看')
    print(image)
    # Compute the brightness histogram
    histogram, _ = np.histogram(image, bins=256, range=(0, 1))

    # Calculate the cumulative distribution of the histogram
    cumulative_distribution = np.cumsum(histogram) / np.sum(histogram)

    # Find the brightness level at the specified percentile
    brightness_level = np.searchsorted(cumulative_distribution, percentile / 100) / 256

    # Calculate a dynamic threshold based on this brightness level
    dynamic_threshold = np.std(image[image < brightness_level])

    return dynamic_threshold


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
        # 取得bmp灰階圖檔
        img = cv2.imread(predict_source+file,0)
        
        img = get_target(img, classes, confs, boxes,file)
        
        

# Define a function to process the image
def process_image(image, classid, file):
    # x, y, w, h = box
    # Calculate global standard deviation of the image
    global_std = np.std(image)
    
    # Calculate local standard deviation using a Laplacian filter for local contrast
    local_contrast = laplace(image)
    local_contrast_std = np.std(local_contrast)
    print(' ▼ global_std: {}'.format(global_std))
    print(' ▼ local_contrast_std: {}'.format(local_contrast_std))
    
    # Calculate gamma_value, alpha_value, and beta_value based on the calculated statistics
    gamma_value = 1.4 / global_std if global_std != 0 else 1  # to avoid division by zero
    alpha_value = 1  # This value could be tuned as needed
    beta_value = 1.2 / local_contrast_std if local_contrast_std != 0 else 1  # to avoid division by zero
    print("beta_value : {}".format(beta_value))
    # Apply Gamma correction and LCGC to the image
    gamma_corrected_image = adjust_gamma(image, gamma=gamma_value)
    lcgc_image = local_contrast_gain_control(image, alpha=alpha_value, beta=beta_value)
    
    # Display the images
    # 指定中文字型
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用思源黑體或其他中文字體
    plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title(file + 'Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(gamma_corrected_image, cmap='gray')
    plt.title('Gamma Corrected Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(lcgc_image, cmap='gray')
    plt.title('LCGC Image')
    plt.axis('off')
    
    plt.show()
    
    # Calculate global and local standard deviations
    global_std = np.std(image)
    local_contrast = laplace(image)
    local_contrast_std = np.std(local_contrast)
    
    # Example usage with an image
    dynamic_threshold = calculate_dynamic_threshold(image, 90)
    
    # 绘制亮度直方图及其CDF

    # histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))
    # 绘制亮度直方图
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 256))
    print("histogram :")
    print(histogram)
    # 正規化直方圖
    normalized_histogram = histogram / histogram.sum()
    
    # 計算CDF
    cdf = np.cumsum(normalized_histogram)
    print("Image shape:", image.shape)
    print("Min pixel value:", np.min(image))
    print("Max pixel value:", np.max(image))
    print(' ▼ histogram.sum: {}'.format(histogram.sum()))
    # cdf = np.cumsum(histogram) / histogram.sum()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(bin_edges[1:], histogram)
    plt.title('light plot')
    plt.xlabel('light val')
    plt.ylabel('pixel num')
    
    plt.subplot(1, 2, 2)
    plt.plot(bin_edges[1:], cdf)
    plt.title('light CDF')
    plt.xlabel('light val ')
    plt.ylabel('sum percent')
    
    plt.show()
    
    abs_num = 0
    if global_std < dynamic_threshold and local_contrast_std > dynamic_threshold:
        print(file + "建議使用Gamma校正")
        abs_num = 1
    else:
        print(file + "建議使用LCGC")
def analyze_gray_image_for_LCGC(image_gray, file):
    """
    分析灰階圖片是否適合進行LCGC（十字強化對比）。
    這裡將檢查圖片的亮度和對比度。

    :param image_path: 灰階圖片的路徑
    :return: 分析結果
    """
    try:

        # 計算圖片的亮度和對比度
        pixels = np.array(image_gray)
        brightness = np.mean(pixels)
        contrast = pixels.std()

        # 展示灰階圖片
        plt.imshow(image_gray, cmap='gray')
        plt.title(file+"gray")
        plt.axis('off')
        plt.show()

        # 設定亮度和對比度的閾值
        brightness_threshold = 100  # 範例閾值
        # brightness_threshold = brightness * 0.5 + contrast * 0.5
        # contrast_threshold = contrast * 0.8
        # contrast_threshold = np.percentile(contrast, 75)  # 取對比度分布的75%分位數作為閾值
        contrast_threshold = 50     # 範例閾值
    
        # 判斷適用的方法
        if brightness > brightness_threshold and contrast > contrast_threshold:
            result_str = "適合LCGC"
        elif brightness < brightness_threshold and contrast < contrast_threshold:
            result_str = "適合Gamma校正"
        else:
            result_str = "無需特殊處理"
    
        # 分析結果
        result = {
            "亮度": brightness,
            "對比度": contrast,
            "結果": result_str,
            "亮度閾值": brightness_threshold,
            "對比度閾值": contrast_threshold,
        }

        print(result)

        return result
    except Exception as e:
        return {"錯誤": str(e)}
    
def get_target(image_o, classes, confs, boxes, file):
    # 先判斷classes裡面有沒有0的target 沒有的話考慮有可能被判斷成類別的尺標所以做例外處理
    has_zero = np.any(classes == 0)
    do_count = 0
    for (classid, conf, box) in zip(classes, confs, boxes):
        x, y, w, h = box
        image = np.zeros((h, w, 1), np.uint8)
        
        for i in range(y, y + h):
            for j in range(x, x + w):
                image[i - y, j - x] = image_o[i, j]
        
        print('------------------------')
        # print('classes', classes)
        # print('classid', classid)
        # print('conf', conf)
        # print('box', box)
        
        # Process the image based on classid
        if classid == 0:
            do_count = do_count + 1 
            # process_image(image, classid, file+'_'+str(do_count))
            analyze_gray_image_for_LCGC(image, file)
        
        # Check for the case where classes do not contain 0 (special handling)
        if not has_zero:
            if classid == 1:
                do_count = do_count + 1 
                print('完蛋啦', file)
                # process_image(image, classid, file+'_'+str(do_count))
                analyze_gray_image_for_LCGC(image, file)
           
        # # classes裡面有沒有0的target 沒有的話考慮有可能被判斷成類別的尺標所以做例外處理
        # if (has_zero == False):
        #     if (classid == 1):
        #         # 各區塊處理
        #         image = np.zeros((h,w,1),np.uint8)
        #         for i in range(y,y+h):
        #             for j in range(x,x+w):
        #                 image[i-y,j-x] = image_o[i,j]
        #         # Calculate global standard deviation of the image
        #         global_std = np.std(image)
                
        #         # Calculate local standard deviation using a Laplacian filter for local contrast
        #         local_contrast = laplace(image)
        #         local_contrast_std = np.std(local_contrast)
                
        #         # Ensure the following functions and the rest of your code are in the same script or
        #         # Jupyter Notebook cell to maintain the context of the defined variables.
                
        #         # Now you can define gamma_value, alpha_value, and beta_value based on the calculated statistics
        #         gamma_value = 0.7 / global_std if global_std != 0 else 1  # to avoid division by zero
        #         alpha_value = 1  # This value could be tuned as needed
        #         beta_value = 1.7 / local_contrast_std if local_contrast_std != 0 else 1  # to avoid division by zero
                
        #         # # Read the image in grayscale
        #         # original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
        #         # Apply Gamma correction and LCGC to the image
        #         gamma_corrected_image = adjust_gamma(image, gamma=gamma_value)
        #         lcgc_image = local_contrast_gain_control(image, alpha=alpha_value, beta=beta_value)
                
        #         # Display the images
        #         # Display the images
                
        #         plt.figure(figsize=(18, 6))
        #         plt.subplot(1, 3, 1)
        #         plt.imshow(image, cmap='gray')
        #         plt.title('Original Image')
        #         plt.axis('off')
                
        #         plt.subplot(1, 3, 2)
        #         plt.imshow(gamma_corrected_image, cmap='gray')
        #         plt.title('Gamma Corrected Image')
        #         plt.axis('off')
                
        #         plt.subplot(1, 3, 3)
        #         plt.imshow(lcgc_image, cmap='gray')
        #         plt.title('LCGC Image')
        #         plt.axis('off')
                
        #         plt.show()
                
        #         # 計算全局和局部標準差
        #         global_std = np.std(image)
        #         local_contrast = laplace(image)
        #         local_contrast_std = np.std(local_contrast)
        #         # print('全局標準差',global_std)
        #         # print('局部標準差',local_contrast_std)
                
        #         # Example usage with an image
        #         dynamic_threshold = calculate_dynamic_threshold(image,80)
        #         # print("Dynamically calculated global standard deviation threshold:", dynamic_threshold)
                
        #         # 绘制亮度直方图及其CDF
        #         histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))
        #         cdf = np.cumsum(histogram) / histogram.sum()
                
        #         plt.figure(figsize=(12, 6))
        #         plt.subplot(1, 2, 1)
        #         plt.plot(bin_edges[1:], histogram)
        #         plt.title('light plot')
        #         plt.xlabel('light val')
        #         plt.ylabel('pixel num')
                
        #         plt.subplot(1, 2, 2)
        #         plt.plot(bin_edges[1:], cdf)
        #         plt.title('light CDF')
        #         plt.xlabel('light val')
        #         plt.ylabel('sum percent')
                
        #         plt.show()
                
        #         abs_num = 0
        #         if global_std < dynamic_threshold and local_contrast_std > dynamic_threshold:
        #             print(file+"沒有0的targe 建議使用Gamma校正")
        #             abs_num = 1
                    
        #         else:
        #             print(file+"沒有0的targe 建議使用LCGC")


if __name__ == '__main__':
    do_pic_otsu()
    # print('※ 程式執行完畢')
    # print('※ 總計：成功 {} 張、失敗 {} 張'.format(success, fail))
    # print('※ 偵測超過兩個字元組 {} 張'.format(uptwo))