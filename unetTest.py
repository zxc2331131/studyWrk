# unetTest.py
# unet model成功
# 用unet kernal環境
# 取unetmask輪廓  重疊原圖
# 改成全資料夾for執行 並存檔
import os
import math
import numpy as np
from PIL import Image
import cv2
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
def detect_dir_image(image_path,result_path):
    
    file_list = os.listdir(image_path)
    for file_name in file_list:
        # 檢查文件擴展名是否為常見的圖像格式，如.jpg、.png等
        if file_name.lower().endswith(('.bmp')):
            # 組成完整的文件路徑
            file_path = os.path.join(image_path, file_name)
            file_root = os.path.join(root_path, file_name)

            image = Image.open(file_path)
            r_image = Image.open(file_root)
            # 將圖片轉為RGB
            image = cvtColor(image)
            r_image = cvtColor(r_image)
            # 先宣告結果image nparray
            r_img = np.array(r_image)
            old_img = copy.deepcopy(image)
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
            # 使用连通区域分析找到红色区域
            gray_seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
            _, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_seg_img, connectivity=8)
            # 创建一个与labels相同大小的掩码，用于保留红色区域
            filtered_mask = np.zeros_like(labels, dtype=np.uint8)
            
            min_area_threshold = 10
            # 遍历每个区域的统计信息
            for label, stat in enumerate(stats):
                area = stat[4]  # 区域的面积在统计信息中的索引为4
                if area >= min_area_threshold:
                    # 如果区域面积大于等于阈值，将该区域标记为有效
                    filtered_mask[labels == label] = 1
                    
            # 将filtered_mask应用到原始图像上，保留红色区域
            seg_img = cv2.bitwise_and(seg_img, seg_img, mask=filtered_mask)   
            # # 骰選掉太小 過度擬和的點
            # # 創建一個掩碼，用於標記紅色區域
            # red_mask = (seg_img[:, :, 0] == 128) & (seg_img[:, :, 1] == 0) & (seg_img[:, :, 2] == 0)
            
            # # 定義周圍的像素偏移
            # # 3*3
            # # neighborhood_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            # # 5*5
            # neighborhood_offsets = [
            #     (2, 0), (-2, 0), (1, 0), (-1, 0), (0, 2), (0, -2), (0, 1), (0, -1),  # 中心像素的上下左右邻居
            #     (1, 1), (-1, 1), (1, -1), (-1, -1)  # 对角线上的像素
            # ]
            # # 7*7
            # # neighborhood_offsets = [
            # #     (3, 0), (-3, 0), (2, 0), (-2, 0), (1, 0), (-1, 0), (0, 3), (0, -3),  # 中心像素的上下左右邻居
            # #     (0, 2), (0, -2), (1, 1), (-1, 1), (1, -1), (-1, -1),  # 对角线上的像素
            # #     (2, 1), (-2, 1), (2, -1), (-2, -1),  # 更远的像素
            # # ]
            
            # # 定義周圍紅色點的最小數量閾值
            # min_red_neighbors = 9  # 根據需要調整閾值
            
            # # 獲取圖像高度和寬度
            # height, width = red_mask.shape
            
            # # 創建一個標記掩碼，用於記錄已檢查的點
            # checked_mask = np.zeros_like(red_mask, dtype=bool)
            
            # # 遍歷標記的紅色區域
            # for y in range(height):
            #     for x in range(width):
            #         if red_mask[y, x] and not checked_mask[y, x]:
            #             red_neighbors = 0
            #             queue = [(x, y)]
            #             checked_mask[y, x] = True
            #             # 使用隊列來叠代周圍的紅色點
            #             while queue:
            #                 cx, cy = queue.pop()
            #                 red_neighbors += 1
            #                 # 檢查周圍的點是否是紅色點，並且未檢查過
            #                 for dx, dy in neighborhood_offsets:
            #                     nx, ny = cx + dx, cy + dy
            #                     if 0 <= nx < width and 0 <= ny < height and red_mask[ny, nx] and not checked_mask[ny, nx]:
            #                         queue.append((nx, ny))
            #                         checked_mask[ny, nx] = True
            #             # 如果紅色點數量小於閾值，將當前點標記為背景
            #             if red_neighbors < min_red_neighbors:
            #                 red_mask[y, x] = False
            
            # # 將標記的結果應用於原始分割圖像
            # seg_img[red_mask == False] = [0, 0, 0]  # 設置背景顏色為黑色

            # 提取seg_img的邊緣  取出輪廓
            edges = cv2.Canny(seg_img, 100, 200)
            
            # 将edges中的白色pixel（canny輪廓）對應的位置在old_img中改成红色
            # 將edges中的白色
            edges_bool = edges.astype(bool)
            # 将 edges_bool 调整为与 r_img 的形状相同
            edges_bool = np.resize(edges_bool, r_img.shape[:2])
            # r_img[edges_bool] = [0, 0, 255]  # BGR 红色為255，绿色和蓝色通道為0
            r_img[edges_bool] = [255, 0, 0]  # BGR 红色為255，绿色和蓝色通道為0
            area_image = Image.fromarray(seg_img)
            # blend 將兩張圖片混合再一起產生新的area_image  方便檢視預測圖形 
            area_image = Image.blend(old_img, area_image, 0.7)
            
            new_file_name = os.path.splitext(file_name)[0] + '.bmp'
            new_file_path = os.path.join(result_path, new_file_name)
            r_img = Image.fromarray(r_img)
            r_img.save(new_file_path)

            image.close()

# print出單張 predict輪廓 刪減躁點
def detect_image(image_path):
    image = Image.open(image_path)
    # 將圖片轉為RGB
    image = cvtColor(image)

    old_img2 = np.array(image)
    old_img = copy.deepcopy(image)
    ori_h = np.array(image).shape[0]
    ori_w = np.array(image).shape[1]

    image_data, nw, nh = resize_image(image, (INPUT_SHAPE[1], INPUT_SHAPE[0]))

    image_data = normalize(np.array(image_data, np.float32))
    # 因為要批量處裡 所以增加批次維度
    # 如果原始的image_data是形狀為 (width, height) 的二維數組，那麽經過 np.expand_dims(image_data, 0) 操作後，它的形狀將變為 (1, width, height)，表示一個包含一個圖像的批次。
    image_data = np.expand_dims(image_data, 0)

    pr = model.predict(image_data)[0]
    
    # 從原始圖像 pr 中提取一個中心裁剪的子圖像，使其具有指定的高度 nh 和寬度 nw
    pr = pr[int((INPUT_SHAPE[0] - nh) // 2) : int((INPUT_SHAPE[0] - nh) // 2 + nh), \
            int((INPUT_SHAPE[1] - nw) // 2) : int((INPUT_SHAPE[1] - nw) // 2 + nw)]

    pr = cv2.resize(pr, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)

    pr = pr.argmax(axis=-1)

    # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
    # for c in range(NUM_CLASSES):
    #     seg_img[:, :, 0] += ((pr[:, :] == c ) * colors[c][0]).astype('uint8')
    #     seg_img[:, :, 1] += ((pr[:, :] == c ) * colors[c][1]).astype('uint8')
    #     seg_img[:, :, 2] += ((pr[:, :] == c ) * colors[c][2]).astype('uint8')
    seg_img = np.reshape(
        np.array(colors, np.uint8)[np.reshape(pr, [-1])], [ori_h, ori_w, -1])
    print('xhere i come')
    print(seg_img)
    # 骰選掉太小 過度擬和的點
    # 創建一個掩碼，用於標記紅色區域
    red_mask = (seg_img[:, :, 0] == 128) & (seg_img[:, :, 1] == 0) & (seg_img[:, :, 2] == 0)

    # 使用连通区域分析找到红色区域
    gray_seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_seg_img, connectivity=8)
    # 创建一个与labels相同大小的掩码，用于保留红色区域
    filtered_mask = np.zeros_like(labels, dtype=np.uint8)
    
    min_area_threshold = 10
    # 遍历每个区域的统计信息
    for label, stat in enumerate(stats):
        area = stat[4]  # 区域的面积在统计信息中的索引为4
        if area >= min_area_threshold:
            # 如果区域面积大于等于阈值，将该区域标记为有效
            filtered_mask[labels == label] = 1
            
    # 将filtered_mask应用到原始图像上，保留红色区域
    seg_img = cv2.bitwise_and(seg_img, seg_img, mask=filtered_mask)
    # return seg_img
    # # 定義周圍的像素偏移
    # # 3*3
    # neighborhood_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    # # 5*5
    # neighborhood_offsets = [
    #     (2, 0), (-2, 0), (1, 0), (-1, 0), (0, 2), (0, -2), (0, 1), (0, -1),  # 中心像素的上下左右邻居
    #     (1, 1), (-1, 1), (1, -1), (-1, -1)  # 对角线上的像素
    # ]
    # # 7*7
    # neighborhood_offsets = [
    #     (3, 0), (-3, 0), (2, 0), (-2, 0), (1, 0), (-1, 0), (0, 3), (0, -3),  # 中心像素的上下左右邻居
    #     (0, 2), (0, -2), (1, 1), (-1, 1), (1, -1), (-1, -1),  # 对角线上的像素
    #     (2, 1), (-2, 1), (2, -1), (-2, -1),  # 更远的像素
    # ]
    
    # # 定義周圍紅色點的最小數量閾值
    # min_red_neighbors = 25  # 根據需要調整閾值
    
    # # 獲取圖像高度和寬度
    # height, width = red_mask.shape
    
    # # 創建一個標記掩碼，用於記錄已檢查的點
    # checked_mask = np.zeros_like(red_mask, dtype=bool)
    
    # # 遍歷標記的紅色區域
    # for y in range(height):
    #     for x in range(width):
    #         if red_mask[y, x] and not checked_mask[y, x]:
    #             red_neighbors = 0
    #             queue = [(x, y)]
    #             checked_mask[y, x] = True
    #             # 使用隊列來叠代周圍的紅色點
    #             while queue:
    #                 cx, cy = queue.pop()
    #                 red_neighbors += 1
    #                 # 檢查周圍的點是否是紅色點，並且未檢查過
    #                 for dx, dy in neighborhood_offsets:
    #                     nx, ny = cx + dx, cy + dy
    #                     if 0 <= nx < width and 0 <= ny < height and red_mask[ny, nx] and not checked_mask[ny, nx]:
    #                         queue.append((nx, ny))
    #                         checked_mask[ny, nx] = True
    #             # 如果紅色點數量小於閾值，將當前點標記為背景
    #             if red_neighbors < min_red_neighbors:
    #                 red_mask[y, x] = False
    
    # # 將標記的結果應用於原始分割圖像
    # seg_img[red_mask == False] = [0, 0, 0]  # 設置背景顏色為黑色

    # 提取seg_img的邊緣
    edges = cv2.Canny(seg_img, 200, 300)
    
    # 将edges中的白色像素（边缘）对应的位置在old_img中改成红色
    # 將edges中的白色
    edges_bool = edges.astype(bool)
    old_img2[edges_bool] = [0, 0, 255]  # 红色通道为255，绿色和蓝色通道为0
    print(edges)
    image = Image.fromarray(seg_img)
    # blend 將兩張圖片混合再一起產生新的image
    image = Image.blend(old_img, image, 0.7)

    return old_img2

# print出單張 predict輪廓  原版
# def detect_image(image_path):
#     image = Image.open(image_path)
#     # 將圖片轉為RGB
#     image = cvtColor(image)

#     old_img2 = np.array(image)
#     old_img = copy.deepcopy(image)
#     ori_h = np.array(image).shape[0]
#     ori_w = np.array(image).shape[1]

#     image_data, nw, nh = resize_image(image, (INPUT_SHAPE[1], INPUT_SHAPE[0]))

#     image_data = normalize(np.array(image_data, np.float32))

#     image_data = np.expand_dims(image_data, 0)

#     pr = model.predict(image_data)[0]
    

#     pr = pr[int((INPUT_SHAPE[0] - nh) // 2) : int((INPUT_SHAPE[0] - nh) // 2 + nh), \
#             int((INPUT_SHAPE[1] - nw) // 2) : int((INPUT_SHAPE[1] - nw) // 2 + nw)]

#     pr = cv2.resize(pr, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)

#     pr = pr.argmax(axis=-1)

#     # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
#     # for c in range(NUM_CLASSES):
#     #     seg_img[:, :, 0] += ((pr[:, :] == c ) * colors[c][0]).astype('uint8')
#     #     seg_img[:, :, 1] += ((pr[:, :] == c ) * colors[c][1]).astype('uint8')
#     #     seg_img[:, :, 2] += ((pr[:, :] == c ) * colors[c][2]).astype('uint8')
#     seg_img = np.reshape(
#         np.array(colors, np.uint8)[np.reshape(pr, [-1])], [ori_h, ori_w, -1])
    
#     # 提取seg_img的邊緣
#     edges = cv2.Canny(seg_img, 100, 200)
    
#     # 将edges中的白色像素（边缘）对应的位置在old_img中改成红色
#     # 將edges中的白色
#     edges_bool = edges.astype(bool)
#     old_img2[edges_bool] = [0, 0, 255]  # 红色通道为255，绿色和蓝色通道为0
#     print(edges)
#     image = Image.fromarray(seg_img)
#     # blend 將兩張圖片混合再一起產生新的image
#     image = Image.blend(old_img, image, 0.7)

#     return old_img2
#     # return image


root_path = 'D:/KC_AOI_Project/wrk/study_wrk/KC_Pic/KC_Pic_demo/'
# 讀取訓練好的模型權重

# ======================================

# 灰階處理 資料model
model = keras.models.load_model('D:/KC_AOI_Project/wrk/study_wrk/Graylogs2/the-last-model.h5')
# 待預測圖片的資料夾
dir_path = 'D:/KC_AOI_Project/wrk/study_wrk/predict_pic/KC_Pic_Original/'
# 產生結果存放的資料夾
result_path = 'D:/KC_AOI_Project/wrk/study_wrk/predict_result/Predict_Pic_Gray2/'


# ======================================

# 高斯處理 + 灰階處理 資料model
# model = keras.models.load_model('D:/KC_AOI_Project/wrk/study_wrk/G_Graylogs/the-last-model.h5')
# # 待預測圖片的資料夾
# dir_path = 'D:/KC_AOI_Project/wrk/study_wrk/predict_pic/G_KC_Pic_Original/'
# # 產生結果存放的資料夾
# result_path = 'D:/KC_AOI_Project/wrk/study_wrk/predict_result/Predict_G_Pic_Gray/'

# ======================================
# test
# 單張
test_image_path = 'KC_Pic/KC_Pic_demo/55.bmp'
test_image_path = 'D:/KC_AOI_Project/wrk/study_wrk/predict_pic/KC_Pic_Cross/14_1.jpg'

# 顏色映射表 用於圖像分割任務中 用不同的顏色來表示不同的類別或分割標籤, (0, 0, 0)代表黑色，(128, 0, 0)代表红色，(0, 128, 0)代表绿色 以此類推
colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
          (128, 0, 128), (0, 128, 128), (128, 128, 128),
          (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
          (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0),
          (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
          (128, 64, 12)]

# 處理整個資料夾
# detect_dir_image(dir_path, result_path)


model = keras.models.load_model('D:/KC_AOI_Project/wrk/study_wrk/Crosslogs/the-last-model.h5')
# 單張
image = detect_image(test_image_path)

plt.figure(figsize=(15, 15))
plt.imshow(image)
plt.axis('off')
plt.show()