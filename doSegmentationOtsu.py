# lebal完資料 SegmentationClassBMP   做otsu處理
# doSegmentationOtsu.py
import os
import cv2

# 設定來源資料夾路徑和目標資料夾路徑
source = 'C:/Windows/System32/labelme/examples/semantic_segmentation/data_output/SegmentationClassBMP/'
target_folder = 'C:/Windows/System32/labelme/examples/semantic_segmentation/data_output/SegmentationClassOTSU/'  # 請確認"AAA"資料夾存在或執行相關程式確保資料夾存在

# 取出路徑下所有圖片 並存成list
files = os.listdir(source)
print('※ 資料夾共有 {} 張圖檔'.format(len(files)))
print('※ 開始執行YOLOV4 test圖片最佳r次方 模糊化處理...')

number = 1
for file in files:
    print(' ▼ 第{}張'.format(number))
    print(' ▼ 檔名: {}'.format(file))
    # 分割檔名編號
    pic_no = file.split('.')

    # 讀取彩色圖片
    img = cv2.imread(source + file, 1)

    # 進行二值化處理，轉換成灰階影像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY)

    # 儲存處理後的圖片到"AAA"資料夾中，注意檔名需要與原始檔名保持一致
    output_path = os.path.join(target_folder, pic_no[0] + '.png')
    cv2.imwrite(output_path, binary_img)

    number += 1