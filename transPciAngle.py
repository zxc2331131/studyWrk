# -*- coding: utf-8 -*-
# transPciAngle.py
# 將tensorflow-unet-labelme-master/tensorflow-unet-labelme-master/datasets/trainPlus/下的圖片分別旋轉 90 180 270度 以及上下翻轉
# 將處理後的資料存到路徑下暫時test資料夾 方便複製下來放到訓練資料夾中後刪除
"""
Created on Thu Sep 28 10:36:07 2023

@author: zihong
"""
from PIL import Image
import json
import os
import base64

# 读取LabelMe数据


def trans90Angle(root_path, do_angle):
    global max_num
    do_count=0
    for filename in os.listdir(root_path):
        do_count = do_count + 1
        if do_count <= file_count:
            if filename.endswith('.json'):  # 假设图像文件都是.json格式
                max_num = max_num + 1
                print(filename)
                with open(root_path + filename, 'r') as json_file:
                    labelme_data = json.load(json_file)
                parts = filename.split('.')
                image_path = root_path + parts[0] + '.jpg'  # 你的图像路径
                print(image_path)
                image = Image.open(image_path)
                print(4546)
                # 旋转图像
                rotated_image = image.rotate(do_angle, expand=True)

                # rotated_image_path = str(max_num) + '.jpg'
                # 為了要同一張圖片的所有翻轉角度都要在同個test or train名單
                rotated_image_path = parts[0]+'_'+str(do_angle) + '.jpg'
                rotated_image.save(root_path+'/test/' + rotated_image_path)
                print(rotated_image.size)
                # 更新LabelMe数据中的图像尺寸信息
                labelme_data['imageWidth'], labelme_data['imageHeight'] = rotated_image.size  

                # 更新标记的坐标
                for shape in labelme_data['shapes']:
                    for point in shape['points']:
                        point[0], point[1] = point[1], rotated_image.height - point[0]

                # 将图像数据转换为Base64编码的字符串
                with open(root_path+'/test/'+rotated_image_path, 'rb') as image_file:
                    image_data_base64 = base64.b64encode(image_file.read()).decode()

                # 更新ImageData字段
                labelme_data['imageData'] = image_data_base64
                # 更新imagePath字段
                labelme_data['imagePath'] = rotated_image_path

                # 保存旋转后的LabelMe数据
                # 為了要同一張圖片的所有翻轉角度都要在同個test or train名單
                with open(root_path+'/test/'+parts[0]+'_'+str(do_angle)+'.json', 'w') as rotated_json_file:
                    json.dump(labelme_data, rotated_json_file, indent=2)
                # with open(root_path+'/test/'+str(max_num)+'.json', 'w') as rotated_json_file:
                #     json.dump(labelme_data, rotated_json_file, indent=2)

                # 关闭原始图像
                image.close()
# 圖片上下翻轉 
def transTopBottom(root_path):
    global max_num
    do_count=0
    for filename in os.listdir(root_path):
        do_count = do_count + 1
        if do_count <= file_count:
            if filename.endswith('.json'):  # 假设图像文件都是.json格式
                max_num = max_num + 1
                
                with open(root_path + filename, 'r') as json_file:
                    labelme_data = json.load(json_file)
                parts = filename.split('.')
                image_path = root_path + parts[0] + '.jpg'  # 你的图像路径
                print(image_path)
                image = Image.open(image_path)
                # 上下翻转图像
                rotated_image = image.transpose(Image.FLIP_TOP_BOTTOM)

                # 為了要同一張圖片的所有翻轉角度都要在同個test or train名單
                rotated_image_path = parts[0]+'_tb' + '.jpg'
                # rotated_image_path = str(max_num) + '.jpg'
                rotated_image.save(root_path+'/test/' + rotated_image_path)
                print(rotated_image.size)
                # 更新LabelMe数据中的图像尺寸信息
                labelme_data['imageWidth'], labelme_data['imageHeight'] = rotated_image.size  

                # 更新标记的坐标
                for shape in labelme_data['shapes']:
                    for point in shape['points']:
                        point[0], point[1] = point[0], rotated_image.height - point[1]

                # 将图像数据转换为Base64编码的字符串
                with open(root_path+'/test/'+rotated_image_path, 'rb') as image_file:
                    image_data_base64 = base64.b64encode(image_file.read()).decode()

                # 更新ImageData字段
                labelme_data['imageData'] = image_data_base64
                # 更新imagePath字段
                labelme_data['imagePath'] = rotated_image_path

                # 保存旋转后的LabelMe数据
                # with open(root_path+'/test/'+str(max_num)+'.json', 'w') as rotated_json_file:
                #     json.dump(labelme_data, rotated_json_file, indent=2)
                # 為了要同一張圖片的所有翻轉角度都要在同個test or train名單
                with open(root_path+'/test/'+parts[0]+'_tb' +'.json', 'w') as rotated_json_file:
                    json.dump(labelme_data, rotated_json_file, indent=2)

                # 关闭原始图像
                image.close()
    


root_path = 'D:/tensorflow-unet-labelme-master/tensorflow-unet-labelme-master/datasets/trainPlus/'
file_count = len(os.listdir(root_path))
max_num = file_count
trans90Angle(root_path, 90)
trans90Angle(root_path, 180)
trans90Angle(root_path, 270)
transTopBottom(root_path)




