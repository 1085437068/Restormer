import os
from PIL import Image

def crop_images_to_320x320(input_folder, output_folder):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 打开图片
            image_path = os.path.join(input_folder, filename)
            img = Image.open(image_path)

            # 获取图片的宽度和高度
            width, height = img.size

            # 计算裁剪区域
            left = (width - 320) // 2
            top = (height - 320) // 2
            right = (width + 320) // 2
            bottom = (height + 320) // 2

            # 裁剪图片
            cropped_img = img.crop((left, top, right, bottom))

            # 保存裁剪后的图片
            output_path = os.path.join(output_folder, filename)
            cropped_img.save(output_path)

# test图片
#input_folder = '/home/hzc/CodeRepository/UnderwaterEnhancement/DeepLearning/CNN/UWEnhancement/DATA/UIEB/Test/test/origin'
#output_folder = '/home/hzc/CodeRepository/UnderwaterEnhancement/DeepLearning/CNN/UWEnhancement/DATA/UIEB/Test/test'

# train图片 -- train文件夹
#input_folder = '/home/hzc/CodeRepository/UnderwaterEnhancement/DeepLearning/CNN/UWEnhancement/DATA/UIEB/Train/train/origin'
#output_folder = '/home/hzc/CodeRepository/UnderwaterEnhancement/DeepLearning/CNN/UWEnhancement/DATA/UIEB/Train/train'

# train图片 -- gt文件夹
input_folder = './data/test'
output_folder = './data'

crop_images_to_320x320(input_folder, output_folder)