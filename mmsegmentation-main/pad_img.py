import os.path

from PIL import Image
import numpy as np
import math
from tqdm import tqdm


def is_power_of_two(n):
    return (n & (n - 1)) == 0

def resize_img(img_path,save_path):
    """
    调整图像的尺寸，检查图像的长宽尺寸是否是2的幂次方，如果不是则使用0将其进行边界填充
    Args:
        img_path:图像所在文件夹目录
        save_path:图像的保存目录

    Returns:

    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    img_list = os.listdir(img_path)
    for image in tqdm(img_list):
        img_open = Image.open(os.path.join(img_path,image))
        width, height = img_open.size
        # 检查长宽是否是2的幂次方,如果是则返回，否则进行pad的填充
        if is_power_of_two(width) and is_power_of_two(height):
            print(f"Image size {width}x{height} is already a power of two. No padding needed.")
            img_open.save(os.path.join(save_path, image))
            return

            # 计算新的宽度和高度，使其为2的幂次方
        new_width = 2 ** math.ceil(math.log2(width))
        new_height = 2 ** math.ceil(math.log2(height))

        # 创建一个新的空白图片，用于填充
        new_img = Image.new(img_open.mode, (new_width, new_height), color=0)  # 使用0填充

        # 将原始图片粘贴到新图片的中心位置
        new_img.paste(img_open, ((new_width - width) // 2, (new_height - height) // 2))

        # 保存新图片
        new_img.save(os.path.join(save_path, image))
    print(f"Image has been padded to size {new_width}x{new_height}")


def pad_image_to_multiple_of_16(image_path, output_path,factor=16):
    # 打开图片
    img_list = os.listdir(image_path)
    img_name = img_list[0]
    img = Image.open(img_list[0])

    # 获取原始尺寸
    original_width, original_height = img.size

    # 计算新的宽度和高度，确保它们是16的倍数
    new_width = (original_width + factor-1) // factor * factor
    new_height = (original_height + factor-1) // factor * factor

    # 创建一个新的白色背景图像（或选择其他颜色）
    padded_img = Image.new(img.mode, (new_width, new_height), "white")

    # 将原始图像粘贴到新图像的中心
    padded_img.paste(img, ((new_width - original_width) // 2, (new_height - original_height) // 2))

    # # 进行一些操作（这里以保存为例）
    # padded_img.save(output_path)

    # 裁剪回原始尺寸
    cropped_img = padded_img.crop((
        (new_width - original_width) // 2,
        (new_height - original_height) // 2,
        (new_width - original_width) // 2 + original_width,
        (new_height - original_height) // 2 + original_height
    ))

    # 保存裁剪后的图像或进行其他操作
    output_path = os.path.join(output_path, img_name)
    # cropped_img.save(output_path)
    padded_img.save(output_path)



# 使用示例
# input_image_path = 'Image_01L.png'  # 替换为你的图片路径
# output_image_path = 'Image_01Lg.jpg'  # 替换为你想要保存的路径
# pad_to_power_of_two(input_image_path, output_image_path)
if __name__ == '__main__':
    # img_root = 'data/CHASE_DB1/annotations/training'
    # save_path = 'data/CHASE_DB1_pad/annotations/training'

    img_path = './img/test_img'
    save_path = './img/save_path'

    # resize_img(img_root,save_path)
    pad_image_to_multiple_of_16(img_path,save_path)

