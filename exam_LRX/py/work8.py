# 8.给出图像增强处理中的空间域增强的加法、减法和乘法的实例
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.family'] = 'SimHei'  # 设置为黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def enhance_image_addition(image, constant):
    """加法增强：将常数加到每个像素值"""
    enhanced_image = cv2.add(image, constant)
    return enhanced_image


def enhance_image_subtraction(image, constant):
    """减法增强：从每个像素值中减去常数"""
    enhanced_image = cv2.subtract(image, constant)
    return enhanced_image


def enhance_image_multiplication(image, constant):
    """乘法增强：将每个像素值与常数相乘"""
    enhanced_image = cv2.multiply(image, constant)
    return enhanced_image


def display_images_in_one_window(original_image, enhanced_addition, enhanced_subtraction, enhanced_multiplication):
    """在同一个窗口中显示原图和所有增强图像"""
    plt.figure(figsize=(12, 8))

    # 原图
    plt.subplot(2, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')

    # 加法增强
    plt.subplot(2, 2, 2)
    plt.imshow(enhanced_addition, cmap='gray')
    plt.title('加法增强')
    plt.axis('off')

    # 减法增强
    plt.subplot(2, 2, 3)
    plt.imshow(enhanced_subtraction, cmap='gray')
    plt.title('减法增强')
    plt.axis('off')

    # 乘法增强
    plt.subplot(2, 2, 4)
    plt.imshow(enhanced_multiplication, cmap='gray')
    plt.title('乘法增强')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    # 读取图像
    image = cv2.imread('../image/image9.png', cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("图像加载失败，请确保文件路径正确。")
        return

    # 选择常数值
    constant = 50

    # 加法增强
    enhanced_addition = enhance_image_addition(image, constant)

    # 减法增强
    enhanced_subtraction = enhance_image_subtraction(image, constant)

    # 乘法增强
    constant = 1.5  # 修改常数值以增强乘法效果
    enhanced_multiplication = enhance_image_multiplication(image, constant)

    # 在同一个窗口内显示原图和所有增强后的图像
    display_images_in_one_window(image, enhanced_addition, enhanced_subtraction, enhanced_multiplication)


if __name__ == "__main__":
    main()
