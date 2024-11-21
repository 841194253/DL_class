# 9.给出图像处理中空间域滤波的均值滤波、中值滤波和高斯滤波实例。
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.family'] = 'SimHei'  # 设置为黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 均值滤波
def mean_filter(image):
    return cv2.blur(image, (11, 11))

# 中值滤波
def median_filter(image):
    return cv2.medianBlur(image, 15)

# 高斯滤波
def gaussian_filter(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

# 显示所有滤波结果
def display_filters(image):
    # 进行滤波处理
    mean_img = mean_filter(image)
    median_img = median_filter(image)
    gaussian_img = gaussian_filter(image)

    # 创建一个包含四个子图的窗口
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 显示原图
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('原始图像')
    axes[0].axis('off')  # 关闭坐标轴

    # 显示均值滤波后的图像
    axes[1].imshow(mean_img, cmap='gray')
    axes[1].set_title('均值滤波')
    axes[1].axis('off')

    # 显示中值滤波后的图像
    axes[2].imshow(median_img, cmap='gray')
    axes[2].set_title('中值滤波')
    axes[2].axis('off')

    # 显示高斯滤波后的图像
    axes[3].imshow(gaussian_img, cmap='gray')
    axes[3].set_title('高斯滤波')
    axes[3].axis('off')

    # 显示结果
    plt.tight_layout()
    plt.show()

# 主函数
def main():
    # 读取图像
    image_path = "../image/image9.png"  # 替换为你的图像路径
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("图像读取失败！请检查路径是否正确。")
        return

    # 显示原图和滤波后的图像
    display_filters(image)

if __name__ == "__main__":
    main()

