import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

def linear_transform(image, alpha=1.5, beta=50):
    """
    线性变换: g(x, y) = alpha * f(x, y) + beta
    :param image: 输入灰度图像
    :param alpha: 对比度增益
    :param beta: 亮度偏移
    :return: 线性变换后的图像
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def log_transform(image):
    """
    对数变换: g(x, y) = c * log(1 + f(x, y))
    :param image: 输入灰度图像
    :return: 对数变换后的图像
    """
    c = 255 / np.log(1 + np.max(image))
    log_image = c * np.log(1 + image.astype(np.float32))
    return np.uint8(log_image)


def gamma_transform(image, gamma=2.0):
    """
    伽马变换: g(x, y) = c * f(x, y)^gamma
    :param image: 输入灰度图像
    :param gamma: 伽马值
    :return: 伽马变换后的图像
    """
    c = 255.0 / (np.max(image) ** gamma)
    gamma_image = c * (image.astype(np.float32) ** gamma)
    return np.uint8(gamma_image)


def main():
    # 读取灰度图像
    image_path = '../image/image9.png'  # 替换为你的图片路径
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"无法读取图像文件：{image_path}")
        return

    # 线性变换
    linear_image = linear_transform(image, alpha=1.5, beta=50)

    # 对数变换
    log_image = log_transform(image)

    # 伽马变换
    gamma_image = gamma_transform(image, gamma=2.0)

    # 可视化结果
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1), plt.title("原图片"), plt.imshow(image, cmap='gray'), plt.axis('off')
    plt.subplot(2, 2, 2), plt.title("线性变换"), plt.imshow(linear_image, cmap='gray'), plt.axis('off')
    plt.subplot(2, 2, 3), plt.title("对数变换"), plt.imshow(log_image, cmap='gray'), plt.axis('off')
    plt.subplot(2, 2, 4), plt.title("伽马变换"), plt.imshow(gamma_image, cmap='gray'), plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
