# 10.给出图像处理中频域滤波法中理想低通滤波器、Butterworth低通滤波器、高斯低通滤波器实例各一个。
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.family'] = 'SimHei'  # 设置为黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义频域滤波器函数
def ideal_lowpass_filter(shape, cutoff):
    """理想低通滤波器"""
    rows, cols = shape
    center = (rows // 2, cols // 2)
    filter_mask = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            if distance <= cutoff:
                filter_mask[i, j] = 1
    return filter_mask

def butterworth_lowpass_filter(shape, cutoff, order=2):
    """Butterworth低通滤波器"""
    rows, cols = shape
    center = (rows // 2, cols // 2)
    filter_mask = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            filter_mask[i, j] = 1 / (1 + (distance / cutoff) ** (2 * order))
    return filter_mask

def gaussian_lowpass_filter(shape, cutoff):
    """高斯低通滤波器"""
    rows, cols = shape
    center = (rows // 2, cols // 2)
    filter_mask = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            filter_mask[i, j] = np.exp(-(distance ** 2) / (2 * (cutoff ** 2)))
    return filter_mask

# 应用频域滤波
def apply_filter(image, filter_mask):
    """对图像应用频域滤波"""
    dft = np.fft.fft2(image)
    dft_shifted = np.fft.fftshift(dft)
    filtered_dft = dft_shifted * filter_mask
    dft_inverse = np.fft.ifftshift(filtered_dft)
    filtered_image = np.abs(np.fft.ifft2(dft_inverse))
    return filtered_image

# 读取灰度图像
image = cv2.imread('../image/image9.png', cv2.IMREAD_GRAYSCALE)

# 定义滤波器参数
cutoff = 50
butterworth_order = 2

# 生成滤波器并应用
ideal_filter = ideal_lowpass_filter(image.shape, cutoff)
butterworth_filter = butterworth_lowpass_filter(image.shape, cutoff, butterworth_order)
gaussian_filter = gaussian_lowpass_filter(image.shape, cutoff)

ideal_result = apply_filter(image, ideal_filter)
butterworth_result = apply_filter(image, butterworth_filter)
gaussian_result = apply_filter(image, gaussian_filter)

# 显示结果
plt.figure(figsize=(12, 8))
plt.subplot(1, 4, 1), plt.imshow(image, cmap='gray')
plt.title("原图"), plt.axis('off')
plt.subplot(1, 4, 2), plt.imshow(ideal_result, cmap='gray')
plt.title("理想滤波图"), plt.axis('off')
plt.subplot(1, 4, 3), plt.imshow(butterworth_result, cmap='gray')
plt.title("Butterworth滤波图"), plt.axis('off')
plt.subplot(1, 4, 4), plt.imshow(gaussian_result, cmap='gray')
plt.title("高斯滤波图"), plt.axis('off')

plt.tight_layout()
plt.show()
